"""contains core training/inference logic used for multiple datasets/tasks"""

import gc
import inspect
import torch
import lightning as L
from datetime import datetime
from lightning.fabric.strategies.model_parallel import ModelParallelStrategy
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Callable, Dict, Optional, Tuple
from models.model_base import ModelBase
from models.model_probe import ProbeModule, ProbeConfig


def initialize_probe_model(
    fabric: L.Fabric,
    config,
    frozen_model,
    tokenizer,
    initialize_optimizer=True,
):
    fabric.print(f"Initializing model with config: {config}")

    # start with model_args from command line
    model_args = dict(
        n_embd=config.model.n_embd,
        bias=config.model.bias,
        vocab_size=config.model.vocab_size,
        dropout=config.model.dropout,
        eos_token_id=tokenizer.eos_token_id,
        use_fused=config.trainer.use_fused_kernels,
        probe_depth=config.trainer.probe_depth,
    )

    model_config = ProbeConfig(**model_args)

    with fabric.init_module():
        model = ProbeModule(model_config, frozen_model)

    # Compile must occur before fabric.setup()
    if config.trainer.compile:
        fabric.print("Compiling model")
        model.compile()

    # FSDP requires model to be setup before initializing the optimizer
    if isinstance(fabric.strategy, ModelParallelStrategy):
        model.setup_fabric(fabric)

    # Initialize optimizer
    if initialize_optimizer:
        is_device_cuda = fabric.device.type == "cuda"
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        # This is native PyTorch fused optimizer, not use_fused_kernels in config
        use_fused = fused_available and is_device_cuda

        model.configure_optimizers(
            weight_decay=config.optimizer.weight_decay,
            learning_rate=config.optimizer.learning_rate,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            use_fused=use_fused,
            optimizer_type=config.optimizer.get("optimizer_type", "adam"),
            mu=config.optimizer.get("mu", 0.95),
            nesterov=config.optimizer.get("nesterov", True),
        )

    # Setup model and optimizer
    # In the FSDP case, model is already setup and this sets up only the optimizer
    model.setup_fabric(fabric)

    return model


class ProbeTrainer:
    """Trainer class for probing hidden states of a frozen GPT/NextLat models"""

    def __init__(
        self,
        fabric: L.Fabric,
        config,
        model: ModelBase,
        show_progress_bar: bool = True,
    ):
        # Check that model has optimizer
        assert (
            hasattr(model, "optimizer") and model.optimizer is not None
        ), "Model must be initialized with optimizer"

        # Initialization state
        self.fabric = fabric
        self.config = config
        self.model = model
        self.show_progress_bar = show_progress_bar

        # Training loop state
        self.train_dataloader: DataLoader = None
        self.val_dataloader: DataLoader = None
        self.tokenizer = None
        self.prepare_batch_func: Optional[Callable] = None
        self.epoch: int = 0
        self.step: int = self.model.training_steps
        self.last_train_loss: Optional[float] = None
        self.last_hidden_state_loss: Optional[float] = None
        self.last_logits_loss: Optional[torch.Tensor] = None

    def train(self, datamodule):
        """
        Call this to start training.
        """
        # Initialize dataloaders
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.tokenizer = datamodule.get_tokenizer()

        # Check if datamodule has a prepare_batch function
        if hasattr(datamodule, "prepare_batch"):
            self.prepare_batch_func = datamodule.prepare_batch
            self.fabric.print(
                f"Using prepare batch function {type(datamodule).__name__}.prepare_batch()"
            )
        else:
            self.prepare_batch_func = None

        # Do training or validation
        if self.config.trainer.val_only:
            self.fabric.print("Running validation only, not training model")
            validation_logs = self._validation_loop()
            self.fabric.print(validation_logs)
        else:
            self.fabric.print(f"Starting training from step={self.step}")
            self._train_loop()

            self.fabric.print("Training complete, running final validation")
            val_logs = self._validation_loop()

    def _train_loop(self):
        """
        Main training loop.
        Iterate over the train dataloader for the configured number of batches.

        For each batch:
            - If we have reached the validation interval, run validation
            - Split batch into micro batches and accumulate gradients
            - Run optimizer step
            - Do logging
        """
        # Set model to train mode
        self.model.train()

        # Sum of train loss over log_interval batches
        running_train_loss = torch.zeros(1, device=self.fabric.device)
        running_mse_loss = torch.zeros(1, device=self.fabric.device)
        running_logits_loss = torch.zeros(
            self.config.trainer.probe_depth, device=self.fabric.device
        )
        # Initialize progress bar
        if self.show_progress_bar:
            pbar = tqdm(total=self.config.trainer.val_interval, leave=False)

        while True:
            # Do one epoch over the entire training dataloader
            for batch in self.train_dataloader:
                # Garbage collect to free up memory
                if (
                    self.config.trainer.garbage_collect > 0
                    and self.step % self.config.trainer.garbage_collect == 0
                ):
                    gc.collect()

                # Validation
                validation_logs = {}
                if (
                    self.step % self.config.trainer.val_interval == 0 and self.step > 0
                ) or self.config.trainer.val_only:
                    # Close the previous training progress bar
                    if self.show_progress_bar:
                        pbar.close()

                    validation_logs = self._validation_loop()
                    self.model.train()

                    if self.config.trainer.garbage_collect > 0:
                        gc.collect()

                    # Initialize a new progress bar
                    if self.show_progress_bar:
                        pbar = tqdm(total=self.config.trainer.val_interval, leave=False)

                # Prepare batch
                if self.prepare_batch_func is not None:
                    batch = self.prepare_batch_func(batch)

                # Accumulate gradients over gradient_accum_steps
                for accum_step in range(self.config.data.gradient_accum_steps):
                    start_idx = accum_step * self.config.data.micro_batch_size
                    end_idx = (accum_step + 1) * self.config.data.micro_batch_size
                    sub_batch = batch[start_idx:end_idx]

                    # Only sync gradients for the last step
                    no_sync = accum_step < self.config.data.gradient_accum_steps - 1
                    loss, logits_loss, mse_loss = self.model.compute_loss(
                        sub_batch,
                        pair_batch_size=self.config.data.pair_batch_size,
                        backpropagate=True,
                        loss_div=self.config.data.gradient_accum_steps,
                        no_sync=no_sync,
                    )
                    running_train_loss += loss
                    running_logits_loss += logits_loss
                    running_mse_loss += mse_loss

                # Optimizer step
                # This should increment model.training_steps
                lr = self._get_lr()
                grad_clip = self.config.optimizer.grad_clip
                grad_norm = self.model.optimizer_step(grad_clip=grad_clip)
                # Logging
                if self.step % self.config.trainer.log_interval == 0 and self.step > 0:
                    # Train loss has been accumulated over log_interval batches
                    self.last_train_loss = (
                        running_train_loss.item() / self.config.trainer.log_interval
                    )
                    running_train_loss.zero_()
                    self.last_logits_loss = (
                        running_logits_loss / self.config.trainer.log_interval
                    )
                    running_logits_loss = torch.zeros_like(running_logits_loss)
                    self.last_mse_loss = (
                        running_mse_loss.item() / self.config.trainer.log_interval
                    )
                    running_mse_loss.zero_()
                    log_dict = {
                        "step": self.step,
                        "epoch": self.epoch,
                        "lr": lr,
                        "train/loss": self.last_train_loss,
                        **{
                            f"train/logits_loss_{i}": self.last_logits_loss[i]
                            for i in range(self.config.trainer.probe_depth)
                        },
                        "train/mse_loss": self.last_mse_loss,
                        **validation_logs,
                    }
                    if grad_norm is not None:
                        log_dict["grad_norm"] = grad_norm

                    self.fabric.log_dict(log_dict)

                # Update progress bar
                progress_str = f"{datetime.now()} Step: {self.step} Epoch: {self.epoch} Train loss: {self.last_train_loss or 0.0:.4f}"
                if self.show_progress_bar:
                    pbar.set_description(progress_str)
                    pbar.update(1)
                else:
                    self.fabric.print(progress_str)

                # Increment batch count
                self.step += 1
                assert (
                    self.step == self.model.training_steps
                ), "Bug in training loop: trainer step count does not match model step count"
                if self.step > self.config.trainer.train_batches:
                    # End of training
                    return

            # End of one full iteration over train dataloader
            self.epoch += 1

    @torch.inference_mode()
    def _validation_loop(self) -> Dict[str, Any]:
        """
        Main validation loop.
        This is called every val_interval steps during training.

        For validation, we do the following:
            - Compute validation loss
            - For BST, compute next/prev token prediction loss
            - Save checkpoint
        """
        # Set model to eval mode
        self.model.eval()

        if self.config.trainer.val_batches > 0:
            # Divide total validation batches by world size
            n_val_batches = self.config.trainer.val_batches // self.fabric.world_size
        else:
            # If validation batches are not specified, validate over entire dataloader
            n_val_batches = None

        validation_logs = {}
        batch_count = 0
        total_loss = torch.zeros(1, device=self.fabric.device)
        total_logits_loss = torch.zeros(
            self.config.trainer.probe_depth, device=self.fabric.device
        )
        total_mse_loss = torch.zeros(1, device=self.fabric.device)
        disable_pbar = (
            True if not self.show_progress_bar else None
        )  # None means automatic

        # Main validation loop
        for batch in tqdm(
            self.val_dataloader,
            desc="Validation",
            total=n_val_batches or len(self.val_dataloader),
            leave=False,
            disable=disable_pbar,
        ):
            if self.prepare_batch_func is not None:
                batch = self.prepare_batch_func(batch)

            # Accumulate loss over gradient_accum_steps
            for accum_step in range(self.config.data.gradient_accum_steps):
                start_idx = accum_step * self.config.data.micro_batch_size
                end_idx = (accum_step + 1) * self.config.data.micro_batch_size
                sub_batch = batch[start_idx:end_idx]

                loss, logits_loss, mse_loss = self.model.compute_loss(
                    sub_batch,
                    pair_batch_size=self.config.data.pair_batch_size,
                    backpropagate=False,
                    loss_div=self.config.data.gradient_accum_steps,
                    return_logits=True,
                    eval_mode=True,
                )
                total_loss += loss
                total_logits_loss += logits_loss
                total_mse_loss += mse_loss

            # End loop if limit_batches is reached
            batch_count += 1
            if n_val_batches is not None and batch_count >= n_val_batches:
                break

        # Sync average loss across all GPUs by uniform mean of per-device averages
        # This assumes that each GPU has roughly the same number of batches
        device_avg_loss = total_loss / batch_count
        device_avg_logits_loss = total_logits_loss / batch_count
        device_avg_mse_loss = total_mse_loss / batch_count
        global_avg_loss = self.fabric.all_reduce(device_avg_loss, reduce_op="mean")
        val_loss = global_avg_loss.item()
        validation_logs["val/loss"] = val_loss
        global_avg_mse_loss = self.fabric.all_reduce(
            device_avg_mse_loss, reduce_op="mean"
        )
        validation_logs["val/mse_loss"] = global_avg_mse_loss.item()
        for i in range(self.config.trainer.probe_depth):
            global_avg_logits_loss = self.fabric.all_reduce(
                device_avg_logits_loss[i], reduce_op="mean"
            )
            validation_logs[f"val/logits_loss_{i}"] = global_avg_logits_loss.item()

        self.fabric.print(
            f"Step: {self.step} Epoch: {self.epoch} Train loss: {self.last_train_loss} Validation loss: {val_loss}"
        )

        return validation_logs

    def _get_lr(self) -> float:
        """
        Learning rate scheduler:
        - constant LR when warmup_iters and warmdown_iters are both 0
        - warmup -> hold -> linear warmdown otherwise
        """
        iter_num = self.step
        warmup_iters = self.config.lr_scheduler.warmup_iters
        warmdown_iters = self.config.lr_scheduler.get("warmdown_iters", 0)
        learning_rate = self.config.optimizer.learning_rate
        train_batches = self.config.trainer.train_batches

        # Constant LR mode.
        if warmup_iters <= 0 and warmdown_iters <= 0:
            return learning_rate

        # 1) linear warmup to base LR
        if iter_num < warmup_iters:
            return learning_rate * iter_num / max(warmup_iters, 1)

        # 2) hold at base LR if warmdown is disabled.
        if warmdown_iters <= 0:
            return learning_rate

        # 3) hold until warmdown starts, then linearly warm down to 0.
        warmdown_start = max(warmup_iters, train_batches - warmdown_iters)
        if iter_num < warmdown_start:
            return learning_rate
        if iter_num <= train_batches:
            warmdown_progress = iter_num - warmdown_start
            warmdown_ratio = warmdown_progress / max(warmdown_iters, 1)
            warmdown_ratio = min(max(warmdown_ratio, 0.0), 1.0)
            return learning_rate * (1.0 - warmdown_ratio)
        return 0.0
