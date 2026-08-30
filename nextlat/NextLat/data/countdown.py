import numpy as np
import torch
import lightning as L
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
import re


def check_eq(left_str, right_str, available_nums):
    left_matches = re.match(r"(\d+)([+\-*/])(\d+)", left_str)
    # get the numbers involved
    used_numbers = re.findall(r"\d+", left_str)
    for n in used_numbers:
        # invalid number used
        if n not in available_nums:
            return False
    if left_matches:
        return eval(left_str) == float(right_str)
    else:
        return False


class Tokenizer:
    def __init__(self, max_intermediate):
        self.max_intermediate = max_intermediate
        self.encoder = {str(i): i for i in range(max_intermediate)}
        # pipe sign will divide context and output equations
        # it will also be our pause token
        self.encoder["|"] = max_intermediate
        self.encoder["*"] = max_intermediate + 1
        self.encoder["/"] = max_intermediate + 2
        self.encoder["+"] = max_intermediate + 3
        self.encoder["-"] = max_intermediate + 4
        self.encoder["="] = max_intermediate + 5
        self.encoder[","] = max_intermediate + 6
        self.encoder[""] = max_intermediate + 7

        self.decoder = {i: str(i) for i in range(max_intermediate)}
        self.decoder[max_intermediate] = "|"
        self.decoder[max_intermediate + 1] = "*"
        self.decoder[max_intermediate + 2] = "/"
        self.decoder[max_intermediate + 3] = "+"
        self.decoder[max_intermediate + 4] = "-"
        self.decoder[max_intermediate + 5] = "="
        self.decoder[max_intermediate + 6] = ","
        self.decoder[max_intermediate + 7] = ""

        self.numbers = set("0123456789")

        # countdown has no eos token, padded sequence masking
        # doesn't need to happen, but this property is needed
        # for compatibility with the bst training code
        self.eos_token_id = max_intermediate + 7

    def encode(self, data, num_pause_tokens=0):
        out = []
        i = 0
        seen_pipe = False
        while i < len(data):
            if data[i] == "," and not seen_pipe:
                i += 1
                continue
            elif data[i] == "|":
                seen_pipe = True
            s = ""
            while i < len(data) and data[i] in self.numbers:
                s += data[i]
                i += 1
            if s:
                out.append(self.encoder[s])
            elif data[i] == "|":
                # insert pause tokens between context and output equations
                for _ in range(num_pause_tokens):
                    out.append(self.encoder["|"])
                i += 1
            else:
                out.append(self.encoder[data[i]])
                i += 1

        return out

    def decode(self, tokens, include_comma=False):
        """Decode a tensor of token indices back to the original sequence"""
        out = ""
        for token_id in tokens:
            # Find the character/token that corresponds to this token_id
            out += self.decoder[token_id.item()]
            if include_comma:
                out += ","
        if include_comma:
            out = out[:-1]
        return out

    def tokenize(self, prefix, num_pause_tokens):
        prefix_tokens = self.encode(prefix, num_pause_tokens)
        prefix_tokens.append(self.eos_token_id)

        seq = np.array(prefix_tokens)

        return seq, len(seq)


class CountdownDataset(torch.utils.data.Dataset):
    def __init__(
        self, data, tokenizer, max_new_tokens, num_equations, num_pause_tokens
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.num_equations = num_equations
        self.num_pause_tokens = num_pause_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].strip()
        seq, _ = self.tokenizer.tokenize(line, self.num_pause_tokens)
        x = torch.tensor(seq, dtype=torch.long)
        return x

    # Modularized evaluation function for Countdown
    def evaluate_countdown(
        self,
        model,
        dataloader,
        config,
        fabric,
        show_progress_bar=True,
        generalization=False,
    ):
        """
        Evaluate Countdown sequence accuracy. Returns a dict with overall and per-equation accuracy.
        """
        model.eval()
        test_logs = {}
        batch_count = 0
        total_correct = torch.zeros(1, device=fabric.device)
        num_sequences = torch.zeros(1, device=fabric.device)
        total_valid_equations = torch.zeros(
            (self.num_equations, 1), device=fabric.device
        )
        prefix_label = "generalization/" if generalization else "val/"

        disable_pbar = True if not show_progress_bar else None
        n_test_batches = None
        if hasattr(config.trainer, "test_batches") and config.trainer.test_batches > 0:
            n_test_batches = config.trainer.test_batches // fabric.world_size

        for batch in tqdm(
            dataloader,
            desc="Countdown Evaluation",
            total=n_test_batches or len(dataloader),
            leave=False,
            disable=disable_pbar,
        ):
            # No special prepare_batch needed for Countdown
            # Assume batch shape: (batch_size, seq_len)
            batch_size, seq_len = batch.shape
            num_target_tokens = self.max_new_tokens
            tokens_including_eos = num_target_tokens + 1
            prefix = batch[:, :-tokens_including_eos].clone()
            target = batch[:, -tokens_including_eos:-1].clone()

            with torch.inference_mode():
                predictions = model.generate(
                    prefix,
                    max_new_tokens=num_target_tokens,
                    temperature=getattr(config.trainer, "temperature", 1.0),
                    top_k=getattr(config.trainer, "top_k", None),
                )
                predictions = predictions[:, -num_target_tokens:]  # remove prefix

            for i in range(batch_size):
                pred = self.tokenizer.decode(predictions[i])
                try:
                    subequations = pred.split(",")  # sub-equations
                except:
                    print(f"Error: {pred}")
                    continue

                match = True
                available_nums = set(
                    self.tokenizer.decode(prefix[i, :-1], include_comma=True).split(
                        ","
                    )[:-1]
                )
                for j, subeq in enumerate(subequations):
                    try:
                        left, right = subeq.split("=")
                        match &= check_eq(left, right, available_nums)
                    except:
                        print(f"Could not split operation into lhs, rhs: {subeq}")
                        match = False
                    if not match:
                        break
                    available_nums.add(right)
                    total_valid_equations[j] += 1

                # last token of target is the answer
                answer = self.tokenizer.decode(target[i]).split("=")[-1]
                # last equation should generate the answer after equals sign
                pred_answer = pred.split("=")[-1]

                # if all equations were valid and the last equation generated the correct answer,
                # then it's correct
                correct = match and (answer == pred_answer)
                total_correct += correct

            num_sequences += batch_size
            batch_count += 1
            if n_test_batches is not None and batch_count >= n_test_batches:
                break

            # print some samples
            # prefix[-1, :-1] => last sample without pipe sign
            print(
                f"Sample {batch_count}: {self.tokenizer.decode(prefix[-1, :-1], include_comma=True)} | Generation: {self.tokenizer.decode(predictions[-1], include_comma=False)} | Correct: {correct}"
            )
            print(f"Available numbers: {available_nums}")

        global_total_correct = fabric.all_reduce(total_correct, reduce_op="sum")
        global_num_sequences = fabric.all_reduce(num_sequences, reduce_op="sum")
        overall_accuracy = (global_total_correct / global_num_sequences).item()

        per_equation_valid = {}
        for i in range(self.num_equations):
            global_token_correct = fabric.all_reduce(
                total_valid_equations[i], reduce_op="sum"
            )
            per_equation_valid[f"{prefix_label}valid_equation_{i+1}"] = (
                global_token_correct / global_num_sequences
            ).item()

        test_logs[f"{prefix_label}test_accuracy"] = overall_accuracy
        test_logs.update(per_equation_valid)

        fabric.print(
            f"{datetime.now()} Countdown Test Accuracy ({prefix_label}): {overall_accuracy:.2%}"
        )

        return test_logs


class CountdownDataModule:
    """
    PyTorch Lightning style DataModule for Countdown dataset
    """

    def __init__(self, fabric: L.Fabric, config):
        self.fabric = fabric
        self.batch_size = config.data.device_batch_size

        # Load data
        with open(config.data.train_countdown_data_path, "r") as f:
            train_data = f.readlines()

        with open(config.data.val_countdown_data_path, "r") as f:
            val_data = f.readlines()

        # each equation has 5 tokens (2 numbers, 1 operator, 1 equals sign, 1 answer)
        # each equation is separated by a comma sign
        self.num_pause_tokens = (
            config.data.num_pause_tokens
        )  # number of pause tokens between context and output equations
        num_equations = config.data.num_equations
        num_target_tokens = 5 * num_equations + (num_equations - 1)

        # Create tokenizer
        max_intermediate = config.data.countdown_max_intermediate
        self.tokenizer = Tokenizer(max_intermediate)

        context_len, total_len = self._measure_index(train_data)
        print(f"[Countdown] Prefix Length: {context_len}, Sequence Length: {total_len}")

        # Create datasets
        self.train_dataset = CountdownDataset(
            train_data,
            self.tokenizer,
            num_target_tokens,
            num_equations,
            self.num_pause_tokens,
        )
        self.val_dataset = CountdownDataset(
            val_data,
            self.tokenizer,
            num_target_tokens,
            num_equations,
            self.num_pause_tokens,
        )

        if config.data.test_generalization:
            with open(config.data.generalization_countdown_data_path, "r") as f:
                generalization_data = f.readlines()

            self.generalization_dataset = CountdownDataset(
                generalization_data,
                self.tokenizer,
                num_target_tokens,
                num_equations,
                self.num_pause_tokens,
            )

        self.vocab_size = max_intermediate + 7 + 1  # Total vocabulary size
        self.context_len = context_len
        self.total_len = total_len

    def _measure_index(self, data):
        line = data[0]
        line = line.strip()  # Remove any trailing whitespace
        prefix = line.split("|")[0] + "|"
        _, prefix_len = self.tokenizer.tokenize(
            prefix, self.num_pause_tokens
        )  # Tokenize the prefix
        _, seq_len = self.tokenizer.tokenize(
            line, self.num_pause_tokens
        )  # Tokenize the entire line

        # one for eos token at end of sequence generated by tokenize(),
        # and one so it appears on index of pipe sign (if num_pause_tokens > 0),
        # else it appears on the last available number.
        context_len = prefix_len - 2

        return context_len, seq_len

    def update_config(self, config):
        config.model.vocab_size = self.vocab_size
        config.model.context_length = self.context_len
        config.model.block_size = self.total_len

    def get_tokenizer(self):
        return self.tokenizer

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        # This will automatically partition data between devices
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        # This will automatically partition data between devices
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)

    def generalization_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.generalization_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)

    @staticmethod
    def collate_fn(batch):
        x_batch = torch.stack([item for item in batch])
        return x_batch
