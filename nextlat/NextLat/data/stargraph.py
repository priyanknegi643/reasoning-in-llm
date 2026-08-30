import numpy as np
import torch
import lightning as L
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime


class Tokenizer:
    def __init__(self, maxNodes):
        self.maxNodes = maxNodes
        self.encoder = {str(i): i for i in range(maxNodes)}
        self.encoder["|"] = maxNodes
        self.encoder["="] = maxNodes + 1
        self.encoder["/"] = maxNodes + 2
        self.encoder["$"] = maxNodes + 3  # Padding token

        self.decoder = {i: str(i) for i in range(maxNodes)}
        self.decoder[maxNodes] = "|"
        self.decoder[maxNodes + 1] = "="
        self.decoder[maxNodes + 2] = "/"
        self.decoder[maxNodes + 3] = "$"
        self.decoder[maxNodes + 4] = ""

        self.numbers = set("0123456789")

        # stargraph has no eos token, padded sequence masking
        # doesn't need to happen, but this property is needed
        # for compatibility with the bst training code
        self.eos_token_id = maxNodes + 4

    def encode(self, data):
        out = []
        i = 0
        while i < len(data):
            if data[i] == ",":
                i += 1
                continue
            s = ""
            while i < len(data) and data[i] in self.numbers:
                s += data[i]
                i += 1
            if s:
                out.append(self.encoder[s])
            else:
                out.append(self.encoder[data[i]])
                i += 1

        return out

    def tokenize(self, prefix):
        prefix_tokens = self.encode(prefix)
        prefix_tokens.append(self.eos_token_id)

        seq = np.array(prefix_tokens)

        return seq, len(seq)


class StarGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, graph_target_len, num_arms=None):
        self.data = data
        self.tokenizer = tokenizer
        self.graph_target_len = graph_target_len
        self.num_arms = num_arms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].strip()
        seq, _ = self.tokenizer.tokenize(line)
        x = torch.tensor(seq, dtype=torch.long)
        return x

    # Modularized evaluation function for StarGraph
    def evaluate_stargraph(
        self,
        model,
        dataloader,
        config,
        fabric,
        show_progress_bar=True,
        generalization=False,
    ):
        """
        Evaluate StarGraph sequence accuracy. Returns a dict with overall and per-token accuracy.
        """
        model.eval()
        test_logs = {}
        batch_count = 0
        total_correct = torch.zeros(1, device=fabric.device)
        total_tokens = torch.zeros(1, device=fabric.device)
        tokens_correct = {}
        prefix_label = (
            f"generalization_({self.num_arms}, {self.graph_target_len})/"
            if generalization
            else f"val_({self.num_arms}, {self.graph_target_len})/"
        )

        disable_pbar = True if not show_progress_bar else None
        n_test_batches = None
        if hasattr(config.trainer, "test_batches") and config.trainer.test_batches > 0:
            n_test_batches = config.trainer.test_batches // fabric.world_size

        for batch in tqdm(
            dataloader,
            desc="StarGraph Evaluation",
            total=n_test_batches or len(dataloader),
            leave=False,
            disable=disable_pbar,
        ):
            # No special prepare_batch needed for StarGraph
            # Assume batch shape: (batch_size, seq_len)
            batch_size, seq_len = batch.shape
            num_target_tokens = self.graph_target_len
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

            correct = target.eq(predictions[:, -num_target_tokens:]).float()
            completely_correct = correct.sum(dim=1).eq(num_target_tokens).float()
            total_correct += completely_correct.sum()
            total_tokens += batch_size

            per_token_acc = correct.mean(dim=0)
            for i in range(num_target_tokens):
                if i not in tokens_correct:
                    tokens_correct[i] = torch.zeros(1, device=fabric.device)
                tokens_correct[i] += per_token_acc[i] * batch_size

            batch_count += 1
            if n_test_batches is not None and batch_count >= n_test_batches:
                break

        global_total_correct = fabric.all_reduce(total_correct, reduce_op="sum")
        global_total_tokens = fabric.all_reduce(total_tokens, reduce_op="sum")
        overall_accuracy = (global_total_correct / global_total_tokens).item()

        per_token_accuracy = {}
        for i in tokens_correct:
            global_token_correct = fabric.all_reduce(tokens_correct[i], reduce_op="sum")
            per_token_accuracy[f"{prefix_label}token_{i+1}"] = (
                global_token_correct / global_total_tokens
            ).item()

        test_logs[f"{prefix_label}test_accuracy"] = overall_accuracy
        test_logs.update(per_token_accuracy)

        fabric.print(
            f"{datetime.now()} StarGraph Test Accuracy ({prefix_label}): {overall_accuracy:.2%}"
        )

        return test_logs


class StarGraphDataModule:
    """
    PyTorch Lightning style DataModule for StarGraph dataset
    """

    def __init__(self, fabric: L.Fabric, config):
        self.fabric = fabric
        self.batch_size = config.data.device_batch_size

        # Create tokenizer
        maxNodes = config.data.stargraph_max_nodes
        self.tokenizer = Tokenizer(maxNodes)

        # Load data
        with open(config.data.stargraph_train_data_path, "r") as f:
            train_data = f.readlines()

        with open(config.data.stargraph_test_data_path, "r") as f:
            val_data = f.readlines()

        graph_description_len, total_len = self._measure_index(train_data)
        num_target_tokens = total_len - graph_description_len - 2
        num_arms = int(config.data.stargraph_train_data_path.split("_")[1])
        assert num_target_tokens == int(
            config.data.stargraph_train_data_path.split("_")[2]
        ), f"num_target_tokens {num_target_tokens} does not match name in file: {config.data.stargraph_train_data_path}"

        print(
            f"[Stargraph ({num_arms}, {num_target_tokens})] Prefix Length: {graph_description_len}, Sequence Length: {total_len}"
        )

        self.train_dataset = StarGraphDataset(
            train_data, self.tokenizer, num_target_tokens, num_arms
        )
        self.val_dataset = StarGraphDataset(
            val_data, self.tokenizer, num_target_tokens, num_arms
        )

        if config.data.test_generalization:
            # Load data
            self.generalization_datasets = []
            for file in config.data.stargraph_generalization_data_path:
                with open(file, "r") as f:
                    generalization_data = f.readlines()

                (
                    generalization_graph_description_len,
                    generalization_total_len,
                ) = self._measure_index(generalization_data)
                num_target_tokens = (
                    generalization_total_len - generalization_graph_description_len - 2
                )
                num_arms = int(file.split("_")[1])
                assert num_target_tokens == int(
                    file.split("_")[2]
                ), f"num_target_tokens {num_target_tokens} does not match name in file: {file}"
                self.generalization_datasets.append(
                    StarGraphDataset(
                        generalization_data, self.tokenizer, num_target_tokens, num_arms
                    )
                )
                total_len = max(
                    total_len, generalization_total_len
                )  # expand total length to include generalization data
                print(
                    f"[Stargraph Generalization ({num_arms}, {num_target_tokens})] Prefix Length: {generalization_graph_description_len}, Sequence Length: {generalization_total_len}"
                )

        self.vocab_size = maxNodes + 5 + 1  # Total vocabulary size
        self.graph_description_len = graph_description_len
        self.total_len = total_len

    def _measure_index(self, data):
        line = data[0]
        line = line.strip()  # Remove any trailing whitespace
        prefix = line.split("=")[0] + "="
        _, prefix_len = self.tokenizer.tokenize(prefix)  # Tokenize the prefix
        _, seq_len = self.tokenizer.tokenize(line)  # Tokenize the entire line

        # one for beginning special token 104 (eos token) at end of sequence, one so it appears on index of equals sign.
        graph_description_len = prefix_len - 2

        return graph_description_len, seq_len

    def update_config(self, config):
        config.model.vocab_size = self.vocab_size
        config.model.context_length = self.graph_description_len
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
        dataloaders = []
        for dataset in self.generalization_datasets:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
                collate_fn=self.collate_fn,
            )
            dataloaders.append(
                self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)
            )
        return dataloaders

    @staticmethod
    def collate_fn(batch):
        x_batch = torch.stack([item for item in batch])
        return x_batch
