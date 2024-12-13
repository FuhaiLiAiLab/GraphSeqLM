import os
import torch
from tqdm import tqdm
from typing import List
from .dna_gpt.dna_gpt import DNAGPT
from .dna_gpt.tokenizer import KmerTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class DNAGPT_LM:
    def __init__(self, model_path: str ="", model_name: str = "dna_gpt0.1b_h", device: str = "cpu"):
        """
        DNAGPT-based language model.
        Args:
            model_name (str): Name of the DNAGPT model. Defaults to 'dna_gpt0.1b_h'.
            device (str): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def get_model(self, model_name):
        """
        Initialize the DNAGPT model and tokenizer.
        """
        special_tokens = (
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] +
            ["+", '-', '*', '/', '=', "&", "|", "!"] +
            ['M', 'B'] + ['P'] + ['R', 'I', 'K', 'L', 'O', 'Q', 'S', 'U', 'V'] + ['W', 'Y', 'X', 'Z']
        )
        if model_name in ('dna_gpt0.1b_h'):
            tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=False)
        else:
            tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=True)

        vocab_size = len(tokenizer)
        model = DNAGPT.from_name(model_name, vocab_size)
        return model, tokenizer

    def load_model(self):
        """
        Load the DNAGPT model and tokenizer from the specified model name.
        """
        self.model, self.tokenizer = self.get_model(self.model_name)
        weight_path = os.path.join(self.model_path, f"{self.model_name}.pth")
        self._load_model_weights(weight_path)
        return self

    def _load_model_weights(self, weight_path, dtype=None):
        """
        Load the model weights from a checkpoint file.
        """
        state = torch.load(weight_path, map_location="cpu")
        if 'model' in state.keys():
            self.model.load_state_dict(state['model'], strict=False)
        else:
            self.model.load_state_dict(state, strict=False)
        print(f"loading model weights from {weight_path}")
        self.model.to(device=self.device, dtype=dtype)
        self.model.eval()
    
    def generate_embeddings(self, sequences: List[str], max_len: int = 256) -> torch.Tensor:
        """
        Generate tensor embeddings for each DNA sequence. 
        Args:
            sequences (List[str]): List of DNA sequences.
            max_len (int): Maximum length of the sequences. Defaults to 256.
        Returns:
            torch.Tensor: A tensor of shape (num_sequences, embedding_dim) where
                        num_sequences is the number of sequences and
                        embedding_dim is the dimension of the reduced embeddings.
        """
        print(f"Generating embeddings for {len(sequences)} sequences.")
        device = self.device
        embeddings = []

        for sequence in tqdm(sequences, desc="Embedding sequences", unit="sequence"):
            # Tokenize and prepare input tensors
            input_ids = self.tokenizer.encode(sequence[0], max_len=max_len, device=device)[None, :]  # Add batch dimension
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            max_new_tokens = max_len - input_ids.shape[1]

            # Forward pass to compute embeddings
            with torch.no_grad():
                outputs = self.model(input_ids, max_new_tokens)
                # Compute the sequence embedding (mean pooling along token dimension)
                sequence_embedding = torch.mean(outputs, dim=1).squeeze()
                embeddings.append(sequence_embedding)

        # Stack embeddings into a single tensor (shape: [num_sequences, embedding_dim])
        embeddings_tensor = torch.stack(embeddings)
        print(f"Generated embeddings shape: {embeddings_tensor.shape}")
        return embeddings_tensor

    def generate_mid_embeddings(self, sequences: List[str], max_len: int = 256) -> torch.Tensor:
        """
        Generate tensor mid embeddings for each DNA sequence. (only used self._embedding_impl()
                                                                and self._transformer_impl(x))
        Args:
            sequences (List[str]): List of DNA sequences.
            max_len (int): Maximum length of the sequences. Defaults to 256.
        Returns:
            torch.Tensor: A tensor of shape (num_sequences, embedding_dim) where
                        num_sequences is the number of sequences and
                        embedding_dim is the dimension of the reduced embeddings.
        """
        print(f"Generating embeddings for {len(sequences)} sequences.")
        device = self.device
        embeddings = []

        for sequence in tqdm(sequences, desc="Embedding sequences", unit="sequence"):
            # Tokenize and prepare input tensors
            input_ids = self.tokenizer.encode(sequence[0], max_len=max_len, device=device)[None, :]  # Add batch dimension
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            max_new_tokens = max_len - input_ids.shape[1]

            # Forward pass to compute embeddings
            with torch.no_grad():
                outputs = self.model.generate_mid_embeddings(input_ids, max_new_tokens)
                # Compute the sequence embedding (mean pooling along token dimension)
                sequence_embedding = torch.mean(outputs, dim=1).squeeze()
                embeddings.append(sequence_embedding)

        # Stack embeddings into a single tensor (shape: [num_sequences, embedding_dim])
        embeddings_tensor = torch.stack(embeddings)
        print(f"Generated embeddings shape: {embeddings_tensor.shape}")
        return embeddings_tensor


class RNAGPT_LM:
    def __init__(self, model_path: str ="", model_name: str = "dna_gpt0.1b_h", device: str = "cpu"):
        """
        RNAGPT-based language model.
        Args:
            model_name (str): Name of the RNAGPT model. Defaults to 'dna_gpt0.1b_h'.
            device (str): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def get_model(self, model_name):
        """
        Initialize the RNAGPT model and tokenizer.
        """
        special_tokens = (
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] +
            ["+", '-', '*', '/', '=', "&", "|", "!"] +
            ['M', 'B'] + ['P'] + ['R', 'I', 'K', 'L', 'O', 'Q', 'S', 'U', 'V'] + ['W', 'Y', 'X', 'Z']
        )
        if model_name in ('dna_gpt0.1b_h'):
            tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=False)
        else:
            tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=True)

        vocab_size = len(tokenizer)
        model = DNAGPT.from_name(model_name, vocab_size)
        return model, tokenizer

    def load_model(self):
        """
        Load the RNAGPT model and tokenizer from the specified model name.
        """
        self.model, self.tokenizer = self.get_model(self.model_name)
        weight_path = os.path.join(self.model_path, f"{self.model_name}.pth")
        self._load_model_weights(weight_path)
        return self

    def _load_model_weights(self, weight_path, dtype=None):
        """
        Load the model weights from a checkpoint file.
        """
        state = torch.load(weight_path, map_location="cpu")
        if 'model' in state.keys():
            self.model.load_state_dict(state['model'], strict=False)
        else:
            self.model.load_state_dict(state, strict=False)
        print(f"loading model weights from {weight_path}")
        self.model.to(device=self.device, dtype=dtype)
        self.model.eval()

    def replace_rna_to_dna(self, sequence):
        sequence = sequence.replace('U', 'T')
        return sequence
    
    def generate_embeddings(self, sequences: List[str], max_len: int = 256) -> torch.Tensor:
        """
        Generate tensor mid embeddings for each RNA sequence. 
        Args:
            sequences (List[str]): List of DNA sequences.
            max_len (int): Maximum length of the sequences. Defaults to 256.
        Returns:
            torch.Tensor: A tensor of shape (num_sequences, embedding_dim) where
                        num_sequences is the number of sequences and
                        embedding_dim is the dimension of the reduced embeddings.
        """
        print(f"Generating embeddings for {len(sequences)} sequences.")
        device = self.device
        embeddings = []

        for sequence in tqdm(sequences, desc="Embedding sequences", unit="sequence"):
            # Tokenize and prepare input tensors
            sequence = self.replace_rna_to_dna(sequence[0])
            input_ids = self.tokenizer.encode(sequence, max_len=max_len, device=device)[None, :]  # Add batch dimension
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            max_new_tokens = max_len - input_ids.shape[1]

            # Forward pass to compute embeddings
            with torch.no_grad():
                outputs = self.model(input_ids, max_new_tokens)
                # Compute the sequence embedding (mean pooling along token dimension)
                sequence_embedding = torch.mean(outputs, dim=1).squeeze()
                embeddings.append(sequence_embedding)

        # Stack embeddings into a single tensor (shape: [num_sequences, embedding_dim])
        embeddings_tensor = torch.stack(embeddings)
        print(f"Generated embeddings shape: {embeddings_tensor.shape}")
        return embeddings_tensor

    def generate_mid_embeddings(self, sequences: List[str], max_len: int = 256) -> torch.Tensor:
        """
        Generate tensor mid embeddings for each RNA sequence. 
        Args:
            sequences (List[str]): List of DNA sequences.
            max_len (int): Maximum length of the sequences. Defaults to 256.
        Returns:
            torch.Tensor: A tensor of shape (num_sequences, embedding_dim) where
                        num_sequences is the number of sequences and
                        embedding_dim is the dimension of the reduced embeddings.
        """
        print(f"Generating embeddings for {len(sequences)} sequences.")
        device = self.device
        embeddings = []

        for sequence in tqdm(sequences, desc="Embedding sequences", unit="sequence"):
            # Tokenize and prepare input tensors
            sequence = self.replace_rna_to_dna(sequence[0])
            input_ids = self.tokenizer.encode(sequence, max_len=max_len, device=device)[None, :]  # Add batch dimension
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            max_new_tokens = max_len - input_ids.shape[1]

            # Forward pass to compute embeddings
            with torch.no_grad():
                outputs = self.model.generate_mid_embeddings(input_ids, max_new_tokens)
                # Compute the sequence embedding (mean pooling along token dimension)
                sequence_embedding = torch.mean(outputs, dim=1).squeeze()
                embeddings.append(sequence_embedding)

        # Stack embeddings into a single tensor (shape: [num_sequences, embedding_dim])
        embeddings_tensor = torch.stack(embeddings)
        print(f"Generated embeddings shape: {embeddings_tensor.shape}")
        return embeddings_tensor


class ProtGPT_LM:
    def __init__(self, model_path: str ="", model_name: str = "prot_gpt2", device: str = "cpu"):
        """
        ProtGPT2-based language model.
        Args:
            model_name (str): Name of the ProtGPT2 model. Defaults to 'prot_gpt2'.
            device (str): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def get_model(self, model_name):
        """
        Initialize the ProtGPT2 model and tokenizer.
        """
        model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        return model, tokenizer

    def load_model(self):
        """
        Load the ProtGPT2 model and tokenizer from the specified model name.
        """
        self.model, self.tokenizer = self.get_model(self.model_name)
        return self

    def generate_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """
        Generate tensor embeddings for each sequence with gradients enabled.
        Args:
            sequences (List[str]): List of DNA sequences.
        Returns:
            torch.Tensor: A tensor of shape (num_sequences, embedding_dim) where
                        num_sequences is the number of sequences and
                        embedding_dim is the dimension of the reduced embeddings.
        """
        print(f"Generating embeddings for {len(sequences)} sequences.")
        device = self.device
        self.model.to(device)
        embeddings = []

        for sequence in tqdm(sequences, desc="Embedding sequences", unit="sequence"):
            try:
                # Tokenize and prepare input tensors
                tokenized = self.tokenizer.encode(sequence[0])
                input_ids = torch.tensor(tokenized).unsqueeze(0).to(device)

                # Truncate input_ids to max length if needed
                max_length = self.model.config.n_positions  # Max length of the model
                input_ids = input_ids[:, :max_length]

                # Forward pass to compute embeddings
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=input_ids)
                    loss, logits = outputs[:2]

                    # Compute the sequence embedding (mean pooling along token dimension)
                    sequence_embedding = torch.mean(logits, dim=1).squeeze(0)  # Shape: [embedding_dim]
                    embeddings.append(sequence_embedding)

            except Exception as e:
                print(f"Error processing sequence: {sequence}. Exception: {e}")
                continue

        # Stack embeddings into a single tensor (shape: [num_sequences, embedding_dim])
        embeddings_tensor = torch.stack(embeddings) if embeddings else torch.empty(0)
        print(f"Generated embeddings shape: {embeddings_tensor.shape}")
        return embeddings_tensor