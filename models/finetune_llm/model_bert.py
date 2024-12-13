import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Dict

from transformers import AutoTokenizer, AutoModel
from multimolecule import RnaTokenizer, RnaBertModel
from transformers import BertModel, BertTokenizer

class DNASeqLM:
    def __init__(self, model_path: str = "zhihan1996/DNA_bert_3", device: str = "cpu", kmer: int = 3):
        """
        DNABERT-based sequence embedding model.
        Args:
            model_path (str): Path to the DNABERT model. Defaults to 'zhihan1996/DNA_bert_3'.
            device (str): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
            kmer (int): K-mer size (e.g., 3 for tri-nucleotide representation). Defaults to 3.
        """
        self.model_path = model_path
        self.device = device
        self.kmer = kmer
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Load the DNABERT model and tokenizer from the specified model path.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        return self

    def _generate_kmers(self, sequence: str) -> str:
        """
        Generate k-mers from a DNA sequence.
        Args:
            sequence (str): Input DNA sequence.
        Returns:
            str: Space-separated k-mer tokens for DNABERT tokenization.
        """
        return " ".join([sequence[i:i+self.kmer] for i in range(len(sequence) - self.kmer + 1)])

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
        embeddings = []
        for sequence in tqdm(sequences, desc="Embedding sequences", unit="sequence"):
            # Generate k-mers for the input sequence
            kmers = self._generate_kmers(sequence[0])
            # Tokenize and prepare input tensors
            inputs = self.tokenizer(kmers, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            # Forward pass with torch.no_grad()
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Compute the sequence embedding (mean pooling along token dimension)
            sequence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
            # Append to the list of embeddings (keep them as tensors)
            embeddings.append(sequence_embedding)
        # Stack embeddings into a single tensor (shape: [num_sequences, embedding_dim])
        return torch.stack(embeddings)


class RNASeqLM:
    def __init__(self, model_path: str = "multimolecule/rnabert", device: str = "cpu", kmer: int = 3):
        """
        RNA-BERT-based sequence embedding model.
        Args:
            model_path (str): Path to the RNA-BERT model. Defaults to 'multimolecule/rnabert'.
            device (str): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
            kmer (int): K-mer size for RNA sequences. Defaults to 3.
        """
        self.model_path = model_path
        self.device = device
        self.kmer = kmer
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Load the RNA-BERT model and tokenizer from the specified model path.
        """
        self.tokenizer = RnaTokenizer.from_pretrained(self.model_path)
        self.model = RnaBertModel.from_pretrained(self.model_path).to(self.device)
        return self

    def _generate_kmers(self, sequence: str) -> str:
        """
        Generate k-mers from an RNA sequence.
        Args:
            sequence (str): Input RNA sequence.

        Returns:
            str: Space-separated k-mer tokens for RNA-BERT tokenization.
        """
        if len(sequence) < self.kmer:
            print(f"Warning: Sequence length ({len(sequence)}) is shorter than k-mer size ({self.kmer}). Skipping this sequence.")
            return None  # Return None for short sequences
        return " ".join([sequence[i:i + self.kmer] for i in range(len(sequence) - self.kmer + 1)])

    def generate_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """
        Generate tensor embeddings for each sequence with gradients enabled.
        Args:
            sequences (List[str]): List of RNA sequences.
        Returns:
            torch.Tensor: A tensor of shape (num_sequences, embedding_dim) where
                          num_sequences is the number of sequences and
                          embedding_dim is the dimension of the reduced embeddings.
        """
        embeddings = []
        for sequence in tqdm(sequences, desc="Embedding RNA sequences", unit="sequence"):
             # Generate k-mers for the input sequence
            kmers = self._generate_kmers(sequence[0])
            if kmers is None:  # Skip short sequences
                embeddings.append(torch.zeros(self.model.config.hidden_size).to(self.device))
                continue
            # Tokenize and prepare input tensors
            inputs = self.tokenizer(
                kmers,  # Pass kmers directly
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=440  # Ensure input length <= model's max_length
            ).to(self.device)
            # Forward pass with torch.no_grad()
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Compute the sequence embedding (mean pooling along token dimension)
            sequence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
            # Append to the list of embeddings (keep them as tensors)
            embeddings.append(sequence_embedding)
        # Stack embeddings into a single tensor (shape: [num_sequences, embedding_dim])
        return torch.stack(embeddings)


class ProteinSeqLM:
    def __init__(self, model_path: str = "Rostlab/prot_bert", device: str = "cpu"):
        """
        Protein-BERT-based sequence embedding model.
        Args:
            model_path (str): Path to the Protein-BERT model. Defaults to 'Rostlab/prot_bert'.
            device (str): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Load the Protein-BERT model and tokenizer from the specified model path.
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path, do_lower_case=False)
        self.model = BertModel.from_pretrained(self.model_path).to(self.device)
        return self

    def _prepare_sequence(self, sequence: str) -> str:
        """
        Prepare a protein sequence by adding spaces between amino acids.
        Args:
            sequence (str): Input protein sequence.
        Returns:
            str: Space-separated amino acid sequence.
        """
        return " ".join(sequence)

    def generate_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """
        Generate tensor embeddings for each sequence with gradients enabled.
        Args:
            sequences (List[str]): List of protein sequences.
        Returns:
            torch.Tensor: A tensor of shape (num_sequences, embedding_dim) where
                          num_sequences is the number of sequences and
                          embedding_dim is the dimension of the reduced embeddings.
        """
        embeddings = []
        for sequence in tqdm(sequences, desc="Embedding protein sequences", unit="sequence"):
            # Prepare the sequence with spaces between amino acids
            prepared_sequence = self._prepare_sequence(sequence[0])
            # Tokenize and prepare input tensors
            inputs = self.tokenizer(prepared_sequence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            # Forward pass with torch.no_grad()
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Compute the sequence embedding (mean pooling along token dimension)
            sequence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
            # Append to the list of embeddings (keep them as tensors)
            embeddings.append(sequence_embedding)
        # Stack embeddings into a single tensor (shape: [num_sequences, embedding_dim])
        return torch.stack(embeddings)