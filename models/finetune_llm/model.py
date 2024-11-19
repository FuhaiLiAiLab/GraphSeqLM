import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Dict


class DNASeqLM:
    def __init__(self, model_path: str = "dmis-lab/biobert-base-cased-v1.1", device: str = "cpu"):
        """
        Args:
            load_model (bool, optional): Load the BioBERT model and tokenizer. Defaults to True.
            model_path (str, optional): Path to the BioBERT model. Defaults to 'dmis-lab/biobert-base-cased-v1.1'.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Load the BioBERT model and tokenizer from the specified model path.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)

    def generate_embeddings(self, sequences: List[str]) -> List[float]:
        """
        Generate a single-dimensional embedding for each sequence.
        """
        embeddings = []
        for sequence in tqdm(sequences, desc="Embedding sequences", unit="sequence"):
            inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            sequence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu()
            single_dim_embedding = torch.mean(sequence_embedding).item()  # Reduce to 1D and convert to float
            embeddings.append(single_dim_embedding)
        return embeddings
    

class RNASeqLM:
    def __init__(self, model_path: str = "dmis-lab/biobert-base-cased-v1.1", device: str = "cpu"):
        """
        Args:
            load_model (bool, optional): Load the BioBERT model and tokenizer. Defaults to True.
            model_path (str, optional): Path to the BioBERT model. Defaults to 'dmis-lab/biobert-base-cased-v1.1'.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Load the BioBERT model and tokenizer from the specified model path.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)

    def generate_embeddings(self, sequences: List[str]) -> List[float]:
        """
        Generate a single-dimensional embedding for each sequence.
        """
        embeddings = []
        for sequence in tqdm(sequences, desc="Embedding sequences", unit="sequence"):
            inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            sequence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu()
            single_dim_embedding = torch.mean(sequence_embedding).item()  # Reduce to 1D and convert to float
            embeddings.append(single_dim_embedding)
        return embeddings
    

class ProteinSeqLM:
    def __init__(self, model_path: str = "dmis-lab/biobert-base-cased-v1.1", device: str = "cpu"):
        """
        Args:
            load_model (bool, optional): Load the BioBERT model and tokenizer. Defaults to True.
            model_path (str, optional): Path to the BioBERT model. Defaults to 'dmis-lab/biobert-base-cased-v1.1'.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Load the BioBERT model and tokenizer from the specified model path.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)

    def generate_embeddings(self, sequences: List[str]) -> List[float]:
        """
        Generate a single-dimensional embedding for each sequence.
        """
        embeddings = []
        for sequence in tqdm(sequences, desc="Embedding sequences", unit="sequence"):
            inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            sequence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu()
            single_dim_embedding = torch.mean(sequence_embedding).item()  # Reduce to 1D and convert to float
            embeddings.append(single_dim_embedding)
        return embeddings