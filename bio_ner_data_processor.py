# -*- coding: utf-8 -*-
"""
BioNER Data Processor: Utilities for processing and preparing biomedical named entity recognition datasets.

This module provides functionality for:
- Reading and parsing NER datasets
- Converting examples to features for model input
- Handling multi-task learning scenarios
- Supporting various tokenization schemes
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np
from filelock import FileLock
from tqdm import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

@dataclass
class NERExample:
    """A single training/test example for token classification."""
    guid: str
    words: List[str]
    labels: Optional[List[str]]
    entity_type: Optional[int] = None

@dataclass
class NERFeatures:
    """A single set of features for NER data."""
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    entity_type_ids: Optional[List[int]] = None

class DatasetSplit(Enum):
    """Enumeration for dataset splits."""
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"

class NERDataset:
    """Dataset class for NER tasks with caching and feature conversion."""

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        model_type: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache: bool = False,
        split: DatasetSplit = DatasetSplit.TRAIN,
        train_datasets: str = '',
        eval_datasets: str = '',
    ):
        self.pad_token_label_id = -100  # CrossEntropyLoss ignore_index
        self.features = self._load_or_create_features(
            data_dir, tokenizer, labels, model_type,
            max_seq_length, overwrite_cache, split,
            train_datasets, eval_datasets
        )

    def _load_or_create_features(self, *args, **kwargs):
        """Load features from cache or create new ones."""
        cached_features_file = self._get_cache_file_path(*args, **kwargs)

        with FileLock(cached_features_file + ".lock"):
            if os.path.exists(cached_features_file) and not kwargs['overwrite_cache']:
                logger.info(f"Loading features from cache: {cached_features_file}")
                return torch.load(cached_features_file)
            else:
                logger.info("Creating new features from dataset")
                examples = self._read_examples(*args, **kwargs)
                return self._convert_to_features(examples, *args, **kwargs)

    def _get_cache_file_path(self, *args, **kwargs):
        """Generate path for cached features file."""
        data_dir = args[0]
        tokenizer = args[1]
        split = kwargs['split'].value
        max_seq_length = kwargs['max_seq_length']

        return os.path.join(
            data_dir,
            f"cached_{split}_{tokenizer.__class__.__name__}_{max_seq_length}"
        )

    def _read_examples(self, *args, **kwargs):
        """Read examples from dataset files."""
        data_dir = args[0]
        split = kwargs['split'].value
        train_datasets = kwargs['train_datasets']
        eval_datasets = kwargs['eval_datasets']

        if split == DatasetSplit.TRAIN.value:
            return self._read_train_examples(data_dir, train_datasets)
        else:
            return self._read_eval_examples(data_dir, eval_datasets)

    def _read_train_examples(self, data_dir, train_datasets):
        """Read training examples from multiple datasets."""
        examples = []
        guid_index = 1

        for dataset in train_datasets.split('+'):
            dataset_path = os.path.join(data_dir, dataset, f"{DatasetSplit.TRAIN.value}.txt")
            if os.path.exists(dataset_path):
                with open(dataset_path, encoding="utf-8") as f:
                    current_words = []
                    current_labels = []
                    entity_type = self._get_entity_type(dataset)

                    for line in f:
                        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                            if current_words:
                                examples.append(NERExample(
                                    guid=f"train-{guid_index}",
                                    words=current_words,
                                    labels=current_labels,
                                    entity_type=entity_type
                                ))
                                guid_index += 1
                                current_words = []
                                current_labels = []
                        else:
                            parts = line.split()
                            current_words.append(parts[0])
                            current_labels.append(parts[-1].replace("\n", "") if len(parts) > 1 else "O")

                    if current_words:  # Add last example
                        examples.append(NERExample(
                            guid=f"train-{guid_index}",
                            words=current_words,
                            labels=current_labels,
                            entity_type=entity_type
                        ))

        return examples

    def _read_eval_examples(self, data_dir, eval_datasets):
        """Read evaluation examples from specified datasets."""
        examples = []
        guid_index = 1

        for dataset in eval_datasets.split('+'):
            dataset_path = os.path.join(data_dir, dataset, f"{DatasetSplit.TEST.value}.txt")
            if os.path.exists(dataset_path):
                with open(dataset_path, encoding="utf-8") as f:
                    current_words = []
                    current_labels = []
                    entity_type = self._get_entity_type(dataset)

                    for line in f:
                        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                            if current_words:
                                examples.append(NERExample(
                                    guid=f"test-{guid_index}",
                                    words=current_words,
                                    labels=current_labels,
                                    entity_type=entity_type
                                ))
                                guid_index += 1
                                current_words = []
                                current_labels = []
                        else:
                            parts = line.split()
                            current_words.append(parts[0])
                            current_labels.append(parts[-1].replace("\n", "") if len(parts) > 1 else "O")

                    if current_words:  # Add last example
                        examples.append(NERExample(
                            guid=f"test-{guid_index}",
                            words=current_words,
                            labels=current_labels,
                            entity_type=entity_type
                        ))

        return examples

    def _get_entity_type(self, dataset_name):
        """Map dataset names to entity type IDs."""
        entity_type_mapping = {
            # Disease datasets
            "NCBI-disease": 1, "BC5CDR-disease": 1, "mirna-di": 1,
            "ncbi_disease": 1, "scai_disease": 1, "variome-di": 1,

            # Chemical datasets
            "BC5CDR-chem": 2, "cdr-ch": 2, "chemdner": 2,
            "scai_chemicals": 2, "chebi-ch": 2, "BC4CHEMD": 2,

            # Gene/Protein datasets
            "BC2GM": 3, "JNLPBA-protein": 3, "bc2gm": 3, "mirna-gp": 3,
            "cell_finder-gp": 3, "chebi-gp": 3, "loctext-gp": 3,
            "deca": 3, "fsu": 3, "gpro": 3, "jnlpba-gp": 3,
            "bio_infer-gp": 3, "variome-gp": 3, "osiris-gp": 3,
            "iepa": 3,

            # Species datasets
            "s800": 4, "linnaeus": 4, "loctext-sp": 4,
            "mirna-sp": 4, "chebi-sp": 4, "cell_finder-sp": 4,
            "variome-sp": 4,

            # Cell line datasets
            "JNLPBA-cl": 5, "cell_finder-cl": 5, "jnlpba-cl": 5,
            "gellus": 5, "cll": 5,

            # DNA datasets
            "JNLPBA-dna": 6, "jnlpba-dna": 6,

            # RNA datasets
            "JNLPBA-rna": 7, "jnlpba-rna": 7,

            # Cell type datasets
            "JNLPBA-ct": 8, "jnlpba-ct": 8
        }

        return entity_type_mapping.get(dataset_name, 0)

    def _convert_to_features(self, examples, *args, **kwargs):
        """Convert examples to input features."""
        tokenizer = args[1]
        label_list = args[2]
        max_seq_length = kwargs['max_seq_length']
        model_type = args[3]

        label_map = {label: i for i, label in enumerate(label_list)}
        features = []

        for example in tqdm(examples, desc="Converting examples to features"):
            tokens = []
            label_ids = []

            for word, label in zip(example.words, example.labels):
                word_tokens = tokenizer.tokenize(word)
                if word_tokens:
                    tokens.extend(word_tokens)
                    label_ids.extend(
                        [label_map[label]] + [self.pad_token_label_id] * (len(word_tokens) - 1)
                    )

            # Truncate if needed
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[:(max_seq_length - special_tokens_count)]
                label_ids = label_ids[:(max_seq_length - special_tokens_count)]

            # Add special tokens
            tokens += [tokenizer.sep_token]
            label_ids += [self.pad_token_label_id]

            if model_type in ["roberta"]:
                tokens += [tokenizer.sep_token]
                label_ids += [self.pad_token_label_id]

            # Create entity type IDs
            entity_type_ids = [example.entity_type] * len(tokens)

            # Add CLS token
            if model_type in ["xlnet"]:
                tokens += [tokenizer.cls_token]
                label_ids += [self.pad_token_label_id]
                entity_type_ids += [example.entity_type]
            else:
                tokens = [tokenizer.cls_token] + tokens
                label_ids = [self.pad_token_label_id] + label_ids
                entity_type_ids = [example.entity_type] + entity_type_ids

            # Convert tokens to IDs
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)

            # Pad sequences
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            token_type_ids = token_type_ids + [tokenizer.pad_token_type_id] * padding_length
            label_ids = label_ids + [self.pad_token_label_id] * padding_length
            entity_type_ids = entity_type_ids + [example.entity_type] * padding_length

            features.append(NERFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_ids=label_ids,
                entity_type_ids=entity_type_ids
            ))

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

def get_ner_labels(label_path: str = None) -> List[str]:
    """Get standard NER labels or load from file."""
    if label_path and os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

def get_bio_labels(label_path: str = None) -> List[str]:
    """Get BIO format labels or load from file."""
    if label_path and os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B", "I"]
