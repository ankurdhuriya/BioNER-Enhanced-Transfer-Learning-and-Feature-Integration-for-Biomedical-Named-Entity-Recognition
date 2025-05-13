#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BioNER Training Pipeline: Fine-tuning transformer models for biomedical named entity recognition.

This script provides functionality for:
- Loading and configuring transformer models for NER tasks
- Training and evaluating models on biomedical datasets
- Supporting multi-task learning scenarios
- Handling both training and inference pipelines
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
import wandb
from bio_ner_data_processor import NERDataset, DatasetSplit, get_ner_labels, get_bio_labels
from bio_ner_multi_task import MultiTaskBertNER, MultiTaskRobertaNER

logger = logging.getLogger(__name__)

@dataclass
class BioModelArguments:
    """Arguments for model configuration and loading."""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path"}
    )
    use_fast_tokenizer: bool = field(
        default=False, metadata={"help": "Use fast tokenization"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory for cached models"}
    )

@dataclass
class BioDataArguments:
    """Arguments for data configuration and processing."""
    data_dir: str = field(
        metadata={"help": "Input data directory containing dataset files"}
    )
    labels: Optional[str] = field(
        default=None, metadata={"help": "Path to file containing all labels"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "Maximum sequence length after tokenization"}
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite cached datasets"}
    )
    wandb_name: str = field(
        default=None, metadata={"help": "Name for Weights & Biases run"}
    )
    train_datasets: str = field(
        default='', metadata={"help": "List of training datasets"}
    )
    eval_datasets: str = field(
        default='', metadata={"help": "List of evaluation datasets"}
    )

def main():
    """Main training and evaluation pipeline."""
    # Parse arguments
    parser = HfArgumentParser(
        (BioModelArguments, BioDataArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize Weights & Biases
    if data_args.wandb_name:
        wandb.init(project="bio_ner", name=data_args.wandb_name)

    # Check output directory
    if (os.path.exists(training_args.output_dir) and
        os.listdir(training_args.output_dir) and
        training_args.do_train and
        not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) exists and is not empty. "
            "Use --overwrite_output_dir to overwrite."
        )

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.info(
        "Process rank: %s, device: %s, n_gpu: %s, distributed: %s, 16-bit: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training parameters: %s", training_args)

    # Set random seed
    set_seed(training_args.seed)

    # Determine entity type and labels
    entity_type = data_args.data_dir.split('/')[-1]
    if entity_type in ["CoNLL2003NER", "OntoNotes5.0", "WNUT2017"]:
        labels = get_ner_labels(data_args.labels)
    else:
        labels = get_bio_labels(data_args.labels)

    label_map = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    # Initialize appropriate model
    if "roberta" in model_args.model_name_or_path.lower():
        model = MultiTaskRobertaNER.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = MultiTaskBertNER.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
        )

    # Load datasets
    train_dataset = (
        NERDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            split=DatasetSplit.TRAIN,
            train_datasets=data_args.train_datasets,
            eval_datasets=data_args.eval_datasets,
        )
        if training_args.do_train else None
    )

    eval_dataset = (
        NERDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            split=DatasetSplit.DEV,
            eval_datasets=data_args.eval_datasets,
        )
        if training_args.do_eval else None
    )

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        """Align predictions with labels for evaluation."""
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape

        preds_list = [[] for _ in range(batch_size)]
        labels_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    labels_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, labels_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        """Compute evaluation metrics for NER predictions."""
        preds_list, labels_list = align_predictions(p.predictions, p.label_ids)

        return {
            "precision": precision_score(labels_list, preds_list),
            "recall": recall_score(labels_list, preds_list),
            "f1": f1_score(labels_list, preds_list),
        }

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training phase
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(
                model_args.model_name_or_path) else None
        )
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation phase
    results = {}
    if training_args.do_eval:
        logger.info("*** Running Evaluation ***")

        eval_results = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")

        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Evaluation Results *****")
                for key, value in eval_results.items():
                    logger.info("  %s = %s", key, value)
                    writer.write(f"{key} = {value}\n")

            # Save evaluation predictions
            eval_predictions_file = os.path.join(training_args.output_dir, "eval_predictions.txt")
            eval_predictions, eval_label_ids, _ = trainer.predict(eval_dataset)
            eval_preds_list, _ = align_predictions(eval_predictions, eval_label_ids)

            with open(eval_predictions_file, "w") as writer:
                with open(os.path.join(data_args.data_dir, data_args.eval_datasets.split('+')[0], "devel.txt"), "r") as f:
                    example_id = 0
                    for line in f:
                        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                            writer.write(line)
                            if not eval_preds_list[example_id]:
                                example_id += 1
                        elif eval_preds_list[example_id]:
                            entity_label = eval_preds_list[example_id].pop(0)
                            output_line = f"{line.split()[0]} {entity_label[0] if entity_label != 'O' else entity_label}\n"
                            writer.write(output_line)

        results.update(eval_results)

    # Prediction phase
    if training_args.do_predict:
        test_dataset = NERDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            split=DatasetSplit.TEST,
            eval_datasets=data_args.eval_datasets,
        )

        test_predictions, test_label_ids, test_metrics = trainer.predict(test_dataset)
        test_preds_list, _ = align_predictions(test_predictions, test_label_ids)

        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_master():
            with open(output_test_results_file, "w") as writer:
                logger.info("***** Test Results *****")
                for key, value in test_metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write(f"{key} = {value}\n")

            # Save test predictions
            test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
            with open(test_predictions_file, "w") as writer:
                with open(os.path.join(data_args.data_dir, data_args.eval_datasets.split('+')[0], "test.txt"), "r") as f:
                    example_id = 0
                    for line in f:
                        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                            writer.write(line)
                            if not test_preds_list[example_id]:
                                example_id += 1
                        elif test_preds_list[example_id]:
                            entity_label = test_preds_list[example_id].pop(0)
                            output_line = f"{line.split()[0]} {entity_label[0] if entity_label != 'O' else entity_label}\n"
                            writer.write(output_line)

    return results

def _mp_fn(index):
    """Entry point for multiprocessing."""
    main()

if __name__ == "__main__":
    main()
