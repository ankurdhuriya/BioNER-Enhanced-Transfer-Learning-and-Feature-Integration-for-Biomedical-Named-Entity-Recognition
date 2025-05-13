import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (
    BertConfig, BertModel, BertForTokenClassification, BertTokenizer,
    RobertaConfig, RobertaModel, RobertaForTokenClassification, RobertaTokenizer
)

class MultiTaskBertNER(nn.Module):
    """
    Multi-task BERT-based model for Biomedical Named Entity Recognition (BioNER)
    with both general and biomedical entity classification capabilities.
    """
    def __init__(self, config, num_labels=3):
        super(MultiTaskBertNER, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Classifiers for different entity types
        self.general_ner_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.bio_ner_classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Intermediate layers for feature transformation
        self.general_ner_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.bio_ner_transform = nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, entity_type_ids=None):
        """
        Forward pass for the multi-task NER model.

        Args:
            input_ids: Input token IDs
            token_type_ids: Token type IDs
            attention_mask: Attention mask
            labels: Ground truth labels
            entity_type_ids: Entity type identifiers (0 for general, 1-9 for biomedical)

        Returns:
            logits: Model predictions
            sequence_output: Transformer output
        """
        # Get sequence output from BERT
        sequence_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=None
        )[0]

        sequence_output = self.dropout(sequence_output)

        # Handle different entity types
        if entity_type_ids[0][0].item() == 0:
            # Process raw text data
            general_features = F.relu(self.general_ner_transform(sequence_output))
            bio_features = F.relu(self.bio_ner_transform(sequence_output))

            general_logits = self.general_ner_classifier(general_features)
            bio_logits = self.bio_ner_classifier(bio_features)

            combined_features = general_features + bio_features
            logits = (bio_logits, general_logits)
        else:
            # Process data with predefined entity types
            bio_mask = (entity_type_ids > 0) & (entity_type_ids <= 9)
            general_mask = (entity_type_ids == 0)

            bio_features = bio_mask.unsqueeze(-1) * sequence_output
            general_features = general_mask.unsqueeze(-1) * sequence_output

            bio_features = F.relu(self.bio_ner_transform(bio_features))
            general_features = F.relu(self.general_ner_transform(general_features))

            bio_logits = self.bio_ner_classifier(bio_features)
            general_logits = self.general_ner_classifier(general_features)

            combined_features = bio_features + general_features
            logits = bio_logits + general_logits

        outputs = (logits, combined_features)

        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_mask = attention_mask.view(-1) == 1

                if entity_type_ids[0][0].item() == 0:
                    bio_logits, general_logits = logits

                    active_bio_logits = bio_logits.view(-1, self.num_labels)
                    active_general_logits = general_logits.view(-1, self.num_labels)

                    active_labels = torch.where(
                        active_mask,
                        labels.view(-1),
                        torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )

                    bio_loss = loss_fct(active_bio_logits, active_labels)
                    general_loss = loss_fct(active_general_logits, active_labels)
                    total_loss = bio_loss + general_loss

                    return (total_loss,) + outputs
                else:
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_mask,
                        labels.view(-1),
                        torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    total_loss = loss_fct(active_logits, active_labels)
                    return (total_loss,) + outputs
            else:
                total_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return total_loss

        return logits

class MultiTaskRobertaNER(nn.Module):
    """
    Multi-task RoBERTa-based model for Biomedical Named Entity Recognition (BioNER)
    with both general and biomedical entity classification capabilities.
    """
    def __init__(self, config, num_labels=3):
        super(MultiTaskRobertaNER, self).__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Classifiers for different entity types
        self.general_ner_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.bio_ner_classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Intermediate layers for feature transformation
        self.general_ner_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.bio_ner_transform = nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, entity_type_ids=None):
        """
        Forward pass for the multi-task NER model.

        Args:
            input_ids: Input token IDs
            token_type_ids: Token type IDs
            attention_mask: Attention mask
            labels: Ground truth labels
            entity_type_ids: Entity type identifiers (0 for general, 1-9 for biomedical)

        Returns:
            logits: Model predictions
            sequence_output: Transformer output
        """
        # Get sequence output from RoBERTa
        sequence_output = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=None
        )[0]

        sequence_output = self.dropout(sequence_output)

        # Handle different entity types
        if entity_type_ids[0][0].item() == 0:
            # Process raw text data
            general_features = F.relu(self.general_ner_transform(sequence_output))
            bio_features = F.relu(self.bio_ner_transform(sequence_output))

            general_logits = self.general_ner_classifier(general_features)
            bio_logits = self.bio_ner_classifier(bio_features)

            combined_features = general_features + bio_features
            logits = (bio_logits, general_logits)
        else:
            # Process data with predefined entity types
            bio_mask = (entity_type_ids > 0) & (entity_type_ids <= 9)
            general_mask = (entity_type_ids == 0)

            bio_features = bio_mask.unsqueeze(-1) * sequence_output
            general_features = general_mask.unsqueeze(-1) * sequence_output

            bio_features = F.relu(self.bio_ner_transform(bio_features))
            general_features = F.relu(self.general_ner_transform(general_features))

            bio_logits = self.bio_ner_classifier(bio_features)
            general_logits = self.general_ner_classifier(general_features)

            combined_features = bio_features + general_features
            logits = bio_logits + general_logits

        outputs = (logits, combined_features)

        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_mask = attention_mask.view(-1) == 1

                if entity_type_ids[0][0].item() == 0:
                    bio_logits, general_logits = logits

                    active_bio_logits = bio_logits.view(-1, self.num_labels)
                    active_general_logits = general_logits.view(-1, self.num_labels)

                    active_labels = torch.where(
                        active_mask,
                        labels.view(-1),
                        torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )

                    bio_loss = loss_fct(active_bio_logits, active_labels)
                    general_loss = loss_fct(active_general_logits, active_labels)
                    total_loss = bio_loss + general_loss

                    return (total_loss,) + outputs
                else:
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_mask,
                        labels.view(-1),
                        torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    total_loss = loss_fct(active_logits, active_labels)
                    return (total_loss,) + outputs
            else:
                total_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return total_loss

        return logits

class SingleTaskBertNER(nn.Module):
    """
    Single-task BERT-based model for Named Entity Recognition (NER)
    """
    def __init__(self, config, num_labels=3):
        super(SingleTaskBertNER, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for the single-task NER model.

        Args:
            input_ids: Input token IDs
            token_type_ids: Token type IDs
            attention_mask: Attention mask
            labels: Ground truth labels

        Returns:
            logits: Model predictions
            sequence_output: Transformer output
        """
        sequence_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=None
        )[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits, sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_mask = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_mask,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                total_loss = loss_fct(active_logits, active_labels)
                return (total_loss,) + outputs
            else:
                total_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return total_loss

        return logits
