"""Pretrained sentence encoder with a trainable intent classification head."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 256


def get_device() -> torch.device:
    """Choose the best available accelerator, falling back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(model_name: str = MODEL_NAME):
    """Load the tokenizer that matches the encoder."""
    return AutoTokenizer.from_pretrained(model_name)


class IntentClassifier(nn.Module):
    """Mean-pool token embeddings and classify each sentence's intent."""

    def __init__(
        self,
        num_labels: int,
        model_name: str = MODEL_NAME,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_labels < 1:
            raise ValueError("num_labels must be at least 1")

        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one unnormalized score (logit) per label and sentence."""
        encoder_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_inputs)
        token_embeddings = outputs.last_hidden_state

        # Ignore padding tokens when averaging the token vectors.
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))


def tokenize(tokenizer, texts: list[str], device: torch.device | str | None = None):
    """Tokenize a batch using the model's standard truncation and padding."""
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}
    return batch
