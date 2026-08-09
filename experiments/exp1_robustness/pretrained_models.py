"""
Pretrained Code Model Wrappers for Experiment 02.

B1: GraphCodeBERT-base  (microsoft/graphcodebert-base)
B2: CodeBERT-base       (microsoft/codebert-base)

Both use full fine-tuning with a classification head.
Input: code-like text (reconstructed from UAST data).
"""

from typing import Any

import torch
import torch.nn as nn


def _transformers() -> tuple[Any, Any]:
    """Load optional pretrained-model dependencies only for B1/B2 runs."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "B1/B2 require the optional research dependencies in requirements.txt."
        ) from error
    return AutoModel, AutoTokenizer


class PretrainedCodeClassifier(nn.Module):
    """
    Pretrained code model + linear classification head.

    Architecture:
        [CLS] embedding → Dropout → Linear → num_classes
    """

    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1,
                 freeze_encoder: bool = False):
        super().__init__()
        auto_model, _ = _transformers()
        self.encoder = auto_model.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]

        Returns:
            logits: [B, num_classes]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_emb)
        return logits

    def get_embeddings(self, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """Return [CLS] embeddings without classification head."""
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]


MODEL_REGISTRY = {
    "B1": "microsoft/graphcodebert-base",
    "B2": "microsoft/codebert-base",
}


def build_pretrained_model(variant: str, num_classes: int,
                           freeze_encoder: bool = False) -> nn.Module:
    if variant not in MODEL_REGISTRY:
        raise ValueError(f"Unknown variant {variant}. Use B1 or B2.")
    return PretrainedCodeClassifier(
        model_name=MODEL_REGISTRY[variant],
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
    )


def build_tokenizer(variant: str) -> Any:
    if variant not in MODEL_REGISTRY:
        raise ValueError(f"Unknown variant {variant}. Use B1 or B2.")
    _, auto_tokenizer = _transformers()
    return auto_tokenizer.from_pretrained(MODEL_REGISTRY[variant])
