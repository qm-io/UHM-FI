"""BioClinicalBERT and offline tiny self-attention text encoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import TextConfig


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


def masked_mean_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
    numerator = (token_embeddings * mask).sum(dim=1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return numerator / denominator


class BioClinicalBERTEncoder(nn.Module):
    """Paper text encoder with trainable projection layers."""

    def __init__(self, config: TextConfig, embedding_dim: int) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, BertConfig, BertModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("transformers is required for BioClinicalBERT.") from exc

        if config.pretrained:
            self.model = AutoModel.from_pretrained(
                config.model_name,
                local_files_only=config.local_files_only,
            )
        else:
            # BioClinicalBERT uses the BERT-base topology. This offline branch
            # exercises that architecture without downloading model weights.
            bert_config = BertConfig(
                vocab_size=config.vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout_prob=config.dropout,
                attention_probs_dropout_prob=config.dropout,
                max_position_embeddings=max(512, config.max_length),
                type_vocab_size=2,
            )
            self.model = BertModel(bert_config)
        hidden_size = int(self.model.config.hidden_size)
        self.projection_head = ProjectionHead(hidden_size, embedding_dim)

        encoder = getattr(self.model, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is not None:
            for layer_index in config.freeze_layers:
                if layer_index < 0 or layer_index >= len(layers):
                    raise ValueError(f"Invalid BERT layer index: {layer_index}")
                for parameter in layers[layer_index].parameters():
                    parameter.requires_grad = False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.model(**kwargs)
        pooled = masked_mean_pooling(output.last_hidden_state, attention_mask)
        return self.projection_head(pooled)


class TinyTransformerTextEncoder(nn.Module):
    """Offline self-attention encoder for environments without BioClinicalBERT."""

    def __init__(self, config: TextConfig, embedding_dim: int) -> None:
        super().__init__()
        self.max_length = config.max_length
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.projection_head = ProjectionHead(config.hidden_dim, embedding_dim)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        del token_type_ids
        batch, sequence_length = input_ids.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds {self.max_length}."
            )
        positions = torch.arange(sequence_length, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=attention_mask.eq(0))
        hidden = self.norm(hidden)
        pooled = masked_mean_pooling(hidden, attention_mask)
        return self.projection_head(pooled)


def build_text_encoder(config: TextConfig, embedding_dim: int) -> nn.Module:
    if config.backend == "bioclinicalbert":
        return BioClinicalBERTEncoder(config, embedding_dim)
    if config.backend == "tiny":
        return TinyTransformerTextEncoder(config, embedding_dim)
    raise ValueError(f"Unsupported text encoder backend: {config.backend}")
