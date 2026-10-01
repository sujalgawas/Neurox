from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


class PatchEmbed(nn.Module):
	def __init__(self, n_features: int, patch_len: int, dim: int):
		super().__init__()
		self.patch_len = patch_len
		self.proj = nn.Linear(patch_len * n_features, dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch, length, features = x.shape
		if length % self.patch_len != 0:
			raise ValueError(
				f"input length {length} must be divisible by patch length {self.patch_len}"
			)
		return self.proj(
			x.reshape(batch, length // self.patch_len, self.patch_len * features)
		)


class TransformerEncoder(nn.Module):
	def __init__(
		self,
		dim: int,
		num_layers: int,
		heads: int,
		mlp_ratio: float = 4.0,
		dropout: float = 0.0,
	):
		super().__init__()
		layer = nn.TransformerEncoderLayer(
			d_model=dim,
			nhead=heads,
			dim_feedforward=int(dim * mlp_ratio),
			dropout=dropout,
			activation="gelu",
			batch_first=True,
			norm_first=True,
		)
		self.blocks = nn.TransformerEncoder(
			layer, num_layers=num_layers, enable_nested_tensor=False
		)
		self.norm = nn.LayerNorm(dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.norm(self.blocks(x))


class LOBEncoder(nn.Module):
	def __init__(
		self,
		n_features: int,
		window: int,
		patch_len: int,
		dim: int,
		num_layers: int,
		heads: int,
		mlp_ratio: float = 4.0,
		dropout: float = 0.0,
	):
		super().__init__()
		self.n_patches = window // patch_len
		self.patch_embed = PatchEmbed(n_features, patch_len, dim)
		self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, dim))
		nn.init.trunc_normal_(self.pos_embed, std=0.02)
		self.encoder = TransformerEncoder(
			dim, num_layers, heads, mlp_ratio, dropout
		)

	def embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
		return self.patch_embed(x) + self.pos_embed

	def forward(
		self, x: torch.Tensor, idx: torch.Tensor | None = None
	) -> torch.Tensor:
		tokens = self.embed_tokens(x)
		if idx is not None:
			tokens = tokens[:, idx, :]
		return self.encoder(tokens)


class JEPAPredictor(nn.Module):
	def __init__(
		self,
		dim: int,
		n_patches: int,
		pred_dim: int,
		num_layer: int,
		heads: int,
		mlp_ratio: float = 4.0,
	):
		super().__init__()
		self.input_proj = nn.Linear(dim, pred_dim)
		self.output_proj = nn.Linear(pred_dim, dim)
		self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
		self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, pred_dim))
		nn.init.trunc_normal_(self.pos_embed, std=0.02)
		nn.init.trunc_normal_(self.mask_token, std=0.02)
		self.encoder = TransformerEncoder(
			pred_dim, num_layer, heads, mlp_ratio
		)

	def forward(
		self, ctx_reps: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor
	) -> torch.Tensor:
		projected = self.input_proj(ctx_reps)
		batch = projected.shape[0]
		sequence = self.mask_token.to(projected.dtype).expand(
			batch, self.pos_embed.shape[1], -1
		).clone()
		sequence[:, ctx_idx, :] = projected
		sequence = self.encoder(sequence)
		return self.output_proj(sequence[:, tgt_idx, :])


class JEPA(nn.Module):
	def __init__(self, **config: Any):
		super().__init__()
		self.config = config
		self.context_encoder = LOBEncoder(
			config["n_features"],
			config["window"],
			config["patch_len"],
			config["pred_dim"],
			config["num_layer"],
			config["heads"],
			config["mlp_ratio"],
			config["dropout"],
		)
		self.target_encoder = copy.deepcopy(self.context_encoder)
		for parameter in self.target_encoder.parameters():
			parameter.requires_grad_(False)
		self.n_patches = self.context_encoder.n_patches
		self.predictor = JEPAPredictor(
			config["dim"],
			self.n_patches,
			config["dim"],
			config["num_layer"],
			config["heads"],
			config["mlp_ratio"],
		)
		self.direction_head = DirectionHead(
			embedding_dim=config["dim"]
		)

	@torch.inference_mode()
	def represent_past(self, x_past: torch.Tensor) -> torch.Tensor:
		"""Encode the live 60-step context without requiring unknown future bars."""
		tokens = self.context_encoder.patch_embed(x_past)
		context_patches = tokens.shape[1]
		tokens = tokens + self.context_encoder.pos_embed[:, :context_patches, :]
		return self.context_encoder.encoder(tokens)


class DirectionHead(nn.Module):
	"""Predict down, flat, or up from JEPA patch representations."""
	def __init__(self, embedding_dim: int = 192):
		super().__init__()
		self.classifier = nn.Sequential(
			nn.Linear(embedding_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 3),
		)

	def forward(self, representation: torch.Tensor) -> torch.Tensor:
		return self.classifier(representation.mean(dim=1))


def get_future_direction(
	y_future_batch: torch.Tensor, threshold: float = 0.0005
) -> torch.Tensor:
	"""Classify known future windows for backtesting, not live prediction."""
	if y_future_batch.ndim != 3 or y_future_batch.shape[-1] < 1:
		raise ValueError("expected y_future_batch with shape [batch, horizon, features]")
	cumulative_returns = torch.sum(y_future_batch[:, :, 0], dim=1)
	direction = torch.zeros_like(cumulative_returns)
	direction[cumulative_returns > threshold] = 1.0
	direction[cumulative_returns < -threshold] = -1.0
	return direction


class JEPAInference:
	"""Load a notebook checkpoint and encode the latest market context."""

	def __init__(
		self,
		checkpoint_path: str | Path,
		device: str | None = None,
	):
		self.checkpoint_path = Path(checkpoint_path)
		if not self.checkpoint_path.is_file():
			raise FileNotFoundError(f"JEPA checkpoint not found: {self.checkpoint_path}")
		self.device = torch.device(
			device or ("cuda" if torch.cuda.is_available() else "cpu")
		)
		checkpoint = torch.load(
			self.checkpoint_path, map_location=self.device, weights_only=False
		)
		self.model = JEPA(**checkpoint["config"])
		model_state = checkpoint["model"]
		embedded_head = {
			key.removeprefix("direction_head."): value
			for key, value in model_state.items()
			if key.startswith("direction_head.")
		}
		if embedded_head:
			self.model.load_state_dict(model_state)
		elif "direction_head" in checkpoint:
			self.model.load_state_dict(model_state, strict=False)
			self.model.direction_head.load_state_dict(checkpoint["direction_head"])
		else:
			raise ValueError(
				"best.pt does not contain direction_head weights; "
				"upload the single combined direction-model checkpoint"
			)
		self.model.to(self.device).eval()

	@torch.inference_mode()
	def encode_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
		features = np.asarray(observation["jepa_features"], dtype=np.float32)
		expected_steps = 60
		if features.ndim != 2 or features.shape[1] != self.model.config["n_features"]:
			raise ValueError(
				f"expected [steps, {self.model.config['n_features']}] features, "
				f"got {features.shape}"
			)
		if features.shape[0] < expected_steps:
			raise ValueError(
				f"expected at least {expected_steps} feature rows, got {features.shape[0]}"
			)
		context = torch.from_numpy(features[-expected_steps:]).unsqueeze(0).to(self.device)
		# Keep the batch dimension: DirectionHead expects [batch, patches, embedding].
		representation = self.model.represent_past(context)
		logits = self.model.direction_head(representation)
		probabilities = torch.softmax(logits, dim=-1)[0]
		class_index = int(probabilities.argmax().item())
		return {
			"latest_price": observation["latest_price"],
			"direction": class_index - 1,
			"direction_label": ("down", "flat", "up")[class_index],
			"direction_confidence": float(probabilities[class_index]),
			"device": str(self.device),
			"checkpoint": str(self.checkpoint_path),
		}

	def predict_signal(self, history: Any, observation: dict[str, Any]) -> float:
		"""Return the single direction model's signal for the trading pipeline."""
		return float(self.encode_observation(observation)["direction"])


def download_checkpoint(
	repo_id: str = "sujalgawas/jepa-trading-direction-Big",
	filename: str = "model.pt",
	token: str | None = None,
	revision: str | None = None,
) -> str:
	"""Download a Hugging Face checkpoint into the Hub cache and return its path."""
	try:
		return hf_hub_download(
			repo_id=repo_id,
			filename=filename,
			repo_type="model",
			token=token,
			revision=revision,
		)
	except Exception as error:
		raise FileNotFoundError(
			f"Unable to download {filename} from Hugging Face repository {repo_id}. "
			"Set HF_TOKEN if the repository is private or gated."
		) from error
