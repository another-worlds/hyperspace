"""Cross-Block Neural Networks: Universal Variance Tensor & Universal Semantic Encoding.

Two interconnected neural networks that map UKT block layers onto one another:

1. **Universal Variance Tensor (UVT)**: A cross-block attention network that learns
   pairwise inter-layer relationships. Produces a coupling matrix showing which
   blocks co-vary, a cross-feature covariance structure, and a variance decomposition
   identifying the strongest cross-modal links.

2. **Universal Semantic Encoding (USE)**: A semantic bottleneck encoder that fuses
   UKT features with cross-block coupling information to produce a single unified
   semantic vector. This vector captures the system's complete multi-modal state
   and maps onto interpretable semantic dimensions.

Both networks are CPU-only, lightweight, and fully traceable.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Universal Variance Tensor — Cross-Block Attention Network                    #
# --------------------------------------------------------------------------- #

class CrossBlockAttention(nn.Module):
    """Multi-head self-attention across pipeline blocks.

    Treats each block's 80-dim feature vector as a token and applies
    scaled dot-product attention to learn which blocks attend to which.
    The attention weights form the cross-block coupling matrix.

    Args:
        feature_dim: Dimension of each block's feature vector (80).
        n_heads: Number of attention heads.
        d_model: Internal projection dimension.
    """

    def __init__(self, feature_dim: int = 80, n_heads: int = 4, d_model: int = 32):
        super().__init__()
        if d_model < n_heads or d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be >= n_heads ({n_heads}) "
                f"and divisible by n_heads."
            )
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads

        # Project features to Q, K, V
        self.W_q = nn.Linear(feature_dim, d_model)
        self.W_k = nn.Linear(feature_dim, d_model)
        self.W_v = nn.Linear(feature_dim, d_model)

        # Output projection back to feature space
        self.W_out = nn.Linear(d_model, feature_dim)

        # Cross-feature covariance head: learns a bilinear form
        # that captures which feature dimensions covary across blocks
        self.cov_encoder = nn.Linear(feature_dim, d_model)
        self.cov_decoder = nn.Linear(d_model, feature_dim)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (n_blocks, feature_dim) block feature matrix.

        Returns:
            attended: (n_blocks, feature_dim) attention-mixed features.
            attn_weights: (n_heads, n_blocks, n_blocks) attention weights per head.
            cross_cov: (feature_dim, feature_dim) learned cross-feature covariance.
        """
        n = x.shape[0]

        Q = self.W_q(x).view(n, self.n_heads, self.d_head).transpose(0, 1)
        K = self.W_k(x).view(n, self.n_heads, self.d_head).transpose(0, 1)
        V = self.W_v(x).view(n, self.n_heads, self.d_head).transpose(0, 1)

        # Scaled dot-product attention: (n_heads, n_blocks, n_blocks)
        scale = self.d_head ** 0.5
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Attended values: (n_heads, n_blocks, d_head)
        attended = torch.bmm(attn_weights, V)
        attended = attended.transpose(0, 1).contiguous().view(n, self.d_model)
        attended = self.W_out(attended)

        # Cross-feature covariance via bilinear encoding
        cov_h = self.cov_encoder(x)  # (n_blocks, d_model)
        cross_cov = cov_h.T @ cov_h / n  # (d_model, d_model)
        # Project back to feature space: W is (feature_dim, d_model)
        W = self.cov_decoder.weight  # (feature_dim, d_model)
        cross_cov_full = W @ cross_cov @ W.T  # (feature_dim, feature_dim)

        return attended, attn_weights, cross_cov_full


class UVTNet(nn.Module):
    """Universal Variance Tensor network.

    Wraps CrossBlockAttention with a reconstruction objective:
    the network must reconstruct each block's features after mixing
    information across blocks. This forces the attention weights to
    capture genuine cross-block variance structure.

    Args:
        feature_dim: Feature dimension (80).
        n_heads: Number of attention heads.
        d_model: Internal dimension.
    """

    def __init__(self, feature_dim: int = 80, n_heads: int = 4, d_model: int = 32):
        super().__init__()
        self.attention = CrossBlockAttention(feature_dim, n_heads, d_model)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with reconstruction loss components.

        Returns:
            x_hat: (n_blocks, feature_dim) reconstructed features.
            attn_weights: (n_heads, n_blocks, n_blocks) cross-block attention.
            cross_cov: (feature_dim, feature_dim) cross-feature covariance.
            attended: (n_blocks, feature_dim) attention-mixed representations.
        """
        attended, attn_weights, cross_cov = self.attention(x)
        # Reconstruct original features from attended representations
        x_hat = self.reconstruction_head(attended)
        return x_hat, attn_weights, cross_cov, attended

    def compute_loss(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute reconstruction + regularization loss."""
        x_hat, attn_weights, cross_cov, _ = self.forward(x)
        recon = F.mse_loss(x_hat, x)
        # Entropy regularization: encourage attention to be neither
        # uniform nor fully peaked — learn genuine structure
        attn_entropy = -(attn_weights * (attn_weights + 1e-8).log()).sum(dim=-1).mean()
        n_blocks = x.shape[0]
        max_entropy = float(np.log(max(n_blocks, 2)))
        entropy_target = 0.5 * max_entropy
        entropy_reg = 0.01 * (attn_entropy - entropy_target).pow(2)
        total = recon + entropy_reg
        return total, {
            "recon": recon.item(),
            "entropy": attn_entropy.item(),
            "total": total.item(),
        }


# --------------------------------------------------------------------------- #
# Universal Semantic Encoder                                                    #
# --------------------------------------------------------------------------- #

class SemanticEncoderNet(nn.Module):
    """Universal Semantic Encoder: fuses UKT features + cross-block coupling
    into a single unified semantic vector.

    Architecture:
        UKT features (n_blocks, 80) → flatten → fuse with coupling features
        → bottleneck (semantic_dim) → decode to reconstructed features

    The bottleneck vector IS the Universal Semantic Encoding: a compact
    representation of the system's entire multi-modal state.

    Args:
        feature_dim: Per-block feature dimension (80).
        max_blocks: Maximum number of blocks (5).
        coupling_dim: Dimension of flattened coupling features.
        semantic_dim: Size of the semantic bottleneck.
    """

    def __init__(
        self,
        feature_dim: int = 80,
        max_blocks: int = 5,
        coupling_dim: int = 25,
        semantic_dim: int = 24,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_blocks = max_blocks
        self.semantic_dim = semantic_dim

        input_dim = max_blocks * feature_dim + coupling_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, semantic_dim),
            nn.Tanh(),  # Bound encoding to [-1, 1]
        )

        self.decoder = nn.Sequential(
            nn.Linear(semantic_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, max_blocks * feature_dim),
        )

    def forward(
        self, features_flat: torch.Tensor, coupling_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode to semantic vector and decode back.

        Args:
            features_flat: (1, max_blocks * feature_dim) padded UKT features.
            coupling_flat: (1, coupling_dim) flattened coupling matrix.

        Returns:
            encoding: (1, semantic_dim) the universal semantic encoding.
            decoded: (1, max_blocks * feature_dim) reconstructed features.
        """
        combined = torch.cat([features_flat, coupling_flat], dim=-1)
        encoding = self.encoder(combined)
        decoded = self.decoder(encoding)
        return encoding, decoded

    def compute_loss(
        self,
        features_flat: torch.Tensor,
        coupling_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Reconstruction loss on the features."""
        encoding, decoded = self.forward(features_flat, coupling_flat)
        recon = F.mse_loss(decoded, features_flat)
        # Encourage use of all semantic dimensions (anti-collapse)
        dim_usage = encoding.var(dim=-1).mean()
        usage_reg = 0.01 / (dim_usage + 1e-6)
        total = recon + usage_reg
        return total, {
            "recon": recon.item(),
            "dim_usage": dim_usage.item(),
            "total": total.item(),
        }


# --------------------------------------------------------------------------- #
# Training functions                                                            #
# --------------------------------------------------------------------------- #

def compute_universal_variance_tensor(
    final_matrix: np.ndarray,
    n_heads: int = 4,
    d_model: int = 32,
    epochs: int = 120,
    lr: float = 0.005,
    registry: Any = None,
    block_names: list[str] | None = None,
) -> dict | None:
    """Train the cross-block attention network and extract the UVT.

    Args:
        final_matrix: (n_blocks, 80) UKT feature matrix.
        n_heads: Number of attention heads.
        d_model: Internal projection dimension.
        epochs: Training epochs.
        lr: Learning rate.
        registry: Optional FeatureRegionRegistry for labeling.

    Returns:
        Dict with coupling_matrix, cross_covariance, variance_decomposition,
        attended_features, attention_per_head, loss_history, block_names.
        None on failure.
    """
    try:
        if final_matrix.shape[0] < 2:
            return None

        n_blocks, feature_dim = final_matrix.shape
        model = UVTNet(feature_dim, n_heads, d_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X = torch.tensor(final_matrix, dtype=torch.float32)

        # Data augmentation: noise perturbations
        augmented = [X]
        for s in [0.02, 0.05, 0.08]:
            augmented.append(X + torch.randn_like(X) * s)

        loss_history: list[float] = []
        for epoch in range(epochs):
            total_loss = 0.0
            for X_batch in augmented:
                loss, _ = model.compute_loss(X_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss_history.append(total_loss / len(augmented))

        # Extract results
        with torch.no_grad():
            x_hat, attn_weights, cross_cov, attended = model.forward(X)

        # Coupling matrix: average attention across heads → (n_blocks, n_blocks)
        # This shows how much each block "attends to" every other block
        attn_np = attn_weights.cpu().numpy()  # (n_heads, n_blocks, n_blocks)
        coupling_matrix = attn_np.mean(axis=0)  # (n_blocks, n_blocks)

        # Cross-feature covariance: (feature_dim, feature_dim)
        cross_cov_np = cross_cov.cpu().numpy()

        # Variance decomposition via eigendecomposition of coupling matrix
        eigenvalues, eigenvectors = np.linalg.eigh(
            coupling_matrix + coupling_matrix.T,  # Symmetrize
        )
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Normalize to get variance explained
        total_var = np.abs(eigenvalues).sum() + 1e-8
        variance_explained = np.abs(eigenvalues) / total_var

        # Cross-feature variance decomposition: top eigenvalues of cross-cov
        cov_sym = (cross_cov_np + cross_cov_np.T) / 2
        cov_eigenvalues, cov_eigenvectors = np.linalg.eigh(cov_sym)
        cov_idx = np.argsort(cov_eigenvalues)[::-1]
        cov_eigenvalues = cov_eigenvalues[cov_idx]
        cov_eigenvectors = cov_eigenvectors[:, cov_idx]
        cov_total = np.abs(cov_eigenvalues).sum() + 1e-8
        cov_variance_explained = np.abs(cov_eigenvalues) / cov_total

        # Resolve block names
        if block_names is None:
            default_names = ["Finance", "Clusters", "Graph", "Spatial", "Agents"]
            block_names = default_names[:n_blocks]

        # Label cross-block coupling modes
        coupling_labels = _label_coupling_modes(
            coupling_matrix, eigenvectors, eigenvalues,
            variance_explained, n_blocks, registry,
            block_names=block_names,
        )

        # Label cross-feature variance modes
        n_feature_modes = min(8, len(cov_eigenvalues))
        feature_variance_modes = _label_feature_variance_modes(
            cov_eigenvectors[:, :n_feature_modes],
            cov_eigenvalues[:n_feature_modes],
            cov_variance_explained[:n_feature_modes],
            registry,
        )

        return dict(
            coupling_matrix=coupling_matrix,
            cross_covariance=cross_cov_np,
            attention_per_head=attn_np,
            attended_features=attended.cpu().numpy(),
            variance_decomposition=dict(
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                variance_explained=variance_explained,
            ),
            feature_variance=dict(
                eigenvalues=cov_eigenvalues[:n_feature_modes],
                eigenvectors=cov_eigenvectors[:, :n_feature_modes],
                variance_explained=cov_variance_explained[:n_feature_modes],
            ),
            coupling_labels=coupling_labels,
            feature_variance_modes=feature_variance_modes,
            reconstruction_error=float(F.mse_loss(x_hat, X).item()),
            loss_history=loss_history,
            block_names=block_names,
        )
    except Exception as exc:
        warnings.warn(
            f"compute_universal_variance_tensor failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def compute_universal_semantic_encoding(
    final_matrix: np.ndarray,
    uvt_result: dict,
    canvas: Any = None,
    semantic_dim: int = 24,
    epochs: int = 150,
    lr: float = 0.003,
    registry: Any = None,
    block_names: list[str] | None = None,
) -> dict | None:
    """Train the semantic encoder and produce the Universal Semantic Encoding.

    Fuses UKT features with cross-block coupling from the UVT into a single
    compact semantic vector that captures the system's multi-modal state.

    Args:
        final_matrix: (n_blocks, 80) UKT feature matrix.
        uvt_result: Output from compute_universal_variance_tensor().
        canvas: Optional SemanticCanvas for dimension labeling.
        semantic_dim: Size of the semantic bottleneck vector.
        epochs: Training epochs.
        lr: Learning rate.
        registry: Optional FeatureRegionRegistry.

    Returns:
        Dict with encoding, decoded_features, dimension_labels,
        alignment_scores, loss_history.
        None on failure.
    """
    try:
        if final_matrix.shape[0] < 2:
            return None

        n_blocks, feature_dim = final_matrix.shape
        max_blocks = 5

        # Pad features to fixed size (max_blocks * feature_dim)
        padded = np.zeros((max_blocks, feature_dim))
        padded[:n_blocks] = final_matrix
        features_flat = padded.flatten().reshape(1, -1)

        # Flatten coupling matrix to fixed size (max_blocks * max_blocks)
        coupling = uvt_result["coupling_matrix"]
        coupling_padded = np.zeros((max_blocks, max_blocks))
        coupling_padded[:n_blocks, :n_blocks] = coupling
        coupling_flat = coupling_padded.flatten().reshape(1, -1)
        coupling_dim = max_blocks * max_blocks

        model = SemanticEncoderNet(
            feature_dim=feature_dim,
            max_blocks=max_blocks,
            coupling_dim=coupling_dim,
            semantic_dim=semantic_dim,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X_feat = torch.tensor(features_flat, dtype=torch.float32)
        X_coup = torch.tensor(coupling_flat, dtype=torch.float32)

        # Augmented training: add noise to both inputs
        loss_history: list[float] = []
        for epoch in range(epochs):
            noise_f = torch.randn_like(X_feat) * 0.03
            noise_c = torch.randn_like(X_coup) * 0.02
            # Train on original + noisy variants
            loss_sum = 0.0
            for xf, xc in [(X_feat, X_coup),
                            (X_feat + noise_f, X_coup + noise_c)]:
                loss, _ = model.compute_loss(xf, xc)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            loss_history.append(loss_sum / 2)

        # Extract the encoding
        with torch.no_grad():
            encoding, decoded = model.forward(X_feat, X_coup)

        encoding_np = encoding.cpu().numpy().flatten()  # (semantic_dim,)
        decoded_np = decoded.cpu().numpy().reshape(max_blocks, feature_dim)

        # Reconstruction quality per block
        recon_per_block = {}
        for i in range(n_blocks):
            recon_err = float(np.mean((decoded_np[i] - final_matrix[i]) ** 2))
            recon_per_block[i] = recon_err

        # Label semantic dimensions
        dimension_labels = _label_semantic_dimensions(
            encoding_np, final_matrix, uvt_result, canvas, registry,
            block_names=block_names,
        )

        # Resolve block names
        if block_names is None:
            default_names = ["Finance", "Clusters", "Graph", "Spatial", "Agents"]
            block_names = default_names[:n_blocks]

        # Cross-modal alignment scores: how well does each pair of blocks
        # align in the semantic encoding space?
        alignment_scores = _compute_alignment_scores(
            model, final_matrix, coupling, max_blocks, feature_dim, coupling_dim,
            block_names=block_names,
        )

        return dict(
            encoding=encoding_np,
            decoded_features=decoded_np[:n_blocks],
            dimension_labels=dimension_labels,
            alignment_scores=alignment_scores,
            reconstruction_per_block=recon_per_block,
            loss_history=loss_history,
            semantic_dim=semantic_dim,
        )
    except Exception as exc:
        warnings.warn(
            f"compute_universal_semantic_encoding failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


# --------------------------------------------------------------------------- #
# Labeling helpers                                                              #
# --------------------------------------------------------------------------- #

def _label_coupling_modes(
    coupling_matrix: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    variance_explained: np.ndarray,
    n_blocks: int,
    registry: Any = None,
    block_names: list[str] | None = None,
) -> list[dict]:
    """Label the cross-block coupling modes from eigendecomposition."""
    labels = []
    if block_names is None:
        block_names = ["Finance", "Clusters", "Graph", "Spatial", "Agents"]

    for mode_idx in range(min(n_blocks, len(eigenvalues))):
        ev = eigenvectors[:, mode_idx]
        # Top 2 contributing blocks
        top_blocks = np.argsort(np.abs(ev[:n_blocks]))[::-1][:2]
        block_pair = [block_names[i] if i < len(block_names) else f"Block_{i}"
                      for i in top_blocks]

        # Coupling strength between top 2 blocks
        i, j = top_blocks[0], top_blocks[1]
        mutual_attention = float((coupling_matrix[i, j] + coupling_matrix[j, i]) / 2)

        # Interpretation
        var_pct = variance_explained[mode_idx]
        if var_pct > 0.3:
            strength = "dominant"
        elif var_pct > 0.15:
            strength = "significant"
        else:
            strength = "minor"

        labels.append(dict(
            mode_id=f"M{mode_idx}",
            mode_idx=mode_idx,
            variance_explained=float(var_pct),
            eigenvalue=float(eigenvalues[mode_idx]),
            primary_blocks=block_pair,
            mutual_attention=mutual_attention,
            loadings={block_names[i] if i < len(block_names) else f"Block_{i}": float(ev[i])
                      for i in range(n_blocks)},
            label=f"M{mode_idx}: {block_pair[0]}↔{block_pair[1]} ({strength}, {var_pct:.0%})",
            narrative=(
                f"Coupling mode M{mode_idx} [{strength}] — explains {var_pct:.0%} of "
                f"cross-block variance. Primary coupling between {block_pair[0]} and "
                f"{block_pair[1]} (mutual attention: {mutual_attention:.3f}). "
                f"This mode captures how {block_pair[0].lower()} signals co-vary with "
                f"{block_pair[1].lower()} signals across the UKT."
            ),
        ))

    return labels


def _label_feature_variance_modes(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    variance_explained: np.ndarray,
    registry: Any = None,
) -> list[dict]:
    """Label the top cross-feature variance modes."""
    from hyperspace.models.knowledge_matrix import FEATURE_REGION_LABELS

    modes = []
    for mode_idx in range(len(eigenvalues)):
        ev = eigenvectors[:, mode_idx]
        # Identify which feature regions this mode spans
        region_energy = {}
        for (lo, hi), region_name in FEATURE_REGION_LABELS.items():
            energy = float(np.abs(ev[lo:hi]).sum())
            region_energy[region_name] = energy

        total_energy = sum(region_energy.values()) + 1e-8
        region_shares = {k: v / total_energy for k, v in region_energy.items()}
        dominant_regions = sorted(region_shares, key=region_shares.get, reverse=True)[:2]

        # Top 3 features
        top_feat_idx = np.argsort(np.abs(ev))[-3:][::-1]
        top_features = []
        for fi in top_feat_idx:
            name = registry.feature_name(int(fi)) if registry else f"feature_{fi}"
            top_features.append(dict(index=int(fi), name=name, loading=float(ev[fi])))

        modes.append(dict(
            mode_id=f"FV{mode_idx}",
            variance_explained=float(variance_explained[mode_idx]),
            eigenvalue=float(eigenvalues[mode_idx]),
            region_shares=region_shares,
            dominant_regions=dominant_regions,
            top_features=top_features,
            label=(
                f"FV{mode_idx}: {' + '.join(r.replace('-', ' ') for r in dominant_regions)} "
                f"({variance_explained[mode_idx]:.0%})"
            ),
        ))

    return modes


def _label_semantic_dimensions(
    encoding: np.ndarray,
    final_matrix: np.ndarray,
    uvt_result: dict,
    canvas: Any = None,
    registry: Any = None,
    block_names: list[str] | None = None,
) -> list[dict]:
    """Label each dimension of the Universal Semantic Encoding."""
    n_blocks = final_matrix.shape[0]
    if block_names is None:
        block_names = ["Finance", "Clusters", "Graph", "Spatial", "Agents"][:n_blocks]

    labels = []
    for dim_idx in range(len(encoding)):
        val = float(encoding[dim_idx])
        abs_val = abs(val)

        # Strength classification
        if abs_val > 0.7:
            strength = "STRONG"
        elif abs_val > 0.3:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        polarity = "positive" if val >= 0 else "negative"

        # Map to canvas dimension if available
        canvas_label = None
        if canvas is not None:
            canvas_state = canvas.get_accumulated_state()
            canvas_dims = canvas_state.get("dimensions", [])
            if dim_idx < len(canvas_dims):
                canvas_label = canvas_dims[dim_idx].get("label")

        labels.append(dict(
            dim_idx=dim_idx,
            value=val,
            abs_value=abs_val,
            strength=strength,
            polarity=polarity,
            canvas_dimension=canvas_label,
            label=f"SE{dim_idx:02d}: {strength} {polarity} ({val:+.3f})",
        ))

    return labels


def _compute_alignment_scores(
    model: SemanticEncoderNet,
    final_matrix: np.ndarray,
    coupling: np.ndarray,
    max_blocks: int,
    feature_dim: int,
    coupling_dim: int,
    block_names: list[str] | None = None,
) -> list[dict]:
    """Compute pairwise cross-modal alignment scores.

    For each pair of blocks, measures how much removing one block changes
    the semantic encoding of the other — high sensitivity indicates strong
    cross-modal alignment.
    """
    n_blocks = final_matrix.shape[0]
    if block_names is None:
        block_names = ["Finance", "Clusters", "Graph", "Spatial", "Agents"][:n_blocks]

    # Get baseline encoding
    padded = np.zeros((max_blocks, feature_dim))
    padded[:n_blocks] = final_matrix
    coup_padded = np.zeros((max_blocks, max_blocks))
    coup_padded[:n_blocks, :n_blocks] = coupling

    X_feat = torch.tensor(padded.flatten().reshape(1, -1), dtype=torch.float32)
    X_coup = torch.tensor(coup_padded.flatten().reshape(1, -1), dtype=torch.float32)

    with torch.no_grad():
        baseline_enc, _ = model.forward(X_feat, X_coup)
        baseline = baseline_enc.cpu().numpy().flatten()

    scores = []
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            # Zero out block j and measure encoding change
            ablated = padded.copy()
            ablated[j] = 0.0
            X_abl = torch.tensor(ablated.flatten().reshape(1, -1), dtype=torch.float32)
            with torch.no_grad():
                abl_enc, _ = model.forward(X_abl, X_coup)
                abl = abl_enc.cpu().numpy().flatten()

            # Cosine distance as alignment measure
            cos_sim = float(np.dot(baseline, abl) / (
                np.linalg.norm(baseline) * np.linalg.norm(abl) + 1e-8
            ))
            sensitivity = 1.0 - cos_sim  # Higher = more aligned/coupled

            scores.append(dict(
                block_a=block_names[i],
                block_b=block_names[j],
                alignment=float(sensitivity),
                cosine_similarity=cos_sim,
                label=(
                    f"{block_names[i]}↔{block_names[j]}: "
                    f"alignment={sensitivity:.3f}"
                ),
            ))

    return scores
