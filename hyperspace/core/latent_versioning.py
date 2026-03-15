"""Latent space versioning for governance reproducibility.

Captures a fingerprint of the latent space configuration at each pipeline
run — feature dimensions, region boundaries, shared-latent encoder status,
and concept vocabulary snapshot.  Enables cross-run auditing to detect when
the latent representation structure changes, supporting governance
requirements for reproducibility and contestability.

Usage:
    version = compute_latent_version(ukt, sae_result, feature_flags)
    trail.record(run_id, version)
    trail.save(path)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LatentVersion:
    """Fingerprint of the latent space configuration for a single run."""

    run_id: str
    timestamp: str
    feature_dim: int
    n_regions: int
    region_boundaries: dict[str, tuple[int, int]]
    n_kernels: int
    shared_latent_active: bool
    concept_count: int
    active_concept_count: int
    config_hash: str  # SHA-256 of deterministic config snapshot


@dataclass
class ConceptVocabularyEntry:
    """Single concept from the SAE vocabulary with provenance."""

    concept_id: str
    dominant_region: str
    mean_activation: float
    active: bool
    top_features: list[dict[str, Any]]


@dataclass
class ConceptAuditRecord:
    """Snapshot of the concept vocabulary at a specific run."""

    run_id: str
    timestamp: str
    total_concepts: int
    active_concepts: int
    vocabulary: list[ConceptVocabularyEntry]
    vocabulary_hash: str  # SHA-256 of concept labels for drift detection


def compute_latent_version(
    *,
    run_id: str,
    timestamp: str,
    feature_dim: int,
    region_labels: dict[tuple[int, int], str],
    n_kernels: int,
    shared_latent_active: bool,
    sae_result: dict | None = None,
) -> LatentVersion:
    """Compute a latent space version fingerprint for the current run.

    Args:
        run_id: Pipeline run identifier.
        timestamp: Run timestamp string.
        feature_dim: UKT feature dimension (e.g. 80).
        region_labels: FEATURE_REGION_LABELS mapping (lo,hi) → name.
        n_kernels: Number of kernels in the final UKT decomposition.
        shared_latent_active: Whether shared-latent prototype was active.
        sae_result: SAE output dict (concept_labels, etc.).

    Returns:
        LatentVersion fingerprint.
    """
    region_boundaries = {
        name: (lo, hi) for (lo, hi), name in region_labels.items()
    }

    concept_count = 0
    active_concept_count = 0
    if sae_result:
        labels = sae_result.get("concept_labels", [])
        concept_count = len(labels)
        active_concept_count = sum(
            1 for c in labels if c.get("active", False)
        )

    # Build deterministic config snapshot for hashing
    config_snapshot = {
        "feature_dim": feature_dim,
        "n_regions": len(region_boundaries),
        "region_boundaries": {
            name: list(bounds)
            for name, bounds in sorted(region_boundaries.items())
        },
        "shared_latent_active": shared_latent_active,
        "concept_count": concept_count,
    }
    config_hash = hashlib.sha256(
        json.dumps(config_snapshot, sort_keys=True).encode()
    ).hexdigest()[:16]

    return LatentVersion(
        run_id=run_id,
        timestamp=timestamp,
        feature_dim=feature_dim,
        n_regions=len(region_boundaries),
        region_boundaries=region_boundaries,
        n_kernels=n_kernels,
        shared_latent_active=shared_latent_active,
        concept_count=concept_count,
        active_concept_count=active_concept_count,
        config_hash=config_hash,
    )


def build_concept_audit_record(
    *,
    run_id: str,
    timestamp: str,
    sae_result: dict | None = None,
) -> ConceptAuditRecord | None:
    """Build a concept vocabulary audit record from SAE results.

    Returns None if no SAE result is available.
    """
    if not sae_result:
        return None

    labels = sae_result.get("concept_labels", [])
    if not labels:
        return None

    vocabulary: list[ConceptVocabularyEntry] = []
    for concept in labels:
        vocabulary.append(ConceptVocabularyEntry(
            concept_id=concept.get("concept_id", f"C{len(vocabulary):02d}"),
            dominant_region=concept.get("dominant_region", "unknown"),
            mean_activation=float(concept.get("mean_activation", 0.0)),
            active=bool(concept.get("active", False)),
            top_features=concept.get("top_features", []),
        ))

    # Hash vocabulary structure for drift detection
    vocab_snapshot = [
        {
            "id": v.concept_id,
            "region": v.dominant_region,
            "active": v.active,
        }
        for v in vocabulary
    ]
    vocab_hash = hashlib.sha256(
        json.dumps(vocab_snapshot, sort_keys=True).encode()
    ).hexdigest()[:16]

    return ConceptAuditRecord(
        run_id=run_id,
        timestamp=timestamp,
        total_concepts=len(vocabulary),
        active_concepts=sum(1 for v in vocabulary if v.active),
        vocabulary=vocabulary,
        vocabulary_hash=vocab_hash,
    )


class LatentVersionTrail:
    """Persistent audit trail of latent space versions across runs.

    Stores a rolling history of LatentVersion and ConceptAuditRecord
    entries for governance export and structural drift detection.
    """

    def __init__(self, max_entries: int = 100) -> None:
        self._versions: list[LatentVersion] = []
        self._concept_records: list[ConceptAuditRecord] = []
        self._max_entries = max_entries

    @property
    def n_versions(self) -> int:
        return len(self._versions)

    @property
    def n_concept_records(self) -> int:
        return len(self._concept_records)

    def record(
        self,
        version: LatentVersion,
        concept_record: ConceptAuditRecord | None = None,
    ) -> None:
        """Record a latent space version and optional concept audit."""
        self._versions.append(version)
        if concept_record is not None:
            self._concept_records.append(concept_record)

        # Enforce rolling window
        if len(self._versions) > self._max_entries:
            self._versions = self._versions[-self._max_entries:]
        if len(self._concept_records) > self._max_entries:
            self._concept_records = self._concept_records[-self._max_entries:]

    def detect_structural_changes(self) -> list[dict[str, Any]]:
        """Detect runs where the latent space structure changed.

        Compares consecutive version hashes and reports any changes with
        details about what shifted.
        """
        changes: list[dict[str, Any]] = []
        for i in range(1, len(self._versions)):
            prev, curr = self._versions[i - 1], self._versions[i]
            if prev.config_hash != curr.config_hash:
                diffs: list[str] = []
                if prev.feature_dim != curr.feature_dim:
                    diffs.append(f"feature_dim: {prev.feature_dim} → {curr.feature_dim}")
                if prev.n_regions != curr.n_regions:
                    diffs.append(f"n_regions: {prev.n_regions} → {curr.n_regions}")
                if prev.shared_latent_active != curr.shared_latent_active:
                    diffs.append(
                        f"shared_latent: {prev.shared_latent_active} → {curr.shared_latent_active}"
                    )
                if prev.concept_count != curr.concept_count:
                    diffs.append(f"concepts: {prev.concept_count} → {curr.concept_count}")
                changes.append({
                    "run_id": curr.run_id,
                    "timestamp": curr.timestamp,
                    "previous_hash": prev.config_hash,
                    "current_hash": curr.config_hash,
                    "diffs": diffs,
                })
        return changes

    def detect_vocabulary_drift(self) -> list[dict[str, Any]]:
        """Detect runs where the concept vocabulary changed."""
        drifts: list[dict[str, Any]] = []
        for i in range(1, len(self._concept_records)):
            prev, curr = self._concept_records[i - 1], self._concept_records[i]
            if prev.vocabulary_hash != curr.vocabulary_hash:
                drifts.append({
                    "run_id": curr.run_id,
                    "timestamp": curr.timestamp,
                    "previous_hash": prev.vocabulary_hash,
                    "current_hash": curr.vocabulary_hash,
                    "active_delta": curr.active_concepts - prev.active_concepts,
                    "total_delta": curr.total_concepts - prev.total_concepts,
                })
        return drifts

    def get_summary(self) -> dict[str, Any]:
        """Return a JSON-safe summary for governance export."""
        latest = self._versions[-1] if self._versions else None
        structural_changes = self.detect_structural_changes()
        vocab_drifts = self.detect_vocabulary_drift()

        summary: dict[str, Any] = {
            "n_versions": self.n_versions,
            "n_concept_records": self.n_concept_records,
            "structural_changes_detected": len(structural_changes),
            "vocabulary_drifts_detected": len(vocab_drifts),
        }

        if latest:
            summary["latest"] = {
                "run_id": latest.run_id,
                "config_hash": latest.config_hash,
                "feature_dim": latest.feature_dim,
                "n_regions": latest.n_regions,
                "n_kernels": latest.n_kernels,
                "shared_latent_active": latest.shared_latent_active,
                "concept_count": latest.concept_count,
                "active_concept_count": latest.active_concept_count,
            }

        if structural_changes:
            summary["structural_changes"] = structural_changes[-3:]

        if vocab_drifts:
            summary["vocabulary_drifts"] = vocab_drifts[-3:]

        return summary

    # ── Disk serialization ─────────────────────────────────────────────── #

    def save(self, path: str | Path) -> None:
        """Persist the version trail to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_entries": self._max_entries,
            "versions": [_version_to_dict(v) for v in self._versions],
            "concept_records": [
                _concept_record_to_dict(r) for r in self._concept_records
            ],
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LatentVersionTrail":
        """Load a version trail from a previously saved JSON file."""
        path = Path(path)
        payload = json.loads(path.read_text())
        trail = cls(max_entries=payload.get("max_entries", 100))
        trail._versions = [
            _version_from_dict(d) for d in payload.get("versions", [])
        ]
        trail._concept_records = [
            _concept_record_from_dict(d)
            for d in payload.get("concept_records", [])
        ]
        return trail


# ── Serialization helpers ──────────────────────────────────────────────── #

def _version_to_dict(v: LatentVersion) -> dict[str, Any]:
    return {
        "run_id": v.run_id,
        "timestamp": v.timestamp,
        "feature_dim": v.feature_dim,
        "n_regions": v.n_regions,
        "region_boundaries": {
            name: list(bounds) for name, bounds in v.region_boundaries.items()
        },
        "n_kernels": v.n_kernels,
        "shared_latent_active": v.shared_latent_active,
        "concept_count": v.concept_count,
        "active_concept_count": v.active_concept_count,
        "config_hash": v.config_hash,
    }


def _version_from_dict(d: dict[str, Any]) -> LatentVersion:
    return LatentVersion(
        run_id=d["run_id"],
        timestamp=d["timestamp"],
        feature_dim=d["feature_dim"],
        n_regions=d["n_regions"],
        region_boundaries={
            name: tuple(bounds)
            for name, bounds in d["region_boundaries"].items()
        },
        n_kernels=d["n_kernels"],
        shared_latent_active=d["shared_latent_active"],
        concept_count=d["concept_count"],
        active_concept_count=d["active_concept_count"],
        config_hash=d["config_hash"],
    )


def _concept_entry_to_dict(e: ConceptVocabularyEntry) -> dict[str, Any]:
    return {
        "concept_id": e.concept_id,
        "dominant_region": e.dominant_region,
        "mean_activation": e.mean_activation,
        "active": e.active,
        "top_features": e.top_features,
    }


def _concept_entry_from_dict(d: dict[str, Any]) -> ConceptVocabularyEntry:
    return ConceptVocabularyEntry(
        concept_id=d["concept_id"],
        dominant_region=d["dominant_region"],
        mean_activation=float(d["mean_activation"]),
        active=bool(d["active"]),
        top_features=d.get("top_features", []),
    )


def _concept_record_to_dict(r: ConceptAuditRecord) -> dict[str, Any]:
    return {
        "run_id": r.run_id,
        "timestamp": r.timestamp,
        "total_concepts": r.total_concepts,
        "active_concepts": r.active_concepts,
        "vocabulary": [_concept_entry_to_dict(e) for e in r.vocabulary],
        "vocabulary_hash": r.vocabulary_hash,
    }


def _concept_record_from_dict(d: dict[str, Any]) -> ConceptAuditRecord:
    return ConceptAuditRecord(
        run_id=d["run_id"],
        timestamp=d["timestamp"],
        total_concepts=d["total_concepts"],
        active_concepts=d["active_concepts"],
        vocabulary=[_concept_entry_from_dict(e) for e in d.get("vocabulary", [])],
        vocabulary_hash=d["vocabulary_hash"],
    )
