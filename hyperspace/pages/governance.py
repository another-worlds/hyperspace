"""Governance module: provenance tracing, policy language mode, annotations, jurisdiction labels.

Features implemented:
    A2 - Provenance Trace Panel: sidebar selectbox for full feature audit chain
    B1 - Policy Language Mode: render kernel narratives in non-technical prose
    C1 - Multi-Stakeholder Annotations: role-tagged notes on kernels/concepts
    C2 - Jurisdiction Labels: data source legal geography tags
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import streamlit as st

from hyperspace.config import (
    DATA_SOURCE_JURISDICTIONS,
    FEATURE_NAMES,
    GLOSSARY,
    GOVERNANCE_FLAG_CODES,
    POLICY_CONFIDENCE_BANDS,
    POLICY_KERNEL_NAMES,
    UKT_FEATURE_DIM,
)


# --------------------------------------------------------------------------- #
# C2: Jurisdiction label helpers                                               #
# --------------------------------------------------------------------------- #

def _infer_jurisdiction(source_string: str) -> dict:
    """Infer jurisdiction metadata from a data source label string."""
    s = source_string.lower()
    if "yfinance" in s or "yahoo" in s or ("live" in s and "finance" in s):
        return DATA_SOURCE_JURISDICTIONS["yfinance"]
    if "gdelt" in s:
        return DATA_SOURCE_JURISDICTIONS["GDELT"]
    if "harvard" in s or "dataverse" in s or "voeten" in s or "un vote" in s:
        return DATA_SOURCE_JURISDICTIONS["Harvard Dataverse"]
    if "synthetic" in s or "fallback" in s or "mock" in s:
        return DATA_SOURCE_JURISDICTIONS["synthetic"]
    return DATA_SOURCE_JURISDICTIONS["fallback"]


def render_jurisdiction_badges(data_sources: dict) -> None:
    """C2: Render jurisdiction tags for all active data sources."""
    if not data_sources:
        return
    st.markdown("**Data Source Jurisdictions**")
    parts = []
    for block, src in data_sources.items():
        jur = _infer_jurisdiction(src)
        tag = jur["tag"]
        note = jur["note"]
        parts.append(
            f'<span class="jurisdiction-badge" title="{note}">'
            f'{block}: [{tag}]</span>'
        )
    st.markdown(" ".join(parts), unsafe_allow_html=True)
    st.caption(
        "Jurisdiction tags reflect the legal/regulatory regime governing each data source. "
        "SYNTHETIC data carries no legal standing as real-world observation."
    )


# --------------------------------------------------------------------------- #
# A2: Provenance Trace Panel                                                   #
# --------------------------------------------------------------------------- #

def _build_feature_chain(
    feature_idx: int,
    snapshots: list[dict],
) -> dict:
    """Build the complete provenance chain for a single feature index.

    Returns a dict with all available provenance information.
    """
    chain: dict[str, Any] = {}

    # Basic feature identity
    chain["feature_index"] = feature_idx
    if feature_idx < len(FEATURE_NAMES):
        chain["feature_name"] = FEATURE_NAMES[feature_idx]
    else:
        chain["feature_name"] = f"feature_{feature_idx}"

    # Source block (derived from feature index ranges)
    from hyperspace.models.knowledge_matrix import _DEFAULT_BLOCK_FEATURE_RANGES, BLOCK_REGION_MAP
    from hyperspace.config import REGION_DESCRIPTIONS
    source_block = "unknown"
    block_description = "Unknown data source"
    for block_name, (start, end) in _DEFAULT_BLOCK_FEATURE_RANGES.items():
        if start <= feature_idx < end:
            source_block = block_name
            break

    # Derive description from registry region (no hardcoded labels)
    region_name = BLOCK_REGION_MAP.get(source_block, (None,))[0]
    if region_name and region_name in REGION_DESCRIPTIONS:
        block_description = REGION_DESCRIPTIONS[region_name].split(".")[0] + "."
    elif source_block != "unknown":
        block_description = f"{source_block} block features."

    chain["source_block"] = source_block
    chain["block_description"] = block_description

    # Metadata from final snapshot
    if snapshots:
        final_snap = snapshots[-1]
        feature_meta = final_snap.get("feature_meta", {})
        meta = feature_meta.get(feature_idx, {})
        chain["label"] = meta.get("label", chain["feature_name"])
        chain["entity"] = meta.get("entity", "N/A")
        chain["metric"] = meta.get("metric", "N/A")
        chain["source"] = meta.get("source", "N/A")
        chain["time_scope"] = meta.get("time_scope", "N/A")
        chain["jurisdiction"] = _infer_jurisdiction(str(meta.get("source", "")))

        # Which kernels load this feature significantly?
        Vt = final_snap.get("Vt")
        importance = final_snap.get("importance", [])
        if Vt is not None and len(Vt.shape) == 2 and feature_idx < Vt.shape[1]:
            kernel_loadings = []
            for k_idx in range(Vt.shape[0]):
                loading = float(Vt[k_idx, feature_idx])
                if abs(loading) > 0.05:
                    kernel_loadings.append({
                        "kernel_id": f"K{k_idx}",
                        "loading": loading,
                        "importance": float(importance[k_idx]) if k_idx < len(importance) else 0.0,
                    })
            kernel_loadings.sort(key=lambda x: abs(x["loading"]), reverse=True)
            chain["kernel_loadings"] = kernel_loadings
        else:
            chain["kernel_loadings"] = []

        # Reality regression weight for this feature
        rr = final_snap.get("reality_regression")
        if rr is not None and feature_idx < len(rr):
            chain["reality_regression_weight"] = float(rr[feature_idx])
        else:
            chain["reality_regression_weight"] = 0.0

    # SAE concept connection
    concept_kernel_map = st.session_state.get("concept_kernel_map", [])
    sae_result = st.session_state.get("sae_result")
    chain["sae_concepts"] = []
    if sae_result and snapshots:
        final_snap = snapshots[-1]
        Vt = final_snap.get("Vt")
        if Vt is not None:
            concept_labels = sae_result.get("concept_labels", [])
            for cl in concept_labels:
                if not cl.get("active"):
                    continue
                top_feats = cl.get("top_features", [])
                for tf in top_feats:
                    if tf.get("index") == feature_idx:
                        concept_num = cl.get("concept_idx", int(cl.get("concept_id", "C00").lstrip("C")))
                        chain["sae_concepts"].append({
                            "concept_id": cl.get("concept_id", "C??"),
                            "label": cl.get("label", cl.get("concept_id", "C??")),
                            "loading": tf.get("loading", 0.0),
                            "mean_activation": sae_result.get(
                                "concept_activations",
                                np.zeros((1, 16))
                            )[:, concept_num].mean()
                            if sae_result.get("concept_activations") is not None
                            else 0.0,
                        })

    return chain


def render_provenance_panel(snapshots: list[dict]) -> None:
    """A2: Render the full provenance trace panel for a selected feature.

    This is designed to be called from the sidebar.
    """
    if not snapshots:
        st.caption("Run the pipeline to enable provenance tracing.")
        return

    # Build feature label list for selectbox
    final_snap = snapshots[-1]
    feature_meta = final_snap.get("feature_meta", {})

    feature_options = []
    for i in range(UKT_FEATURE_DIM):
        meta = feature_meta.get(i, {})
        label = meta.get("label") or (FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}")
        feature_options.append(f"[{i:02d}] {label}")

    selected = st.selectbox(
        "Select feature to trace:",
        feature_options,
        key="provenance_feature_select",
    )
    feature_idx = int(selected.split("]")[0].strip("["))

    chain = _build_feature_chain(feature_idx, snapshots)

    st.markdown("---")
    st.markdown(f"**Feature {feature_idx}: `{chain['feature_name']}`**")

    # Source Block
    st.markdown(
        f'<span class="concept-badge">{chain["source_block"]}</span>',
        unsafe_allow_html=True,
    )
    st.caption(chain.get("block_description", ""))

    # Source metadata
    jur = chain.get("jurisdiction", {})
    if jur.get("tag"):
        st.markdown(
            f'<span class="jurisdiction-badge">[{jur["tag"]}]</span> '
            f'<small>{jur.get("note", "")}</small>',
            unsafe_allow_html=True,
        )

    st.markdown("**Provenance Chain:**")
    st.markdown(
        f"- **Source:** {chain.get('source', 'N/A')}\n"
        f"- **Entity:** {chain.get('entity', 'N/A')}\n"
        f"- **Metric:** {chain.get('metric', 'N/A')}\n"
        f"- **Time scope:** {chain.get('time_scope', 'N/A')}\n"
        f"- **Source Block:** {chain.get('source_block', 'N/A')}"
    )

    # Kernel loadings
    kernel_loadings = chain.get("kernel_loadings", [])
    if kernel_loadings:
        st.markdown("**Kernel contributions:**")
        for kl in kernel_loadings[:3]:
            direction = "↑" if kl["loading"] > 0 else "↓"
            strength = "Strong" if abs(kl["loading"]) > 0.2 else "Moderate" if abs(kl["loading"]) > 0.1 else "Weak"
            st.caption(
                f"{direction} {kl['kernel_id']}: loading={kl['loading']:+.3f} "
                f"({strength}), kernel importance={kl['importance']:.1%}"
            )
    else:
        st.caption("Feature has negligible kernel loading.")

    # Reality regression
    rr_weight = chain.get("reality_regression_weight", 0.0)
    st.markdown(
        f"**Reality Regression weight:** `{rr_weight:+.4f}`"
    )
    if abs(rr_weight) > 0.05:
        st.caption("This feature materially influences the system's overall conclusion.")
    else:
        st.caption("This feature has minimal influence on the final conclusion.")

    # SAE concepts
    sae_concepts = chain.get("sae_concepts", [])
    if sae_concepts:
        st.markdown("**Discovered concepts containing this feature:**")
        for sc in sae_concepts:
            st.caption(
                f"• {sc['label']} — loading={sc['loading']:+.3f}, "
                f"mean activation={float(sc['mean_activation']):.3f}"
            )
    else:
        st.caption("Feature not prominently loaded in any active SAE concept.")


# --------------------------------------------------------------------------- #
# B1: Policy Language Mode helpers                                             #
# --------------------------------------------------------------------------- #

def _confidence_label(importance: float) -> tuple[str, str]:
    """Return (label, explanation) for a given importance value."""
    for threshold, label, explanation in POLICY_CONFIDENCE_BANDS:
        if importance >= threshold:
            return label, explanation
    return "Negligible", "This factor explains very little variance."


def render_kernel_policy_mode(kernel_labels: list[dict]) -> None:
    """B1: Render kernel narratives in policy-friendly language.

    Replaces technical terms with plain-English equivalents.
    """
    st.markdown("#### 📋 Policy Briefing: Key Analytical Findings")
    st.caption(
        "Governance Language Mode is active. Technical loading values have been "
        "replaced with confidence descriptors suitable for policy documents."
    )

    for i, kl in enumerate(kernel_labels):
        dominant_feature_block = kl.get("dominant_feature_block", "")
        importance = kl.get("importance", 0.0)
        policy_name = POLICY_KERNEL_NAMES.get(dominant_feature_block, f"Pattern {i + 1}")
        conf_label, conf_explanation = _confidence_label(importance)

        with st.expander(
            f"Finding {i + 1}: {policy_name} — {conf_label}",
            expanded=importance > 0.2,
        ):
            # Confidence badge
            badge_class = "gov-pass" if importance > 0.25 else "gov-flag"
            st.markdown(
                f'<span class="{badge_class}">{conf_label}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"*{conf_explanation}*")
            st.markdown("")

            # Plain-English narrative rewrite
            dominant_block = kl.get("dominant_block", "Unknown")
            top_features = kl.get("top_features", [])

            st.markdown(f"**Primary data domain:** {dominant_block}")
            st.markdown(
                f"**What this finding represents:** "
                f"This pattern draws primarily from {dominant_feature_block.replace('-', ' ')} "
                f"signals. {_policy_block_description(dominant_feature_block)}"
            )

            if top_features:
                st.markdown("**Key contributing signals:**")
                for tf in top_features[:3]:
                    direction = "positively" if tf.get("loading", 0) > 0 else "negatively"
                    st.markdown(
                        f"  - `{tf.get('name', 'unknown')}` contributes {direction} "
                        f"(from {tf.get('source_block', 'unknown')} block)"
                    )

            block_scores = kl.get("block_scores", {})
            if block_scores:
                total = sum(block_scores.values()) + 1e-8
                st.markdown("**Cross-block signal breakdown:**")
                for block_name, score in sorted(block_scores.items(), key=lambda x: x[1], reverse=True):
                    share = score / total
                    policy_b = POLICY_KERNEL_NAMES.get(block_name, block_name.replace("-", " "))
                    bar = "█" * int(share * 20) + "░" * (20 - int(share * 20))
                    st.caption(f"{policy_b}: {bar} {share:.0%}")

            # Copy-ready briefing text
            briefing = (
                f"Finding {i + 1} ({policy_name}): {conf_label}. "
                f"{conf_explanation} "
                f"Primary data domain: {dominant_block}. "
            )
            if top_features:
                signal_names = [tf.get("name", "") for tf in top_features[:2]]
                briefing += f"Key signals: {', '.join(signal_names)}."
            st.text_area(
                "📋 Copy briefing text:",
                briefing,
                height=80,
                key=f"policy_briefing_{i}",
                label_visibility="collapsed",
            )


def _policy_block_description(block_name: str) -> str:
    """Return a one-sentence policy-friendly description of a block."""
    from hyperspace.models.knowledge_matrix import BLOCK_REGION_MAP
    from hyperspace.config import REGION_DESCRIPTIONS
    region_name = BLOCK_REGION_MAP.get(block_name, (None,))[0]
    if region_name and region_name in REGION_DESCRIPTIONS:
        return REGION_DESCRIPTIONS[region_name]
    return "Signals from multiple data domains contribute to this pattern."


# --------------------------------------------------------------------------- #
# C1: Multi-Stakeholder Annotation Layer                                       #
# --------------------------------------------------------------------------- #

ANNOTATION_ROLES: list[str] = ["Technical", "Policy", "Legal", "Civil Society", "Diplomatic"]


def render_annotation_widget(
    kernel_id: str,
    label: str = "",
    key_suffix: str = "",
) -> None:
    """C1: Render annotation input for a single kernel or concept.

    Stores annotations in st.session_state.stakeholder_annotations.
    """
    # Show existing annotations for this kernel
    existing = [
        a for a in st.session_state.get("stakeholder_annotations", [])
        if a.get("kernel_id") == kernel_id
    ]
    if existing:
        st.markdown("**Stakeholder notes on this finding:**")
        for ann in existing:
            role = ann.get("role", "Unknown")
            text = ann.get("text", "")
            ts = ann.get("timestamp", "")
            st.markdown(
                f'<div class="contest-note">'
                f'<span class="annotation-tag">{role}</span> '
                f'<small style="color:#667788;">{ts}</small><br/>{text}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Input for new annotation
    with st.expander("➕ Add stakeholder note", expanded=False):
        role = st.selectbox(
            "Role",
            ANNOTATION_ROLES,
            key=f"ann_role_{kernel_id}_{key_suffix}",
        )
        text = st.text_area(
            "Note",
            placeholder="Enter your observation, objection, or contextual note...",
            height=80,
            key=f"ann_text_{kernel_id}_{key_suffix}",
        )
        if st.button("Save Note", key=f"ann_save_{kernel_id}_{key_suffix}"):
            if text.strip():
                ann_entry = {
                    "kernel_id": kernel_id,
                    "role": role,
                    "text": text.strip(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                if "stakeholder_annotations" not in st.session_state:
                    st.session_state.stakeholder_annotations = []
                st.session_state.stakeholder_annotations.append(ann_entry)
                st.success("Note saved.")


def render_annotations_summary() -> None:
    """C1: Render a summary of all stakeholder annotations for export."""
    annotations = st.session_state.get("stakeholder_annotations", [])
    if not annotations:
        st.caption("No stakeholder annotations recorded.")
        return

    st.markdown(f"**{len(annotations)} stakeholder annotation(s) recorded:**")
    import pandas as pd
    ann_df = pd.DataFrame(annotations)
    st.dataframe(ann_df, use_container_width=True, hide_index=True)

    run_id = st.session_state.get("run_id", "UNKNOWN")
    run_ts = st.session_state.get("run_timestamp", "")
    ann_df_export = ann_df.copy()
    ann_df_export.insert(0, "RunID", run_id)
    ann_df_export.insert(1, "RunTimestamp", run_ts)
    st.download_button(
        "Download Annotations (CSV)",
        ann_df_export.to_csv(index=False),
        f"hyperspace_annotations_{run_id}.csv",
        "text/csv",
        key="download_annotations",
    )


# --------------------------------------------------------------------------- #
# A1: Contest This — per-kernel contestability widget                          #
# --------------------------------------------------------------------------- #

def render_contest_popover(
    kernel_label: dict,
    stability: dict | None = None,
    key_suffix: str = "",
) -> None:
    """A1: Render a 'Contest This' popover for a kernel finding.

    Shows stability variance, uncertainty context, and annotation field.
    """
    kernel_id = kernel_label.get("kernel_id", "K?")
    importance = kernel_label.get("importance", 0.0)
    top_features = kernel_label.get("top_features", [])

    with st.popover(f"⚠️ Contest [{kernel_id}]"):
        st.markdown(f"### Contesting: {kernel_label.get('label', kernel_id)}")
        st.markdown("---")

        # Stability context
        st.markdown("**① Stability Assessment**")
        if stability and stability.get("n_runs", 0) > 0:
            mean_c = stability["mean_cosine"]
            min_c = stability["min_cosine"]
            std_c = stability["std_cosine"]

            if mean_c >= 0.90:
                stability_verdict = "🟢 **Highly stable** — conclusions are robust to minor data variations."
            elif mean_c >= 0.75:
                stability_verdict = "🟡 **Moderately stable** — conclusions hold under most perturbations but may shift with larger data changes."
            else:
                stability_verdict = "🔴 **Unstable** — conclusions change significantly under small data variations. High governance risk."

            st.markdown(stability_verdict)
            st.caption(
                f"Mean cosine similarity: {mean_c:.3f} | "
                f"Min: {min_c:.3f} | Std: {std_c:.3f} | "
                f"Runs: {stability['n_runs']}"
            )
        else:
            st.caption("Stability data not available.")

        # Uncertainty per feature
        st.markdown("**② Feature Uncertainty**")
        if top_features:
            for tf in top_features[:3]:
                abs_load = abs(tf.get("loading", 0.0))
                certainty = "High confidence" if abs_load > 0.3 else "Moderate confidence" if abs_load > 0.15 else "Low confidence"
                st.caption(
                    f"• `{tf.get('name', '?')}`: loading={tf.get('loading', 0):+.3f} → {certainty}"
                )
        else:
            st.caption("No feature data available.")

        # Importance context
        st.markdown("**③ Variance Explained**")
        st.caption(
            f"This kernel explains **{importance:.1%}** of total cross-modal variance. "
            + (
                "This is a dominant factor — contestation is high-priority."
                if importance > 0.35
                else "This is a secondary factor — conclusions have limited overall weight."
            )
        )

        st.markdown("---")
        st.markdown("**④ Record Your Objection**")

        # Contest annotation (pre-fills role as "Policy" for governance context)
        ann_key = f"contest_{kernel_id}_{key_suffix}"
        role = st.selectbox(
            "Your role:", ANNOTATION_ROLES,
            index=1,  # default to "Policy"
            key=f"contest_role_{ann_key}",
        )
        objection = st.text_area(
            "Objection / Challenge:",
            placeholder=(
                "Describe your objection to this finding. "
                "Reference specific data sources, assumptions, or missing context..."
            ),
            height=100,
            key=f"contest_text_{ann_key}",
        )
        if st.button("Submit Objection", key=f"contest_submit_{ann_key}"):
            if objection.strip():
                ann_entry = {
                    "kernel_id": kernel_id,
                    "role": role,
                    "text": f"[OBJECTION] {objection.strip()}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                if "stakeholder_annotations" not in st.session_state:
                    st.session_state.stakeholder_annotations = []
                st.session_state.stakeholder_annotations.append(ann_entry)
                st.success("Objection recorded and will appear in the exported report.")
