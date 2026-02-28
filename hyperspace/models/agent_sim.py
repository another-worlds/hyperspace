"""Agentic simulation: data-driven cluster agents with UKT feature extraction."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hyperspace.config import GEOPOLITICAL_NODES, UKT_FEATURE_DIM


@dataclass
class ClusterAgent:
    """Bounded-rational agent representing a geopolitical cluster."""
    name: str
    x: float
    y: float
    resources: float = 100.0
    alliances: dict[str, float] = field(default_factory=dict)
    history: list[float] = field(default_factory=list)
    capability_multiplier: float = 1.0  # from spatial geographic resilience proxy

    def step(self, agents: dict[str, "ClusterAgent"], rng: np.random.Generator,
             resource_flow: float, alliance_fluidity: float, shock_prob: float) -> str:
        log = ""
        for ally_name, strength in list(self.alliances.items()):
            if ally_name in agents:
                transfer = resource_flow * strength * rng.uniform(0.5, 1.5)
                self.resources += transfer * 0.1
                agents[ally_name].resources -= transfer * 0.05
        for ally_name in list(self.alliances.keys()):
            drift = rng.normal(0, alliance_fluidity * 0.05)
            self.alliances[ally_name] = np.clip(
                self.alliances[ally_name] + drift, -1, 1)
        if rng.random() < shock_prob:
            loss = rng.uniform(5, 25)
            self.resources = max(10, self.resources - loss)
            log = f"SHOCK: {self.name} lost {loss:.1f} resources"
        self.history.append(self.resources)
        return log


def initialize_agents_from_data(
    graph_analysis: dict | None = None,
    agreement_matrix: pd.DataFrame | None = None,
    spatial_features: dict | None = None,
) -> dict[str, ClusterAgent]:
    """Create agents with data-driven parameters.

    Args:
        graph_analysis: Output from graph_engine.analyze_graph().
        agreement_matrix: Pairwise country agreement from political data.
        spatial_features: Output from spatial_kernels.get_spatial_features().
            If provided, per-node (10,) spatial vectors enrich resource init
            and set capability_multiplier from geographic resilience proxy.

    Returns:
        Dict of name -> ClusterAgent.
    """
    agents: dict[str, ClusterAgent] = {}
    node_names = list(GEOPOLITICAL_NODES.keys())

    # Pre-extract spatial vectors for enrichment if available
    per_node_vectors: dict[str, np.ndarray] = {}
    scalar_names: list[str] = []
    if spatial_features is not None:
        per_node_vectors = spatial_features.get("per_node_vectors", {})
        scalar_names = spatial_features.get("scalar_names", [])

    for name in node_names:
        attrs = GEOPOLITICAL_NODES[name]
        # Base resources from eigenvector centrality if available
        base_resources = attrs["influence"] * 100 + 20
        if graph_analysis and "eigenvector" in graph_analysis:
            eig = graph_analysis["eigenvector"].get(name, 0.5)
            base_resources = eig * 150 + 20

        capability_mult = 1.0

        if name in per_node_vectors and len(scalar_names) >= 6:
            vec = per_node_vectors[name]  # (10,) normalized
            # Scalar indices in the full (10,) vector:
            # 0=elevation, 1=temp, 2=humidity, 3=precip (physical, rows 0-3)
            # 4=gdp_ppp, 5=debt_pct_gdp, 6=military_pct_gdp, 7=tertiary_enroll
            # 8=conflict_event_density, 9=conflict_fatality_density
            gdp_norm       = float(vec[4]) if len(vec) > 4 else 0.5
            enroll_norm    = float(vec[7]) if len(vec) > 7 else 0.5
            conflict_norm  = float(vec[8]) if len(vec) > 8 else 0.0
            elev_norm      = float(vec[0]) if len(vec) > 0 else 0.5
            temp_norm      = float(vec[1]) if len(vec) > 1 else 0.5

            economic_capacity = (gdp_norm + enroll_norm) / 2.0
            conflict_stress   = np.clip(conflict_norm, 0.0, 1.0)
            # Blended resource: 60% centrality base, 30% economic capacity, 10% conflict relief
            base_resources = (
                base_resources * 0.6
                + economic_capacity * 150 * 0.3
                + (1.0 - conflict_stress) * 20 * 0.1
            )
            # capability_multiplier: geographic resilience proxy (elevation × temperature diversity)
            capability_mult = float(np.clip(
                0.5 + 0.5 * (elev_norm + (1.0 - temp_norm)) / 2.0,
                0.5, 2.0,
            ))

        agent = ClusterAgent(
            name=name,
            x=attrs["lon"] / 40,
            y=attrs["lat"] / 20,
            resources=base_resources,
            capability_multiplier=capability_mult,
        )

        # Set alliances from agreement matrix or graph weights
        for other in node_names:
            if other != name:
                if agreement_matrix is not None and name in agreement_matrix.columns:
                    if other in agreement_matrix.columns:
                        agent.alliances[other] = float(
                            agreement_matrix.loc[name, other])
                        continue
                # Fall back to influence-based default
                agent.alliances[other] = 0.0

        agents[name] = agent

    # Fill in missing alliances from hardcoded edges
    from hyperspace.config import GEOPOLITICAL_EDGES
    for src, dst, w, _, _ in GEOPOLITICAL_EDGES:
        if src in agents and dst in agents:
            if agents[src].alliances.get(dst, 0.0) == 0.0:
                agents[src].alliances[dst] = w
            if agents[dst].alliances.get(src, 0.0) == 0.0:
                agents[dst].alliances[src] = w

    return agents


def run_simulation(
    agents: dict[str, ClusterAgent],
    steps: int = 50,
    resource_flow: float = 5.0,
    alliance_fluidity: float = 0.5,
    shock_prob: float = 0.1,
    s: int = 42,
) -> tuple[dict[str, ClusterAgent], list[str], np.ndarray, dict[int, dict]]:
    """Run the agent simulation.

    Returns:
        (agents, log_entries, features_for_ukt, feature_meta)
    """
    rng = np.random.default_rng(s)
    log_entries: list[str] = []

    for step in range(steps):
        for agent in agents.values():
            msg = agent.step(agents, rng, resource_flow, alliance_fluidity, shock_prob)
            if msg:
                log_entries.append(f"[Step {step:03d}] {msg}")

    # Build UKT feature vector (dynamic-agent region: indices 48-63)
    features_for_ukt = np.zeros(UKT_FEATURE_DIM)
    feature_meta: dict[int, dict] = {}
    agent_names = list(agents.keys())
    resources = np.array([a.resources for a in agents.values()])
    # Normalized resource distribution
    res_norm = resources / (resources.sum() + 1e-8)
    n_res = min(8, len(res_norm))
    features_for_ukt[48:48 + n_res] = res_norm[:n_res]
    for i in range(n_res):
        entity = agent_names[i] if i < len(agent_names) else f"agent_{i}"
        feature_meta[48 + i] = {
            "label": f"{entity}_resource_share",
            "entity": entity,
            "metric": "resource_share",
            "block": "agents",
            "source": "agent_simulation",
        }

    # Alliance matrix eigenvalues (captures structural properties)
    node_names = list(agents.keys())
    n = len(node_names)
    alliance_mat = np.zeros((n, n))
    for i, name in enumerate(node_names):
        for j, other in enumerate(node_names):
            if other in agents[name].alliances:
                alliance_mat[i, j] = agents[name].alliances[other]
    eigenvalues = np.sort(np.linalg.eigvalsh(alliance_mat))[::-1]
    n_eig = min(8, len(eigenvalues))
    features_for_ukt[56:56 + n_eig] = eigenvalues[:n_eig]
    for i in range(n_eig):
        feature_meta[56 + i] = {
            "label": f"alliance_eigenvalue_{i+1}",
            "metric": "alliance_spectrum",
            "block": "agents",
            "source": "agent_simulation",
        }

    return agents, log_entries, features_for_ukt, feature_meta
