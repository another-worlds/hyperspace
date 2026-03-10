"""Agentic Simulation tab: data-driven multi-agent resource/alliance sim."""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hyperspace.config import GEOPOLITICAL_NODES, PLOTLY_LAYOUT
from hyperspace.data.political import get_political_data
from hyperspace.models.agent_sim import (
    ClusterAgent, initialize_agents_from_data, run_simulation,
)
from hyperspace.pages._report_section import render_interpretability_report
from hyperspace.viz.charts import source_badge


def render() -> None:
    """Render the Agentic Simulation tab."""
    st.markdown("## Agentic Simulation")
    st.markdown(
        "Bounded-rational agents with data-driven resource initialization "
        "and alliance structures derived from graph centrality and voting agreement."
    )

    sc1, sc2, sc3, sc4 = st.columns(4)
    sim_steps = sc1.slider("Steps", 20, 100, 50, key="sim_steps")
    resource_flow = sc2.slider("Resource Flow", 1.0, 15.0, 5.0, key="sim_rf")
    alliance_fluidity = sc3.slider("Alliance Fluidity", 0.1, 1.0, 0.5, key="sim_af")
    shock_prob = sc4.slider("Shock Probability", 0.0, 0.3, 0.1, key="sim_sp")

    sim_btn = st.button("Run Simulation", type="primary", key="sim_run")

    sim_result = st.session_state.get("sim_result")

    if sim_btn or sim_result:
        if sim_btn:
            with st.spinner("Running agent simulation..."):
                # Get graph analysis for data-driven init
                graph_result = st.session_state.get("graph_result")
                graph_analysis = graph_result.get("analysis") if graph_result else None
                try:
                    _, agreement, _ = get_political_data()
                except RuntimeError as exc:
                    st.error(str(exc))
                    return

                agents = initialize_agents_from_data(graph_analysis, agreement)
                agents, log_entries, features, feature_meta = run_simulation(
                    agents, steps=sim_steps,
                    resource_flow=resource_flow,
                    alliance_fluidity=alliance_fluidity,
                    shock_prob=shock_prob,
                )
                sim_result = dict(
                    agents=agents, log=log_entries,
                    features_for_ukt=features, feature_meta=feature_meta,
                )
                st.session_state.sim_result = sim_result

        if sim_result:
            agents = sim_result["agents"]
            log_entries = sim_result.get("log", [])

            # Resource trajectories
            st.markdown("### Agent Resource Trajectories")
            fig = go.Figure()
            for name, agent in agents.items():
                color = GEOPOLITICAL_NODES.get(name, {}).get("color", "#ffffff")
                fig.add_trace(go.Scatter(
                    y=agent.history, mode="lines",
                    name=name, line=dict(color=color, width=2),
                ))
            fig.update_layout(
                **PLOTLY_LAYOUT, title="Resource Evolution Over Time",
                height=400, xaxis_title="Step", yaxis_title="Resources",
            )
            st.plotly_chart(fig, use_container_width=True, key="agents_resource_trajectories")
            st.caption(
                "v3.0 — Resource trajectories capture bounded-rational agent "
                "dynamics. Per-agent resource shares populate UKT indices 51–55 "
                "(dynamic-agent region), enabling traceability of power concentration."
            )

            # Final resource distribution
            st.markdown("### Final Resource Distribution")
            names = list(agents.keys())
            resources = [agents[n].resources for n in names]
            colors = [GEOPOLITICAL_NODES.get(n, {}).get("color", "#888") for n in names]
            fig = go.Figure(go.Bar(
                x=names, y=resources, marker_color=colors,
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="Final Resources", height=350)
            st.plotly_chart(fig, use_container_width=True, key="agents_final_resources")
            st.caption(
                "v3.0 — Final resource distribution shows equilibrium power "
                "balance after N simulation steps. Concentration is quantified "
                "in the UKT and triggers governance flag GOV-001 if one modality dominates."
            )

            # Alliance matrix heatmap
            st.markdown("### Final Alliance Matrix")
            node_names = list(agents.keys())
            n = len(node_names)
            alliance_mat = np.zeros((n, n))
            for i, name in enumerate(node_names):
                for j, other in enumerate(node_names):
                    alliance_mat[i, j] = agents[name].alliances.get(other, 0)

            fig = px.imshow(
                alliance_mat, x=node_names, y=node_names,
                color_continuous_scale="RdBu_r", text_auto=".2f",
                title="Alliance Strengths After Simulation",
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig, use_container_width=True, key="agents_alliance_matrix")
            st.caption(
                "v3.0 — Alliance matrix eigenvalues populate UKT indices 56–63. "
                "A single dominant eigenvalue signals a unipolar bloc; multiple "
                "comparable eigenvalues indicate a multipolar structure — both "
                "are interpretable via the Semantic Canvas."
            )

            # ----------------------------------------------------------- #
            # Power Concentration Assessment (governance signal)          #
            # ----------------------------------------------------------- #
            st.markdown("---")
            res_arr = np.array(resources, dtype=float)
            if len(res_arr) > 1 and res_arr.sum() > 0:
                sorted_res = np.sort(res_arr)
                n_r = len(sorted_res)
                index = np.arange(1, n_r + 1)
                gini = float(
                    (2 * np.sum(index * sorted_res) - (n_r + 1) * np.sum(sorted_res))
                    / (n_r * np.sum(sorted_res) + 1e-8)
                )
                top_share = float(max(resources) / (res_arr.sum() + 1e-8))
                max_agent = max(agents.values(), key=lambda a: a.resources)

                if gini > 0.50:
                    st.error(
                        f"**Power Concentration: HIGH** — Resource Gini = {gini:.2f}. "
                        f"{max_agent.name} holds {top_share:.0%} of all resources. "
                        "This signals a highly unipolar equilibrium — governance risk for "
                        "coercive influence or single-actor dependency."
                    )
                elif gini > 0.25:
                    st.warning(
                        f"**Power Concentration: MODERATE** — Resource Gini = {gini:.2f}. "
                        f"Leading actor ({max_agent.name}) holds {top_share:.0%} of resources. "
                        "Some imbalance detected; multipolarity partially maintained."
                    )
                else:
                    st.success(
                        f"**Power Concentration: LOW** — Resource Gini = {gini:.2f}. "
                        "Resources are broadly distributed across actors. "
                        "Multipolar equilibrium maintained at simulation end."
                    )

            # ----------------------------------------------------------- #
            # Interpretability Report                                      #
            # ----------------------------------------------------------- #
            st.markdown("---")
            render_interpretability_report("Agents")
