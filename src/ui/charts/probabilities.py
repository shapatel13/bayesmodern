from __future__ import annotations

import plotly.graph_objects as go

from llm.structured_output import ResearchReport


def differential_figure(report: ResearchReport) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Bar(
                x=[entry.posterior for entry in report.differential.ranked],
                y=[entry.name for entry in report.differential.ranked],
                orientation="h",
                marker=dict(
                    color=["#0b7285", "#1971c2", "#2f9e44", "#f08c00", "#c92a2a"][: len(report.differential.ranked)]
                ),
            )
        ]
    )
    figure.update_layout(
        title="Posterior Differential",
        xaxis_title="Posterior Probability",
        yaxis_title="Diagnosis",
        template="plotly_white",
        height=380,
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return figure


def calibration_figure(report: ResearchReport) -> go.Figure:
    top = report.differential.ranked[0] if report.differential.ranked else None
    value = top.posterior * 100 if top else 0.0
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%"},
            title={"text": "Top-Hypothesis Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0b7285"},
                "steps": [
                    {"range": [0, 45], "color": "#ffe8cc"},
                    {"range": [45, 75], "color": "#d0ebff"},
                    {"range": [75, 100], "color": "#d3f9d8"},
                ],
            },
        )
    )
    figure.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    return figure

