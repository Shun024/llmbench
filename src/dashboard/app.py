"""
LLMBench — Interactive Leaderboard Dashboard
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from pathlib import Path


# Load results
df = pd.read_parquet("data/results/benchmark_results.parquet")

with open("data/results/detailed_results.json") as f:
    detailed = json.load(f)

# Model display names
MODEL_NAMES = {
    "gpt-4o-mini": "GPT-4o-mini",
    "llama-3.1-8b-instant": "Llama 3.1 8B",
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
}

MODEL_COLOURS = {
    "gpt-4o-mini": "#74AA9C",
    "llama-3.1-8b-instant": "#7B68EE",
    "llama-3.3-70b-versatile": "#FF8C00",
}

DIMENSIONS = [
    "factuality", "reasoning", "safety",
    "instruction_following", "consistency",
]

# Overall scores
overall = df.groupby("model")["score"].mean().reset_index()
overall["display_name"] = overall["model"].map(MODEL_NAMES)
overall = overall.sort_values("score", ascending=False)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="LLMBench",
)


def medal(rank):
    return ["🥇", "🥈", "🥉"][rank] if rank < 3 else f"#{rank+1}"


def metric_card(title, value, subtitle, color="primary"):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted mb-1",
                    style={"fontSize": "0.75rem"}),
            html.H3(value, className=f"text-{color} mb-0"),
            html.Small(subtitle, className="text-muted"),
        ])
    ], className="mb-3")


# Pivot for radar chart
pivot = df.pivot_table(
    index="model", columns="dimension", values="score"
).fillna(0)


def make_radar_chart():
    fig = go.Figure()
    categories = DIMENSIONS + [DIMENSIONS[0]]

    for model_id in pivot.index:
        values = [pivot.loc[model_id, d] for d in DIMENSIONS]
        values += [values[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[d.replace("_", " ").title() for d in categories],
            fill="toself",
            name=MODEL_NAMES.get(model_id, model_id),
            line_color=MODEL_COLOURS.get(model_id, "#888"),
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        title="Model Capability Radar",
        height=450,
    )
    return fig


def make_dimension_bar():
    fig = go.Figure()

    for model_id in pivot.index:
        fig.add_trace(go.Bar(
            name=MODEL_NAMES.get(model_id, model_id),
            x=[d.replace("_", " ").title() for d in DIMENSIONS],
            y=[pivot.loc[model_id, d] for d in DIMENSIONS],
            marker_color=MODEL_COLOURS.get(model_id, "#888"),
        ))

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        title="Score by Dimension",
        yaxis=dict(range=[0, 1], title="Score"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1),
        height=400,
    )
    return fig


def make_cost_performance_chart():
    cost_df = df.groupby("model").agg(
        score=("score", "mean"),
        cost=("total_cost_usd", "sum"),
        latency=("avg_latency_ms", "mean"),
    ).reset_index()
    cost_df["display"] = cost_df["model"].map(MODEL_NAMES)

    fig = go.Figure()
    for _, row in cost_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["cost"]],
            y=[row["score"]],
            mode="markers+text",
            name=row["display"],
            text=[row["display"]],
            textposition="top center",
            marker=dict(
                size=20,
                color=MODEL_COLOURS.get(row["model"], "#888"),
            ),
        ))

    fig.update_layout(
        template="plotly_dark",
        title="Cost vs Performance",
        xaxis_title="Total API Cost (USD)",
        yaxis_title="Overall Score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=350,
    )
    return fig


app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H2("🏆 LLMBench", className="mt-4 mb-0"),
            html.P(
                "Comprehensive LLM Evaluation · Factuality · Reasoning · "
                "Safety · Instruction Following · Consistency",
                className="text-muted mb-4",
            ),
        ])
    ]),

    # Overall ranking cards
    dbc.Row([
        dbc.Col(metric_card(
            f"{medal(i)} {row['display_name']}",
            f"{row['score']:.3f}",
            "Overall Score",
            ["success", "warning", "secondary"][i],
        ), md=4)
        for i, (_, row) in enumerate(overall.iterrows())
    ]),

    # Leaderboard table
    dbc.Row([
        dbc.Col([
            html.H5("Leaderboard", className="mt-2 mb-3"),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Rank"),
                    html.Th("Model"),
                    html.Th("Factuality"),
                    html.Th("Reasoning"),
                    html.Th("Safety"),
                    html.Th("Instruction Following"),
                    html.Th("Consistency"),
                    html.Th("Overall"),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(medal(i)),
                        html.Td(MODEL_NAMES.get(row["model"], row["model"])),
                        html.Td(f"{pivot.loc[row['model'], 'factuality']:.3f}"
                                if row["model"] in pivot.index else "—"),
                        html.Td(f"{pivot.loc[row['model'], 'reasoning']:.3f}"
                                if row["model"] in pivot.index else "—"),
                        html.Td(f"{pivot.loc[row['model'], 'safety']:.3f}"
                                if row["model"] in pivot.index else "—"),
                        html.Td(f"{pivot.loc[row['model'], 'instruction_following']:.3f}"
                                if row["model"] in pivot.index else "—"),
                        html.Td(f"{pivot.loc[row['model'], 'consistency']:.3f}"
                                if row["model"] in pivot.index else "—"),
                        html.Td(
                            html.Strong(f"{row['score']:.3f}"),
                            style={"color": ["#00FF7F", "#FFD700", "#888"][i]},
                        ),
                    ])
                    for i, (_, row) in enumerate(overall.iterrows())
                ])
            ], bordered=True, hover=True, striped=True),
        ])
    ]),

    # Charts
    dbc.Row([
        dbc.Col([dcc.Graph(figure=make_radar_chart())], md=6),
        dbc.Col([dcc.Graph(figure=make_dimension_bar())], md=6),
    ]),

    dbc.Row([
        dbc.Col([dcc.Graph(figure=make_cost_performance_chart())], md=6),
        dbc.Col([
            html.H5("Key Findings", className="mt-4 mb-3"),
            dbc.ListGroup([
                dbc.ListGroupItem(
                    "🥇 Llama 3.3 70B wins overall (0.713) — perfect safety score",
                    color="success",
                ),
                dbc.ListGroupItem(
                    "🎯 GPT-4o-mini leads factuality (0.867) — best at avoiding hallucinations",
                    color="info",
                ),
                dbc.ListGroupItem(
                    "🧮 Reasoning is the weakest dimension across ALL models (~0.45 avg)",
                    color="warning",
                ),
                dbc.ListGroupItem(
                    "🛡️ All models score >0.85 on safety — refusal training is effective",
                    color="success",
                ),
                dbc.ListGroupItem(
                    "💰 Llama 3.3 70B via Groq: best performance at $0 cost",
                    color="primary",
                ),
            ]),
        ], md=6),
    ]),

    html.Hr(className="mt-4"),
    html.P(
        "Benchmarks: TruthfulQA (factuality) · GSM8K (reasoning) · "
        "Custom safety set · IFEval-style (instruction following) · "
        "Paraphrase pairs (consistency)",
        className="text-muted text-center mb-4",
        style={"fontSize": "0.75rem"},
    ),

], fluid=True)


if __name__ == "__main__":
    app.run(debug=True, port=8053)