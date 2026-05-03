"""
src/style.py
─────────────────────────────────────────────────────────────────────────────
Design system for the Dash dashboard.

Provides:
  - C              : Colour palette dict
  - TICKER_COLORS  : Per-ticker accent colours
  - BASE_LAYOUT    : Default Plotly figure layout
  - CARD / CARD2   : Div style dicts
  - LBL / VAL      : Label / value typography styles
  - apply_layout() : Apply BASE_LAYOUT to a figure
  - metric()       : Stat card component
  - slide_intro()  : Presentation slide intro panel
  - chart_note()   : Chart annotation footer

Author: Muhideen Ogunlowo
"""

from dash import html
import dash_bootstrap_components as dbc


# ── Colour Palette ────────────────────────────────────────────────────────────

C = dict(
    bg       = "#0d1117",
    surface  = "#161b22",
    surface2 = "#21262d",
    border   = "#30363d",
    text     = "#e6edf3",
    muted    = "#8b949e",
    teal     = "#39d0a0",
    purple   = "#a78bfa",
    amber    = "#fbbf24",
    coral    = "#fb7185",
    blue     = "#60a5fa",
    green    = "#4ade80",
    orange   = "#fb923c",
    pink     = "#f472b6",
)

MONO = "'JetBrains Mono','Fira Code',monospace"

TICKER_COLORS = {
    "DIS":  C["teal"],
    "AAPL": C["blue"],
    "MSFT": C["purple"],
    "GOOGL":C["amber"],
    "AMZN": C["orange"],
    "TSLA": C["coral"],
    "NVDA": C["green"],
    "META": C["pink"],
    "NFLX": C["coral"],
    "JPM":  C["blue"],
}


# ── Plotly Base Layout ────────────────────────────────────────────────────────

BASE_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(color=C["text"], family="Inter,sans-serif", size=12),
    colorway      = [C["teal"], C["purple"], C["amber"], C["coral"],
                     C["blue"], C["green"],  C["orange"], C["pink"]],
    xaxis = dict(gridcolor="#21262d", linecolor=C["border"], zerolinecolor="#21262d"),
    yaxis = dict(gridcolor="#21262d", linecolor=C["border"], zerolinecolor="#21262d"),
    legend = dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"],
                  borderwidth=1, font=dict(size=11)),
    margin = dict(l=48, r=20, t=44, b=36),
    hoverlabel = dict(bgcolor=C["surface2"], bordercolor=C["border"],
                      font=dict(color=C["text"], family="Inter")),
)


# ── Component Style Dicts ─────────────────────────────────────────────────────

CARD  = {"background": C["surface"],  "border": f"1px solid {C['border']}",
          "borderRadius": "12px", "padding": "14px 18px"}

CARD2 = {"background": C["surface2"], "border": f"1px solid {C['border']}",
          "borderRadius": "8px",  "padding": "12px 16px"}

LBL = {"color": C["muted"], "fontSize": "10px", "fontWeight": "600",
        "letterSpacing": "0.09em", "textTransform": "uppercase", "marginBottom": "4px"}

VAL = {"fontSize": "20px", "fontWeight": "600",
        "fontFamily": MONO, "lineHeight": "1.15"}


# ── Helper Functions ──────────────────────────────────────────────────────────

def apply_layout(fig, title="", height=300, legend_h=False, **kw):
    """Apply BASE_LAYOUT to a Plotly figure with optional overrides."""
    lo = dict(
        **BASE_LAYOUT,
        height=height,
        title=dict(text=title, font=dict(size=13, color=C["muted"]), x=0, xanchor="left"),
    )
    if legend_h:
        lo["legend"].update(orientation="h", y=1.12, x=0)
    lo.update(kw)
    fig.update_layout(**lo)
    return fig


def metric(label, value, color, width=2):
    """Render a single stat card (label + large value)."""
    return dbc.Col(
        html.Div([
            html.Div(label, style=LBL),
            html.Div(value, style=VAL | {"color": color}),
        ], style=CARD),
        width=width,
        className="mb-2",
    )


def slide_intro(phase_num, accent, title, body_lines, insight):
    """
    Render a presentation-style slide intro panel.

    Left column: phase badge, title, body paragraphs.
    Right column: tinted 'Key Insight' callout box.
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div(f"Phase {phase_num}", style={
                    "color": accent, "fontSize": "11px", "fontWeight": "700",
                    "letterSpacing": "0.12em", "textTransform": "uppercase", "marginBottom": "6px",
                }),
                html.Div(title, style={
                    "color": C["text"], "fontSize": "19px", "fontWeight": "600", "marginBottom": "10px",
                }),
                *[html.P(ln, style={"color": C["muted"], "fontSize": "13px",
                                     "lineHeight": "1.7", "margin": "0 0 5px 0"})
                  for ln in body_lines],
            ], width=8),
            dbc.Col(
                html.Div([
                    html.Div("Key insight", style={
                        "color": accent, "fontSize": "10px", "fontWeight": "700",
                        "letterSpacing": "0.1em", "textTransform": "uppercase", "marginBottom": "8px",
                    }),
                    html.Div(insight, style={"color": C["text"], "fontSize": "13px", "lineHeight": "1.65"}),
                ], style={
                    "background": f"{accent}12", "border": f"1px solid {accent}40",
                    "borderLeft": f"3px solid {accent}", "borderRadius": "8px",
                    "padding": "14px 16px", "height": "100%",
                }),
                width=4,
            ),
        ], className="g-3"),
    ], style={
        "background": C["surface"], "border": f"1px solid {C['border']}",
        "borderRadius": "12px", "padding": "20px 24px", "marginBottom": "20px",
    })


def chart_note(*cols):
    """
    Render a chart annotation footer row.

    Each argument is a (text, width) tuple.
    """
    return html.Div(
        dbc.Row([
            dbc.Col(
                html.Div(txt, style={"color": C["muted"], "fontSize": "12px", "lineHeight": "1.7"}),
                width=w,
            )
            for txt, w in cols
        ], className="g-3"),
        style=CARD2 | {"marginTop": "14px"},
    )
