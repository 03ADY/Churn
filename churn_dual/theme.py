"""ChurnGuard — dark enterprise Streamlit theme."""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ACCENT = ("#7c3aed", "#db2777")
PRIMARY = "#a855f7"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; }
.block-container { padding-top: 1.25rem; max-width: 1400px; }
.ep-hero {
  background: linear-gradient(135deg, ACCENT_A 0%, ACCENT_B 100%);
  padding: 1.75rem 2rem; border-radius: 16px; margin-bottom: 1.25rem; color: #fff;
  box-shadow: 0 16px 48px rgba(124, 58, 237, 0.35);
  border: 1px solid rgba(255,255,255,0.08);
}
.ep-hero h1 { margin: 0; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; }
.ep-hero p { margin: 0.45rem 0 0; opacity: 0.92; font-size: 0.95rem; }
div[data-testid="stMetric"] {
  background: rgba(30, 41, 59, 0.85); border: 1px solid rgba(148, 163, 184, 0.15);
  border-radius: 12px; padding: 0.65rem 0.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
div[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 0.78rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f8fafc !important; }
div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
  border-right: 1px solid rgba(148, 163, 184, 0.12);
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
.stTabs [data-baseweb="tab"] {
  border-radius: 10px; padding: 0.5rem 1rem; font-weight: 500;
  background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(148,163,184,0.12);
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, ACCENT_A, ACCENT_B) !important; color: #fff !important;
  border-color: transparent !important;
}
.insight-card {
  background: rgba(30, 41, 59, 0.9); border: 1px solid rgba(148, 163, 184, 0.15);
  border-radius: 12px; padding: 1rem; border-left: 4px solid PRIMARY;
}
</style>
""".replace("ACCENT_A", ACCENT[0]).replace("ACCENT_B", ACCENT[1]).replace("PRIMARY", PRIMARY)


def inject_theme() -> None:
    pio.templates.default = "plotly_dark"
    st.markdown(_CSS, unsafe_allow_html=True)


def hero_html(title: str, subtitle: str, icon: str = "") -> str:
    return f"""
<div class="ep-hero">
  <h1>{icon} {title}</h1>
  <p>{subtitle}</p>
</div>
"""


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=48, r=24, t=48, b=40),
    )
    return fig
