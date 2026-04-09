"""
RADCF Fair Pricing Engine — Actuarial Evaluation of Consumer Overpricing
in Kenya's Hire-Purchase Market
Egerton University · Department of Mathematics
"""

import re
import math
import io
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ─────────────────────────────────────────────────────────
#  CALIBRATED MODEL CONSTANTS  (from Egerton KCHSP 2022 study)
#  β0 = 3.0267 | β1 = −1.2553  — DO NOT EXPOSE IN UI
# ─────────────────────────────────────────────────────────
_BETA0 = 3.0267
_BETA1 = -1.2553
_INCOME_SCALE = 1000.0          # ln(income / 1000)

# ─────────────────────────────────────────────────────────
#  CORE ACTUARIAL FUNCTIONS
# ─────────────────────────────────────────────────────────

def logistic_pd(income_ksh: float) -> float:
    """PD = 1 / (1 + exp(-(β0 + β1·ln(income/1000))))"""
    if income_ksh <= 0:
        return 1.0
    z = _BETA0 + _BETA1 * math.log(income_ksh / _INCOME_SCALE)
    return float(max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-z)))))


def annuity_factor(r: float, n: int) -> float:
    if n <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return float(n)
    return float((1.0 - (1.0 + r) ** (-n)) / r)


def fair_installment(
    cash_price: float,
    deposit_pct: float,
    admin_cost_pct: float,
    n_months: int,
    r_monthly: float,
    pd_est: float,
) -> dict:
    op = float(cash_price)
    deposit = op * (deposit_pct / 100.0)
    admin_cost = op * (admin_cost_pct / 100.0)
    cf_revised = op + admin_cost - deposit
    af = annuity_factor(r_monthly, int(n_months))
    repay_prob = max(1e-9, 1.0 - float(pd_est))
    m = (cf_revised / (repay_prob * af)) if af > 0 else float("nan")
    fair_total = deposit + (m * n_months)
    return {
        "op": op,
        "deposit_amount": deposit,
        "admin_cost_amount": admin_cost,
        "cf_revised": cf_revised,
        "annuity_factor": af,
        "fair_monthly_installment": m,
        "fair_total_paid": fair_total,
    }


def implied_monthly_rate(P: float, payment: float, n: int) -> float:
    if P <= 0 or n <= 0 or payment * n < P:
        return float("nan")
    lo, hi = 0.0, 3.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        denom = 1.0 - (1.0 + mid) ** (-n)
        if denom <= 0:
            lo = mid
            continue
        (hi if P * mid / denom > payment else lo).__class__  # dummy
        if P * mid / denom > payment:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def effective_apr(i: float) -> float:
    return float((1.0 + i) ** 12 - 1.0) if np.isfinite(i) else float("nan")


# ─────────────────────────────────────────────────────────
#  CLASSIFICATION HELPERS
# ─────────────────────────────────────────────────────────

def pd_label(pd_val: float) -> tuple:
    if pd_val >= 0.50:
        return "Very High", "#FF4B4B"
    if pd_val >= 0.35:
        return "High", "#FF8C00"
    if pd_val >= 0.20:
        return "Moderate", "#FFD700"
    if pd_val >= 0.10:
        return "Low-Moderate", "#9ACD32"
    return "Low", "#00CC88"


def affordability_label(iti: float) -> tuple:
    """Returns (label, color, icon)"""
    if iti < 0.20:
        return "Highly Affordable ✅", "#00CC88", "green"
    if iti <= 0.30:
        return "Affordable ✅", "#9ACD32", "green"
    if iti <= 0.40:
        return "Moderately Affordable ⚠️", "#FFD700", "orange"
    return "Low Affordability 🚨", "#FF4B4B", "red"


def fairness_badge(over_pct: float) -> tuple:
    if not np.isfinite(over_pct):
        return "", "#888888"
    if over_pct >= 0.50:
        return "Severely Overpriced 🚨", "#FF4B4B"
    if over_pct >= 0.25:
        return "Overpriced ⚠️", "#FF8C00"
    if over_pct >= -0.10:
        return "Near Fair ✅", "#00CC88"
    return "Below Fair 💰", "#3399FF"


# ─────────────────────────────────────────────────────────
#  MAX AFFORDABLE PHONE PRICE (30 % ITI threshold)
# ─────────────────────────────────────────────────────────

def max_affordable_phone(
    income_ksh: float,
    deposit_pct: float,
    admin_cost_pct: float,
    n_months: int,
    r_monthly: float,
    iti_threshold: float = 0.30,
) -> float:
    """Binary-search for cash price where ITI == threshold."""
    max_m = income_ksh * iti_threshold
    pd_est = logistic_pd(income_ksh)
    af = annuity_factor(r_monthly, n_months)
    repay_prob = max(1e-9, 1.0 - pd_est)
    d = deposit_pct / 100.0
    a = admin_cost_pct / 100.0
    # M = (OP + OP*a - OP*d) / (repay_prob * af)
    # max_m = OP * (1 + a - d) / (repay_prob * af)
    denom = (1.0 + a - d)
    if denom <= 0 or af <= 0:
        return 0.0
    return (max_m * repay_prob * af) / denom


# ─────────────────────────────────────────────────────────
#  CONTRACT TEXT EXTRACTION
# ─────────────────────────────────────────────────────────

def extract_deal_fields(text: str) -> dict:
    t = (text or "").lower().replace(",", " ")
    money = r"(?:ksh|kes)\s*([0-9]{3,})"
    pct = r"([0-9]{1,2}(?:\.[0-9]+)?)\s*%"
    cash_price = None
    m = re.search(r"(cash price|cash|price)\s*[:\-]?\s*" + money, t)
    if m:
        cash_price = float(m.group(2))
    if cash_price is None:
        m2 = re.search(money, t)
        if m2:
            cash_price = float(m2.group(1))
    deposit_pct = None
    mdp = re.search(r"(deposit|downpayment|down payment)\s*[:\-]?\s*" + pct, t)
    if mdp:
        deposit_pct = float(mdp.group(2))
    deposit_amount = None
    mda = re.search(r"(deposit|downpayment|down payment)\s*[:\-]?\s*" + money, t)
    if mda:
        deposit_amount = float(mda.group(2))
    term_months = None
    mt = re.search(r"([0-9]{1,2})\s*(months|month|mos|mo)\b", t)
    if mt:
        term_months = int(mt.group(1))
    monthly_installment = None
    mm = re.search(r"(installment|instalment|monthly|per month)\s*[:\-]?\s*" + money, t)
    if mm:
        monthly_installment = float(mm.group(2))
    admin_pct = None
    mapct = re.search(r"(admin|administration|processing)\s*(fee|cost)?\s*[:\-]?\s*" + pct, t)
    if mapct:
        admin_pct = float(mapct.group(3))
    return {
        "cash_price": cash_price,
        "deposit_pct": deposit_pct,
        "deposit_amount": deposit_amount,
        "term_months": term_months,
        "monthly_installment": monthly_installment,
        "admin_pct": admin_pct,
    }


# ─────────────────────────────────────────────────────────
#  PDF REPORT GENERATOR
# ─────────────────────────────────────────────────────────

def generate_pdf_report(params: dict, res: dict, pd_val: float, iti: float,
                         market_monthly: float, scenarios: list) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                  fontSize=16, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                          fontSize=12, textColor=colors.HexColor("#E63946"), spaceAfter=4)
    normal = styles["Normal"]
    small = ParagraphStyle("Small", parent=normal, fontSize=8, textColor=colors.grey)

    elements = []
    elements.append(Paragraph("RADCF Fair Pricing Report", title_style))
    elements.append(Paragraph("Actuarial Evaluation of Consumer Overpricing · Kenya Hire-Purchase Market", small))
    elements.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}", small))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E63946")))
    elements.append(Spacer(1, 0.4*cm))

    elements.append(Paragraph("Contract Parameters", h2))
    param_data = [
        ["Parameter", "Value"],
        ["Cash Price (KSh)", f"{params['cash_price']:,.2f}"],
        ["Deposit (%)", f"{params['deposit_pct']:.1f}%"],
        ["Admin Cost (%)", f"{params['admin_cost_pct']:.1f}%"],
        ["Repayment Term", f"{params['n_months']} months"],
        ["Monthly Discount Rate", f"{params['r_monthly']*100:.2f}%"],
        ["Borrower Monthly Income (KSh)", f"{params['income_ksh']:,.2f}"],
    ]
    t = Table(param_data, colWidths=[9*cm, 7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E63946")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f9f9f9"), colors.white]),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.4*cm))

    elements.append(Paragraph("RADCF Pricing Results", h2))
    pd_lbl, _ = pd_label(pd_val)
    iti_lbl, _, _ = affordability_label(iti)
    result_data = [
        ["Metric", "Value"],
        ["Estimated Probability of Default (PD)", f"{pd_val:.4f}  ({pd_lbl} Risk)"],
        ["Fair Monthly Installment (KSh)", f"{res['fair_monthly_installment']:,.2f}"],
        ["Deposit Amount (KSh)", f"{res['deposit_amount']:,.2f}"],
        ["Admin Cost Amount (KSh)", f"{res['admin_cost_amount']:,.2f}"],
        ["Fair Total Paid (KSh)", f"{res['fair_total_paid']:,.2f}"],
        ["Income-to-Installment Ratio (ITI)", f"{iti*100:.1f}%  — {iti_lbl}"],
    ]
    if market_monthly > 0:
        mkt_total = res["deposit_amount"] + market_monthly * params["n_months"]
        over_amt = mkt_total - res["fair_total_paid"]
        over_pct = over_amt / res["fair_total_paid"] if res["fair_total_paid"] > 0 else 0
        lbl, _ = fairness_badge(over_pct)
        result_data += [
            ["Market Monthly Installment (KSh)", f"{market_monthly:,.2f}"],
            ["Market Total Repayment (KSh)", f"{mkt_total:,.2f}"],
            ["Overpricing Amount (KSh)", f"{over_amt:,.2f}"],
            ["Overpricing (%)", f"{over_pct*100:.2f}%  — {lbl}"],
        ]
    t2 = Table(result_data, colWidths=[10*cm, 6*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f9f9f9"), colors.white]),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 0.4*cm))

    elements.append(Paragraph("Sensitivity Analysis", h2))
    sens_data = [["Scenario", "PD", "Admin %", "r", "Fair Monthly (KSh)", "Fair Total (KSh)"]]
    for s in scenarios:
        sens_data.append([
            s["Scenario"], f"{s['PD']:.3f}", f"{s['Admin%']:.0f}%",
            f"{s['r']:.3f}", f"{s['Fair Monthly (KSh)']:,.2f}", f"{s['Fair Total (KSh)']:,.2f}"
        ])
    t3 = Table(sens_data, colWidths=[3.5*cm, 2*cm, 2*cm, 2*cm, 3.5*cm, 3.5*cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E63946")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f9f9f9"), colors.white]),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
    ]))
    elements.append(t3)
    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph(
        "This report was generated by the RADCF Pricing Engine based on research by "
        "Mureithi M., Kariuki D., Kiprotich P. & Onyango C. — Egerton University, 2025.", small))
    doc.build(elements)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────
#  PLOTLY CHART HELPERS
# ─────────────────────────────────────────────────────────

def make_gauge(value: float, title: str, max_val: float = 1.0,
               thresholds=None, colors_list=None) -> go.Figure:
    if thresholds is None:
        thresholds = [0.10, 0.25, 0.50, 1.0]
    if colors_list is None:
        colors_list = ["#00CC88", "#9ACD32", "#FFD700", "#FF4B4B"]
    steps = []
    prev = 0
    for thr, col in zip(thresholds, colors_list):
        steps.append({"range": [prev * max_val, thr * max_val], "color": col})
        prev = thr
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13, "color": "#ccc"}},
        number={"font": {"size": 22, "color": "#fff"}, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#888"},
            "bar": {"color": "#E63946", "thickness": 0.25},
            "bgcolor": "#1e1e2e",
            "bordercolor": "#333",
            "steps": steps,
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc"
    )
    return fig


def make_bar_comparison(radcf_m: float, market_m: float, n_months: int,
                         deposit: float) -> go.Figure:
    radcf_total = deposit + radcf_m * n_months
    market_total = deposit + market_m * n_months if market_m > 0 else None
    cats = ["RADCF Fair Value"]
    vals = [radcf_total]
    cols = ["#00CC88"]
    if market_total:
        cats.append("Market Price")
        vals.append(market_total)
        cols.append("#E63946")
    fig = go.Figure(go.Bar(
        x=cats, y=vals, marker_color=cols,
        text=[f"KSh {v:,.0f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#fff", size=12),
    ))
    fig.update_layout(
        title=dict(text="Total Repayment Comparison", font=dict(color="#ccc", size=14)),
        yaxis=dict(title="KSh", gridcolor="#333", color="#888"),
        xaxis=dict(color="#888"),
        height=300, margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
    )
    return fig


def make_tornado(scenarios: list, base_val: float, metric: str) -> go.Figure:
    names, impacts = [], []
    for s in scenarios[1:]:
        imp = s[metric] - base_val
        names.append(s["Scenario"])
        impacts.append(imp)
    sorted_pairs = sorted(zip(impacts, names))
    impacts_s, names_s = zip(*sorted_pairs) if sorted_pairs else ([], [])
    colors_bar = ["#3399FF" if v < 0 else "#E63946" for v in impacts_s]
    fig = go.Figure(go.Bar(
        x=list(impacts_s), y=list(names_s), orientation="h",
        marker_color=colors_bar,
        text=[f"{v:+,.2f}" for v in impacts_s],
        textposition="outside", textfont=dict(color="#ccc", size=10),
    ))
    fig.update_layout(
        title=dict(text=f"Tornado Chart — Sensitivity vs Base ({metric})",
                   font=dict(color="#ccc", size=14)),
        xaxis=dict(title="Impact vs Base (KSh)", gridcolor="#333", color="#888",
                   zeroline=True, zerolinecolor="#666"),
        yaxis=dict(color="#888"),
        height=320, margin=dict(l=120, r=80, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
    )
    return fig


def make_pd_curve(income_ksh: float) -> go.Figure:
    incomes = np.linspace(3000, 100000, 400)
    pds = [logistic_pd(i) for i in incomes]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=incomes, y=pds, mode="lines",
        line=dict(color="#E63946", width=2.5),
        name="PD Curve",
    ))
    fig.add_trace(go.Scatter(
        x=[income_ksh], y=[logistic_pd(income_ksh)],
        mode="markers",
        marker=dict(size=12, color="#FFD700", symbol="diamond",
                    line=dict(color="#fff", width=1)),
        name="Your Income",
        text=[f"KSh {income_ksh:,.0f}<br>PD={logistic_pd(income_ksh):.3f}"],
        hoverinfo="text",
    ))
    fig.update_layout(
        title=dict(text="Probability of Default vs Monthly Income",
                   font=dict(color="#ccc", size=14)),
        xaxis=dict(title="Monthly Income (KSh)", gridcolor="#333", color="#888",
                   tickformat=","),
        yaxis=dict(title="PD", gridcolor="#333", color="#888",
                   tickformat=".0%", range=[0, 1]),
        height=280, margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", legend=dict(font=dict(color="#ccc")),
        hovermode="closest",
    )
    return fig


def make_iti_gauge(iti: float) -> go.Figure:
    return make_gauge(
        iti * 100, "Affordability Ratio (ITI %)", max_val=80,
        thresholds=[20, 30, 40, 80],
        colors_list=["#00CC88", "#9ACD32", "#FFD700", "#FF4B4B"]
    )


# ─────────────────────────────────────────────────────────
#  PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RADCF Fair Pricing Engine · Kenya HP Market",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── global reset ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1300px; }

/* ── metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e1e2e 0%, #16213e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
div[data-testid="metric-container"] label {
    color: #aaa !important; font-size: 11px; text-transform: uppercase; letter-spacing: .8px;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #fff !important; font-size: 1.5rem !important; font-weight: 700;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 11px; }

/* ── info/warning/error boxes ── */
.radcf-card {
    background: linear-gradient(135deg, #1e1e2e, #16213e);
    border-radius: 14px;
    border: 1px solid #2a2a4a;
    padding: 20px 24px;
    margin: 8px 0;
}
.stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; }
.stat-chip {
    background: #2a2a4a; border-radius: 8px; padding: 6px 14px;
    font-size: 13px; color: #ccc;
}
.badge-green  { background: #00CC8822; color: #00CC88; border: 1px solid #00CC8844; border-radius: 8px; padding: 4px 12px; font-weight: 600; font-size: 13px; }
.badge-yellow { background: #FFD70022; color: #FFD700; border: 1px solid #FFD70044; border-radius: 8px; padding: 4px 12px; font-weight: 600; font-size: 13px; }
.badge-orange { background: #FF8C0022; color: #FF8C00; border: 1px solid #FF8C0044; border-radius: 8px; padding: 4px 12px; font-weight: 600; font-size: 13px; }
.badge-red    { background: #FF4B4B22; color: #FF4B4B; border: 1px solid #FF4B4B44; border-radius: 8px; padding: 4px 12px; font-weight: 600; font-size: 13px; }
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #E63946;
    text-transform: uppercase; letter-spacing: 1px;
    border-left: 3px solid #E63946; padding-left: 10px; margin: 1.2rem 0 0.6rem;
}
.formula-box {
    background: #0d1117; border-radius: 10px; border: 1px solid #30363d;
    padding: 16px 20px; font-family: monospace; color: #79c0ff; font-size: 13px;
}
/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 8px 20px;
    background: #1e1e2e;
    color: #aaa;
    border: 1px solid #2a2a4a;
    font-weight: 600;
}
.stTabs [aria-selected="true"] { background: #E63946 !important; color: #fff !important; border-color: #E63946 !important; }

/* ── number inputs ── */
.stNumberInput input { background: #1e1e2e !important; color: #fff !important; border: 1px solid #2a2a4a !important; border-radius: 8px !important; }
.stTextArea textarea { background: #1e1e2e !important; color: #ccc !important; border: 1px solid #2a2a4a !important; border-radius: 8px !important; }

/* ── buttons ── */
.stDownloadButton button {
    background: linear-gradient(135deg, #E63946, #c1121f) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 10px 20px !important; font-weight: 700 !important; font-size: 14px !important;
}
.stButton button {
    background: linear-gradient(135deg, #3a3a5c, #2a2a4a) !important;
    color: #ddd !important; border: 1px solid #4a4a7a !important; border-radius: 10px !important;
}

/* ── dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── divider ── */
hr { border-color: #2a2a4a !important; }

/* ── expander ── */
.streamlit-expanderHeader { background: #1e1e2e; border-radius: 10px; font-weight: 600; color: #ccc; }

/* ── sidebar ── */
.css-1d391kg, [data-testid="stSidebar"] { background: #0d0d1a !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────

st.markdown("""
<div style="background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 60%, #16213e 100%);
            border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
            border: 1px solid #E6394622; box-shadow: 0 8px 32px rgba(230,57,70,0.15);">
  <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 8px;">
    <span style="font-size: 3rem;">📱</span>
    <div>
      <h1 style="margin:0; color:#fff; font-size:1.8rem; font-weight:800; letter-spacing:-0.5px;">
        Actuarial Evaluation of Consumer Overpricing
      </h1>
      <h2 style="margin:0; color:#E63946; font-size:1.15rem; font-weight:600;">
        Kenya's Hire-Purchase Market · RADCF Pricing Engine
      </h2>
    </div>
  </div>
  <p style="color:#888; font-size:13px; margin:0;">
    Risk-Adjusted Discounted Cash Flow (RADCF) model calibrated on <strong style="color:#aaa">17,452 households</strong>
    from the KCHSP 2022 national survey · Egerton University, Dept. of Mathematics
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  EXPANDERS: Overview + Formulas
# ─────────────────────────────────────────────────────────

with st.expander("📚 Model Overview", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**How this tool works**

1. **Probability of Default (PD)** — estimated via logistic regression using borrower income as predictor,
   calibrated on KCHSP 2022 national survey data (17,452 households).

2. **Risk-Adjusted Cash Flow** — each monthly installment is adjusted by *(1 − PD)* to reflect the
   expected fraction of borrowers who actually repay.

3. **Discounting** — expected cash flows are discounted at a monthly rate *r* using an annuity factor.

4. **Affordability** — the Income-to-Installment Ratio (ITI) flags whether a phone is financially
   accessible at the borrower's income level.
        """)
    with c2:
        st.markdown("""
**Key findings from Egerton research**

| Income Band | Default Rate | RADCF Monthly |
|---|---|---|
| Below KSh 10,000 | 62.6% | KSh 3,751 |
| KSh 10k–15k | 46.8% | KSh 2,647 |
| KSh 15k–25k | 33.3% | KSh 2,099 |
| KSh 25k–40k | 21.5% | KSh 1,789 |
| KSh 40k–70k | 13.9% | KSh 1,610 |
| Above KSh 70k | 9.6% | KSh 1,529 |

*For a KSh 20,000 phone · 30% deposit · 12-month term*
        """)

with st.expander("🔢 Mathematical Framework", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Probability of Default (Logistic Model)**")
        st.latex(r"PD_i = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \ln(Income_i / 1000))}}")
        st.markdown("*Coefficients calibrated from KCHSP 2022 · statistically significant at 1% level · AUC = 0.694*")
        st.latex(r"AF = \frac{1-(1+r)^{-n}}{r}")
    with c2:
        st.markdown("**RADCF Installment Formula**")
        st.latex(r"M = \frac{CF_{revised}}{(1 - PD) \times AF}")
        st.latex(r"CF_{revised} = OP + AdminCost - Deposit")
        st.markdown("**Affordability Ratio**")
        st.latex(r"ITI = \frac{M}{MonthlyIncome}")
        st.markdown("*< 20% Highly Affordable · 20–30% Affordable · 30–40% Moderate · > 40% Low*")


# ─────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────

tabs = st.tabs(["🧮 Manual Calculator", "📄 Paste Contract (Auto-fill)", "📊 Income Band Explorer"])


# ═══════════════════════════════════════════════
#  TAB 1 — MANUAL CALCULATOR
# ═══════════════════════════════════════════════

with tabs[0]:

    # ── INPUTS ──────────────────────────────────
    st.markdown('<div class="section-title">Contract & Borrower Inputs</div>', unsafe_allow_html=True)

    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown("**📋 Phone & Contract**")
        cash_price = st.number_input("Cash price (KSh)", min_value=0.0, value=25000.0, step=500.0, key="cp1")
        deposit_pct = st.number_input("Deposit (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0, key="dp1")
        admin_cost_pct = st.number_input("Administrative cost (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.5, key="ac1")

    with colB:
        st.markdown("**⏱️ Term & Rate**")
        n_months = st.number_input("Repayment term (months)", min_value=1, max_value=60, value=12, step=1, key="nm1")
        r_monthly = st.number_input("Monthly discount rate (e.g. 0.02 = 2%)", min_value=0.0, max_value=0.30,
                                     value=0.02, step=0.005, format="%.3f", key="rm1")
        income_ksh = st.number_input("Borrower monthly income (KSh)", min_value=0.0, value=20000.0,
                                      step=1000.0, key="inc1")

    with colC:
        st.markdown("**📊 Market Comparison (Optional)**")
        market_monthly = st.number_input("Market monthly installment (KSh)", min_value=0.0, value=3150.0,
                                          step=100.0, key="mm1")
        market_total_input = st.number_input("Market total repayment (KSh)", min_value=0.0, value=0.0,
                                              step=500.0, key="mt1")
        iti_threshold = st.slider("Affordability threshold (%)", min_value=10, max_value=50, value=30,
                                   step=5, key="iti1") / 100.0

    st.markdown("---")

    # ── COMPUTE ─────────────────────────────────
    pd_val = logistic_pd(income_ksh)
    res = fair_installment(cash_price, deposit_pct, admin_cost_pct, int(n_months), float(r_monthly), pd_val)

    fair_m = res["fair_monthly_installment"]
    fair_total = res["fair_total_paid"]
    iti = fair_m / income_ksh if income_ksh > 0 else 0.0

    pd_lbl, pd_col = pd_label(pd_val)
    iti_lbl, iti_col, iti_tag = affordability_label(iti)
    max_phone = max_affordable_phone(income_ksh, deposit_pct, admin_cost_pct,
                                      int(n_months), float(r_monthly), iti_threshold)

    # Market figures
    if market_total_input > 0:
        mkt_total = market_total_input
    elif market_monthly > 0:
        mkt_total = res["deposit_amount"] + market_monthly * int(n_months)
    else:
        mkt_total = 0.0

    over_amt = (mkt_total - fair_total) if mkt_total > 0 else float("nan")
    over_pct = (over_amt / fair_total) if (mkt_total > 0 and fair_total > 0) else float("nan")
    fairness_lbl, fairness_col = fairness_badge(over_pct) if mkt_total > 0 else ("", "")

    # ── KPI CARDS ───────────────────────────────
    st.markdown('<div class="section-title">RADCF Pricing Results</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Estimated PD", f"{pd_val:.3f}",
              delta=f"Risk: {pd_lbl}", delta_color="off" if pd_val < 0.35 else "inverse")
    k2.metric("Fair Monthly (KSh)", f"{fair_m:,.2f}")
    k3.metric("Fair Total Paid (KSh)", f"{fair_total:,.2f}")
    k4.metric("Deposit Amount (KSh)", f"{res['deposit_amount']:,.2f}")
    k5.metric("Affordability (ITI)", f"{iti*100:.1f}%",
              delta=f"{iti_lbl}", delta_color="off")

    if mkt_total > 0 and np.isfinite(over_pct):
        k6, k7, k8, _ = st.columns(4)
        k6.metric("Market Total (KSh)", f"{mkt_total:,.2f}")
        k7.metric("Overpricing Amount (KSh)", f"{over_amt:,.2f}",
                  delta_color="inverse" if over_amt > 0 else "normal")
        k8.metric("Overpricing %", f"{over_pct*100:.1f}%")
        principal_financed = cash_price - res["deposit_amount"]
        eff_m = market_monthly if market_monthly > 0 else (mkt_total - res["deposit_amount"]) / int(n_months)
        im = implied_monthly_rate(principal_financed, eff_m, int(n_months))
        apr = effective_apr(im)

    # ── GAUGES + CHART ROW ──────────────────────
    g1, g2, g3 = st.columns([1, 1, 2])
    with g1:
        st.plotly_chart(make_gauge(pd_val, "Default Probability", max_val=1.0,
                                    thresholds=[0.10, 0.25, 0.50, 1.0],
                                    colors_list=["#00CC88", "#9ACD32", "#FFD700", "#FF4B4B"]),
                        use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(make_iti_gauge(iti), use_container_width=True,
                        config={"displayModeBar": False})
    with g3:
        if mkt_total > 0:
            st.plotly_chart(
                make_bar_comparison(fair_m, market_monthly, int(n_months), res["deposit_amount"]),
                use_container_width=True, config={"displayModeBar": False})
        else:
            st.plotly_chart(make_pd_curve(income_ksh), use_container_width=True,
                            config={"displayModeBar": False})

    # ── AFFORDABILITY INSIGHT BOX ───────────────
    st.markdown('<div class="section-title">Affordability & Eligibility</div>', unsafe_allow_html=True)
    af_col1, af_col2 = st.columns(2)

    badge_map = {
        "green": "badge-green", "orange": "badge-orange", "red": "badge-red"
    }

    with af_col1:
        st.markdown(f"""
<div class="radcf-card">
  <p style="color:#aaa; font-size:12px; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Affordability Assessment</p>
  <p style="color:#fff; font-size:1.05rem; margin:0 0 6px;">
    Monthly installment = <strong style="color:#E63946;">KSh {fair_m:,.0f}</strong>
    on income of <strong style="color:#4FC3F7;">KSh {income_ksh:,.0f}</strong>
  </p>
  <p style="font-size: 1.5rem; font-weight: 800; color: {iti_col}; margin: 4px 0;">{iti*100:.1f}% ITI</p>
  <span class="{badge_map.get(iti_tag, 'badge-yellow')}">{iti_lbl}</span>
  <p style="color:#aaa; font-size:12px; margin-top: 10px;">
    Sustainability threshold: <strong style="color:#FFD700;">{iti_threshold*100:.0f}%</strong> of income.
    {"✅ Within budget — repayment is manageable." if iti <= iti_threshold
     else "⚠️ Exceeds threshold — high risk of financial stress and default."}
  </p>
</div>""", unsafe_allow_html=True)

    with af_col2:
        eligibility = "✅ Eligible" if iti <= iti_threshold else "❌ Not Recommended"
        elig_col = "#00CC88" if iti <= iti_threshold else "#FF4B4B"
        st.markdown(f"""
<div class="radcf-card">
  <p style="color:#aaa; font-size:12px; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Eligibility Framework</p>
  <p style="color:#fff; font-size:1rem; margin: 0 0 6px;">
    Recommended max phone price at <strong style="color:#4FC3F7;">KSh {income_ksh:,.0f}</strong> income:
  </p>
  <p style="font-size: 1.6rem; font-weight: 800; color: #FFD700; margin: 4px 0;">
    KSh {max_phone:,.0f}
  </p>
  <p style="color:{elig_col}; font-size: 1rem; font-weight: 700; margin: 6px 0;">
    Current deal: {eligibility}
  </p>
  <p style="color:#aaa; font-size:12px; margin: 4px 0;">
    KSh {cash_price:,.0f} phone {"is" if iti <= iti_threshold else "exceeds"} the
    {iti_threshold*100:.0f}% affordability ceiling at this income level.
  </p>
</div>""", unsafe_allow_html=True)

    if mkt_total > 0 and np.isfinite(over_pct):
        savings = over_amt
        st.markdown(f"""
<div style="background: linear-gradient(135deg, {'#1a2e1a' if over_amt > 0 else '#1a1a2e'}, #16213e);
            border-radius: 12px; border: 1px solid {'#E6394633' if over_amt > 0 else '#00CC8833'};
            padding: 16px 20px; margin: 8px 0;">
  <span style="font-size:1.2rem; font-weight:800; color:{fairness_col};">{fairness_lbl}</span>
  <p style="color:#ccc; margin: 8px 0 0; font-size:14px;">
    {'You are being overcharged' if over_amt > 0 else 'Market is below RADCF fair value'} by
    <strong style="color:{fairness_col};">KSh {abs(savings):,.2f}</strong> total
    (<strong>{abs(over_pct)*100:.1f}%</strong> {'above' if over_amt > 0 else 'below'} the actuarially fair price).
    {'Over a 12-month contract, this excess payment could fund school fees, food, or savings.' if over_amt > 0 else ''}
    {"Implied effective APR: " + f"{apr*100:.1f}%" if np.isfinite(apr) else ""}
  </p>
</div>""", unsafe_allow_html=True)

    # ── PD CURVE (always shown in bottom row if market data shown above) ──
    if mkt_total > 0:
        st.plotly_chart(make_pd_curve(income_ksh), use_container_width=True,
                        config={"displayModeBar": False})

    # ── SENSITIVITY ANALYSIS ────────────────────
    st.markdown('<div class="section-title">Sensitivity Analysis (Stress Test)</div>', unsafe_allow_html=True)

    pd_low = max(0.0, pd_val * 0.9)
    pd_high = min(1.0, pd_val * 1.1)

    scenarios_raw = [
        ("Base", pd_val, admin_cost_pct, r_monthly),
        ("PD -10%", pd_low, admin_cost_pct, r_monthly),
        ("PD +10%", pd_high, admin_cost_pct, r_monthly),
        ("Admin 8%", pd_val, 8.0, r_monthly),
        ("r +2pp", pd_val, admin_cost_pct, r_monthly + 0.02),
        ("r -1pp", pd_val, admin_cost_pct, max(0.0, r_monthly - 0.01)),
    ]

    rows = []
    for name, pd_s, admin_s, r_s in scenarios_raw:
        rr = fair_installment(cash_price, deposit_pct, admin_s, int(n_months), float(r_s), float(pd_s))
        rows.append({
            "Scenario": name,
            "PD": round(float(pd_s), 3),
            "Admin%": float(admin_s),
            "r": round(float(r_s), 3),
            "Fair Monthly (KSh)": round(rr["fair_monthly_installment"], 2),
            "Fair Total (KSh)": round(rr["fair_total_paid"], 2),
        })

    df_sens = pd.DataFrame(rows)
    st.dataframe(df_sens.style.format({
        "PD": "{:.3f}", "Admin%": "{:.0f}", "r": "{:.3f}",
        "Fair Monthly (KSh)": "{:,.2f}", "Fair Total (KSh)": "{:,.2f}"
    }).background_gradient(subset=["Fair Total (KSh)"], cmap="RdYlGn_r"),
        use_container_width=True)

    # Tornado charts side by side
    tc1, tc2 = st.columns(2)
    base_total = rows[0]["Fair Total (KSh)"]
    base_monthly = rows[0]["Fair Monthly (KSh)"]
    with tc1:
        st.plotly_chart(make_tornado(rows, base_total, "Fair Total (KSh)"),
                        use_container_width=True, config={"displayModeBar": False})
    with tc2:
        st.plotly_chart(make_tornado(rows, base_monthly, "Fair Monthly (KSh)"),
                        use_container_width=True, config={"displayModeBar": False})

    most_sensitive = max(rows[1:], key=lambda x: abs(x["Fair Total (KSh)"] - base_total))
    st.caption(f"🎯 Most sensitive factor: **{most_sensitive['Scenario']}** → "
               f"KSh {abs(most_sensitive['Fair Total (KSh)'] - base_total):,.2f} change from base.")

    # ── PDF DOWNLOAD ────────────────────────────
    st.markdown('<div class="section-title">Download Report</div>', unsafe_allow_html=True)
    pdf_bytes = generate_pdf_report(
        params={"cash_price": cash_price, "deposit_pct": deposit_pct,
                "admin_cost_pct": admin_cost_pct, "n_months": int(n_months),
                "r_monthly": float(r_monthly), "income_ksh": income_ksh},
        res=res, pd_val=pd_val, iti=iti,
        market_monthly=market_monthly if market_monthly > 0 else 0.0,
        scenarios=rows,
    )
    st.download_button(
        "📥 Download RADCF Report (PDF)",
        data=pdf_bytes,
        file_name=f"RADCF_Report_{datetime.date.today()}.pdf",
        mime="application/pdf",
    )
    st.caption("Report includes contract parameters, fair pricing results, affordability analysis, and sensitivity scenarios.")


# ═══════════════════════════════════════════════
#  TAB 2 — PASTE CONTRACT TEXT
# ═══════════════════════════════════════════════

with tabs[1]:
    st.markdown('<div class="section-title">Paste a Hire-Purchase Offer</div>', unsafe_allow_html=True)
    st.write("Paste a WhatsApp message, SMS, or advert text. The tool will extract contract terms automatically.")

    sample_text = (
        "Phone: Samsung A16. Cash price: KSh 18,000. Deposit 30%. "
        "Pay KSh 3,150 per month for 12 months. Admin fee 5%."
    )
    txt = st.text_area("Paste text here", value=sample_text, height=120, key="contract_text")

    extracted = extract_deal_fields(txt)

    with st.expander("🔍 Extracted Fields (raw)", expanded=False):
        st.json({k: v for k, v in extracted.items() if v is not None})

    st.markdown('<div class="section-title">Review / Edit Extracted Inputs</div>', unsafe_allow_html=True)
    cX, cY, cZ = st.columns(3)

    with cX:
        cp2 = st.number_input("Cash price (KSh)", value=float(extracted["cash_price"] or 18000.0),
                               min_value=0.0, step=500.0, key="cp2")
        n2 = st.number_input("Repayment term (months)", value=int(extracted["term_months"] or 12),
                              min_value=1, step=1, key="nm2")

    with cY:
        dep2_guess = extracted["deposit_pct"]
        if dep2_guess is None and extracted["deposit_amount"] and cp2 > 0:
            dep2_guess = 100.0 * float(extracted["deposit_amount"]) / cp2
        dep2 = st.number_input("Deposit (%)", value=float(dep2_guess or 30.0),
                                min_value=0.0, max_value=100.0, step=1.0, key="dp2")

        adm2_guess = extracted["admin_pct"]
        adm2 = st.number_input("Admin cost (%)", value=float(adm2_guess or 5.0),
                                min_value=0.0, max_value=30.0, step=0.5, key="adm2")

    with cZ:
        inc2 = st.number_input("Borrower monthly income (KSh)", value=20000.0,
                                min_value=0.0, step=1000.0, key="inc2")
        r2 = st.number_input("Monthly discount rate", value=0.02, min_value=0.0,
                              max_value=0.30, step=0.005, format="%.3f", key="r2")

    pd2 = logistic_pd(inc2)
    res2 = fair_installment(cp2, dep2, adm2, int(n2), float(r2), pd2)
    iti2 = res2["fair_monthly_installment"] / inc2 if inc2 > 0 else 0.0
    pd_lbl2, pd_col2 = pd_label(pd2)
    iti_lbl2, iti_col2, _ = affordability_label(iti2)

    st.markdown("---")
    st.markdown('<div class="section-title">RADCF Results</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estimated PD", f"{pd2:.3f}", delta=f"Risk: {pd_lbl2}", delta_color="off")
    k2.metric("Fair Monthly (KSh)", f"{res2['fair_monthly_installment']:,.2f}")
    k3.metric("Fair Total (KSh)", f"{res2['fair_total_paid']:,.2f}")
    k4.metric("ITI", f"{iti2*100:.1f}%", delta=iti_lbl2, delta_color="off")

    if extracted["monthly_installment"] is not None:
        mkt_m2 = float(extracted["monthly_installment"])
        mkt_t2 = res2["deposit_amount"] + mkt_m2 * int(n2)
        ov2 = mkt_t2 - res2["fair_total_paid"]
        ovp2 = ov2 / res2["fair_total_paid"] if res2["fair_total_paid"] > 0 else float("nan")
        fl2, fc2 = fairness_badge(ovp2)

        st.markdown('<div class="section-title">Market Comparison (from extracted data)</div>',
                    unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Market Monthly (KSh)", f"{mkt_m2:,.2f}")
        m2.metric("Market Total (KSh)", f"{mkt_t2:,.2f}")
        m3.metric("Overpricing (KSh)", f"{ov2:,.2f}")
        m4.metric("Overpricing (%)", f"{ovp2*100:.1f}%" if np.isfinite(ovp2) else "N/A")

        st.markdown(f"""
<div style="background:#1a1a2e; border-radius:10px; border:1px solid #E6394633;
            padding:14px 18px; margin:8px 0;">
  <span style="font-size:1.1rem; font-weight:700; color:{fc2};">{fl2}</span>
  <p style="color:#ccc; font-size:13px; margin:6px 0 0;">
    Market price is {'above' if ov2 > 0 else 'below'} RADCF fair value by
    <strong style="color:{fc2};">KSh {abs(ov2):,.2f}</strong> over the contract term.
  </p>
</div>""", unsafe_allow_html=True)

        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.plotly_chart(make_bar_comparison(res2["fair_monthly_installment"], mkt_m2,
                                                 int(n2), res2["deposit_amount"]),
                            use_container_width=True, config={"displayModeBar": False})
        with c_chart2:
            st.plotly_chart(make_pd_curve(inc2), use_container_width=True,
                            config={"displayModeBar": False})
    else:
        st.plotly_chart(make_pd_curve(inc2), use_container_width=True,
                        config={"displayModeBar": False})

    st.caption("ℹ️ Text extraction uses regex-based parsing. For best results, include keywords: "
               "'cash price', 'deposit', 'per month', 'admin fee'.")


# ═══════════════════════════════════════════════
#  TAB 3 — INCOME BAND EXPLORER
# ═══════════════════════════════════════════════

with tabs[2]:
    st.markdown('<div class="section-title">Explore RADCF Pricing Across Income Bands</div>',
                unsafe_allow_html=True)
    st.write("See how the RADCF fair installment, probability of default, and affordability vary "
             "across Kenya's income distribution for any phone price.")

    band_col1, band_col2, band_col3 = st.columns(3)
    with band_col1:
        phone_price_band = st.number_input("Phone cash price (KSh)", value=20000.0,
                                            min_value=5000.0, step=500.0, key="ppb")
    with band_col2:
        dep_band = st.number_input("Deposit (%)", value=30.0, min_value=0.0,
                                    max_value=100.0, step=5.0, key="dpb")
        adm_band = st.number_input("Admin cost (%)", value=5.0, min_value=0.0,
                                    max_value=20.0, step=1.0, key="adb")
    with band_col3:
        term_band = st.number_input("Term (months)", value=12, min_value=6,
                                     max_value=36, step=3, key="tmb")
        r_band = st.number_input("Monthly discount rate", value=0.02, min_value=0.0,
                                  max_value=0.20, step=0.005, format="%.3f", key="rb")
        market_band = st.number_input("Market monthly installment (KSh, for comparison)",
                                       value=3150.0, min_value=0.0, step=100.0, key="mkb")

    BANDS = [
        ("Below 10k", 7500, "Below KSh 10,000"),
        ("10k–15k", 12500, "KSh 10,000–15,000"),
        ("15k–25k", 20000, "KSh 15,001–25,000"),
        ("25k–40k", 32500, "KSh 25,001–40,000"),
        ("40k–70k", 55000, "KSh 40,001–70,000"),
        ("Above 70k", 85000, "Above KSh 70,000"),
    ]

    band_rows = []
    for label, mid, full_label in BANDS:
        pd_b = logistic_pd(mid)
        r_b = fair_installment(phone_price_band, dep_band, adm_band,
                                int(term_band), float(r_band), pd_b)
        m_b = r_b["fair_monthly_installment"]
        t_b = r_b["fair_total_paid"]
        iti_b = m_b / mid
        iti_lbl_b, _, _ = affordability_label(iti_b)
        markup_b = (t_b - phone_price_band) / phone_price_band * 100
        over_b = ((market_band * term_band + r_b["deposit_amount"]) - t_b) if market_band > 0 else float("nan")
        band_rows.append({
            "Income Band": full_label,
            "Mid Income": f"KSh {mid:,}",
            "Est. PD": f"{pd_b:.3f}",
            "RADCF Monthly (KSh)": round(m_b, 2),
            "RADCF Total (KSh)": round(t_b, 2),
            "ITI": f"{iti_b*100:.1f}%",
            "Affordability": iti_lbl_b,
            "Markup over Cash": f"+{markup_b:.1f}%",
            "Mkt Overcharge (KSh)": round(over_b, 2) if np.isfinite(over_b) else "—",
        })

    df_bands = pd.DataFrame(band_rows)
    st.dataframe(df_bands, use_container_width=True, hide_index=True)

    # Visualisations
    vis1, vis2 = st.columns(2)
    mids = [r[1] for r in BANDS]
    pds_b = [logistic_pd(m) for m in mids]
    labels_b = [r[0] for r in BANDS]
    radcf_ms = []
    mkt_ms = [market_band] * 6 if market_band > 0 else []
    for mid in mids:
        pd_b = logistic_pd(mid)
        r_b = fair_installment(phone_price_band, dep_band, adm_band,
                                int(term_band), float(r_band), pd_b)
        radcf_ms.append(r_b["fair_monthly_installment"])

    with vis1:
        fig_bands = go.Figure()
        fig_bands.add_trace(go.Bar(
            name="RADCF Fair", x=labels_b, y=radcf_ms,
            marker_color="#00CC88",
            text=[f"KSh {v:,.0f}" for v in radcf_ms],
            textposition="outside", textfont=dict(size=9, color="#ccc"),
        ))
        if market_band > 0:
            fig_bands.add_trace(go.Bar(
                name="Market Price", x=labels_b, y=[market_band]*6,
                marker_color="#E63946",
                text=[f"KSh {market_band:,.0f}"] * 6,
                textposition="outside", textfont=dict(size=9, color="#ccc"),
            ))
        fig_bands.update_layout(
            title=dict(text="RADCF vs Market Monthly Installment by Income Band",
                       font=dict(color="#ccc", size=13)),
            barmode="group",
            xaxis=dict(color="#888"), yaxis=dict(title="KSh", gridcolor="#333", color="#888"),
            height=320, margin=dict(l=20, r=20, t=50, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", legend=dict(font=dict(color="#ccc")),
        )
        st.plotly_chart(fig_bands, use_container_width=True, config={"displayModeBar": False})

    with vis2:
        fig_pd = go.Figure()
        fig_pd.add_trace(go.Bar(
            x=labels_b, y=[p*100 for p in pds_b],
            marker_color=["#FF4B4B","#FF8C00","#FFD700","#9ACD32","#4FC3F7","#00CC88"],
            text=[f"{p*100:.1f}%" for p in pds_b],
            textposition="outside", textfont=dict(size=10, color="#ccc"),
        ))
        fig_pd.update_layout(
            title=dict(text="Probability of Default by Income Band",
                       font=dict(color="#ccc", size=13)),
            xaxis=dict(color="#888"),
            yaxis=dict(title="PD (%)", gridcolor="#333", color="#888", range=[0, 80]),
            height=320, margin=dict(l=20, r=20, t=50, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
        )
        st.plotly_chart(fig_pd, use_container_width=True, config={"displayModeBar": False})

    # Eligibility table
    st.markdown('<div class="section-title">Income-Based Eligibility Framework</div>',
                unsafe_allow_html=True)

    elig_rows = []
    for label, mid, full_label in BANDS:
        max_p = max_affordable_phone(mid, dep_band, adm_band, int(term_band), float(r_band), 0.30)
        pd_b = logistic_pd(mid)
        r_b = fair_installment(phone_price_band, dep_band, adm_band,
                                int(term_band), float(r_band), pd_b)
        iti_b = r_b["fair_monthly_installment"] / mid
        eligible = iti_b <= 0.30
        elig_rows.append({
            "Income Band": full_label,
            "Est. PD": f"{pd_b:.3f}",
            "Max Affordable Phone (KSh)": f"KSh {max_p:,.0f}",
            f"KSh {phone_price_band:,.0f} Eligible?": "✅ Yes" if eligible else "❌ No",
            "Policy Note": (
                "High financial burden — opt for lower-priced phone" if iti_b > 0.40
                else "Borderline — monitor repayment" if iti_b > 0.30
                else "Affordable — low default risk"
            ),
        })
    st.dataframe(pd.DataFrame(elig_rows), use_container_width=True, hide_index=True)
    st.caption("Max Affordable Phone calculated at 30% ITI threshold per Hulchanski (1995) sustainability standard.")


# ─────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555; font-size:12px; padding:16px 0;">
  <strong style="color:#aaa;">RADCF Fair Pricing Engine</strong> ·
  Research by Martin Mureithi, Dominick Kariuki, Purity Kiprotich & Caleb Onyango ·
  Egerton University, Department of Mathematics · 2025 <br>
  Calibrated on KCHSP 2022 (17,452 households) · Supervisor: Mr Francis Ndung'u ·
  <em>Model AUC = 0.694 · β₀ = 3.0267 · β₁ = −1.2553</em>
</div>
""", unsafe_allow_html=True)
