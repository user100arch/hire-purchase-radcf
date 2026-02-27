import re
import math
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ============================================================
# Core actuarial/pricing functions
# ============================================================
def logistic_pd(income_ksh: float, beta0: float, beta1: float) -> float:
    """
    PD = 1 / (1 + exp(-(beta0 + beta1 * ln(income/100))))
    Clamps to [0,1].
    """
    if income_ksh <= 0:
        return 1.0
    x = math.log(income_ksh / 100.0)
    z = beta0 + beta1 * x
    pd_est = 1.0 / (1.0 + math.exp(-z))
    return float(max(0.0, min(1.0, pd_est)))


def annuity_factor(r: float, n: int) -> float:
    """
    AF = (1 - (1+r)^-n) / r
    If r == 0: AF = n
    """
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
    pd_est: float
) -> dict:
    """
    Consistent with your draft:
    OP = cash_price
    Deposit = OP * deposit_pct
    Admin = OP * admin_cost_pct
    CF_revised = OP + Admin - Deposit
    AF = annuity_factor(r, n)
    M = CF_revised / ((1 - PD) * AF)

    Fair total assumes full payment (deposit + M*n)
    RADCF PV is expected PV after default-risk adjustment:
        PV = deposit + (M * AF * (1-PD))
    """
    op = float(cash_price)
    deposit = op * (deposit_pct / 100.0)
    admin_cost = op * (admin_cost_pct / 100.0)
    cf_revised = op + admin_cost - deposit

    af = annuity_factor(r_monthly, int(n_months))
    repay_prob = max(1e-9, (1.0 - float(pd_est)))  # avoid divide-by-zero

    if af <= 0:
        m = float("nan")
    else:
        m = cf_revised / (repay_prob * af)

    fair_total = deposit + (m * n_months)
    radcf_pv = deposit + (m * af * repay_prob)

    return {
        "op": op,
        "deposit_amount": deposit,
        "admin_cost_amount": admin_cost,
        "cf_revised": cf_revised,
        "annuity_factor": af,
        "repay_prob": repay_prob,
        "fair_monthly_installment": m,
        "fair_total_paid_if_no_default": fair_total,
        "radcf_present_value": radcf_pv,
    }


def implied_monthly_rate_from_payment(P: float, payment: float, n: int) -> float:
    """
    Solve for i in: payment = P * i / (1 - (1+i)^-n) using binary search.
    Returns monthly i.
    """
    if P <= 0 or n <= 0:
        return float("nan")
    if payment * n < P:
        return float("nan")

    lo, hi = 0.0, 3.0  # 0% to 300% monthly
    for _ in range(80):
        mid = (lo + hi) / 2.0
        denom = 1.0 - (1.0 + mid) ** (-n)
        if denom <= 0:
            lo = mid
            continue
        pmid = P * mid / denom
        if pmid > payment:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def effective_apr_from_monthly(i: float) -> float:
    """
    Effective APR: (1+i)^12 - 1
    """
    if not np.isfinite(i):
        return float("nan")
    return float((1.0 + i) ** 12 - 1.0)


# ============================================================
# Simple text extraction (regex)
# ============================================================
def extract_deal_fields(text: str) -> dict:
    """
    Attempts to find:
      cash price, deposit (%/amount), term (months), monthly installment, admin fee (%/amount)
    """
    t = (text or "").lower()
    t = t.replace(",", " ")

    money = r"(?:ksh|kes)\s*([0-9]{2,})"
    pct = r"([0-9]{1,2}(?:\.[0-9]+)?)\s*%"

    # Cash price
    cash_price = None
    m = re.search(r"(cash price|cash|price)\s*[:\-]?\s*" + money, t)
    if m:
        cash_price = float(m.group(2))
    if cash_price is None:
        m2 = re.search(money, t)
        if m2:
            cash_price = float(m2.group(1))

    # Deposit
    deposit_pct = None
    mdp = re.search(r"(deposit|downpayment|down payment)\s*[:\-]?\s*" + pct, t)
    if mdp:
        deposit_pct = float(mdp.group(2))

    deposit_amount = None
    mda = re.search(r"(deposit|downpayment|down payment)\s*[:\-]?\s*" + money, t)
    if mda:
        deposit_amount = float(mda.group(2))

    # Term
    term_months = None
    mt = re.search(r"([0-9]{1,2})\s*(months|month|mos|mo)\b", t)
    if mt:
        term_months = int(mt.group(1))

    # Monthly installment
    monthly_installment = None
    mm = re.search(r"(installment|instalment|monthly|per month)\s*[:\-]?\s*" + money, t)
    if mm:
        monthly_installment = float(mm.group(2))

    # Admin cost
    admin_pct = None
    mapct = re.search(r"(admin|administration|processing)\s*(fee|cost)?\s*[:\-]?\s*" + pct, t)
    if mapct:
        admin_pct = float(mapct.group(3))

    admin_amount = None
    maamt = re.search(r"(admin|administration|processing)\s*(fee|cost)?\s*[:\-]?\s*" + money, t)
    if maamt:
        admin_amount = float(maamt.group(3))

    return {
        "cash_price": cash_price,
        "deposit_pct": deposit_pct,
        "deposit_amount": deposit_amount,
        "term_months": term_months,
        "monthly_installment": monthly_installment,
        "admin_pct": admin_pct,
        "admin_amount": admin_amount,
    }


# ============================================================
# Interpretation helpers
# ============================================================
def pd_bucket(pd_val: float) -> tuple[str, str]:
    if pd_val >= 0.50:
        return ("High", "High estimated repayment risk; fair installments increase to compensate expected default losses.")
    if pd_val >= 0.25:
        return ("Moderate", "Moderate repayment risk; pricing includes a meaningful credit-risk adjustment.")
    return ("Low", "Low repayment risk; pricing requires a smaller credit-risk adjustment.")


def fairness_tag(over_pct: float) -> tuple[str, str]:
    if not np.isfinite(over_pct):
        return ("", "")
    if over_pct >= 0.25:
        return ("Severely Overpriced", "Market pricing is far above RADCF fair value.")
    if over_pct >= 0.10:
        return ("Overpriced", "Market pricing is above RADCF fair value.")
    if over_pct >= -0.10:
        return ("Near Fair", "Market pricing is close to RADCF fair value.")
    return ("Below Fair", "Market pricing is below RADCF fair value (possible subsidy, promotion, or different risk structure).")


def ksh(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"KSh {x:,.2f}"


def pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x*100:.2f}%"


# ============================================================
# PDF generator (ReportLab)
# ============================================================
def build_pdf_report(
    report_title: str,
    generated_dt: datetime,
    inputs: dict,
    pd_params: dict,
    pd_value: float,
    radcf: dict,
    market: dict,
    sensitivity_df: pd.DataFrame,
) -> bytes:
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="RADCF Fair Pricing Report",
        author="RADCF Pricing Engine",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=16, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=10, leading=14))

    story = []

    # Header
    story.append(Paragraph("RADCF Fair Pricing Report", styles["H1x"]))
    story.append(Paragraph(report_title, styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {generated_dt.strftime('%Y-%m-%d %H:%M:%S')} (EAT)", styles["Small"]))
    story.append(Paragraph(f"Report ID: RADCF-{generated_dt.strftime('%Y%m%d-%H%M%S')}", styles["Small"]))
    story.append(Spacer(1, 12))

    # 1. Contract Summary
    story.append(Paragraph("1. Contract Summary (Inputs)", styles["H2x"]))
    input_rows = [
        ["Cash Price (OP)", ksh(inputs["cash_price"])],
        ["Repayment Term", f'{inputs["n_months"]} months'],
        ["Deposit", f'{inputs["deposit_pct"]:.2f}%  →  {ksh(radcf["deposit_amount"])}'],
        ["Admin Cost", f'{inputs["admin_cost_pct"]:.2f}%  →  {ksh(radcf["admin_cost_amount"])}'],
        ["Monthly Discount Rate r", f'{inputs["r_monthly"]:.4f}  ({inputs["r_monthly"]*100:.2f}% per month)'],
        ["Borrower Monthly Income", ksh(inputs["income_ksh"])],
    ]
    t1 = Table(input_rows, colWidths=[180, 320])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.Color(0.98, 0.98, 0.98)]),
    ]))
    story.append(t1)
    story.append(Spacer(1, 12))

    # 2. Risk Model
    story.append(Paragraph("2. Risk Model (PD Estimation)", styles["H2x"]))
    pd_level, pd_explain = pd_bucket(pd_value)
    story.append(Paragraph(
        "PD model form (logistic): PD = 1 / (1 + exp(-(β0 + β1 * ln(income/100))))",
        styles["Body"]
    ))
    story.append(Spacer(1, 6))

    pd_rows = [
        ["β0", f'{pd_params["beta0"]:.2f}'],
        ["β1", f'{pd_params["beta1"]:.2f}'],
        ["Estimated PD", f"{pd_value:.3f}"],
        ["Risk Interpretation", f"{pd_level} — {pd_explain}"],
    ]
    t2 = Table(pd_rows, colWidths=[180, 320])
    t2.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.Color(0.98, 0.98, 0.98)]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))

    # 3. RADCF Computation
    story.append(Paragraph("3. RADCF Pricing Computation (Core)", styles["H2x"]))

    story.append(Paragraph("Step A: Compute revised cashflow requirement", styles["Body"]))
    story.append(Paragraph(
        f"CF_revised = OP + AdminCost − Deposit = {ksh(radcf['op'])} + {ksh(radcf['admin_cost_amount'])} − {ksh(radcf['deposit_amount'])} = <b>{ksh(radcf['cf_revised'])}</b>",
        styles["Body"]
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Step B: Compute annuity factor", styles["Body"]))
    story.append(Paragraph(
        f"AF = (1 − (1+r)^(-n)) / r, where r={inputs['r_monthly']:.4f}, n={inputs['n_months']} → AF ≈ <b>{radcf['annuity_factor']:.4f}</b>",
        styles["Body"]
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Step C: Compute fair monthly installment", styles["Body"]))
    story.append(Paragraph(
        f"M = CF_revised / ((1−PD) * AF) = {ksh(radcf['cf_revised'])} / ({(1-pd_value):.3f} * {radcf['annuity_factor']:.4f}) → <b>{ksh(radcf['fair_monthly_installment'])}</b>",
        styles["Body"]
    ))
    story.append(Spacer(1, 12))

    # 4. Outputs
    story.append(Paragraph("4. Fair Price Outputs (Main Results)", styles["H2x"]))
    out_rows = [
        ["Deposit", ksh(radcf["deposit_amount"])],
        ["Fair Monthly Installment (M)", ksh(radcf["fair_monthly_installment"])],
        ["Fair Total Paid (Deposit + M*n)", ksh(radcf["fair_total_paid_if_no_default"])],
        ["RADCF Present Value (expected PV)", ksh(radcf["radcf_present_value"])],
    ]
    t3 = Table(out_rows, colWidths=[230, 270])
    t3.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.Color(0.98, 0.98, 0.98)]),
    ]))
    story.append(t3)
    story.append(Spacer(1, 12))

    # 5. Market comparison (optional)
    story.append(Paragraph("5. Market Comparison (if provided)", styles["H2x"]))
    if market.get("provided"):
        mc_rows = [
            ["Market Monthly Installment", ksh(market.get("market_monthly"))],
            ["Market Total Repayment", ksh(market.get("market_total"))],
            ["Overpricing Amount", ksh(market.get("over_amt"))],
            ["Overpricing (%)", f'{market.get("over_pct")*100:.2f}%'],
            ["Fairness Score (0–100)", f'{market.get("fairness_score"):.1f}'],
            ["Assessment", f'{market.get("tag")} — {market.get("tag_explain")}'],
        ]
        if np.isfinite(market.get("implied_apr", float("nan"))):
            mc_rows.append(["Implied APR (effective)", f'{market["implied_apr"]*100:.1f}%'])

        t4 = Table(mc_rows, colWidths=[230, 270])
        t4.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.Color(0.98, 0.98, 0.98)]),
        ]))
        story.append(t4)
    else:
        story.append(Paragraph("No market comparison values were provided in this run.", styles["Body"]))
    story.append(Spacer(1, 12))

    # 6. Sensitivity analysis
    story.append(Paragraph("6. Sensitivity Analysis (Stress Test)", styles["H2x"]))
    if sensitivity_df is not None and len(sensitivity_df) > 0:
        df = sensitivity_df.copy()
        # keep it readable in PDF
        cols = ["Scenario", "PD", "Admin%", "r", "Fair Monthly (KSh)", "Fair Total (KSh)"]
        df = df[cols]

        table_data = [cols] + df.values.tolist()
        t5 = Table(table_data, colWidths=[120, 50, 55, 45, 110, 110])
        t5.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t5)
    else:
        story.append(Paragraph("Sensitivity table not available.", styles["Body"]))
    story.append(Spacer(1, 12))
   

    # 7. Conclusion
    story.append(Paragraph("7. Conclusion & Recommendation (Auto-generated)", styles["H2x"]))
    conclusion = (
        f"Based on the RADCF framework and the inputs provided, the actuarially fair repayment plan is: "
        f"Deposit {ksh(radcf['deposit_amount'])}, monthly installment {ksh(radcf['fair_monthly_installment'])}, "
        f"and fair total {ksh(radcf['fair_total_paid_if_no_default'])}. "
    )
    if market.get("provided"):
        conclusion += (
            f"The market deal was assessed as {market.get('tag')}. "
            f"Overpricing was {market.get('over_pct')*100:.2f}% relative to RADCF fair value. "
        )
    conclusion += (
        "For consumer protection purposes, large deviations above RADCF fair value may indicate potential overpricing. "
        "Sensitivity results indicate which parameters most influence fair pricing."
    )
    story.append(Paragraph(conclusion, styles["Body"]))
    story.append(Spacer(1, 12))

    # 8. Assumptions
    story.append(Paragraph("8. Assumptions & Limitations", styles["H2x"]))
    assumptions = [
        "PD model is an income-based proxy; real lenders may use richer behavioral and credit history data.",
        "PD is treated as constant across the repayment term (simplifying assumption).",
        "No recovery after default is assumed (LGD ≈ 100%) unless the model is extended.",
        "Fair total assumes full payment of installments; RADCF PV reflects expected PV after default adjustment.",
        "Some vendors may subsidize products or bundle services, causing market prices to appear below fair value."
    ]
    for a in assumptions:
        story.append(Paragraph(f"• {a}", styles["Body"]))
    story.append(Spacer(1, 6))

    doc.build(story)
    return buf.getvalue()


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="RADCF Fair Pricing Engine", layout="wide")

st.title("Actuarial Evaluation of Consumer Overpricing in Kenya’s Hire-Purchase Market")
st.markdown("### Risk-Adjusted Discounted Cash Flow (RADCF) Pricing Engine")

with st.expander("Model Overview", expanded=True):
    st.markdown(
        """
This tool estimates an **actuarially fair** hire-purchase repayment plan using a **Risk-Adjusted Discounted Cash Flow (RADCF)** approach.

**Workflow**
1. Estimate **Probability of Default (PD)** via a logistic model (income-based proxy).
2. Adjust repayments using *(1 − PD)*.
3. Discount expected cashflows using monthly discount rate *r*.
4. Include deposit and administrative cost assumptions.

Use the **Market Deal Comparison** section to estimate overpricing and generate a formal PDF report.
        """
    )

with st.expander("Mathematical Framework (Formulas)", expanded=False):
    st.latex(r"PD = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \ln(\text{income}/100))}}")
    st.latex(r"AF = \frac{1-(1+r)^{-n}}{r}")
    st.latex(r"CF_{revised} = OP + \text{AdminCost} - \text{Deposit}")
    st.latex(r"M = \frac{CF_{revised}}{(1-PD)\cdot AF}")

tabs = st.tabs(["Manual Calculator", "Paste Contract Text (Auto-fill)"])


# ============================================================
# TAB 1: Manual Calculator
# ============================================================
with tabs[0]:
    st.subheader("Manual RADCF Calculator")

    colA, colB, colC = st.columns(3)

    with colA:
        cash_price = st.number_input("Cash price (KSh)", min_value=0.0, value=25000.0, step=500.0, key="cp1")
        n_months = st.number_input("Repayment term (months)", min_value=1, value=12, step=1, key="n1")
        income_ksh = st.number_input("Borrower monthly income (KSh)", min_value=0.0, value=30000.0, step=1000.0, key="inc1")

    with colB:
        deposit_pct = st.number_input("Deposit (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0, key="dp1")
        admin_cost_pct = st.number_input("Administrative cost (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.5, key="ad1")
        r_monthly = st.number_input("Monthly discount rate r (e.g. 0.02 = 2%)",
                                    min_value=0.0, value=0.02, step=0.005, format="%.3f", key="r1")

    with colC:
        st.markdown("**PD model parameters**")
        beta0 = st.number_input("β0", value=2.5, step=0.1, format="%.2f", key="b01")
        beta1 = st.number_input("β1", value=-0.4, step=0.05, format="%.2f", key="b11")

    pd_val = logistic_pd(income_ksh, beta0, beta1)
    res = fair_installment(cash_price, deposit_pct, admin_cost_pct, int(n_months), float(r_monthly), pd_val)

    st.divider()

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("### Fair Pricing Outputs")
        lvl, expl = pd_bucket(pd_val)
        st.metric("Estimated PD", f"{pd_val:.3f}")
        st.caption(f"Risk Level: **{lvl}** — {expl}")

        st.metric("Fair monthly installment (KSh)", f"{res['fair_monthly_installment']:.2f}")
        st.metric("Deposit amount (KSh)", f"{res['deposit_amount']:.2f}")
        st.metric("Admin cost amount (KSh)", f"{res['admin_cost_amount']:.2f}")
        st.metric("Fair total paid (KSh)", f"{res['fair_total_paid_if_no_default']:.2f}")
        st.metric("RADCF PV (expected PV)", f"{res['radcf_present_value']:.2f}")

    # Market comparison + PDF data
    market_info = {"provided": False}

    with right:
        st.markdown("### Market Deal Comparison (optional)")
        market_monthly = st.number_input("Market monthly installment (KSh)", min_value=0.0, value=0.0, step=100.0, key="m_m1")
        market_total = st.number_input("Market total repayment (KSh)", min_value=0.0, value=0.0, step=500.0, key="m_t1")

        fair_total = res["fair_total_paid_if_no_default"]

        over_amt = float("nan")
        over_pct = float("nan")
        implied_apr = float("nan")
        mkt_total_used = 0.0

        if market_total > 0:
            mkt_total_used = float(market_total)
            over_amt = mkt_total_used - fair_total
            over_pct = (over_amt / fair_total) if fair_total > 0 else float("nan")
        elif market_monthly > 0:
            mkt_total_used = res["deposit_amount"] + float(market_monthly) * int(n_months)
            over_amt = mkt_total_used - fair_total
            over_pct = (over_amt / fair_total) if fair_total > 0 else float("nan")

            principal_financed = cash_price - res["deposit_amount"]
            im = implied_monthly_rate_from_payment(principal_financed, float(market_monthly), int(n_months))
            implied_apr = effective_apr_from_monthly(im)

        if np.isfinite(over_pct):
            tag, tag_explain = fairness_tag(over_pct)
            fairness_score = max(0.0, min(100.0, 100.0 - over_pct * 100.0))

            st.metric("Overpricing amount (KSh)", f"{over_amt:.2f}")
            st.metric("Overpricing (%)", f"{over_pct*100:.2f}%")
            st.metric("Fairness Score (0–100)", f"{fairness_score:.1f}")
            st.caption(f"Assessment: **{tag}** — {tag_explain}")

            if np.isfinite(implied_apr):
                st.metric("Implied APR (effective)", f"{implied_apr*100:.1f}%")

            market_info = {
                "provided": True,
                "market_monthly": float(market_monthly) if market_monthly > 0 else float("nan"),
                "market_total": float(mkt_total_used),
                "over_amt": float(over_amt),
                "over_pct": float(over_pct),
                "fairness_score": float(fairness_score),
                "tag": tag,
                "tag_explain": tag_explain,
                "implied_apr": float(implied_apr),
            }

    # Sensitivity table (for PDF + UI)
    st.divider()
    st.markdown("### Sensitivity Analysis (Stress Test)")

    pd_low = max(0.0, pd_val * 0.9)
    pd_high = min(1.0, pd_val * 1.1)

    scenarios = [
        ("Base", pd_val, admin_cost_pct, r_monthly),
        ("PD -10%", pd_low, admin_cost_pct, r_monthly),
        ("PD +10%", pd_high, admin_cost_pct, r_monthly),
        ("Admin 8%", pd_val, 8.0, r_monthly),
        ("r +2pp", pd_val, admin_cost_pct, r_monthly + 0.02),
        ("r -1pp", pd_val, admin_cost_pct, max(0.0, r_monthly - 0.01)),
    ]

    rows = []
    for name, pd_s, admin_s, r_s in scenarios:
        rr = fair_installment(cash_price, deposit_pct, admin_s, int(n_months), float(r_s), float(pd_s))
        rows.append({
            "Scenario": name,
            "PD": round(float(pd_s), 3),
            "Admin%": float(admin_s),
            "r": round(float(r_s), 3),
            "Fair Monthly (KSh)": round(rr["fair_monthly_installment"], 2),
            "Fair Total (KSh)": round(rr["fair_total_paid_if_no_default"], 2),
        })

    sens_df = pd.DataFrame(rows)
    st.dataframe(sens_df, use_container_width=True)

    # -----------------------------
# Tornado chart (Sensitivity visualization)
# -----------------------------
st.markdown("### Tornado Chart (Sensitivity Impact vs Base)")

# Make sure the expected columns exist
required_cols = {"Scenario", "Fair Total (KSh)", "Fair Monthly (KSh)"}
if not required_cols.issubset(df.columns):
    st.warning("Sensitivity table is missing expected columns for tornado chart.")
else:
    # Choose which metric to visualize
    metric = st.radio(
        "Show sensitivity for:",
        options=["Fair Total (KSh)", "Fair Monthly (KSh)"],
        horizontal=True,
        index=0,
        key="tornado_metric"
    )

    # Get base value
    base_row = df[df["Scenario"] == "Base"]
    if base_row.empty:
        st.warning("Base scenario not found in sensitivity table.")
    else:
        base_val = float(base_row.iloc[0][metric])

        # Compute impacts (difference from base)
        plot_df = df.copy()
        plot_df["Impact"] = plot_df[metric].astype(float) - base_val

        # Exclude "Base" from chart bars (optional)
        plot_df = plot_df[plot_df["Scenario"] != "Base"].copy()

        # Sort by absolute impact (tornado style)
        plot_df["AbsImpact"] = plot_df["Impact"].abs()
        plot_df = plot_df.sort_values("AbsImpact", ascending=True)

        # Split into decreases and increases for coloring/legend
        dec = plot_df[plot_df["Impact"] < 0]
        inc = plot_df[plot_df["Impact"] > 0]

        fig = go.Figure()

        # Negative impacts (left)
        fig.add_trace(go.Bar(
            y=dec["Scenario"],
            x=dec["Impact"],
            orientation="h",
            name="Decrease vs Base"
        ))

        # Positive impacts (right)
        fig.add_trace(go.Bar(
            y=inc["Scenario"],
            x=inc["Impact"],
            orientation="h",
            name="Increase vs Base"
        ))

        fig.update_layout(
            barmode="relative",
            height=420,
            xaxis_title=f"Impact on {metric} (KSh)",
            yaxis_title="Scenario",
            title=f"Sensitivity Tornado Chart (Base = {base_val:,.2f} KSh)",
            margin=dict(l=20, r=20, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Add a vertical zero line (visual base reference)
        fig.add_vline(x=0, line_width=1)

        st.plotly_chart(fig, use_container_width=True)

        # Quick interpretation (auto)
        most_sensitive = plot_df.iloc[-1]
        st.caption(
            f"Most sensitive factor (by absolute impact): **{most_sensitive['Scenario']}** "
            f"→ {most_sensitive['Impact']:,.2f} KSh change from base."
        )

    # PDF download (manual tab)
    st.divider()
    st.markdown("### Download Report")

    pdf_bytes = build_pdf_report(
        report_title="Actuarial Evaluation of Consumer Overpricing in Kenya’s Hire-Purchase Market (RADCF Engine)",
        generated_dt=datetime.now(),
        inputs={
            "cash_price": cash_price,
            "n_months": int(n_months),
            "deposit_pct": float(deposit_pct),
            "admin_cost_pct": float(admin_cost_pct),
            "r_monthly": float(r_monthly),
            "income_ksh": float(income_ksh),
        },
        pd_params={"beta0": float(beta0), "beta1": float(beta1)},
        pd_value=float(pd_val),
        radcf=res,
        market=market_info,
        sensitivity_df=sens_df,
    )

    st.download_button(
        label="Download RADCF Report (PDF)",
        data=pdf_bytes,
        file_name=f"RADCF_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        key="dl_pdf_manual"
    )


# ============================================================
# TAB 2: Paste Contract Text (Auto-fill)
# ============================================================
with tabs[1]:
    st.subheader("Paste Contract / Offer Text → Auto-fill")
    st.write("Paste a hire-purchase offer (WhatsApp message, advert text). We'll extract fields and compute RADCF fair price.")

    sample = "Cash price: KSh 25000. Deposit 30%. Pay KES 2500 per month for 12 months. Admin fee 5%."
    txt = st.text_area("Paste text here", value=sample, height=140, key="txt_offer")

    extracted = extract_deal_fields(txt)
    st.markdown("### Extracted (best guesses)")
    st.json(extracted)

    st.markdown("### Auto-filled calculator")
    colX, colY = st.columns(2)

    with colX:
        cash_price2 = st.number_input(
            "Cash price (KSh)",
            min_value=0.0,
            value=float(extracted["cash_price"] or 25000.0),
            step=500.0,
            key="cp2"
        )
        n_months2 = st.number_input(
            "Repayment term (months)",
            min_value=1,
            value=int(extracted["term_months"] or 12),
            step=1,
            key="n2"
        )
        income2 = st.number_input(
            "Borrower monthly income (KSh)",
            min_value=0.0,
            value=30000.0,
            step=1000.0,
            key="inc2"
        )

    with colY:
        dep_pct_guess = extracted["deposit_pct"]
        if dep_pct_guess is None and extracted["deposit_amount"] is not None and cash_price2 > 0:
            dep_pct_guess = 100.0 * float(extracted["deposit_amount"]) / float(cash_price2)

        deposit_pct2 = st.number_input(
            "Deposit (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(dep_pct_guess or 30.0),
            step=1.0,
            key="dp2"
        )

        admin_pct_guess = extracted["admin_pct"]
        if admin_pct_guess is None and extracted["admin_amount"] is not None and cash_price2 > 0:
            admin_pct_guess = 100.0 * float(extracted["admin_amount"]) / float(cash_price2)

        admin2 = st.number_input(
            "Administrative cost (%)",
            min_value=0.0,
            max_value=30.0,
            value=float(admin_pct_guess or 5.0),
            step=0.5,
            key="ad2"
        )

        r2 = st.number_input(
            "Monthly discount rate r",
            min_value=0.0,
            value=0.02,
            step=0.005,
            format="%.3f",
            key="r2"
        )

    st.markdown("**PD parameters**")
    colP1, colP2 = st.columns(2)
    with colP1:
        beta0_2 = st.number_input("β0", value=2.5, step=0.1, format="%.2f", key="b02")
    with colP2:
        beta1_2 = st.number_input("β1", value=-0.4, step=0.05, format="%.2f", key="b12")

    pd2 = logistic_pd(income2, beta0_2, beta1_2)
    res2 = fair_installment(cash_price2, deposit_pct2, admin2, int(n_months2), float(r2), pd2)

    st.divider()
    st.metric("Estimated PD", f"{pd2:.3f}")
    st.metric("Fair monthly installment (KSh)", f"{res2['fair_monthly_installment']:.2f}")
    st.metric("Fair total paid (KSh)", f"{res2['fair_total_paid_if_no_default']:.2f}")

    # Market comparison from extracted monthly installment
    market_info2 = {"provided": False}
    sens_df2 = pd.DataFrame()

    if extracted["monthly_installment"] is not None:
        market_m = float(extracted["monthly_installment"])
        market_total_est = res2["deposit_amount"] + market_m * int(n_months2)

        over_amt2 = market_total_est - res2["fair_total_paid_if_no_default"]
        over_pct2 = (over_amt2 / res2["fair_total_paid_if_no_default"]) if res2["fair_total_paid_if_no_default"] > 0 else float("nan")

        tag2, tag_explain2 = fairness_tag(over_pct2)
        fairness_score2 = max(0.0, min(100.0, 100.0 - over_pct2 * 100.0))

        principal_financed2 = cash_price2 - res2["deposit_amount"]
        im2 = implied_monthly_rate_from_payment(principal_financed2, market_m, int(n_months2))
        apr2 = effective_apr_from_monthly(im2)

        st.divider()
        st.markdown("### Market comparison (from extracted monthly installment)")
        st.metric("Market monthly installment (KSh)", f"{market_m:.2f}")
        st.metric("Estimated market total paid (KSh)", f"{market_total_est:.2f}")
        st.metric("Overpricing amount (KSh)", f"{over_amt2:.2f}")
        st.metric("Overpricing (%)", f"{over_pct2*100:.2f}%")
        st.metric("Fairness Score (0–100)", f"{fairness_score2:.1f}")
        st.caption(f"Assessment: **{tag2}** — {tag_explain2}")

        if np.isfinite(apr2):
            st.metric("Implied APR (effective)", f"{apr2*100:.1f}%")

        market_info2 = {
            "provided": True,
            "market_monthly": float(market_m),
            "market_total": float(market_total_est),
            "over_amt": float(over_amt2),
            "over_pct": float(over_pct2),
            "fairness_score": float(fairness_score2),
            "tag": tag2,
            "tag_explain": tag_explain2,
            "implied_apr": float(apr2),
        }

    # Build a small sensitivity table also in auto-fill mode (so PDF is complete)
    pd_low2 = max(0.0, pd2 * 0.9)
    pd_high2 = min(1.0, pd2 * 1.1)
    scenarios2 = [
        ("Base", pd2, admin2, r2),
        ("PD -10%", pd_low2, admin2, r2),
        ("PD +10%", pd_high2, admin2, r2),
        ("Admin 8%", pd2, 8.0, r2),
        ("r +2pp", pd2, admin2, r2 + 0.02),
        ("r -1pp", pd2, admin2, max(0.0, r2 - 0.01)),
    ]
    rows2 = []
    for name, pd_s, admin_s, r_s in scenarios2:
        rr = fair_installment(cash_price2, deposit_pct2, admin_s, int(n_months2), float(r_s), float(pd_s))
        rows2.append({
            "Scenario": name,
            "PD": round(float(pd_s), 3),
            "Admin%": float(admin_s),
            "r": round(float(r_s), 3),
            "Fair Monthly (KSh)": round(rr["fair_monthly_installment"], 2),
            "Fair Total (KSh)": round(rr["fair_total_paid_if_no_default"], 2),
        })
    sens_df2 = pd.DataFrame(rows2)

    st.divider()
    st.markdown("### Download Report")

    pdf_bytes2 = build_pdf_report(
        report_title="Actuarial Evaluation of Consumer Overpricing in Kenya’s Hire-Purchase Market (RADCF Engine)",
        generated_dt=datetime.now(),
        inputs={
            "cash_price": float(cash_price2),
            "n_months": int(n_months2),
            "deposit_pct": float(deposit_pct2),
            "admin_cost_pct": float(admin2),
            "r_monthly": float(r2),
            "income_ksh": float(income2),
        },
        pd_params={"beta0": float(beta0_2), "beta1": float(beta1_2)},
        pd_value=float(pd2),
        radcf=res2,
        market=market_info2,
        sensitivity_df=sens_df2,
    )

    st.download_button(
        label="Download RADCF Report (PDF)",
        data=pdf_bytes2,
        file_name=f"RADCF_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        key="dl_pdf_autofill"
    )

    st.caption("Extraction is basic regex for now. Next step: upload images/PDFs + OCR.")
