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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ============================================================
# FINAL MODEL PARAMETERS
# Fitted on World Bank Global Findex 2024 Kenya Microdata
# Income distribution from Kenya Household Income Dataset (n=1,000)
# AUC = 0.7409
# ============================================================
DEFAULT_BETA0 =  4.8497   # intercept
DEFAULT_BETA1 = -1.0054   # log_income    (higher income → lower default)
DEFAULT_BETA2 = -0.1700   # stable_job    (stable income → lower default)
DEFAULT_BETA3 =  0.5851   # age_scaled    (older age → higher default)
MEAN_AGE      = 31.4      # population mean age (Findex Kenya)
SD_AGE        = 12.4      # population SD age   (Findex Kenya)
MEAN_STABLE   = 0.22      # proportion with stable income (Findex Kenya)


# ============================================================
# Core actuarial/pricing functions
# ============================================================

def logistic_pd(income_ksh: float,
                stable_job: float = MEAN_STABLE,
                age: float        = MEAN_AGE,
                beta0: float      = DEFAULT_BETA0,
                beta1: float      = DEFAULT_BETA1,
                beta2: float      = DEFAULT_BETA2,
                beta3: float      = DEFAULT_BETA3) -> float:
    """
    3-variable logistic PD model:
      PD = 1 / (1 + exp(-(β₀ + β₁·ln(income/100) + β₂·StableJob + β₃·AgeScaled)))

    Parameters
    ----------
    income_ksh : monthly income in KES
    stable_job : 1 = stable income, 0 = unstable/irregular income
    age        : borrower age in years (will be standardised internally)
    beta0–3    : model coefficients
    """
    if income_ksh <= 0:
        return 1.0
    x        = math.log(income_ksh / 100.0)
    age_sc   = (age - MEAN_AGE) / SD_AGE
    z        = beta0 + beta1 * x + beta2 * float(stable_job) + beta3 * age_sc
    pd_est   = 1.0 / (1.0 + math.exp(-z))
    return float(max(0.0, min(1.0, pd_est)))


def annuity_factor(r: float, n: int) -> float:
    """AF = (1 - (1+r)^-n) / r ; if r==0 then AF=n"""
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
    """
    OP       = cash_price
    Deposit  = OP × deposit_pct
    Admin    = OP × admin_cost_pct
    CF_rev   = OP + Admin − Deposit
    AF       = annuity_factor(r, n)
    M        = CF_rev / ((1 − PD) × AF)
    """
    op         = float(cash_price)
    deposit    = op * (deposit_pct / 100.0)
    admin_cost = op * (admin_cost_pct / 100.0)
    cf_revised = op + admin_cost - deposit

    af          = annuity_factor(float(r_monthly), int(n_months))
    repay_prob  = max(1e-9, (1.0 - float(pd_est)))

    m          = float("nan") if af <= 0 else cf_revised / (repay_prob * af)
    fair_total = deposit + (m * n_months)
    radcf_pv   = deposit + (m * af * repay_prob)

    return {
        "op":                          op,
        "deposit_amount":              deposit,
        "admin_cost_amount":           admin_cost,
        "cf_revised":                  cf_revised,
        "annuity_factor":              af,
        "repay_prob":                  repay_prob,
        "fair_monthly_installment":    m,
        "fair_total_paid_if_no_default": fair_total,
        "radcf_present_value":         radcf_pv,
    }


def implied_monthly_rate_from_payment(P: float, payment: float, n: int) -> float:
    """Solve i: payment = P·i / (1-(1+i)^-n) via binary search."""
    if P <= 0 or n <= 0:
        return float("nan")
    if payment * n < P:
        return float("nan")
    lo, hi = 0.0, 3.0
    for _ in range(80):
        mid   = (lo + hi) / 2.0
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
    if not np.isfinite(i):
        return float("nan")
    return float((1.0 + i) ** 12 - 1.0)


# ============================================================
# Simple text extraction (regex)
# ============================================================
def extract_deal_fields(text: str) -> dict:
    t     = (text or "").lower().replace(",", " ")
    money = r"(?:ksh|kes)\s*([0-9]{2,})"
    pct   = r"([0-9]{1,2}(?:\.[0-9]+)?)\s*%"

    cash_price = None
    m = re.search(r"(cash price|cash|price)\s*[:\-]?\s*" + money, t)
    if m:
        cash_price = float(m.group(2))
    if cash_price is None:
        m2 = re.search(money, t)
        if m2:
            cash_price = float(m2.group(1))

    deposit_pct, deposit_amount = None, None
    mdp = re.search(r"(deposit|downpayment|down payment)\s*[:\-]?\s*" + pct, t)
    if mdp:
        deposit_pct = float(mdp.group(2))
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

    admin_pct, admin_amount = None, None
    mapct = re.search(r"(admin|administration|processing)\s*(fee|cost)?\s*[:\-]?\s*" + pct, t)
    if mapct:
        admin_pct = float(mapct.group(3))
    maamt = re.search(r"(admin|administration|processing)\s*(fee|cost)?\s*[:\-]?\s*" + money, t)
    if maamt:
        admin_amount = float(maamt.group(3))

    return {
        "cash_price": cash_price, "deposit_pct": deposit_pct,
        "deposit_amount": deposit_amount, "term_months": term_months,
        "monthly_installment": monthly_installment,
        "admin_pct": admin_pct, "admin_amount": admin_amount,
    }


# ============================================================
# Interpretation helpers
# ============================================================
def pd_bucket(pd_val: float) -> tuple[str, str]:
    if pd_val >= 0.50:
        return ("High", "High estimated repayment risk; fair installments increase significantly to compensate expected default losses.")
    if pd_val >= 0.25:
        return ("Moderate", "Moderate repayment risk; pricing includes a meaningful credit-risk adjustment.")
    return ("Low", "Low repayment risk; pricing requires a smaller credit-risk adjustment.")


def age_risk_label(age: float) -> str:
    if age < 26:  return "18–25 (young borrower, lower observed default)"
    if age < 36:  return "26–35 (prime age, moderate risk)"
    if age < 46:  return "36–45 (mid-career, elevated risk)"
    if age < 61:  return "46–60 (senior borrower, high observed default)"
    return "60+ (very high observed default rate in Kenya data)"


def fairness_tag(over_pct: float) -> tuple[str, str]:
    if not np.isfinite(over_pct):
        return ("", "")
    if over_pct >= 0.25:
        return ("Severely Overpriced", "Market pricing is far above RADCF fair value.")
    if over_pct >= 0.10:
        return ("Overpriced", "Market pricing is above RADCF fair value.")
    if over_pct >= -0.10:
        return ("Near Fair", "Market pricing is close to RADCF fair value.")
    return ("Below Fair", "Market pricing is below RADCF fair value.")


def ksh(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"KSh {x:,.2f}"


# ============================================================
# PDF generator (ReportLab)
# ============================================================
def build_pdf_report(
    report_title, generated_dt, inputs, pd_params,
    pd_value, radcf, market, sensitivity_df,
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=36, rightMargin=36,
                            topMargin=36, bottomMargin=36,
                            title="RADCF Fair Pricing Report")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"],
                              fontSize=16, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"],
                              fontSize=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"],
                              fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"],
                              fontSize=10, leading=14))
    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("RADCF Fair Pricing Report", styles["H1x"]))
    story.append(Paragraph(report_title, styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated: {generated_dt.strftime('%Y-%m-%d %H:%M:%S')} (EAT)", styles["Small"]))
    story.append(Paragraph(
        f"Report ID: RADCF-{generated_dt.strftime('%Y%m%d-%H%M%S')}", styles["Small"]))
    story.append(Spacer(1, 12))

    # ── 1. Contract Summary ────────────────────────────────────────────────────
    story.append(Paragraph("1. Contract Summary (Inputs)", styles["H2x"]))
    input_rows = [
        ["Cash Price (OP)",            ksh(inputs["cash_price"])],
        ["Repayment Term",             f'{inputs["n_months"]} months'],
        ["Deposit",                    f'{inputs["deposit_pct"]:.2f}%  →  {ksh(radcf["deposit_amount"])}'],
        ["Admin Cost",                 f'{inputs["admin_cost_pct"]:.2f}%  →  {ksh(radcf["admin_cost_amount"])}'],
        ["Monthly Discount Rate r",    f'{inputs["r_monthly"]:.4f}  ({inputs["r_monthly"]*100:.2f}% per month)'],
        ["Borrower Monthly Income",    ksh(inputs["income_ksh"])],
        ["Job/Income Stability",       "Stable income" if inputs["stable_job"] == 1 else "Unstable/Irregular income"],
        ["Borrower Age",               f'{inputs["age"]:.0f} years  ({age_risk_label(inputs["age"])})'],
    ]
    t1 = Table(input_rows, colWidths=[180, 320])
    t1.setStyle(TableStyle([
        ("GRID",           (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME",       (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.white, colors.Color(0.98, 0.98, 0.98)]),
    ]))
    story.append(t1)
    story.append(Spacer(1, 12))

    # ── 2. Risk Model ──────────────────────────────────────────────────────────
    story.append(Paragraph("2. Risk Model (3-Variable PD Estimation)", styles["H2x"]))
    story.append(Paragraph(
        "PD = 1 / (1 + exp(-(β₀ + β₁·ln(Income/100) + β₂·StableJob + β₃·AgeScaled)))",
        styles["Body"]))
    story.append(Paragraph(
        "Model fitted on World Bank Global Findex 2024 Kenya Microdata (n=934 credit buyers). "
        "AUC = 0.7409. Income distribution from Kenya Household Income Dataset (n=1,000).",
        styles["Small"]))
    story.append(Spacer(1, 6))

    pd_level, pd_explain = pd_bucket(pd_value)
    age_sc = (inputs["age"] - MEAN_AGE) / SD_AGE
    pd_rows = [
        ["β₀ (intercept)",        f'{pd_params["beta0"]:.4f}'],
        ["β₁ (log income)",       f'{pd_params["beta1"]:.4f}  — higher income lowers default risk'],
        ["β₂ (stable job)",       f'{pd_params["beta2"]:.4f}  — stable income lowers default risk'],
        ["β₃ (age scaled)",       f'{pd_params["beta3"]:.4f}  — older age raises default risk'],
        ["AUC (test set)",        "0.7409"],
        ["Age (standardised)",    f'{age_sc:.3f}  (age {inputs["age"]:.0f} yrs)'],
        ["Job Stability",         "Stable (1)" if inputs["stable_job"]==1 else "Unstable (0)"],
        ["Estimated PD",          f'{pd_value:.4f}  ({pd_value*100:.1f}%)'],
        ["Risk Level",            f'{pd_level} — {pd_explain}'],
    ]
    t2 = Table(pd_rows, colWidths=[180, 320])
    t2.setStyle(TableStyle([
        ("GRID",           (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.white, colors.Color(0.98, 0.98, 0.98)]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))

    # ── 3. RADCF Computation ──────────────────────────────────────────────────
    story.append(Paragraph("3. RADCF Pricing Computation (Core)", styles["H2x"]))
    story.append(Paragraph(
        f"CF_revised = OP + AdminCost − Deposit = "
        f"{ksh(radcf['op'])} + {ksh(radcf['admin_cost_amount'])} − "
        f"{ksh(radcf['deposit_amount'])} = <b>{ksh(radcf['cf_revised'])}</b>",
        styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"AF = (1 − (1+r)^(−n)) / r,  r={inputs['r_monthly']:.4f},  "
        f"n={inputs['n_months']}  →  AF ≈ <b>{radcf['annuity_factor']:.4f}</b>",
        styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"M = CF_revised / ((1−PD) × AF)  →  <b>{ksh(radcf['fair_monthly_installment'])}</b>",
        styles["Body"]))
    story.append(Spacer(1, 12))

    # ── 4. Outputs ────────────────────────────────────────────────────────────
    story.append(Paragraph("4. Fair Price Outputs (Main Results)", styles["H2x"]))
    out_rows = [
        ["Deposit",                          ksh(radcf["deposit_amount"])],
        ["Fair Monthly Installment (M)",     ksh(radcf["fair_monthly_installment"])],
        ["Fair Total Paid (Deposit + M×n)",  ksh(radcf["fair_total_paid_if_no_default"])],
        ["RADCF Present Value (expected PV)", ksh(radcf["radcf_present_value"])],
    ]
    t3 = Table(out_rows, colWidths=[230, 270])
    t3.setStyle(TableStyle([
        ("GRID",           (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.white, colors.Color(0.98, 0.98, 0.98)]),
    ]))
    story.append(t3)
    story.append(Spacer(1, 12))

    # ── 5. Market Comparison ──────────────────────────────────────────────────
    story.append(Paragraph("5. Market Comparison (if provided)", styles["H2x"]))
    if market.get("provided"):
        mc_rows = [
            ["Market Monthly Installment",  ksh(market.get("market_monthly"))],
            ["Market Total Repayment",      ksh(market.get("market_total"))],
            ["Overpricing Amount",          ksh(market.get("over_amt"))],
            ["Overpricing (%)",             f'{market.get("over_pct")*100:.2f}%'],
            ["Fairness Score (0–100)",      f'{market.get("fairness_score"):.1f}'],
            ["Assessment",                  f'{market.get("tag")} — {market.get("tag_explain")}'],
        ]
        t4 = Table(mc_rows, colWidths=[230, 270])
        t4.setStyle(TableStyle([
            ("GRID",           (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.white, colors.Color(0.98, 0.98, 0.98)]),
        ]))
        story.append(t4)
    else:
        story.append(Paragraph(
            "No market comparison values were provided.", styles["Body"]))
    story.append(Spacer(1, 12))

    # ── 6. Sensitivity ────────────────────────────────────────────────────────
    story.append(Paragraph("6. Sensitivity Analysis (Stress Test)", styles["H2x"]))
    if sensitivity_df is not None and len(sensitivity_df) > 0:
        cols       = ["Scenario", "PD", "Admin%", "r", "Fair Monthly (KSh)", "Fair Total (KSh)"]
        dfp        = sensitivity_df[cols].copy()
        table_data = [cols] + dfp.values.tolist()
        t5 = Table(table_data, colWidths=[120, 50, 55, 45, 110, 110])
        t5.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID",       (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ALIGN",      (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t5)
    else:
        story.append(Paragraph("Sensitivity table not available.", styles["Body"]))
    story.append(Spacer(1, 12))

    # ── 7. Conclusion ─────────────────────────────────────────────────────────
    story.append(Paragraph("7. Conclusion & Recommendation", styles["H2x"]))
    conclusion = (
        f"Based on the RADCF framework, the actuarially fair plan for this borrower is: "
        f"Deposit {ksh(radcf['deposit_amount'])}, monthly installment "
        f"{ksh(radcf['fair_monthly_installment'])}, fair total "
        f"{ksh(radcf['fair_total_paid_if_no_default'])}. "
        f"The estimated probability of default is {pd_value*100:.1f}%, "
        f"reflecting the borrower's income of {ksh(inputs['income_ksh'])}/month, "
        f"{'stable' if inputs['stable_job']==1 else 'unstable'} income, "
        f"and age of {inputs['age']:.0f} years. "
    )
    if market.get("provided"):
        conclusion += (
            f"The market deal was assessed as {market.get('tag')}. "
            f"Overpricing was {market.get('over_pct')*100:.2f}% relative to RADCF fair value. "
        )
    conclusion += "Sensitivity results indicate which parameters most influence fair pricing."
    story.append(Paragraph(conclusion, styles["Body"]))
    story.append(Spacer(1, 12))

    # ── 8. Assumptions ────────────────────────────────────────────────────────
    story.append(Paragraph("8. Assumptions & Limitations", styles["H2x"]))
    assumptions = [
        "PD model uses 3 predictors: income, job stability, and age. "
        "Fitted on Findex 2024 Kenya microdata (n=934 credit buyers). AUC=0.7409.",
        "Job stability (fin22e) and age sourced from World Bank Global Findex 2024 Kenya.",
        "Income distribution parameters (μ=5.1967, σ=0.7381) from Kenya Household "
        "Income Dataset (n=1,000).",
        "PD is constant across the repayment term (simplifying assumption).",
        "No recovery after default assumed (LGD ≈ 100%).",
        "Market markup of 89% sourced from Citizen Digital (2025) and "
        "Business Daily Africa (2024).",
    ]
    for a in assumptions:
        story.append(Paragraph(f"• {a}", styles["Body"]))

    doc.build(story)
    return buf.getvalue()


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(
    page_title="RADCF Fair Pricing Engine",
    layout="wide",
    page_icon="📱"
)

st.title("Actuarial Evaluation of Consumer Overpricing in Kenya's Hire-Purchase Market")
st.markdown("### Risk-Adjusted Discounted Cash Flow (RADCF) Pricing Engine")
st.caption(
    "Model: 3-variable logistic regression | "
    "Data: Findex 2024 Kenya + Kenya Income Dataset | "
    "AUC = 0.7409"
)

with st.expander("Model Overview", expanded=True):
    st.markdown("""
This tool estimates an **actuarially fair** hire-purchase repayment plan using a
**Risk-Adjusted Discounted Cash Flow (RADCF)** approach.

**Workflow**
1. Estimate **Probability of Default (PD)** via a 3-variable logistic model
   *(income, job stability, age)*.
2. Adjust repayments using *(1 − PD)*.
3. Discount expected cashflows at monthly rate *r*.
4. Include deposit and administrative cost assumptions.

Use the **Market Deal Comparison** section to estimate overpricing and generate a PDF report.
    """)

with st.expander("Mathematical Framework", expanded=False):
    st.latex(r"""
        PD = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \ln(\text{Income}/100)
             + \beta_2 \cdot \text{StableJob}
             + \beta_3 \cdot \text{AgeScaled})}}
    """)
    st.latex(r"\text{AgeScaled} = \frac{\text{Age} - 31.4}{12.4}")
    st.latex(r"AF = \frac{1-(1+r)^{-n}}{r}")
    st.latex(r"CF_{revised} = OP + \text{AdminCost} - \text{Deposit}")
    st.latex(r"M = \frac{CF_{revised}}{(1-PD)\cdot AF}")

    st.markdown("**Model Coefficients (Findex 2024 Kenya)**")
    coef_df = pd.DataFrame({
        "Parameter": ["β₀ (intercept)", "β₁ (log income)", "β₂ (stable job)", "β₃ (age scaled)"],
        "Value":     [DEFAULT_BETA0, DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_BETA3],
        "Direction": ["Baseline", "↑ income → ↓ PD", "Stable → ↓ PD", "↑ age → ↑ PD"],
    })
    st.dataframe(coef_df, use_container_width=False, hide_index=True)
    st.caption("AUC = 0.7409 | Data: World Bank Global Findex 2024 Kenya (n=934 credit buyers)")

tabs = st.tabs(["Manual Calculator", "Paste Contract Text (Auto-fill)"])


# ============================================================
# TAB 1: Manual Calculator
# ============================================================
with tabs[0]:
    st.subheader("Manual RADCF Calculator")

    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown("**Contract Details**")
        cash_price = st.number_input(
            "Cash price (KSh)", min_value=0.0, value=25000.0, step=500.0, key="cp1")
        n_months = st.number_input(
            "Repayment term (months)", min_value=1, value=12, step=1, key="n1")
        income_ksh = st.number_input(
            "Borrower monthly income (KSh)", min_value=0.0,
            value=30000.0, step=1000.0, key="inc1")

    with colB:
        st.markdown("**Pricing Parameters**")
        deposit_pct = st.number_input(
            "Deposit (%)", min_value=0.0, max_value=100.0,
            value=30.0, step=1.0, key="dp1")
        admin_cost_pct = st.number_input(
            "Administrative cost (%)", min_value=0.0, max_value=30.0,
            value=5.0, step=0.5, key="ad1")
        r_monthly = st.number_input(
            "Monthly discount rate r  (CBK 13% → 0.0108)",
            min_value=0.0, value=round(0.13/12, 4),
            step=0.001, format="%.4f", key="r1")

    with colC:
        st.markdown("**Borrower Risk Profile**")

        borrower_age = st.number_input(
            "Borrower age (years)", min_value=18, max_value=90,
            value=31, step=1, key="age1",
            help="Older borrowers have higher observed default rates in Kenya data.")
        st.caption(f"Risk signal: {age_risk_label(borrower_age)}")

        stable_job = st.radio(
            "Income stability",
            options=[0, 1],
            format_func=lambda x: "✅ Stable / Regular income" if x == 1
                                  else "⚠️ Unstable / Irregular income",
            index=0, key="stab1",
            help="Stable job = salary employment or regular business income. "
                 f"{MEAN_STABLE*100:.0f}% of Findex Kenya credit buyers have stable income.")

        st.divider()
        st.markdown("**Advanced: PD model coefficients**")
        with st.expander("Override coefficients (optional)", expanded=False):
            beta0 = st.number_input("β₀", value=DEFAULT_BETA0, step=0.01, format="%.4f", key="b01")
            beta1 = st.number_input("β₁ (income)", value=DEFAULT_BETA1, step=0.01, format="%.4f", key="b11")
            beta2 = st.number_input("β₂ (stable job)", value=DEFAULT_BETA2, step=0.01, format="%.4f", key="b21")
            beta3 = st.number_input("β₃ (age scaled)", value=DEFAULT_BETA3, step=0.01, format="%.4f", key="b31")
        # If not expanded, use defaults
        if "b01" not in st.session_state:
            beta0, beta1, beta2, beta3 = DEFAULT_BETA0, DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_BETA3

    # ── Compute PD and RADCF ───────────────────────────────────────────────────
    pd_val = logistic_pd(income_ksh, stable_job=stable_job, age=borrower_age,
                          beta0=beta0, beta1=beta1, beta2=beta2, beta3=beta3)
    res    = fair_installment(cash_price, deposit_pct, admin_cost_pct,
                               int(n_months), float(r_monthly), pd_val)

    st.divider()

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("### Fair Pricing Outputs")
        lvl, expl = pd_bucket(pd_val)

        # PD breakdown
        age_sc_display = (borrower_age - MEAN_AGE) / SD_AGE
        st.metric("Estimated Probability of Default", f"{pd_val:.3f}  ({pd_val*100:.1f}%)")
        st.caption(f"Risk Level: **{lvl}** — {expl}")

        with st.expander("PD component breakdown", expanded=False):
            inc_component  = beta1 * math.log(max(income_ksh,1) / 100)
            job_component  = beta2 * float(stable_job)
            age_component  = beta3 * age_sc_display
            st.write(f"β₀ (baseline)         : {beta0:+.4f}")
            st.write(f"β₁ × ln(income/100)   : {inc_component:+.4f}  ({'↓' if inc_component<0 else '↑'} risk)")
            st.write(f"β₂ × stable_job       : {job_component:+.4f}  ({'↓' if job_component<0 else '↑'} risk)")
            st.write(f"β₃ × age_scaled       : {age_component:+.4f}  ({'↓' if age_component<0 else '↑'} risk)")
            lin_comb = beta0 + inc_component + job_component + age_component
            st.write(f"Linear combination (z): {lin_comb:+.4f}")
            st.write(f"PD = 1/(1+e^(-z))     : {pd_val:.4f}")

        st.metric("Fair monthly installment (KSh)",
                  f"{res['fair_monthly_installment']:,.2f}")
        st.metric("Deposit amount (KSh)",
                  f"{res['deposit_amount']:,.2f}")
        st.metric("Admin cost amount (KSh)",
                  f"{res['admin_cost_amount']:,.2f}")
        st.metric("Fair total paid (KSh)",
                  f"{res['fair_total_paid_if_no_default']:,.2f}")
        st.metric("RADCF PV (expected PV)",
                  f"{res['radcf_present_value']:,.2f}")

        # PD comparison: how does stability change the PD?
        pd_unstable = logistic_pd(income_ksh, stable_job=0,
                                   age=borrower_age,
                                   beta0=beta0, beta1=beta1,
                                   beta2=beta2, beta3=beta3)
        pd_stable   = logistic_pd(income_ksh, stable_job=1,
                                   age=borrower_age,
                                   beta0=beta0, beta1=beta1,
                                   beta2=beta2, beta3=beta3)
        st.info(
            f"📊 **Stability impact for this income & age:**  "
            f"Unstable → PD = {pd_unstable*100:.1f}%  |  "
            f"Stable → PD = {pd_stable*100:.1f}%  "
            f"(difference: {abs(pd_stable-pd_unstable)*100:.1f} pp)"
        )

    # ── Market comparison ──────────────────────────────────────────────────────
    market_info = {"provided": False}

    with right:
        st.markdown("### Market Deal Comparison (optional)")
        market_monthly = st.number_input(
            "Market monthly installment (KSh)", min_value=0.0,
            value=0.0, step=100.0, key="m_m1")
        market_total = st.number_input(
            "Market total repayment (KSh)", min_value=0.0,
            value=0.0, step=500.0, key="m_t1")

        fair_total   = res["fair_total_paid_if_no_default"]
        over_amt     = float("nan")
        over_pct     = float("nan")
        implied_apr  = float("nan")
        mkt_total_used = 0.0

        if market_total > 0:
            mkt_total_used = float(market_total)
            over_amt = mkt_total_used - fair_total
            over_pct = (over_amt / fair_total) if fair_total > 0 else float("nan")
        elif market_monthly > 0:
            mkt_total_used = res["deposit_amount"] + float(market_monthly) * int(n_months)
            over_amt       = mkt_total_used - fair_total
            over_pct       = (over_amt / fair_total) if fair_total > 0 else float("nan")
            principal      = cash_price - res["deposit_amount"]
            im             = implied_monthly_rate_from_payment(
                                 principal, float(market_monthly), int(n_months))
            implied_apr    = effective_apr_from_monthly(im)

        if np.isfinite(over_pct):
            tag, tag_explain = fairness_tag(over_pct)
            fairness_score   = max(0.0, min(100.0, 100.0 - over_pct * 100.0))

            st.metric("Overpricing amount (KSh)", f"{over_amt:,.2f}")
            st.metric("Overpricing (%)",          f"{over_pct*100:.2f}%")
            st.metric("Fairness Score (0–100)",   f"{fairness_score:.1f}")
            st.caption(f"Assessment: **{tag}** — {tag_explain}")

            if np.isfinite(implied_apr):
                st.metric("Implied APR (effective)", f"{implied_apr*100:.1f}%")

            market_info = {
                "provided":       True,
                "market_monthly": float(market_monthly) if market_monthly > 0 else float("nan"),
                "market_total":   float(mkt_total_used),
                "over_amt":       float(over_amt),
                "over_pct":       float(over_pct),
                "fairness_score": float(fairness_score),
                "tag":            tag,
                "tag_explain":    tag_explain,
                "implied_apr":    float(implied_apr),
            }

    # ── Sensitivity analysis ───────────────────────────────────────────────────
    st.divider()
    st.markdown("### Sensitivity Analysis (Stress Test)")
    st.caption(
        "Base scenario uses your inputs above. "
        "PD ±10% shows how small changes in default probability affect pricing.")

    pd_low  = max(0.0, pd_val * 0.9)
    pd_high = min(1.0, pd_val * 1.1)

    # Additional scenarios: stable vs unstable job, young vs old
    pd_young = logistic_pd(income_ksh, stable_job=stable_job,
                            age=22, beta0=beta0, beta1=beta1,
                            beta2=beta2, beta3=beta3)
    pd_older = logistic_pd(income_ksh, stable_job=stable_job,
                            age=50, beta0=beta0, beta1=beta1,
                            beta2=beta2, beta3=beta3)

    scenarios = [
        ("Base",              pd_val,    admin_cost_pct, r_monthly),
        ("PD −10%",           pd_low,    admin_cost_pct, r_monthly),
        ("PD +10%",           pd_high,   admin_cost_pct, r_monthly),
        ("Stable income",     pd_stable,   admin_cost_pct, r_monthly),
        ("Unstable income",   pd_unstable, admin_cost_pct, r_monthly),
        ("Age 22 (young)",    pd_young,  admin_cost_pct, r_monthly),
        ("Age 50 (senior)",   pd_older,  admin_cost_pct, r_monthly),
        ("Admin 8%",          pd_val,    8.0,            r_monthly),
        ("Rate +2pp",         pd_val,    admin_cost_pct, r_monthly + 0.02),
        ("Rate −1pp",         pd_val,    admin_cost_pct, max(0.0, r_monthly - 0.01)),
    ]

    rows = []
    for name, pd_s, admin_s, r_s in scenarios:
        rr = fair_installment(cash_price, deposit_pct, admin_s,
                               int(n_months), float(r_s), float(pd_s))
        rows.append({
            "Scenario":          name,
            "PD":                round(float(pd_s), 3),
            "Admin%":            float(admin_s),
            "r":                 round(float(r_s), 4),
            "Fair Monthly (KSh)": round(rr["fair_monthly_installment"], 2),
            "Fair Total (KSh)":  round(rr["fair_total_paid_if_no_default"], 2),
        })

    sens_df = pd.DataFrame(rows)
    st.dataframe(sens_df, use_container_width=True)

    # ── Tornado chart ──────────────────────────────────────────────────────────
    st.markdown("### Tornado Chart (Sensitivity Impact vs Base)")

    metric = st.radio(
        "Show sensitivity for:",
        options=["Fair Total (KSh)", "Fair Monthly (KSh)"],
        horizontal=True, index=0, key="tornado_metric")

    base_row = sens_df[sens_df["Scenario"] == "Base"]
    if not base_row.empty:
        base_val = float(base_row.iloc[0][metric])
        plot_df  = sens_df[sens_df["Scenario"] != "Base"].copy()
        plot_df["Impact"]    = plot_df[metric].astype(float) - base_val
        plot_df["AbsImpact"] = plot_df["Impact"].abs()
        plot_df = plot_df.sort_values("AbsImpact", ascending=True)

        dec = plot_df[plot_df["Impact"] < 0]
        inc = plot_df[plot_df["Impact"] > 0]

        fig = go.Figure()
        fig.add_trace(go.Bar(y=dec["Scenario"], x=dec["Impact"],
                             orientation="h", name="Decrease vs Base",
                             marker_color="#2E75B6"))
        fig.add_trace(go.Bar(y=inc["Scenario"], x=inc["Impact"],
                             orientation="h", name="Increase vs Base",
                             marker_color="#CC3300"))
        fig.add_vline(x=0, line_width=1)
        fig.update_layout(
            barmode="relative", height=450,
            xaxis_title=f"Impact on {metric} (KSh)",
            yaxis_title="Scenario",
            title=f"Sensitivity Tornado Chart (Base = KSh {base_val:,.2f})",
            margin=dict(l=20, r=20, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        most_sensitive = plot_df.iloc[-1]
        st.caption(
            f"Most sensitive factor: **{most_sensitive['Scenario']}** "
            f"→ KSh {most_sensitive['Impact']:,.2f} change from base.")

    # ── PDF download ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Download Report")

    pdf_bytes = build_pdf_report(
        report_title="Actuarial Evaluation of Consumer Overpricing in Kenya's "
                     "Hire-Purchase Market (RADCF Engine)",
        generated_dt=datetime.now(),
        inputs={
            "cash_price":     cash_price,
            "n_months":       int(n_months),
            "deposit_pct":    float(deposit_pct),
            "admin_cost_pct": float(admin_cost_pct),
            "r_monthly":      float(r_monthly),
            "income_ksh":     float(income_ksh),
            "stable_job":     int(stable_job),
            "age":            float(borrower_age),
        },
        pd_params={"beta0": beta0, "beta1": beta1,
                   "beta2": beta2, "beta3": beta3},
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
        key="dl_pdf_manual",
    )


# ============================================================
# TAB 2: Paste Contract Text (Auto-fill)
# ============================================================
with tabs[1]:
    st.subheader("Paste Contract / Offer Text → Auto-fill")
    st.write("Paste a hire-purchase offer (WhatsApp message, advert text). "
             "We'll extract fields and compute the RADCF fair price.")

    sample = ("Cash price: KSh 25000. Deposit 30%. "
              "Pay KES 2500 per month for 12 months. Admin fee 5%.")
    txt = st.text_area("Paste text here", value=sample, height=140, key="txt_offer")

    extracted = extract_deal_fields(txt)
    st.markdown("### Extracted (best guesses)")
    st.json(extracted)

    st.markdown("### Auto-filled calculator")
    colX, colY = st.columns(2)

    with colX:
        cash_price2 = st.number_input(
            "Cash price (KSh)", min_value=0.0,
            value=float(extracted["cash_price"] or 25000.0),
            step=500.0, key="cp2")
        n_months2 = st.number_input(
            "Repayment term (months)", min_value=1,
            value=int(extracted["term_months"] or 12),
            step=1, key="n2")
        income2 = st.number_input(
            "Borrower monthly income (KSh)", min_value=0.0,
            value=30000.0, step=1000.0, key="inc2")

    with colY:
        dep_pct_guess = extracted["deposit_pct"]
        if dep_pct_guess is None and extracted["deposit_amount"] and cash_price2 > 0:
            dep_pct_guess = 100.0 * float(extracted["deposit_amount"]) / float(cash_price2)
        deposit_pct2 = st.number_input(
            "Deposit (%)", min_value=0.0, max_value=100.0,
            value=float(dep_pct_guess or 30.0), step=1.0, key="dp2")

        admin_pct_guess = extracted["admin_pct"]
        if admin_pct_guess is None and extracted["admin_amount"] and cash_price2 > 0:
            admin_pct_guess = 100.0 * float(extracted["admin_amount"]) / float(cash_price2)
        admin2 = st.number_input(
            "Administrative cost (%)", min_value=0.0, max_value=30.0,
            value=float(admin_pct_guess or 5.0), step=0.5, key="ad2")

        r2 = st.number_input(
            "Monthly discount rate r", min_value=0.0,
            value=round(0.13/12, 4), step=0.001, format="%.4f", key="r2")

    st.markdown("**Borrower Risk Profile**")
    colP1, colP2, colP3 = st.columns(3)
    with colP1:
        age2 = st.number_input("Age (years)", min_value=18, max_value=90,
                                value=31, step=1, key="age2")
    with colP2:
        stable2 = st.radio(
            "Income stability",
            options=[0, 1],
            format_func=lambda x: "✅ Stable" if x==1 else "⚠️ Unstable",
            index=0, key="stab2")
    with colP3:
        st.markdown("&nbsp;")
        st.caption(f"{age_risk_label(age2)}")

    pd2  = logistic_pd(income2, stable_job=stable2, age=age2)
    res2 = fair_installment(cash_price2, deposit_pct2, admin2,
                             int(n_months2), float(r2), pd2)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated PD", f"{pd2:.3f}  ({pd2*100:.1f}%)")
    c2.metric("Fair monthly installment (KSh)",
              f"{res2['fair_monthly_installment']:,.2f}")
    c3.metric("Fair total paid (KSh)",
              f"{res2['fair_total_paid_if_no_default']:,.2f}")

    st.divider()
    st.markdown("### Download Report")

    pd_low2  = max(0.0, pd2 * 0.9)
    pd_high2 = min(1.0, pd2 * 1.1)
    pd_s2    = logistic_pd(income2, stable_job=1, age=age2)
    pd_u2    = logistic_pd(income2, stable_job=0, age=age2)

    scenarios2 = [
        ("Base",            pd2,    admin2, r2),
        ("PD −10%",         pd_low2, admin2, r2),
        ("PD +10%",         pd_high2, admin2, r2),
        ("Stable income",   pd_s2,  admin2, r2),
        ("Unstable income", pd_u2,  admin2, r2),
        ("Admin 8%",        pd2,    8.0,   r2),
        ("Rate +2pp",       pd2,    admin2, r2 + 0.02),
        ("Rate −1pp",       pd2,    admin2, max(0.0, r2 - 0.01)),
    ]
    rows2 = []
    for name, pd_s, admin_s, r_s in scenarios2:
        rr = fair_installment(cash_price2, deposit_pct2, admin_s,
                               int(n_months2), float(r_s), float(pd_s))
        rows2.append({
            "Scenario":          name,
            "PD":                round(float(pd_s), 3),
            "Admin%":            float(admin_s),
            "r":                 round(float(r_s), 4),
            "Fair Monthly (KSh)": round(rr["fair_monthly_installment"], 2),
            "Fair Total (KSh)":  round(rr["fair_total_paid_if_no_default"], 2),
        })
    sens_df2 = pd.DataFrame(rows2)

    pdf_bytes2 = build_pdf_report(
        report_title="Actuarial Evaluation of Consumer Overpricing in Kenya's "
                     "Hire-Purchase Market (RADCF Engine)",
        generated_dt=datetime.now(),
        inputs={
            "cash_price":     float(cash_price2),
            "n_months":       int(n_months2),
            "deposit_pct":    float(deposit_pct2),
            "admin_cost_pct": float(admin2),
            "r_monthly":      float(r2),
            "income_ksh":     float(income2),
            "stable_job":     int(stable2),
            "age":            float(age2),
        },
        pd_params={"beta0": DEFAULT_BETA0, "beta1": DEFAULT_BETA1,
                   "beta2": DEFAULT_BETA2, "beta3": DEFAULT_BETA3},
        pd_value=float(pd2),
        radcf=res2,
        market={"provided": False},
        sensitivity_df=sens_df2,
    )

    st.download_button(
        label="Download RADCF Report (PDF)",
        data=pdf_bytes2,
        file_name=f"RADCF_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        key="dl_pdf_autofill",
    )
    st.caption("Extraction is basic regex. Next step: upload images/PDFs + OCR.")
