import re
import math
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Core actuarial/pricing functions
# -----------------------------
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
    CF_revised = OP + AdminCosts(OP) - Deposit(OP)
    M = CF_revised / ((1 - PD) * AF)
    RADCF PV ~= Deposit + M * AF * (1-PD)
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

    fair_total = deposit + (m * n_months)            # total if installments fully paid
    radcf_pv = deposit + (m * af * repay_prob)       # PV of expected installments + deposit

    return {
        "deposit_amount": deposit,
        "admin_cost_amount": admin_cost,
        "cf_revised": cf_revised,
        "annuity_factor": af,
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

    lo, hi = 0.0, 3.0  # 0% to 300% monthly (very wide)
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


# -----------------------------
# Simple text extraction (regex)
# -----------------------------
def extract_deal_fields(text: str) -> dict:
    """
    Lightweight extraction from pasted text.
    Attempts to find:
      cash price, deposit (%/amount), term (months), monthly installment, admin fee (%/amount)
    """
    t = (text or "").lower()
    t = t.replace(",", " ")

    money = r"(?:ksh|kes)\s*([0-9]{3,})"
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

    # Deposit (% then amount)
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


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hire-Purchase Fair Pricing (RADCF)", layout="wide")
st.title("Hire-Purchase Fair Pricing + Transparency Tool (RADCF Core)")

st.markdown(
    "### Actuarial Risk-Adjusted Pricing Engine (RADCF Model)"
)

tabs = st.tabs(["1) Manual Calculator", "2) Paste Contract Text (Auto-fill)"])

# -----------------------------
# TAB 1: Manual Calculator
# -----------------------------
with tabs[0]:
    st.subheader("Manual RADCF Calculator")

    colA, colB, colC = st.columns(3)

    with colA:
        cash_price = st.number_input("Cash price (KSh)", min_value=0.0, value=25000.0, step=500.0)
        n_months = st.number_input("Repayment term (months)", min_value=1, value=12, step=1)
        income_ksh = st.number_input("Borrower monthly income (KSh)", min_value=0.0, value=30000.0, step=1000.0)

    with colB:
        deposit_pct = st.number_input("Deposit (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        admin_cost_pct = st.number_input("Admin cost (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
        r_monthly = st.number_input(
            "Monthly discount rate r (e.g. 0.02 = 2%)",
            min_value=0.0,
            value=0.02,
            step=0.005,
            format="%.3f",
        )

    with colC:
        st.markdown("**PD model parameters (for now)**")
        beta0 = st.number_input("β0", value=2.5, step=0.1, format="%.2f")
        beta1 = st.number_input("β1", value=-0.4, step=0.05, format="%.2f")

    pd_val = logistic_pd(income_ksh, beta0, beta1)
    results = fair_installment(
        cash_price, deposit_pct, admin_cost_pct, int(n_months), float(r_monthly), pd_val
    )

    st.divider()

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown("### Fair Pricing Outputs")
        st.metric("Estimated PD", f"{pd_val:.3f}")
        st.metric("Fair monthly installment (KSh)", f"{results['fair_monthly_installment']:.2f}")
        st.metric("Deposit amount (KSh)", f"{results['deposit_amount']:.2f}")
        st.metric("Fair total paid (deposit + installments) (KSh)", f"{results['fair_total_paid_if_no_default']:.2f}")
        st.caption("Fair total assumes full payment. RADCF PV is expected PV after default-risk adjustment.")

    with col2:
        st.markdown("### Market Deal Comparison (optional)")
        market_monthly = st.number_input("Market monthly installment (KSh) [optional]", min_value=0.0, value=0.0, step=100.0)
        market_total = st.number_input("Market total repayment (KSh) [optional]", min_value=0.0, value=0.0, step=500.0)

        fair_total = results["fair_total_paid_if_no_default"]

        if market_total > 0:
            over_amt = market_total - fair_total
            over_pct = (over_amt / fair_total) if fair_total > 0 else float("nan")
            st.metric("Overpricing amount (KSh)", f"{over_amt:.2f}")
            st.metric("Overpricing (%)", f"{over_pct*100:.2f}%")

        elif market_monthly > 0:
            market_total_est = results["deposit_amount"] + market_monthly * int(n_months)
            over_amt = market_total_est - fair_total
            over_pct = (over_amt / fair_total) if fair_total > 0 else float("nan")

            st.metric("Overpricing amount (KSh)", f"{over_amt:.2f}")
            st.metric("Overpricing (%)", f"{over_pct*100:.2f}%")

            # Implied APR for financed part (principal = cash - deposit)
            principal_financed = cash_price - results["deposit_amount"]
            im = implied_monthly_rate_from_payment(principal_financed, market_monthly, int(n_months))
            apr = effective_apr_from_monthly(im)
            if np.isfinite(apr):
                st.metric("Implied APR (effective)", f"{apr*100:.1f}%")

    st.divider()
    st.markdown("### Sensitivity Band (quick stress test)")

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
            "PD": round(pd_s, 3),
            "Admin%": float(admin_s),
            "r": round(float(r_s), 3),
            "Fair Monthly (KSh)": round(rr["fair_monthly_installment"], 2),
            "Fair Total (KSh)": round(rr["fair_total_paid_if_no_default"], 2),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


# -----------------------------
# TAB 2: Paste Contract Text
# -----------------------------
with tabs[1]:
    st.subheader("Paste Contract / Offer Text → Auto-fill")
    st.write("Paste a hire-purchase offer (WhatsApp message, advert text). We'll extract fields and compute RADCF fair price.")

    sample = (
        "Cash price: KSh 25000. Deposit 30%. Pay KES 2500 per month for 12 months. Admin fee 5%."
    )
    txt = st.text_area("Paste text here", value=sample, height=140)

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
            "Term (months)",
            min_value=1,
            value=int(extracted["term_months"] or 12),
            step=1,
            key="n2"
        )
        income2 = st.number_input(
            "Borrower income (KSh)",
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
            "Admin cost (%)",
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

    # If extracted monthly installment exists, compare + implied APR
    if extracted["monthly_installment"] is not None:
        market_m = float(extracted["monthly_installment"])
        market_total = res2["deposit_amount"] + market_m * int(n_months2)

        over_amt = market_total - res2["fair_total_paid_if_no_default"]
        over_pct = (over_amt / res2["fair_total_paid_if_no_default"]) if res2["fair_total_paid_if_no_default"] > 0 else float("nan")

        st.divider()
        st.markdown("### Market comparison (from extracted monthly installment)")
        st.metric("Market monthly installment (KSh)", f"{market_m:.2f}")
        st.metric("Estimated market total paid (KSh)", f"{market_total:.2f}")
        st.metric("Overpricing amount (KSh)", f"{over_amt:.2f}")
        st.metric("Overpricing (%)", f"{over_pct*100:.2f}%")

        principal_financed = cash_price2 - res2["deposit_amount"]
        im = implied_monthly_rate_from_payment(principal_financed, market_m, int(n_months2))
        apr = effective_apr_from_monthly(im)
        if np.isfinite(apr):
            st.metric("Implied APR (effective)", f"{apr*100:.1f}%")

    st.caption("Extraction is basic regex for now. Next step: upload images/PDFs + OCR.")