"""Streamlit dashboard: upload leads, see which look fraudulent and why.

Run with:  streamlit run src/adshield/dashboard/app.py
Needs the scoring API running (default http://localhost:8000, or set ADSHIELD_API_URL).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from adshield.dashboard.client import ApiError, ScoringClient
from adshield.dashboard.logic import (
    UploadError,
    describe_feature,
    parse_leads,
    read_upload,
    results_table,
    score_upload,
    tradeoff_curve,
)
from adshield.models.rule_engine import RuleEngine

SAMPLE_PATH = Path(__file__).with_name("sample_clicks.csv")
DEFAULT_API = os.environ.get("ADSHIELD_API_URL", "http://localhost:8000")

st.set_page_config(page_title="AdShield - lead quality review", layout="wide")


@st.cache_resource
def rule_info() -> tuple[str, dict[str, tuple[str, float]]]:
    engine = RuleEngine.from_yaml()
    return engine.config_hash, {r.name: (r.description, r.points) for r in engine.config.rules}


# ------------------------------------------------------------------ sidebar: connection + data

st.title("AdShield: which of these leads look fake?")
st.caption(
    "Upload a list of ad clicks. Each one gets a risk score from 0 (looks normal) to 1 (looks suspicious), "
    "and a plain-English reason. You choose how strict to be with the slider."
)

with st.sidebar:
    st.header("1. Connect")
    api_url = st.text_input("Scoring service address", DEFAULT_API)
    client = ScoringClient(api_url)
    try:
        health = client.health()
    except ApiError as exc:
        st.error(str(exc))
        st.stop()
    if health["status"] == "ok":
        st.success(f"Connected. Model version: {health['model_version']}")
    else:
        st.warning(
            "No trained model is loaded, so scores come from the simple rule checks only. "
            "Treat results with extra caution."
        )

    st.header("2. Choose a file")
    uploaded = st.file_uploader("CSV of clicks", type="csv")
    use_sample = st.checkbox("Or use the built-in example file (made-up data)")
    st.caption(
        "Needed columns: ip, app, device, os, channel, click_time. Optional: lead_id (otherwise rows are "
        "numbered), is_attributed (1 = the click later converted; enables the cost/benefit view)."
    )

if uploaded is not None:
    file_bytes, file_name = uploaded.getvalue(), uploaded.name
elif use_sample:
    file_bytes, file_name = SAMPLE_PATH.read_bytes(), "example file (made-up data)"
else:
    st.info("Pick a CSV in the sidebar to begin.")
    st.stop()

# ------------------------------------------------------------------ parse + score (cached per file)

cache_key = (hashlib.sha256(file_bytes).hexdigest(), api_url, health["model_version"])
rescore = st.sidebar.button("Score again")
if rescore or st.session_state.get("cache_key") != cache_key:
    try:
        parsed, errors = parse_leads(read_upload(file_bytes))
    except UploadError as exc:
        st.error(str(exc))
        st.stop()
    if parsed is None:
        st.error(
            f"{len(errors)} problem(s) found, so nothing was scored. Fix these rows and upload again "
            "(row 1 is the header):"
        )
        st.code("\n".join(str(e) for e in errors[:20]))
        st.stop()
    with st.spinner(f"Scoring {len(parsed.records):,} leads..."):
        try:
            st.session_state["results"] = score_upload(parsed, client)
        except ApiError as exc:
            st.error(str(exc))
            st.stop()
    st.session_state["cache_key"] = cache_key

results: pd.DataFrame = st.session_state["results"]
has_outcomes = "converted" in results
rule_hash, rule_descriptions = rule_info()

# ------------------------------------------------------------------ threshold + summary

default_threshold = round(float(health["model_threshold"] or 0.5), 2)
st.subheader(f"Results for {file_name}")
threshold = st.slider(
    "How suspicious must a lead look before we flag it?  (lower = flag more, higher = flag fewer)",
    0.0,
    1.0,
    default_threshold,
    0.01,
    key=f"threshold-{health['model_version']}",
    help="Starts at the setting the model was tuned for. Move it to see the trade-off below.",
)
flagged_mask = results["risk_score"] >= threshold
n, n_flagged = len(results), int(flagged_mask.sum())

c1, c2, c3 = st.columns(3)
c1.metric("Leads scored", f"{n:,}")
c2.metric("Flagged", f"{n_flagged:,}")
c3.metric("Share flagged", f"{100 * n_flagged / n:.1f}%")
if (results["source"] != "model").any():
    st.warning("Some leads were scored by the simple rule checks because the model could not score them.")

hist = (
    alt.Chart(results)
    .mark_bar()
    .encode(
        alt.X("risk_score:Q", bin=alt.Bin(step=0.05, extent=[0, 1]), title="Risk score"),
        alt.Y("count()", title="Number of leads"),
    )
)
line = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color="red", strokeWidth=2).encode(x="t:Q")
st.altair_chart(
    (hist + line).properties(height=220, title="How risk scores are spread (red line = your threshold)"),
    use_container_width=True,
)

# ------------------------------------------------------------------ threshold tradeoff

st.subheader("What changes when you move the threshold?")
curve = tradeoff_curve(results, np.round(np.arange(0, 1.0001, 0.01), 2))
now = curve.iloc[int(round(threshold * 100))]
if has_outcomes:
    a, b = st.columns(2)
    a.metric(
        "Non-converting clicks we would flag",
        f"{now['pct_nonconverting_flagged']:.1f}%",
        help="Of all clicks that never converted, the share flagged. Higher = more waste caught.",
    )
    b.metric(
        "Real conversions we would wrongly flag",
        f"{now['pct_conversions_flagged']:.1f}%",
        f"{int(now['conversions_flagged']):,} lead(s)",
        delta_color="inverse",
        help="Of all clicks that did convert, the share flagged. Lower = fewer good leads blocked.",
    )
    long = curve.melt(
        "threshold",
        ["pct_nonconverting_flagged", "pct_conversions_flagged"],
        var_name="measure",
        value_name="percent",
    )
    long["measure"] = long["measure"].map(
        {
            "pct_nonconverting_flagged": "Non-converting clicks flagged (waste caught)",
            "pct_conversions_flagged": "Real conversions flagged (good leads blocked)",
        }
    )
    chart = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("threshold:Q", title="Threshold"),
            y=alt.Y("percent:Q", title="% of that group flagged"),
            color=alt.Color("measure:N", title=None, legend=alt.Legend(orient="bottom")),
        )
    )
    st.altair_chart((chart + line).properties(height=260), use_container_width=True)
    st.caption(
        "This is the precision/recall trade-off made concrete: a lower threshold catches more wasted "
        "clicks but also blocks more real conversions. 'Never converted' is a stand-in for fraud: "
        "many honest clicks do not convert either."
    )
else:
    chart = (
        alt.Chart(curve)
        .mark_line()
        .encode(x=alt.X("threshold:Q", title="Threshold"), y=alt.Y("flagged:Q", title="Leads flagged"))
    )
    st.altair_chart((chart + line).properties(height=240), use_container_width=True)
    st.caption(
        "Add an is_attributed column (1 = converted) to your file to also see how many real "
        "conversions each threshold would wrongly flag."
    )

# ------------------------------------------------------------------ results table + detail

st.subheader("All leads, riskiest first")
st.caption("Click a row to see exactly why it scored the way it did.")
table = results_table(results, threshold)
selection = st.dataframe(
    table,
    hide_index=True,
    use_container_width=True,
    height=320,
    on_select="rerun",
    selection_mode="single-row",
    key="lead-table",
)

picked = selection.selection.rows
if picked:
    pos = table.index[picked[0]]
    row = results.loc[pos]
    is_flagged = row["risk_score"] >= threshold
    st.subheader(f"Lead {row['lead_id']}: " + ("FLAGGED" if is_flagged else "not flagged"))
    st.write(f"Risk score **{row['risk_score']:.3f}** against your threshold of **{threshold:.2f}**.")
    if row["source"] != "model":
        st.warning("This lead was scored by the simple rule checks only; there is no model breakdown.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Rule checks** (simple, fixed rules)")
        st.write(f"Rule score: **{row['rule_score']:.0f} / 100**")
        if row["fired_rules"]:
            same_config = rule_hash == health["rules_config_hash"]
            for name in row["fired_rules"]:
                desc, pts = rule_descriptions.get(name, (name.replace("_", " "), None))
                suffix = f" (+{pts:g} points)" if pts is not None and same_config else ""
                st.write(f"- {desc if same_config else name.replace('_', ' ')}{suffix}")
        else:
            st.write("No rule triggered.")
    with right:
        st.markdown("**The model's reasons** (▲ pushes toward suspicious, ▼ pushes toward normal)")
        contribs = sorted(row["contributions"], key=lambda c: abs(c["contribution"]), reverse=True)
        if not contribs:
            st.write("No model reasons for this lead.")
        for c in contribs[:5]:
            st.write(
                f"- {'▲' if c['contribution'] > 0 else '▼'} {describe_feature(c['feature'], c['value'])}"
            )

    if contribs:
        detail = pd.DataFrame(
            {
                "What was measured": [describe_feature(c["feature"], c["value"]) for c in contribs],
                "Push": [c["contribution"] for c in contribs],
                "Direction": [
                    "toward suspicious" if c["contribution"] > 0 else "toward normal" for c in contribs
                ],
            }
        )
        bars = (
            alt.Chart(detail)
            .mark_bar()
            .encode(
                x=alt.X("Push:Q", title="Strength of push (right = more suspicious)"),
                y=alt.Y("What was measured:N", sort=None, title=None, axis=alt.Axis(labelLimit=420)),
                color=alt.Color(
                    "Direction:N",
                    scale=alt.Scale(
                        domain=["toward suspicious", "toward normal"], range=["#d62728", "#1f77b4"]
                    ),
                    legend=None,
                ),
            )
        )
        st.altair_chart(bars.properties(height=28 * len(detail) + 40), use_container_width=True)
        with st.expander("Full feature table"):
            st.dataframe(
                pd.DataFrame(row["contributions"]).rename(
                    columns={"feature": "Feature", "value": "Value", "contribution": "Push (log-odds)"}
                ),
                hide_index=True,
                use_container_width=True,
            )

st.download_button(
    "Download results (CSV)", table.to_csv(index=False).encode(), "adshield_results.csv", "text/csv"
)
