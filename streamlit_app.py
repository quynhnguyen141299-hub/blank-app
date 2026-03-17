import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_NAME = "CBDC Sentinel: AI Attack & Detection Analytics"
RL_AGENTS = ["Q-Learning", "DQN", "REINFORCE", "A2C"]

st.set_page_config(page_title=APP_NAME, layout="wide")


def safe_json(value):
    if isinstance(value, (dict, list)):
        return value
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    repaired = (
        text.replace("'", '"')
        .replace("None", "null")
        .replace("True", "true")
        .replace("False", "false")
    )
    try:
        return json.loads(repaired)
    except Exception:
        return None


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@st.cache_data
def load_logs(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if "details" in df.columns:
        df["details"] = df["details"].apply(safe_json)
    else:
        df["details"] = None

    if "stride_tags" in df.columns:
        df["stride_tags"] = df["stride_tags"].apply(safe_json).apply(ensure_list)
    else:
        df["stride_tags"] = [[] for _ in range(len(df))]

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    else:
        df["ts"] = pd.NaT

    df["agent_id"] = df["details"].apply(
        lambda d: d.get("agent_id") if isinstance(d, dict) else None
    )
    df["amount"] = df["details"].apply(
        lambda d: d.get("amount") if isinstance(d, dict) else np.nan
    )
    df["label"] = df["details"].apply(
        lambda d: d.get("label") if isinstance(d, dict) else np.nan
    )
    df["ok"] = df["details"].apply(
        lambda d: d.get("ok") if isinstance(d, dict) else None
    )
    df["complexity"] = df["details"].apply(
        lambda d: d.get("complexity") if isinstance(d, dict) else np.nan
    )
    df["wallet_id"] = df["details"].apply(
        lambda d: d.get("wallet_id") if isinstance(d, dict) else None
    )

    df["process"] = df["process"].astype(str) if "process" in df.columns else "unknown"
    df["asap_layer"] = df["asap_layer"].astype(str) if "asap_layer" in df.columns else "unknown"

    df["is_attack_event"] = (
        (df["label"] == 1)
        | (
            df["agent_id"].isin(RL_AGENTS)
            & (
                df["complexity"].notna()
                | df["wallet_id"].notna()
                | df["amount"].notna()
            )
        )
    )

    df["is_risk_check"] = df["ok"].notna() | df["process"].astype(str).str.contains("P5", na=False)

    return df


def metric_delta(current, baseline):
    if baseline in (None, 0) or pd.isna(baseline):
        return "n/a"
    delta = current - baseline
    return f"{delta:+.1%}"


st.title(APP_NAME)
st.caption(
    "Interactive dashboard for exploring RL-driven attack behaviour, STRIDE patterns, and detection signals in your CBDC simulation logs."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload cbdc_logs.csv",
    type=["csv"],
    help="Use the aggregated log file exported from your notebook.",
)

st.sidebar.header("Filters")
selected_agents = st.sidebar.multiselect("RL agents", RL_AGENTS, default=RL_AGENTS)
attack_only = st.sidebar.toggle("Show only attack-related events", value=True)
show_benign = st.sidebar.toggle("Include benign/system events", value=False)
amount_threshold = st.sidebar.slider(
    "Detection threshold (amount-based what-if)",
    min_value=0,
    max_value=10000,
    value=500,
    step=50,
    help="A simple interactive threshold to estimate which payment amounts would be flagged.",
)

if uploaded_file is None:
    st.info(
        "Upload the CSV exported from your notebook to activate the dashboard.\n\n"
        "Expected columns include: ts, process, asap_layer, stride_tags, details."
    )
    st.stop()

logs = load_logs(uploaded_file)
filtered = logs.copy()

if selected_agents:
    agent_mask = filtered["agent_id"].isin(selected_agents)
    if show_benign:
        agent_mask = agent_mask | filtered["agent_id"].isin(["benign", "system"]) | filtered["agent_id"].isna()
    filtered = filtered[agent_mask]

if attack_only:
    filtered = filtered[filtered["is_attack_event"]]

if filtered.empty:
    st.warning("No rows matched the current filter settings.")
    st.stop()

attack_df = filtered[filtered["is_attack_event"]].copy()
rl_df = filtered[filtered["agent_id"].isin(RL_AGENTS)].copy()

col1, col2, col3, col4 = st.columns(4)

n_attack_events = int(attack_df.shape[0])
mean_amount = float(attack_df["amount"].dropna().mean()) if attack_df["amount"].notna().any() else 0.0
risk_checks = attack_df[attack_df["is_risk_check"]]
passed_checks = risk_checks["ok"].eq(True).sum() if not risk_checks.empty else 0
pass_rate = (passed_checks / len(risk_checks)) if len(risk_checks) else np.nan
flagged_rate = attack_df["amount"].ge(amount_threshold).mean() if attack_df["amount"].notna().any() else np.nan

baseline_df = logs[(logs["agent_id"].isin(RL_AGENTS)) & (logs["is_attack_event"])]
baseline_flagged = baseline_df["amount"].ge(amount_threshold).mean() if baseline_df["amount"].notna().any() else np.nan

col1.metric("Attack-related events", f"{n_attack_events:,}")
col2.metric("Average attack amount", f"{mean_amount:,.1f}")
col3.metric("Risk-check pass rate", "n/a" if pd.isna(pass_rate) else f"{pass_rate:.1%}")
col4.metric(
    f"Flagged at threshold >= {amount_threshold}",
    "n/a" if pd.isna(flagged_rate) else f"{flagged_rate:.1%}",
    metric_delta(flagged_rate, baseline_flagged),
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["System Overview", "Agent Behavior", "Detection Lab", "Threat Intelligence"]
)

with tab1:
    left, right = st.columns([1.2, 1])
    with left:
        timeline = (
            attack_df.dropna(subset=["ts"])
            .set_index("ts")
            .resample("1min")
            .size()
            .reset_index(name="events")
        )
        if timeline.empty:
            st.info("No timestamped events available for the timeline.")
        else:
            fig = px.line(
                timeline,
                x="ts",
                y="events",
                markers=True,
                title="Attack Event Timeline",
            )
            fig.update_layout(xaxis_title="Timestamp", yaxis_title="Event count")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        proc_counts = attack_df["process"].value_counts().reset_index()
        proc_counts.columns = ["process", "count"]
        fig = px.bar(proc_counts, x="process", y="count", title="Events by Process")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        layer_counts = attack_df["asap_layer"].value_counts().reset_index()
        layer_counts.columns = ["asap_layer", "count"]
        fig = px.pie(layer_counts, names="asap_layer", values="count", title="ASAP Layer Share")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        amount_series = attack_df["amount"].dropna()
        if amount_series.empty:
            st.info("No amount field found in filtered records.")
        else:
            fig = px.histogram(
                attack_df.dropna(subset=["amount"]),
                x="amount",
                nbins=min(20, max(5, attack_df["amount"].nunique())),
                title="Attack Amount Distribution",
            )
            fig.add_vline(x=amount_threshold, line_dash="dash", annotation_text="threshold")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filtered event log")
    display_cols = [c for c in ["ts", "agent_id", "process", "asap_layer", "amount", "label", "ok", "stride_tags", "details"] if c in attack_df.columns]
    st.dataframe(attack_df[display_cols].sort_values("ts", ascending=False), use_container_width=True, height=320)

with tab2:
    if rl_df.empty:
        st.info("No RL-agent rows available under the current filters.")
    else:
        left, right = st.columns(2)
        with left:
            agent_counts = (
                rl_df[rl_df["is_attack_event"]]["agent_id"].value_counts().rename_axis("agent_id").reset_index(name="count")
            )
            fig = px.bar(agent_counts, x="agent_id", y="count", title="Attack Event Volume by Agent")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            amount_by_agent = rl_df.dropna(subset=["amount"])
            if amount_by_agent.empty:
                st.info("No amount values found for RL-agent events.")
            else:
                fig = px.box(
                    amount_by_agent,
                    x="agent_id",
                    y="amount",
                    points="all",
                    title="Attack Amount Spread by Agent",
                )
                st.plotly_chart(fig, use_container_width=True)

        success_df = rl_df[rl_df["ok"].notna()].copy()
        if not success_df.empty:
            success_summary = (
                success_df.groupby("agent_id")["ok"]
                .mean()
                .sort_values(ascending=False)
                .reset_index(name="pass_rate")
            )
            fig = px.bar(
                success_summary,
                x="agent_id",
                y="pass_rate",
                title="Risk-Check Pass Rate by Agent",
                text=success_summary["pass_rate"].map(lambda x: f"{x:.1%}"),
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Agent summary")
        summary = (
            rl_df.groupby("agent_id")
            .agg(
                events=("agent_id", "size"),
                attack_events=("is_attack_event", "sum"),
                avg_amount=("amount", "mean"),
                risk_check_pass_rate=("ok", "mean"),
            )
            .reset_index()
            .sort_values("attack_events", ascending=False)
        )
        st.dataframe(summary, use_container_width=True)

with tab3:
    st.write(
        "This tab gives you a fast what-if detector. It is not a full anomaly model retraining pipeline; it is a practical threshold sandbox built on the log amounts your notebook already produced."
    )

    det_df = attack_df.dropna(subset=["amount"]).copy()
    if det_df.empty:
        st.info("No numeric amount data available for threshold analysis.")
    else:
        det_df["predicted_flag"] = det_df["amount"] >= amount_threshold
        if det_df["label"].notna().any():
            labeled = det_df[det_df["label"].isin([0, 1])].copy()
        else:
            labeled = det_df.copy()
            labeled["label"] = 1

        tp = int(((labeled["label"] == 1) & (labeled["predicted_flag"])).sum())
        fn = int(((labeled["label"] == 1) & (~labeled["predicted_flag"])).sum())
        fp = int(((labeled["label"] == 0) & (labeled["predicted_flag"])).sum())
        tn = int(((labeled["label"] == 0) & (~labeled["predicted_flag"])).sum())

        a, b, c, d = st.columns(4)
        a.metric("True positives", tp)
        b.metric("False negatives", fn)
        c.metric("False positives", fp)
        d.metric("True negatives", tn)

        cm = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual benign", "Actual attack"],
            columns=["Pred benign", "Pred flagged"],
        )
        fig = px.imshow(cm, text_auto=True, aspect="auto", title="Threshold Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

        curve_thresholds = sorted(det_df["amount"].dropna().unique())
        rows = []
        for thr in curve_thresholds:
            pred = det_df["amount"] >= thr
            y = det_df["label"].fillna(1).astype(int)
            tp_i = ((y == 1) & pred).sum()
            fn_i = ((y == 1) & (~pred)).sum()
            fp_i = ((y == 0) & pred).sum()
            tn_i = ((y == 0) & (~pred)).sum()
            tpr = tp_i / (tp_i + fn_i) if (tp_i + fn_i) else 0
            fpr = fp_i / (fp_i + tn_i) if (fp_i + tn_i) else 0
            precision = tp_i / (tp_i + fp_i) if (tp_i + fp_i) else 0
            rows.append({"threshold": thr, "TPR": tpr, "FPR": fpr, "Precision": precision})
        curve_df = pd.DataFrame(rows)

        left, right = st.columns(2)
        with left:
            roc_fig = px.line(curve_df, x="FPR", y="TPR", markers=True, title="Proxy ROC Curve")
            roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
            st.plotly_chart(roc_fig, use_container_width=True)
        with right:
            pr_fig = px.line(curve_df, x="TPR", y="Precision", markers=True, title="Proxy Precision-Recall View")
            st.plotly_chart(pr_fig, use_container_width=True)

        st.dataframe(curve_df.sort_values("threshold"), use_container_width=True)

with tab4:
    stride_exploded = attack_df.explode("stride_tags")
    stride_exploded = stride_exploded[stride_exploded["stride_tags"].notna()]

    left, right = st.columns(2)
    with left:
        if stride_exploded.empty:
            st.info("No STRIDE tags were found in the filtered rows.")
        else:
            stride_counts = stride_exploded["stride_tags"].value_counts().reset_index()
            stride_counts.columns = ["stride_tag", "count"]
            fig = px.bar(stride_counts, x="stride_tag", y="count", title="Overall STRIDE Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if stride_exploded.empty:
            st.empty()
        else:
            per_agent_stride = (
                stride_exploded[stride_exploded["agent_id"].isin(RL_AGENTS)]
                .groupby(["agent_id", "stride_tags"])
                .size()
                .reset_index(name="count")
            )
            fig = px.bar(
                per_agent_stride,
                x="agent_id",
                y="count",
                color="stride_tags",
                barmode="group",
                title="STRIDE Preference by Agent",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quick narrative")
    if stride_exploded.empty:
        st.write("No STRIDE evidence is available under the current filters.")
    else:
        top_stride = stride_exploded["stride_tags"].value_counts().idxmax()
        top_agent_series = attack_df[attack_df["agent_id"].isin(RL_AGENTS)]["agent_id"].value_counts()
        top_agent = top_agent_series.idxmax() if not top_agent_series.empty else "n/a"
        st.write(
            f"The dominant threat pattern in the current view is **{top_stride}**, and the most active RL agent is **{top_agent}**. "
            "That gives you a quick read on whether your simulated attacker population is leaning toward tampering, disclosure, or resource abuse."
        )

st.markdown("---")
st.caption(
    "Run with: `streamlit run app.py`  |  Upload the `cbdc_logs.csv` exported from your notebook."
)
