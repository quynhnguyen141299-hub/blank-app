import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_NAME = "CBDC Sentinel: AI Attack & Detection Analytics"
RL_AGENTS = ["Q-Learning", "DQN", "REINFORCE", "A2C"]

# ── Colour palette (consistent across all tabs) ──────────────────────────
AGENT_COLOURS = {
    "Q-Learning": "#636EFA",
    "DQN": "#EF553B",
    "REINFORCE": "#00CC96",
    "A2C": "#AB63FA",
}
STRIDE_COLOURS = {
    "Spoofing": "#636EFA",
    "Tampering": "#EF553B",
    "Repudiation": "#00CC96",
    "Information Disclosure": "#AB63FA",
    "Denial of Service": "#FFA15A",
    "Elevation of Privilege": "#19D3F3",
}

st.set_page_config(page_title=APP_NAME, layout="wide")

# ── Global CSS tweaks ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* tighten metric cards */
    [data-testid="stMetric"] {
        background: rgba(28, 131, 225, .04);
        border-radius: 8px;
        padding: 12px 16px 8px;
    }
    /* scrollable data-frames */
    .stDataFrame { max-height: 380px; overflow-y: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

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
    # Extra fields for new features
    df["action"] = df["details"].apply(
        lambda d: d.get("action") if isinstance(d, dict) else None
    )
    df["reward"] = df["details"].apply(
        lambda d: d.get("reward") if isinstance(d, dict) else np.nan
    )
    df["episode"] = df["details"].apply(
        lambda d: d.get("episode") if isinstance(d, dict) else np.nan
    )
    df["target_wallet"] = df["details"].apply(
        lambda d: d.get("target_wallet") if isinstance(d, dict) else None
    )
    df["source_wallet"] = df["details"].apply(
        lambda d: d.get("source_wallet", d.get("wallet_id"))
        if isinstance(d, dict)
        else None
    )

    df["process"] = (
        df["process"].astype(str) if "process" in df.columns else "unknown"
    )
    df["asap_layer"] = (
        df["asap_layer"].astype(str) if "asap_layer" in df.columns else "unknown"
    )

    # Cast boolean / label columns to proper numeric types so .mean()/.sum() never
    # hit TypeError on object-dtype columns.
    df["ok"] = pd.to_numeric(df["ok"].map({True: 1, False: 0, "True": 1, "False": 0}), errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["complexity"] = pd.to_numeric(df["complexity"], errors="coerce")
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")

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
    ).astype(bool)

    df["is_risk_check"] = (
        df["ok"].notna()
        | df["process"].astype(str).str.contains("P5", na=False)
    ).astype(bool)

    return df


def metric_delta(current, baseline):
    try:
        if baseline in (None, 0) or pd.isna(baseline) or pd.isna(current):
            return "n/a"
        delta = float(current) - float(baseline)
        return f"{delta:+.1%}"
    except (TypeError, ValueError):
        return "n/a"


# ═══════════════════════════════════════════════════════════════════════════
# HEADER & SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
st.title(APP_NAME)
st.caption(
    "Interactive dashboard for exploring RL-driven attack behaviour, "
    "STRIDE patterns, and detection signals in your CBDC simulation logs."
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
        "Expected columns include: `ts`, `process`, `asap_layer`, `stride_tags`, `details`."
    )
    st.stop()

logs = load_logs(uploaded_file)
filtered = logs.copy()

if selected_agents:
    agent_mask = filtered["agent_id"].isin(selected_agents)
    if show_benign:
        agent_mask = agent_mask | filtered["agent_id"].isin(
            ["benign", "system"]
        ) | filtered["agent_id"].isna()
    filtered = filtered[agent_mask]

if attack_only:
    filtered = filtered[filtered["is_attack_event"]]

if filtered.empty:
    st.warning("No rows matched the current filter settings.")
    st.stop()

attack_df = filtered[filtered["is_attack_event"]].copy()
rl_df = filtered[filtered["agent_id"].isin(RL_AGENTS)].copy()


# ═══════════════════════════════════════════════════════════════════════════
# TOP-LEVEL KPI METRICS
# ═══════════════════════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)

n_attack_events = int(attack_df.shape[0])
mean_amount = (
    float(attack_df["amount"].dropna().mean())
    if attack_df["amount"].notna().any()
    else 0.0
)
risk_checks = attack_df[attack_df["is_risk_check"]]
passed_checks = int(risk_checks["ok"].eq(1).sum()) if not risk_checks.empty else 0
pass_rate = (passed_checks / len(risk_checks)) if len(risk_checks) else np.nan
flagged_rate = (
    attack_df["amount"].ge(amount_threshold).mean()
    if attack_df["amount"].notna().any()
    else np.nan
)

baseline_df = logs[(logs["agent_id"].isin(RL_AGENTS)) & (logs["is_attack_event"])]
baseline_flagged = (
    baseline_df["amount"].ge(amount_threshold).mean()
    if baseline_df["amount"].notna().any()
    else np.nan
)

col1.metric("Attack-related events", f"{n_attack_events:,}")
col2.metric("Average attack amount", f"{mean_amount:,.1f}")
col3.metric(
    "Risk-check pass rate",
    "n/a" if pd.isna(pass_rate) else f"{pass_rate:.1%}",
)
col4.metric(
    f"Flagged at threshold >= {amount_threshold}",
    "n/a" if pd.isna(flagged_rate) else f"{flagged_rate:.1%}",
    metric_delta(flagged_rate, baseline_flagged),
)


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "System Overview",
        "Agent Behavior",
        "Detection Lab",
        "Threat Intelligence",
        "Attack Replay",
        "Network Graph",
        "Agent Strategy",
    ]
)


# ─── TAB 1: System Overview (original) ───────────────────────────────────
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
        fig = px.pie(
            layer_counts,
            names="asap_layer",
            values="count",
            title="ASAP Layer Share",
        )
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
            fig.add_vline(
                x=amount_threshold,
                line_dash="dash",
                annotation_text="threshold",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filtered event log")
    display_cols = [
        c
        for c in [
            "ts",
            "agent_id",
            "process",
            "asap_layer",
            "amount",
            "label",
            "ok",
            "stride_tags",
            "details",
        ]
        if c in attack_df.columns
    ]
    st.dataframe(
        attack_df[display_cols].sort_values("ts", ascending=False),
        use_container_width=True,
        height=320,
    )


# ─── TAB 2: Agent Behavior (original) ────────────────────────────────────
with tab2:
    if rl_df.empty:
        st.info("No RL-agent rows available under the current filters.")
    else:
        left, right = st.columns(2)
        with left:
            agent_counts = (
                rl_df[rl_df["is_attack_event"]]["agent_id"]
                .value_counts()
                .rename_axis("agent_id")
                .reset_index(name="count")
            )
            fig = px.bar(
                agent_counts,
                x="agent_id",
                y="count",
                title="Attack Event Volume by Agent",
                color="agent_id",
                color_discrete_map=AGENT_COLOURS,
            )
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
                    color="agent_id",
                    color_discrete_map=AGENT_COLOURS,
                )
                st.plotly_chart(fig, use_container_width=True)

        success_df = rl_df[rl_df["ok"].notna()].copy()
        if not success_df.empty:
            success_summary = (
                success_df.groupby("agent_id")["ok"]
                .mean(numeric_only=False)
                .sort_values(ascending=False)
                .reset_index(name="pass_rate")
            )
            fig = px.bar(
                success_summary,
                x="agent_id",
                y="pass_rate",
                title="Risk-Check Pass Rate by Agent",
                text=success_summary["pass_rate"].map(lambda x: f"{x:.1%}"),
                color="agent_id",
                color_discrete_map=AGENT_COLOURS,
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Agent summary")
        summary = (
            rl_df.groupby("agent_id")
            .agg(
                events=("agent_id", "size"),
                attack_events=("is_attack_event", lambda x: x.astype(int).sum()),
                avg_amount=("amount", "mean"),
                risk_check_pass_rate=("ok", "mean"),
            )
            .reset_index()
            .sort_values("attack_events", ascending=False)
        )
        st.dataframe(summary, use_container_width=True)


# ─── TAB 3: Detection Lab (ENHANCED with ML anomaly scoring) ─────────────
with tab3:
    st.write(
        "**Threshold sandbox** plus **ML-based anomaly scoring**. "
        "Compare a simple amount-threshold detector against Isolation Forest, "
        "Z-score, and DBSCAN clustering — all fitted on-the-fly to your log data."
    )

    det_df = filtered.dropna(subset=["amount"]).copy()
    if det_df.empty:
        st.info("No numeric amount data available for threshold analysis.")
    else:
        # ── Threshold-based detection (original) ─────────────────────────
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
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Threshold Confusion Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Proxy ROC / PR (original) ────────────────────────────────────
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
            rows.append(
                {"threshold": thr, "TPR": tpr, "FPR": fpr, "Precision": precision}
            )
        curve_df = pd.DataFrame(rows)

        left, right = st.columns(2)
        with left:
            roc_fig = px.line(
                curve_df, x="FPR", y="TPR", markers=True, title="Proxy ROC Curve"
            )
            roc_fig.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash")
            )
            st.plotly_chart(roc_fig, use_container_width=True)
        with right:
            pr_fig = px.line(
                curve_df,
                x="TPR",
                y="Precision",
                markers=True,
                title="Proxy Precision-Recall View",
            )
            st.plotly_chart(pr_fig, use_container_width=True)

        st.dataframe(curve_df.sort_values("threshold"), use_container_width=True)

        # ══════════════════════════════════════════════════════════════════
        # NEW: ML Anomaly Scoring Section
        # ══════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("ML Anomaly Scoring")
        st.caption(
            "Scikit-learn models fitted on the filtered data. "
            "Features: `amount`, `complexity` (where available)."
        )

        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import (
                precision_score,
                recall_score,
                f1_score,
                accuracy_score,
            )

            ml_df = det_df.copy()
            feat_cols = ["amount"]
            if ml_df["complexity"].notna().sum() > 5:
                feat_cols.append("complexity")
            X_raw = ml_df[feat_cols].fillna(0).values

            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)

            # ── Z-Score ──────────────────────────────────────────────────
            zscore_sensitivity = st.slider(
                "Z-score sensitivity (σ threshold)",
                min_value=1.0,
                max_value=5.0,
                value=2.5,
                step=0.1,
                help="Points more than this many standard deviations from the mean are flagged.",
            )
            z_scores = np.abs(X[:, 0])  # already standardised
            ml_df["zscore_flag"] = z_scores > zscore_sensitivity

            # ── Isolation Forest ─────────────────────────────────────────
            iso_contamination = st.slider(
                "Isolation Forest contamination",
                min_value=0.01,
                max_value=0.50,
                value=0.10,
                step=0.01,
                help="Expected fraction of outliers in the data.",
            )
            iso = IsolationForest(
                contamination=iso_contamination, random_state=42, n_jobs=-1
            )
            ml_df["iso_pred"] = iso.fit_predict(X)
            ml_df["iso_flag"] = ml_df["iso_pred"] == -1
            ml_df["iso_score"] = -iso.score_samples(X)  # higher = more anomalous

            # ── DBSCAN clustering ────────────────────────────────────────
            dbscan_eps = st.slider(
                "DBSCAN ε (neighbourhood radius)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Maximum distance between two points in the same neighbourhood.",
            )
            db = DBSCAN(eps=dbscan_eps, min_samples=3)
            ml_df["cluster"] = db.fit_predict(X)
            ml_df["dbscan_flag"] = ml_df["cluster"] == -1  # noise = anomaly

            # ── Ensemble vote ────────────────────────────────────────────
            ml_df["ensemble_votes"] = (
                ml_df["zscore_flag"].astype(int)
                + ml_df["iso_flag"].astype(int)
                + ml_df["dbscan_flag"].astype(int)
            )
            ml_df["ensemble_flag"] = ml_df["ensemble_votes"] >= 2  # majority

            # ── Comparison metrics ───────────────────────────────────────
            y_true = ml_df["label"].fillna(1).astype(int)

            model_metrics = []
            for name, col in [
                ("Threshold", "predicted_flag"),
                ("Z-Score", "zscore_flag"),
                ("Isolation Forest", "iso_flag"),
                ("DBSCAN", "dbscan_flag"),
                ("Ensemble (≥2/3)", "ensemble_flag"),
            ]:
                y_pred = ml_df[col].astype(int)
                model_metrics.append(
                    {
                        "Model": name,
                        "Accuracy": accuracy_score(y_true, y_pred),
                        "Precision": precision_score(y_true, y_pred, zero_division=0),
                        "Recall": recall_score(y_true, y_pred, zero_division=0),
                        "F1": f1_score(y_true, y_pred, zero_division=0),
                        "Flagged": int(y_pred.sum()),
                    }
                )
            metrics_df = pd.DataFrame(model_metrics)

            st.dataframe(
                metrics_df.style.highlight_max(
                    subset=["Accuracy", "Precision", "Recall", "F1"],
                    color="#2ecc71",
                    axis=0,
                ),
                use_container_width=True,
            )

            # ── Visual comparison ────────────────────────────────────────
            left, right = st.columns(2)
            with left:
                bar_data = metrics_df.melt(
                    id_vars="Model",
                    value_vars=["Precision", "Recall", "F1"],
                    var_name="Metric",
                )
                fig = px.bar(
                    bar_data,
                    x="Model",
                    y="value",
                    color="Metric",
                    barmode="group",
                    title="Detector Comparison",
                )
                fig.update_layout(yaxis_title="Score", yaxis_range=[0, 1.05])
                st.plotly_chart(fig, use_container_width=True)

            with right:
                fig = go.Figure()
                if ml_df["iso_score"].notna().any():
                    fig.add_trace(
                        go.Histogram(
                            x=ml_df.loc[ml_df["label"] == 1, "iso_score"],
                            name="Attack",
                            opacity=0.7,
                        )
                    )
                    fig.add_trace(
                        go.Histogram(
                            x=ml_df.loc[ml_df["label"] == 0, "iso_score"],
                            name="Benign",
                            opacity=0.7,
                        )
                    )
                    fig.update_layout(
                        barmode="overlay",
                        title="Isolation Forest Anomaly Score Distribution",
                        xaxis_title="Anomaly Score",
                        yaxis_title="Count",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # ── Scatter: Isolation Forest score vs amount ────────────────
            if "complexity" in feat_cols:
                fig = px.scatter(
                    ml_df,
                    x="amount",
                    y="complexity",
                    color="ensemble_flag",
                    color_discrete_map={True: "#EF553B", False: "#636EFA"},
                    title="Ensemble Anomaly Map (Amount vs Complexity)",
                    labels={"ensemble_flag": "Flagged"},
                    hover_data=["agent_id", "iso_score", "ensemble_votes"],
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.scatter(
                    ml_df,
                    x="amount",
                    y="iso_score",
                    color="ensemble_flag",
                    color_discrete_map={True: "#EF553B", False: "#636EFA"},
                    title="Ensemble Anomaly Map (Amount vs IF Score)",
                    labels={"ensemble_flag": "Flagged"},
                    hover_data=["agent_id", "ensemble_votes"],
                )
                st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.warning(
                "scikit-learn is required for ML anomaly scoring. "
                "Add `scikit-learn` to your requirements.txt and redeploy."
            )


# ─── TAB 4: Threat Intelligence (original) ───────────────────────────────
with tab4:
    stride_exploded = attack_df.explode("stride_tags")
    stride_exploded = stride_exploded[stride_exploded["stride_tags"].notna()]

    left, right = st.columns(2)
    with left:
        if stride_exploded.empty:
            st.info("No STRIDE tags were found in the filtered rows.")
        else:
            stride_counts = (
                stride_exploded["stride_tags"].value_counts().reset_index()
            )
            stride_counts.columns = ["stride_tag", "count"]
            fig = px.bar(
                stride_counts,
                x="stride_tag",
                y="count",
                title="Overall STRIDE Distribution",
                color="stride_tag",
                color_discrete_map=STRIDE_COLOURS,
            )
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
                color_discrete_map=STRIDE_COLOURS,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quick narrative")
    if stride_exploded.empty:
        st.write("No STRIDE evidence is available under the current filters.")
    else:
        top_stride = stride_exploded["stride_tags"].value_counts().idxmax()
        top_agent_series = attack_df[attack_df["agent_id"].isin(RL_AGENTS)][
            "agent_id"
        ].value_counts()
        top_agent = (
            top_agent_series.idxmax() if not top_agent_series.empty else "n/a"
        )
        st.write(
            f"The dominant threat pattern in the current view is **{top_stride}**, "
            f"and the most active RL agent is **{top_agent}**. "
            "That gives you a quick read on whether your simulated attacker "
            "population is leaning toward tampering, disclosure, or resource abuse."
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: Attack Simulation Replay  (NEW)
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.write(
        "**Step-by-step replay** of attack sequences. Watch how RL agents "
        "escalate their strategy over time — event by event."
    )

    replay_df = attack_df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    if replay_df.empty:
        st.info("No timestamped attack events available for replay.")
    else:
        # ── Playback controls ────────────────────────────────────────────
        replay_agent = st.selectbox(
            "Focus agent (or All)",
            ["All"] + [a for a in RL_AGENTS if a in replay_df["agent_id"].values],
            key="replay_agent",
        )
        if replay_agent != "All":
            replay_df = replay_df[replay_df["agent_id"] == replay_agent].reset_index(
                drop=True
            )

        if replay_df.empty:
            st.info("No events for the selected agent.")
        else:
            max_step = len(replay_df) - 1
            step = st.slider(
                "Event step",
                min_value=0,
                max_value=max_step,
                value=max_step,
                help="Drag to scrub through the attack sequence.",
            )

            window = replay_df.iloc[: step + 1]
            current = replay_df.iloc[step]

            # ── Current event card ───────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Step", f"{step + 1} / {max_step + 1}")
            c2.metric("Agent", str(current.get("agent_id", "—")))
            c3.metric("Amount", f"{current.get('amount', 0):,.1f}")
            c4.metric("Process", str(current.get("process", "—")))

            # ── Animated cumulative timeline ─────────────────────────────
            window_agg = (
                window.set_index("ts")
                .resample("1min")
                .size()
                .cumsum()
                .reset_index(name="cumulative_events")
            )
            fig = px.area(
                window_agg,
                x="ts",
                y="cumulative_events",
                title="Cumulative Attack Events (up to current step)",
            )
            fig.update_layout(xaxis_title="Time", yaxis_title="Cumulative events")
            st.plotly_chart(fig, use_container_width=True)

            # ── Strategy evolution: rolling average amount ────────────────
            left, right = st.columns(2)
            with left:
                if window["amount"].notna().sum() > 1:
                    window_amt = window.dropna(subset=["amount"]).copy()
                    window_amt["rolling_avg"] = (
                        window_amt["amount"].expanding().mean()
                    )
                    fig = px.line(
                        window_amt,
                        x="ts",
                        y="rolling_avg",
                        title="Running Average Attack Amount",
                        markers=True,
                    )
                    fig.update_layout(
                        xaxis_title="Time", yaxis_title="Avg amount"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough amount data for running-average chart.")

            with right:
                # STRIDE tag evolution over the replay window
                stride_window = window.explode("stride_tags")
                stride_window = stride_window[stride_window["stride_tags"].notna()]
                if not stride_window.empty:
                    stride_window["ts_bin"] = stride_window["ts"].dt.floor("1min")
                    stride_time = (
                        stride_window.groupby(["ts_bin", "stride_tags"])
                        .size()
                        .reset_index(name="count")
                    )
                    fig = px.bar(
                        stride_time,
                        x="ts_bin",
                        y="count",
                        color="stride_tags",
                        title="STRIDE Tag Evolution (Replay Window)",
                        color_discrete_map=STRIDE_COLOURS,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No STRIDE tags in the replay window.")

            # ── Event detail table ───────────────────────────────────────
            with st.expander("Event details for current step", expanded=False):
                st.json(
                    {
                        k: (
                            str(v)
                            if isinstance(v, (pd.Timestamp, np.floating, np.integer))
                            else v
                        )
                        for k, v in current.to_dict().items()
                        if not (isinstance(v, float) and np.isnan(v))
                    }
                )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: Network / Graph Visualisation  (NEW)
# ═══════════════════════════════════════════════════════════════════════════
with tab6:
    st.write(
        "**Transaction network** showing wallet-to-wallet flows and attack "
        "propagation paths. Nodes are wallets; edges are transactions."
    )

    # Build edges from source_wallet → target_wallet
    net_df = attack_df.copy()
    # Fallback: if target_wallet is missing, use wallet_id as source, "CBDC_LEDGER" as target
    net_df["src"] = net_df["source_wallet"].fillna(net_df["wallet_id"])
    net_df["dst"] = net_df["target_wallet"].fillna("CBDC_LEDGER")
    edges = net_df.dropna(subset=["src"]).copy()

    if edges.empty:
        st.info(
            "No wallet-level transaction data found. "
            "Your logs need `wallet_id`, `source_wallet`, or `target_wallet` "
            "fields inside the `details` JSON."
        )
    else:
        # Aggregate edges
        edge_agg = (
            edges.groupby(["src", "dst"])
            .agg(
                tx_count=("amount", "size"),
                total_amount=("amount", "sum"),
                agents=("agent_id", lambda x: list(x.dropna().unique())),
            )
            .reset_index()
        )

        # Build node set
        all_nodes = list(set(edge_agg["src"].tolist() + edge_agg["dst"].tolist()))
        node_idx = {n: i for i, n in enumerate(all_nodes)}

        # Determine node risk: how many attack events touch this wallet
        node_attack_counts = (
            edges.groupby("src")["is_attack_event"]
            .sum()
            .reindex(all_nodes, fill_value=0)
        )
        dst_attack_counts = (
            edges.groupby("dst")["is_attack_event"]
            .sum()
            .reindex(all_nodes, fill_value=0)
        )
        node_risk = (node_attack_counts + dst_attack_counts).fillna(0)

        # Layout: force-directed approximation via random + spring-like
        np.random.seed(42)
        n = len(all_nodes)
        pos_x = np.random.randn(n)
        pos_y = np.random.randn(n)

        # Simple iterative force-directed layout (capped for performance)
        iterations = 50 if n < 100 else 15
        for _ in range(iterations):
            for idx_i in range(n):
                for idx_j in range(idx_i + 1, n):
                    dx = pos_x[idx_i] - pos_x[idx_j]
                    dy = pos_y[idx_i] - pos_y[idx_j]
                    dist = max(np.sqrt(dx * dx + dy * dy), 0.01)
                    repulsion = 0.5 / (dist * dist)
                    pos_x[idx_i] += repulsion * dx / dist
                    pos_y[idx_i] += repulsion * dy / dist
                    pos_x[idx_j] -= repulsion * dx / dist
                    pos_y[idx_j] -= repulsion * dy / dist

            for _, row in edge_agg.iterrows():
                i = node_idx[row["src"]]
                j = node_idx[row["dst"]]
                dx = pos_x[j] - pos_x[i]
                dy = pos_y[j] - pos_y[i]
                dist = max(np.sqrt(dx * dx + dy * dy), 0.01)
                attraction = 0.1 * dist
                pos_x[i] += attraction * dx / dist
                pos_y[i] += attraction * dy / dist
                pos_x[j] -= attraction * dx / dist
                pos_y[j] -= attraction * dy / dist

        # Build Plotly figure
        edge_traces = []
        for _, row in edge_agg.iterrows():
            i = node_idx[row["src"]]
            j = node_idx[row["dst"]]
            width = max(1, min(8, row["tx_count"]))
            edge_traces.append(
                go.Scatter(
                    x=[pos_x[i], pos_x[j], None],
                    y=[pos_y[i], pos_y[j], None],
                    mode="lines",
                    line=dict(width=width, color="rgba(150,150,150,0.4)"),
                    hoverinfo="text",
                    text=f"{row['src']} → {row['dst']}<br>"
                    f"Txns: {row['tx_count']}<br>"
                    f"Amount: {row['total_amount']:,.0f}<br>"
                    f"Agents: {', '.join(row['agents'])}",
                    showlegend=False,
                )
            )

        node_sizes = 10 + 3 * np.sqrt(node_risk.values)
        node_colors = node_risk.values

        node_trace = go.Scatter(
            x=pos_x,
            y=pos_y,
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale="YlOrRd",
                colorbar=dict(title="Attack<br>exposure"),
                line=dict(width=1, color="white"),
            ),
            text=[n[:12] for n in all_nodes],
            textposition="top center",
            textfont=dict(size=9),
            hovertext=[
                f"{n}<br>Attack exposure: {int(node_risk[n])}" for n in all_nodes
            ],
            hoverinfo="text",
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title="Wallet Transaction Network (Attack Events)",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Cluster analysis ─────────────────────────────────────────────
        st.subheader("Top compromised wallets")
        top_wallets = (
            node_risk.sort_values(ascending=False).head(10).reset_index()
        )
        top_wallets.columns = ["wallet", "attack_exposure"]
        st.dataframe(top_wallets, use_container_width=True)

        # ── Edge weight distribution ─────────────────────────────────────
        left, right = st.columns(2)
        with left:
            fig = px.histogram(
                edge_agg,
                x="tx_count",
                nbins=15,
                title="Edge Transaction Count Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.histogram(
                edge_agg,
                x="total_amount",
                nbins=15,
                title="Edge Total Amount Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7: Agent Strategy Comparison  (NEW)
# ═══════════════════════════════════════════════════════════════════════════
with tab7:
    st.write(
        "**Side-by-side RL agent comparison**: reward curves, action distributions, "
        "convergence analysis, and STRIDE strategy heatmaps."
    )

    strat_df = rl_df.copy()
    if strat_df.empty:
        st.info("No RL-agent data for strategy comparison.")
    else:
        agents_present = [a for a in RL_AGENTS if a in strat_df["agent_id"].values]

        # ── 1. Reward Curves ─────────────────────────────────────────────
        st.subheader("Reward Curves")
        reward_data = strat_df.dropna(subset=["reward"])
        if reward_data.empty:
            # Fallback: synthesise a proxy reward from amount * (1 if ok else -1)
            st.caption(
                "No explicit `reward` field found in logs — using a proxy reward "
                "computed as `amount × (1 if risk-check passed else −0.5)`."
            )
            strat_df["proxy_reward"] = strat_df["amount"].fillna(0) * np.where(
                strat_df["ok"].fillna(0) == 1, 1.0, -0.5
            )
            reward_col = "proxy_reward"
        else:
            reward_col = "reward"

        # Cumulative reward per agent over time
        cum_rewards = []
        for agent in agents_present:
            agent_rows = strat_df[strat_df["agent_id"] == agent].sort_values("ts")
            agent_rows = agent_rows[agent_rows[reward_col].notna()]
            if agent_rows.empty:
                continue
            agent_rows["cum_reward"] = agent_rows[reward_col].cumsum()
            agent_rows["step_idx"] = range(len(agent_rows))
            cum_rewards.append(agent_rows[["step_idx", "cum_reward", "agent_id"]])

        if cum_rewards:
            cum_df = pd.concat(cum_rewards, ignore_index=True)
            fig = px.line(
                cum_df,
                x="step_idx",
                y="cum_reward",
                color="agent_id",
                title="Cumulative Reward by Agent",
                color_discrete_map=AGENT_COLOURS,
            )
            fig.update_layout(
                xaxis_title="Event step", yaxis_title="Cumulative reward"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient reward data for cumulative curves.")

        # ── 2. Action Distribution ───────────────────────────────────────
        st.subheader("Action Distribution")
        action_data = strat_df[strat_df["action"].notna()]
        if action_data.empty:
            # Fallback: use process as proxy for action
            st.caption(
                "No `action` field found — using `process` column as action proxy."
            )
            action_data = strat_df.copy()
            action_data["action"] = action_data["process"]

        action_counts = (
            action_data.groupby(["agent_id", "action"])
            .size()
            .reset_index(name="count")
        )
        if not action_counts.empty:
            fig = px.bar(
                action_counts,
                x="agent_id",
                y="count",
                color="action",
                barmode="stack",
                title="Action Distribution by Agent",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── 3. STRIDE Strategy Heatmap ───────────────────────────────────
        st.subheader("STRIDE Strategy Heatmap")
        stride_strat = strat_df.explode("stride_tags")
        stride_strat = stride_strat[stride_strat["stride_tags"].notna()]
        if stride_strat.empty:
            st.info("No STRIDE tags available for strategy heatmap.")
        else:
            heatmap_data = (
                stride_strat.groupby(["agent_id", "stride_tags"])
                .size()
                .reset_index(name="count")
            )
            pivot = heatmap_data.pivot(
                index="agent_id", columns="stride_tags", values="count"
            ).fillna(0)

            # Normalise by row to show preference distribution
            pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

            fig = px.imshow(
                pivot_norm,
                text_auto=".1%",
                aspect="auto",
                title="Agent × STRIDE Preference (row-normalised)",
                color_continuous_scale="Viridis",
                labels=dict(color="Share"),
            )
            fig.update_layout(xaxis_title="STRIDE Category", yaxis_title="RL Agent")
            st.plotly_chart(fig, use_container_width=True)

            # Raw count heatmap
            with st.expander("Raw count heatmap"):
                fig = px.imshow(
                    pivot,
                    text_auto=True,
                    aspect="auto",
                    title="Agent × STRIDE (raw counts)",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── 4. Convergence Analysis ──────────────────────────────────────
        st.subheader("Convergence Analysis")
        st.caption(
            "Rolling standard deviation of reward/amount — lower σ signals convergence."
        )
        convergence_traces = []
        window_size = st.slider(
            "Rolling window size",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="convergence_window",
        )
        for agent in agents_present:
            agent_rows = strat_df[strat_df["agent_id"] == agent].sort_values("ts")
            series = agent_rows[reward_col].fillna(0)
            if len(series) < window_size:
                continue
            rolling_std = series.rolling(window_size).std().dropna()
            conv_df = pd.DataFrame(
                {
                    "step": range(len(rolling_std)),
                    "rolling_std": rolling_std.values,
                    "agent_id": agent,
                }
            )
            convergence_traces.append(conv_df)

        if convergence_traces:
            conv_all = pd.concat(convergence_traces, ignore_index=True)
            fig = px.line(
                conv_all,
                x="step",
                y="rolling_std",
                color="agent_id",
                title=f"Rolling σ of Reward (window = {window_size})",
                color_discrete_map=AGENT_COLOURS,
            )
            fig.update_layout(
                xaxis_title="Event step", yaxis_title="Rolling Std Dev"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                f"Not enough data points (need ≥ {window_size}) for convergence analysis."
            )

        # ── 5. Per-Agent Summary Cards ───────────────────────────────────
        st.subheader("Agent comparison cards")
        card_cols = st.columns(len(agents_present) if agents_present else 1)
        for idx, agent in enumerate(agents_present):
            adf = strat_df[strat_df["agent_id"] == agent]
            with card_cols[idx]:
                st.markdown(f"#### {agent}")
                st.metric("Events", int(len(adf)))
                st.metric(
                    "Avg amount",
                    f"{adf['amount'].mean():,.1f}"
                    if adf["amount"].notna().any()
                    else "—",
                )
                ok_vals = pd.to_numeric(adf["ok"], errors="coerce")
                st.metric(
                    "Risk pass rate",
                    f"{ok_vals.mean():.1%}" if ok_vals.notna().any() else "—",
                )
                top_stride_agent = adf.explode("stride_tags")["stride_tags"]
                top_stride_agent = top_stride_agent[top_stride_agent.notna()]
                if not top_stride_agent.empty:
                    st.metric("Top STRIDE", top_stride_agent.value_counts().idxmax())
                cum_r = adf[reward_col].fillna(0).sum()
                st.metric("Total reward", f"{cum_r:,.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "Run with: `streamlit run streamlit_app.py`  |  "
    "Upload the `cbdc_logs.csv` exported from your notebook."
)
