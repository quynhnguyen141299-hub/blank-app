import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import LabelEncoder

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

    # ── V2 Feature Engineering (categorical + STRIDE one-hot) ────────────
    _le_process = LabelEncoder()
    _le_layer   = LabelEncoder()
    df["process_enc"]    = _le_process.fit_transform(df["process"].fillna("unknown"))
    df["asap_layer_enc"] = _le_layer.fit_transform(df["asap_layer"].fillna("unknown"))

    _STRIDE_TAGS = [
        "Spoofing", "Tampering", "Repudiation",
        "Information Disclosure", "Denial of Service", "Elevation of Privilege",
    ]
    for _tag in _STRIDE_TAGS:
        df[f"stride_{_tag.lower().replace(' ', '_')}"] = df["stride_tags"].apply(
            lambda tags, t=_tag: 1 if t in tags else 0
        )
    df["stride_count"] = df["stride_tags"].apply(len)

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

has_csv = uploaded_file is not None

if has_csv:
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

    csv_empty = filtered.empty
    if not csv_empty:
        attack_df = filtered[filtered["is_attack_event"]].copy()
        rl_df = filtered[filtered["agent_id"].isin(RL_AGENTS)].copy()
    else:
        attack_df = pd.DataFrame()
        rl_df = pd.DataFrame()
else:
    _empty_cols = {
        "ts": pd.Series(dtype="datetime64[ns]"),
        "process": pd.Series(dtype="str"),
        "asap_layer": pd.Series(dtype="str"),
        "stride_tags": pd.Series(dtype="object"),
        "details": pd.Series(dtype="object"),
        "agent_id": pd.Series(dtype="str"),
        "amount": pd.Series(dtype="float64"),
        "label": pd.Series(dtype="float64"),
        "ok": pd.Series(dtype="float64"),
        "complexity": pd.Series(dtype="float64"),
        "reward": pd.Series(dtype="float64"),
        "episode": pd.Series(dtype="float64"),
        "wallet_id": pd.Series(dtype="str"),
        "action": pd.Series(dtype="str"),
        "target_wallet": pd.Series(dtype="str"),
        "source_wallet": pd.Series(dtype="str"),
        "is_attack_event": pd.Series(dtype="bool"),
        "is_risk_check": pd.Series(dtype="bool"),
        "process_enc": pd.Series(dtype="int64"),
        "asap_layer_enc": pd.Series(dtype="int64"),
        "stride_spoofing": pd.Series(dtype="int64"),
        "stride_tampering": pd.Series(dtype="int64"),
        "stride_repudiation": pd.Series(dtype="int64"),
        "stride_information_disclosure": pd.Series(dtype="int64"),
        "stride_denial_of_service": pd.Series(dtype="int64"),
        "stride_elevation_of_privilege": pd.Series(dtype="int64"),
        "stride_count": pd.Series(dtype="int64"),
    }
    _empty_df = pd.DataFrame(_empty_cols)
    logs = _empty_df.copy()
    filtered = _empty_df.copy()
    attack_df = _empty_df.copy()
    rl_df = _empty_df.copy()
    csv_empty = True


# ═══════════════════════════════════════════════════════════════════════════
# TOP-LEVEL KPI METRICS
# ═══════════════════════════════════════════════════════════════════════════
if has_csv and not csv_empty:
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
elif not has_csv:
    st.info(
        "Upload the CSV exported from your notebook to activate simulation tabs.\n\n"
        "The **CBDC Architecture**, **Threat Catalogue**, **Adversary Profiles**, "
        "and **Risk Register** tabs are available without CSV data."
    )


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(
    [
        "System Overview",
        "Agent Behavior",
        "Detection Lab",
        "Threat Intelligence",
        "Attack Replay",
        "Network Graph",
        "Agent Strategy",
        "CBDC Architecture",
        "Threat Catalogue",
        "Adversary Profiles",
        "Risk Register",
    ]
)

_CSV_REQUIRED_MSG = (
    "Upload a CSV file using the sidebar to activate this tab."
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
        # ML Anomaly Scoring — V2 Features (categorical + STRIDE)
        # ══════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("ML Anomaly Scoring (V2)")
        st.caption(
            "V2 feature set: `amount` + process/ASAP-layer encoding + "
            "STRIDE one-hot + `complexity` (where available). "
            "All models are fitted on-the-fly to your filtered data."
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

            # ── V2 feature set ───────────────────────────────────────────
            feat_cols = [
                "amount",
                "process_enc",
                "asap_layer_enc",
                "stride_count",
                "stride_spoofing",
                "stride_tampering",
                "stride_repudiation",
                "stride_information_disclosure",
                "stride_denial_of_service",
                "stride_elevation_of_privilege",
            ]
            if ml_df["complexity"].notna().sum() > 5:
                feat_cols.append("complexity")
            # Keep only columns that exist in the data
            feat_cols = [c for c in feat_cols if c in ml_df.columns]

            X_raw = ml_df[feat_cols].fillna(0).values

            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)

            st.info(f"V2 feature set: **{len(feat_cols)}** features — {', '.join(feat_cols)}")

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
    net_df["src"] = net_df["source_wallet"].fillna(net_df["wallet_id"]).astype(str)
    net_df["dst"] = net_df["target_wallet"].fillna("CBDC_LEDGER").astype(str)
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
        all_nodes = [str(n) for n in set(edge_agg["src"].tolist() + edge_agg["dst"].tolist())]
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
            text=[str(n)[:12] for n in all_nodes],
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
# EMBEDDED REFERENCE DATA FOR CBDC THREAT MODEL
# ═══════════════════════════════════════════════════════════════════════════

ASAP_LAYERS = {
    "Platform": {
        "order": 0,
        "color": "#636EFA",
        "functions": "Execution, Storage/Ledger, Communication, Consensus, Identification, Authentication, Authorization",
        "components": ["Core Ledger", "Consensus Nodes", "Settlement Engine", "Key Management Service (KMS)"],
        "threats": "Consensus mechanism exploits, ledger tampering, 51% attacks, smart contract vulnerabilities, key compromise",
        "tier": "Central Bank",
    },
    "Asset": {
        "order": 1,
        "color": "#EF553B",
        "functions": "Issuance (minting), Transfer, Redemption (burning), Access Control, State Representation",
        "components": ["Minting Authority", "Token/Account State", "UTXO Store", "Asset Lifecycle Manager"],
        "threats": "Double-spending, counterfeiting/unauthorized minting, replay attacks, state manipulation",
        "tier": "Central Bank",
    },
    "Service": {
        "order": 2,
        "color": "#00CC96",
        "functions": "Asset Exchange, DVP/PVP, Conditional Transfer, Lending, Collateralisation, Programmable Lock, Escrow, Identity, Admin, Behavioural Constraints",
        "components": ["Payment Processor", "AML/CFT Engine", "Smart Contract Runtime", "Interoperability Bridge", "Offline Payment Module", "Oracle Interface"],
        "threats": "Smart contract exploits, oracle manipulation, AML bypass, bridge attacks, offline replay",
        "tier": "Both",
    },
    "Access": {
        "order": 3,
        "color": "#AB63FA",
        "functions": "Presentation, Credential Management, Formatting, Client-side Processing, Translation, User Authentication, Data Exchange",
        "components": ["Wallet Providers", "Mobile Apps", "Web Portals", "API Gateways", "Aggregators", "KYC/IAM System"],
        "threats": "Wallet compromise, phishing/credential theft, API abuse, MitM attacks, SIM swap, malicious wallet apps",
        "tier": "Intermediary / Retail",
    },
}

COMPONENT_DETAILS = {
    "Core Ledger": {
        "layer": "Platform", "tier": "Central Bank",
        "functions": "Immutable record of all CBDC transactions; maintains global state of balances and ownership.",
        "attack_surface": "Direct ledger manipulation, Byzantine faults, storage corruption, unauthorized state changes.",
        "stride": ["Tampering", "Repudiation", "Denial of Service"],
        "mitre": ["T1485", "CBDC-NEW-10"],
        "connected": ["Consensus Nodes", "Settlement Engine", "Minting Authority"],
    },
    "Consensus Nodes": {
        "layer": "Platform", "tier": "Central Bank",
        "functions": "Validate transactions and maintain agreement on ledger state across distributed nodes.",
        "attack_surface": "51% attacks, Sybil attacks, consensus delay exploitation, node compromise.",
        "stride": ["Tampering", "Denial of Service", "Spoofing"],
        "mitre": ["CBDC-NEW-04", "CBDC-NEW-10", "T1489"],
        "connected": ["Core Ledger", "Settlement Engine"],
    },
    "Settlement Engine": {
        "layer": "Platform", "tier": "Central Bank",
        "functions": "Processes final settlement of wholesale and retail transactions; interfaces with RTGS.",
        "attack_surface": "Settlement manipulation, timing attacks, race conditions in finality.",
        "stride": ["Tampering", "Denial of Service"],
        "mitre": ["T1489", "T1485"],
        "connected": ["Core Ledger", "Consensus Nodes", "Payment Processor"],
    },
    "Key Management Service (KMS)": {
        "layer": "Platform", "tier": "Central Bank",
        "functions": "Generates, stores, rotates, and manages cryptographic keys for signing and encryption.",
        "attack_surface": "Key extraction via side-channel, insider theft, HSM bypass, weak key generation.",
        "stride": ["Information Disclosure", "Elevation of Privilege", "Tampering"],
        "mitre": ["CBDC-NEW-07", "T1068"],
        "connected": ["Core Ledger", "Minting Authority", "Wallet Providers"],
    },
    "Minting Authority": {
        "layer": "Asset", "tier": "Central Bank",
        "functions": "Controls CBDC issuance (minting) and destruction (burning); enforces monetary policy.",
        "attack_surface": "Unauthorized minting, insider abuse of issuance controls, policy bypass.",
        "stride": ["Elevation of Privilege", "Tampering"],
        "mitre": ["CBDC-NEW-11", "T1098"],
        "connected": ["Core Ledger", "Asset Lifecycle Manager", "Token/Account State"],
    },
    "Token/Account State": {
        "layer": "Asset", "tier": "Central Bank",
        "functions": "Tracks current ownership, balances, and token metadata for all CBDC units.",
        "attack_surface": "State manipulation, double-spend via state forking, balance inflation.",
        "stride": ["Tampering", "Repudiation"],
        "mitre": ["CBDC-NEW-10", "T1070"],
        "connected": ["Core Ledger", "UTXO Store", "Minting Authority"],
    },
    "UTXO Store": {
        "layer": "Asset", "tier": "Central Bank",
        "functions": "Manages unspent transaction outputs for token-based CBDC models.",
        "attack_surface": "UTXO replay, double-spend via unspent output reuse.",
        "stride": ["Tampering", "Spoofing"],
        "mitre": ["CBDC-NEW-10", "CBDC-NEW-12"],
        "connected": ["Token/Account State", "Core Ledger"],
    },
    "Asset Lifecycle Manager": {
        "layer": "Asset", "tier": "Central Bank",
        "functions": "Orchestrates the full lifecycle: issuance → circulation → redemption of CBDC tokens.",
        "attack_surface": "Lifecycle bypass allowing tokens to skip validation stages.",
        "stride": ["Tampering", "Elevation of Privilege"],
        "mitre": ["T1098", "CBDC-NEW-11"],
        "connected": ["Minting Authority", "Token/Account State", "Payment Processor"],
    },
    "Payment Processor": {
        "layer": "Service", "tier": "Both",
        "functions": "Routes and processes retail and wholesale payment transactions.",
        "attack_surface": "Transaction manipulation, routing exploits, fee bypass.",
        "stride": ["Tampering", "Spoofing", "Denial of Service"],
        "mitre": ["T1036", "CBDC-NEW-06", "T1489"],
        "connected": ["Settlement Engine", "AML/CFT Engine", "Wallet Providers"],
    },
    "AML/CFT Engine": {
        "layer": "Service", "tier": "Both",
        "functions": "Screens transactions for anti-money laundering and counter-terrorism financing compliance.",
        "attack_surface": "Rule evasion via structuring, false negative injection, threshold gaming.",
        "stride": ["Tampering", "Spoofing", "Repudiation"],
        "mitre": ["CBDC-NEW-06", "T1036"],
        "connected": ["Payment Processor", "KYC/IAM System", "Oracle Interface"],
    },
    "Smart Contract Runtime": {
        "layer": "Service", "tier": "Both",
        "functions": "Executes programmable logic for conditional payments, escrow, and automated compliance.",
        "attack_surface": "Reentrancy, logic bombs, gas manipulation, malicious contract deployment.",
        "stride": ["Tampering", "Elevation of Privilege"],
        "mitre": ["T1059", "CBDC-NEW-02"],
        "connected": ["Core Ledger", "Payment Processor", "Oracle Interface"],
    },
    "Interoperability Bridge": {
        "layer": "Service", "tier": "Both",
        "functions": "Enables cross-platform and cross-border CBDC transactions.",
        "attack_surface": "Bridge exploits, cross-chain replay, liquidity drain.",
        "stride": ["Tampering", "Spoofing", "Information Disclosure"],
        "mitre": ["CBDC-NEW-09", "T1021"],
        "connected": ["Payment Processor", "Settlement Engine"],
    },
    "Offline Payment Module": {
        "layer": "Service", "tier": "Intermediary / Retail",
        "functions": "Enables CBDC transactions without network connectivity using stored value.",
        "attack_surface": "Offline token replay, double-spend in disconnected mode, delayed reconciliation exploits.",
        "stride": ["Spoofing", "Repudiation", "Tampering"],
        "mitre": ["CBDC-NEW-12", "CBDC-NEW-10"],
        "connected": ["Wallet Providers", "Payment Processor"],
    },
    "Oracle Interface": {
        "layer": "Service", "tier": "Both",
        "functions": "Feeds external data (exchange rates, compliance lists) into smart contracts and AML engines.",
        "attack_surface": "Oracle manipulation, data poisoning, feed spoofing.",
        "stride": ["Tampering", "Spoofing"],
        "mitre": ["CBDC-NEW-03"],
        "connected": ["Smart Contract Runtime", "AML/CFT Engine"],
    },
    "Wallet Providers": {
        "layer": "Access", "tier": "Intermediary / Retail",
        "functions": "Provide end-user wallet applications for storing and transacting CBDC.",
        "attack_surface": "Malicious wallet apps, SDK backdoors, supply chain compromise.",
        "stride": ["Spoofing", "Tampering", "Information Disclosure"],
        "mitre": ["CBDC-NEW-01", "T1566", "T1110"],
        "connected": ["API Gateways", "KYC/IAM System", "Payment Processor"],
    },
    "Mobile Apps": {
        "layer": "Access", "tier": "Intermediary / Retail",
        "functions": "Native mobile applications for CBDC wallet access and transaction management.",
        "attack_surface": "App tampering, reverse engineering, credential theft, MitM on mobile.",
        "stride": ["Spoofing", "Tampering", "Information Disclosure"],
        "mitre": ["CBDC-NEW-01", "T1557", "T1110"],
        "connected": ["Wallet Providers", "API Gateways"],
    },
    "Web Portals": {
        "layer": "Access", "tier": "Intermediary / Retail",
        "functions": "Browser-based interfaces for CBDC management by intermediaries and end-users.",
        "attack_surface": "XSS, CSRF, session hijacking, credential phishing.",
        "stride": ["Spoofing", "Tampering", "Information Disclosure"],
        "mitre": ["T1190", "T1566"],
        "connected": ["API Gateways", "KYC/IAM System"],
    },
    "API Gateways": {
        "layer": "Access", "tier": "Both",
        "functions": "Central entry point for all API calls; handles rate limiting, authentication, routing.",
        "attack_surface": "API abuse, injection attacks, DDoS, authentication bypass.",
        "stride": ["Denial of Service", "Spoofing", "Elevation of Privilege"],
        "mitre": ["T1190", "T1595", "T1489"],
        "connected": ["Wallet Providers", "Payment Processor", "KYC/IAM System"],
    },
    "Aggregators": {
        "layer": "Access", "tier": "Intermediary / Retail",
        "functions": "Third-party services that aggregate multiple wallet and payment providers.",
        "attack_surface": "Supply chain compromise, data aggregation abuse, API key theft.",
        "stride": ["Information Disclosure", "Tampering"],
        "mitre": ["T1005", "T1119"],
        "connected": ["API Gateways", "Wallet Providers"],
    },
    "KYC/IAM System": {
        "layer": "Access", "tier": "Both",
        "functions": "Manages identity verification, KYC compliance, and access management.",
        "attack_surface": "Identity fraud, credential stuffing, synthetic identity, PII harvesting.",
        "stride": ["Spoofing", "Information Disclosure", "Repudiation"],
        "mitre": ["T1078", "T1136", "T1119", "T1589"],
        "connected": ["Wallet Providers", "AML/CFT Engine", "API Gateways"],
    },
}

MITRE_TECHNIQUES = pd.DataFrame([
    {"tactic": "Reconnaissance", "tactic_id": "TA0043", "technique_id": "T1595", "technique_name": "Active Scanning", "cbdc_application": "Scanning CBDC API endpoints for vulnerabilities", "target_layer": "Access", "stride": "Information Disclosure", "adversaries": "Nation-State, Organised Crime, Hacktivist", "is_cbdc_new": False, "example_scenario": "Attacker scans API gateway endpoints to discover unprotected admin interfaces.", "controls": "Rate limiting, WAF, API authentication, endpoint obfuscation"},
    {"tactic": "Reconnaissance", "tactic_id": "TA0043", "technique_id": "T1592", "technique_name": "Gather Victim Host Info", "cbdc_application": "Profiling wallet software versions", "target_layer": "Access", "stride": "Information Disclosure", "adversaries": "Nation-State, Organised Crime", "is_cbdc_new": False, "example_scenario": "Fingerprinting wallet app versions to identify known vulnerabilities.", "controls": "Version obfuscation, runtime integrity checks"},
    {"tactic": "Reconnaissance", "tactic_id": "TA0043", "technique_id": "T1589", "technique_name": "Gather Victim Identity", "cbdc_application": "Mapping wallet addresses to real identities", "target_layer": "Access", "stride": "Information Disclosure", "adversaries": "Nation-State, Organised Crime", "is_cbdc_new": False, "example_scenario": "Correlating on-chain CBDC transactions with social media to de-anonymize users.", "controls": "Privacy-preserving design, address rotation, zero-knowledge proofs"},
    {"tactic": "Initial Access", "tactic_id": "TA0001", "technique_id": "T1566", "technique_name": "Phishing", "cbdc_application": "Credential harvesting for wallet/intermediary access", "target_layer": "Access", "stride": "Spoofing", "adversaries": "Organised Crime, End-User Fraud", "is_cbdc_new": False, "example_scenario": "Phishing emails impersonating CBDC wallet provider to steal login credentials.", "controls": "MFA, user awareness training, email filtering, domain monitoring"},
    {"tactic": "Initial Access", "tactic_id": "TA0001", "technique_id": "T1190", "technique_name": "Exploit Public-Facing Application", "cbdc_application": "Exploiting API gateway vulnerabilities", "target_layer": "Access", "stride": "Elevation of Privilege", "adversaries": "Nation-State, Organised Crime, Hacktivist", "is_cbdc_new": False, "example_scenario": "Exploiting unpatched API gateway to gain unauthorized access to backend services.", "controls": "WAF, regular patching, penetration testing, input validation"},
    {"tactic": "Initial Access", "tactic_id": "TA0001", "technique_id": "T1078", "technique_name": "Valid Accounts", "cbdc_application": "Using compromised intermediary credentials", "target_layer": "Access", "stride": "Spoofing", "adversaries": "Malicious Insider, Organised Crime", "is_cbdc_new": False, "example_scenario": "Using stolen credentials of a commercial bank operator to access wholesale network.", "controls": "MFA, privileged access management, behavioral analytics, session monitoring"},
    {"tactic": "Initial Access", "tactic_id": "TA0001", "technique_id": "CBDC-NEW-01", "technique_name": "Malicious Wallet Distribution", "cbdc_application": "Distributing trojanised wallet apps", "target_layer": "Access", "stride": "Tampering", "adversaries": "Organised Crime, Compromised Third-Party", "is_cbdc_new": True, "example_scenario": "Publishing a fake CBDC wallet app on app stores that steals private keys.", "controls": "App store verification, code signing, wallet attestation, supply chain security"},
    {"tactic": "Execution", "tactic_id": "TA0002", "technique_id": "T1059", "technique_name": "Command and Scripting", "cbdc_application": "Exploiting smart contract execution", "target_layer": "Service", "stride": "Tampering", "adversaries": "Nation-State, Organised Crime", "is_cbdc_new": False, "example_scenario": "Injecting malicious code into smart contract execution to redirect funds.", "controls": "Formal verification, code auditing, execution sandboxing, gas limits"},
    {"tactic": "Execution", "tactic_id": "TA0002", "technique_id": "CBDC-NEW-02", "technique_name": "Malicious Smart Contract Deployment", "cbdc_application": "Deploying contracts that drain funds", "target_layer": "Service", "stride": "Tampering", "adversaries": "Nation-State, Organised Crime", "is_cbdc_new": True, "example_scenario": "Deploying a contract with hidden reentrancy vulnerability to drain escrow funds.", "controls": "Contract whitelisting, formal verification, deployment approvals, upgrade controls"},
    {"tactic": "Execution", "tactic_id": "TA0002", "technique_id": "CBDC-NEW-03", "technique_name": "Oracle Data Injection", "cbdc_application": "Feeding false data to price/state oracles", "target_layer": "Service", "stride": "Tampering", "adversaries": "Nation-State, Organised Crime", "is_cbdc_new": True, "example_scenario": "Manipulating exchange rate oracle to profit from cross-border CBDC transactions.", "controls": "Multiple oracle sources, median aggregation, anomaly detection, trusted execution environments"},
    {"tactic": "Persistence", "tactic_id": "TA0003", "technique_id": "T1098", "technique_name": "Account Manipulation", "cbdc_application": "Modifying intermediary access controls", "target_layer": "Access", "stride": "Elevation of Privilege", "adversaries": "Malicious Insider, Nation-State", "is_cbdc_new": False, "example_scenario": "Modifying role assignments to elevate retail operator to wholesale access level.", "controls": "Role-based access control, audit logging, separation of duties, periodic access reviews"},
    {"tactic": "Persistence", "tactic_id": "TA0003", "technique_id": "T1136", "technique_name": "Create Account", "cbdc_application": "Creating fraudulent wallet identities", "target_layer": "Access", "stride": "Spoofing", "adversaries": "Organised Crime, End-User Fraud", "is_cbdc_new": False, "example_scenario": "Creating synthetic identities to open multiple wallets for money laundering.", "controls": "Enhanced KYC, biometric verification, identity graph analysis, velocity checks"},
    {"tactic": "Persistence", "tactic_id": "TA0003", "technique_id": "CBDC-NEW-04", "technique_name": "Consensus Node Infiltration", "cbdc_application": "Gaining persistent access to validator nodes", "target_layer": "Platform", "stride": "Tampering", "adversaries": "Nation-State", "is_cbdc_new": True, "example_scenario": "Compromising a validator node to persistently influence consensus outcomes.", "controls": "Node integrity monitoring, hardware attestation, validator rotation, Byzantine fault tolerance"},
    {"tactic": "Privilege Escalation", "tactic_id": "TA0004", "technique_id": "T1068", "technique_name": "Exploitation for Privilege Escalation", "cbdc_application": "Escalating from retail to wholesale access", "target_layer": "Platform", "stride": "Elevation of Privilege", "adversaries": "Nation-State, Organised Crime, Malicious Insider", "is_cbdc_new": False, "example_scenario": "Exploiting a vulnerability to escalate from intermediary access to central bank tier.", "controls": "Network segmentation, tier isolation, least privilege, vulnerability management"},
    {"tactic": "Privilege Escalation", "tactic_id": "TA0004", "technique_id": "CBDC-NEW-05", "technique_name": "Intermediary Role Escalation", "cbdc_application": "Exploiting two-tier boundary to gain central bank tier access", "target_layer": "Service", "stride": "Elevation of Privilege", "adversaries": "Organised Crime, Malicious Insider", "is_cbdc_new": True, "example_scenario": "Exploiting misconfigured API to access central bank minting functions from retail tier.", "controls": "Strict tier boundary enforcement, API gateway policies, zero-trust architecture"},
    {"tactic": "Defense Evasion", "tactic_id": "TA0005", "technique_id": "T1070", "technique_name": "Indicator Removal", "cbdc_application": "Erasing transaction traces", "target_layer": "Platform", "stride": "Repudiation", "adversaries": "Nation-State, Malicious Insider", "is_cbdc_new": False, "example_scenario": "Deleting or modifying audit logs to hide unauthorized minting activity.", "controls": "Immutable audit logs, write-once storage, log integrity verification, SIEM monitoring"},
    {"tactic": "Defense Evasion", "tactic_id": "TA0005", "technique_id": "T1036", "technique_name": "Masquerading", "cbdc_application": "Disguising attack transactions as legitimate payments", "target_layer": "Service", "stride": "Spoofing", "adversaries": "Organised Crime, Compromised Third-Party", "is_cbdc_new": False, "example_scenario": "Disguising money laundering transactions as legitimate merchant payments.", "controls": "Behavioral analytics, transaction pattern analysis, anomaly detection"},
    {"tactic": "Defense Evasion", "tactic_id": "TA0005", "technique_id": "CBDC-NEW-06", "technique_name": "Transaction Structuring", "cbdc_application": "Splitting amounts below AML thresholds (smurfing)", "target_layer": "Service", "stride": "Tampering", "adversaries": "Organised Crime, End-User Fraud", "is_cbdc_new": True, "example_scenario": "Breaking large transfers into many small transactions to evade AML detection rules.", "controls": "Aggregate monitoring, velocity checks, graph-based analytics, adaptive thresholds"},
    {"tactic": "Credential Access", "tactic_id": "TA0006", "technique_id": "T1110", "technique_name": "Brute Force", "cbdc_application": "Attacking wallet PINs/passwords", "target_layer": "Access", "stride": "Spoofing", "adversaries": "Organised Crime, End-User Fraud", "is_cbdc_new": False, "example_scenario": "Brute-forcing wallet PINs to gain access to user funds.", "controls": "Account lockout, rate limiting, biometric authentication, hardware tokens"},
    {"tactic": "Credential Access", "tactic_id": "TA0006", "technique_id": "T1557", "technique_name": "Adversary-in-the-Middle", "cbdc_application": "Intercepting wallet-to-intermediary communications", "target_layer": "Access", "stride": "Information Disclosure", "adversaries": "Nation-State, Organised Crime", "is_cbdc_new": False, "example_scenario": "MitM attack on wallet-to-bank API to intercept transaction signing keys.", "controls": "Certificate pinning, mutual TLS, end-to-end encryption, channel binding"},
    {"tactic": "Credential Access", "tactic_id": "TA0006", "technique_id": "CBDC-NEW-07", "technique_name": "Private Key Extraction", "cbdc_application": "Side-channel attacks on hardware wallet/KMS", "target_layer": "Platform", "stride": "Information Disclosure", "adversaries": "Nation-State", "is_cbdc_new": True, "example_scenario": "Using power analysis side-channel attack to extract KMS master key from HSM.", "controls": "Side-channel resistant HSMs, key sharding, multi-party computation, regular key rotation"},
    {"tactic": "Lateral Movement", "tactic_id": "TA0008", "technique_id": "T1021", "technique_name": "Remote Services", "cbdc_application": "Moving between intermediary systems", "target_layer": "Service", "stride": "Elevation of Privilege", "adversaries": "Nation-State, Organised Crime", "is_cbdc_new": False, "example_scenario": "Using compromised intermediary credentials to pivot to other financial institutions.", "controls": "Network segmentation, micro-segmentation, just-in-time access, behavioral monitoring"},
    {"tactic": "Lateral Movement", "tactic_id": "TA0008", "technique_id": "CBDC-NEW-08", "technique_name": "Cross-Tier Pivot", "cbdc_application": "Leveraging retail compromise to access wholesale network", "target_layer": "Platform", "stride": "Elevation of Privilege", "adversaries": "Nation-State, Organised Crime", "is_cbdc_new": True, "example_scenario": "Pivoting from compromised retail wallet provider to wholesale settlement network.", "controls": "Air-gapped tier separation, strict firewall rules, zero-trust architecture"},
    {"tactic": "Lateral Movement", "tactic_id": "TA0008", "technique_id": "CBDC-NEW-09", "technique_name": "Bridge Exploitation", "cbdc_application": "Attacking interoperability bridges to reach other CBDC platforms", "target_layer": "Service", "stride": "Tampering", "adversaries": "Nation-State, Organised Crime", "is_cbdc_new": True, "example_scenario": "Exploiting cross-border CBDC bridge to drain liquidity from foreign CBDC system.", "controls": "Bridge security audits, rate limiting, multi-sig validation, circuit breakers"},
    {"tactic": "Collection", "tactic_id": "TA0009", "technique_id": "T1005", "technique_name": "Data from Local System", "cbdc_application": "Harvesting wallet transaction history", "target_layer": "Access", "stride": "Information Disclosure", "adversaries": "Nation-State, Organised Crime, Malicious Insider", "is_cbdc_new": False, "example_scenario": "Extracting complete transaction history from compromised wallet application.", "controls": "Data encryption at rest, secure enclaves, minimal data retention, access logging"},
    {"tactic": "Collection", "tactic_id": "TA0009", "technique_id": "T1119", "technique_name": "Automated Collection", "cbdc_application": "Bulk extraction of PII from KYC databases", "target_layer": "Access", "stride": "Information Disclosure", "adversaries": "Nation-State, Organised Crime, Malicious Insider", "is_cbdc_new": False, "example_scenario": "Automated scraping of KYC database to harvest personal identity documents.", "controls": "Data loss prevention, query rate limiting, database activity monitoring, encryption"},
    {"tactic": "Impact", "tactic_id": "TA0040", "technique_id": "T1485", "technique_name": "Data Destruction", "cbdc_application": "Destroying ledger records", "target_layer": "Platform", "stride": "Denial of Service", "adversaries": "Nation-State, Hacktivist", "is_cbdc_new": False, "example_scenario": "Wiping or corrupting ledger data to disrupt national payment infrastructure.", "controls": "Immutable ledger design, geo-distributed backups, Byzantine fault tolerance, disaster recovery"},
    {"tactic": "Impact", "tactic_id": "TA0040", "technique_id": "T1489", "technique_name": "Service Stop", "cbdc_application": "DDoS on CBDC payment infrastructure", "target_layer": "Access", "stride": "Denial of Service", "adversaries": "Hacktivist, Nation-State", "is_cbdc_new": False, "example_scenario": "Massive DDoS campaign against API gateways to halt CBDC payments nationwide.", "controls": "DDoS mitigation, CDN, traffic scrubbing, auto-scaling, geographic distribution"},
    {"tactic": "Impact", "tactic_id": "TA0040", "technique_id": "CBDC-NEW-10", "technique_name": "Double Spending", "cbdc_application": "Exploiting consensus delays for double-spend", "target_layer": "Asset", "stride": "Tampering", "adversaries": "Organised Crime, Nation-State", "is_cbdc_new": True, "example_scenario": "Racing two conflicting transactions during consensus delay window to spend same funds twice.", "controls": "Finality guarantees, consensus speed optimization, double-spend proofs, real-time monitoring"},
    {"tactic": "Impact", "tactic_id": "TA0040", "technique_id": "CBDC-NEW-11", "technique_name": "Unauthorized Minting", "cbdc_application": "Exploiting issuance controls to create counterfeit CBDC", "target_layer": "Asset", "stride": "Elevation of Privilege", "adversaries": "Nation-State, Malicious Insider", "is_cbdc_new": True, "example_scenario": "Insider exploiting minting authority access to create unauthorized CBDC tokens.", "controls": "Multi-party authorization, hardware security modules, issuance audit trails, separation of duties"},
    {"tactic": "Impact", "tactic_id": "TA0040", "technique_id": "CBDC-NEW-12", "technique_name": "Offline Payment Replay", "cbdc_application": "Replaying offline payment tokens", "target_layer": "Service", "stride": "Spoofing", "adversaries": "End-User Fraud, Organised Crime", "is_cbdc_new": True, "example_scenario": "Copying and replaying offline payment tokens to spend the same value multiple times.", "controls": "Token uniqueness enforcement, reconciliation on reconnect, hardware token storage, expiry mechanisms"},
])

ADVERSARIES = {
    "Nation-State": {
        "capability": 9, "resources": 10, "sophistication": 9, "persistence": 10, "motivation_score": 9,
        "motivation": "Espionage / Disruption",
        "primary_targets": ["Platform Layer (consensus, ledger)", "Service Layer (bridges)"],
        "stride_focus": ["Tampering", "Information Disclosure", "Denial of Service"],
        "top_components": ["Consensus Nodes", "Core Ledger", "Interoperability Bridge"],
        "top_techniques": ["CBDC-NEW-04", "CBDC-NEW-07", "CBDC-NEW-08"],
        "example": "Targeting consensus nodes to disrupt national payment infrastructure",
        "layer_scores": {"Platform": 3, "Asset": 2, "Service": 3, "Access": 1},
    },
    "Organised Crime": {
        "capability": 7, "resources": 7, "sophistication": 7, "persistence": 7, "motivation_score": 8,
        "motivation": "Financial gain",
        "primary_targets": ["Asset Layer (double-spend)", "Service Layer (AML bypass)", "Access Layer (wallet theft)"],
        "stride_focus": ["Spoofing", "Tampering", "Elevation of Privilege"],
        "top_components": ["Payment Processor", "AML/CFT Engine", "Wallet Providers"],
        "top_techniques": ["CBDC-NEW-06", "CBDC-NEW-10", "T1566"],
        "example": "Structuring transactions to launder proceeds through CBDC",
        "layer_scores": {"Platform": 1, "Asset": 3, "Service": 3, "Access": 3},
    },
    "Malicious Insider": {
        "capability": 6, "resources": 4, "sophistication": 5, "persistence": 8, "motivation_score": 6,
        "motivation": "Financial gain / Coercion",
        "primary_targets": ["Platform Layer (KMS)", "Asset Layer (minting)", "Service Layer (admin)"],
        "stride_focus": ["Repudiation", "Elevation of Privilege", "Information Disclosure"],
        "top_components": ["Key Management Service (KMS)", "Minting Authority", "Smart Contract Runtime"],
        "top_techniques": ["CBDC-NEW-11", "T1098", "T1070"],
        "example": "Abusing privileged access to mint unauthorized CBDC",
        "layer_scores": {"Platform": 3, "Asset": 3, "Service": 2, "Access": 1},
    },
    "Compromised Third-Party": {
        "capability": 5, "resources": 5, "sophistication": 6, "persistence": 6, "motivation_score": 5,
        "motivation": "Supply chain attack",
        "primary_targets": ["Access Layer (wallet SDK)", "Platform Layer (cloud infra)", "Service Layer (oracle)"],
        "stride_focus": ["Tampering", "Information Disclosure"],
        "top_components": ["Wallet Providers", "Oracle Interface", "API Gateways"],
        "top_techniques": ["CBDC-NEW-01", "CBDC-NEW-03", "T1190"],
        "example": "Injecting backdoor into wallet SDK used by multiple intermediaries",
        "layer_scores": {"Platform": 2, "Asset": 1, "Service": 2, "Access": 3},
    },
    "End-User Fraud": {
        "capability": 2, "resources": 1, "sophistication": 2, "persistence": 3, "motivation_score": 4,
        "motivation": "Petty fraud",
        "primary_targets": ["Access Layer (own wallet)", "Service Layer (offline payments)"],
        "stride_focus": ["Spoofing", "Repudiation"],
        "top_components": ["Wallet Providers", "Offline Payment Module", "Mobile Apps"],
        "top_techniques": ["CBDC-NEW-12", "T1110", "T1136"],
        "example": "Attempting offline payment replay attacks",
        "layer_scores": {"Platform": 0, "Asset": 1, "Service": 2, "Access": 3},
    },
    "Hacktivist": {
        "capability": 4, "resources": 3, "sophistication": 4, "persistence": 5, "motivation_score": 7,
        "motivation": "Disruption / Ideology",
        "primary_targets": ["Access Layer (DDoS on wallets)", "Platform Layer (DDoS on nodes)"],
        "stride_focus": ["Denial of Service"],
        "top_components": ["API Gateways", "Consensus Nodes", "Web Portals"],
        "top_techniques": ["T1489", "T1485", "T1595"],
        "example": "DDoS campaign against CBDC infrastructure to protest surveillance",
        "layer_scores": {"Platform": 2, "Asset": 0, "Service": 1, "Access": 3},
    },
}

RISK_REGISTER = pd.DataFrame([
    {"risk_id": "R-001", "layer": "Platform", "component": "Core Ledger", "threat": "Ledger tampering via compromised validator node", "stride": "Tampering", "mitre": "CBDC-NEW-04", "adversary": "Nation-State", "likelihood": 2, "impact": 5, "controls": "Byzantine fault tolerance, immutable audit logs, node integrity monitoring", "control_effectiveness": 4},
    {"risk_id": "R-002", "layer": "Platform", "component": "Consensus Nodes", "threat": "51% attack on consensus mechanism", "stride": "Tampering", "mitre": "CBDC-NEW-04", "adversary": "Nation-State", "likelihood": 1, "impact": 5, "controls": "Permissioned consensus, validator vetting, geographic distribution", "control_effectiveness": 5},
    {"risk_id": "R-003", "layer": "Platform", "component": "Key Management Service (KMS)", "threat": "Private key extraction via side-channel attack", "stride": "Information Disclosure", "mitre": "CBDC-NEW-07", "adversary": "Nation-State", "likelihood": 2, "impact": 5, "controls": "Side-channel resistant HSMs, key sharding, multi-party computation", "control_effectiveness": 4},
    {"risk_id": "R-004", "layer": "Platform", "component": "Settlement Engine", "threat": "Settlement manipulation causing financial loss", "stride": "Tampering", "mitre": "T1485", "adversary": "Nation-State", "likelihood": 2, "impact": 5, "controls": "Real-time reconciliation, dual authorization, audit trails", "control_effectiveness": 4},
    {"risk_id": "R-005", "layer": "Asset", "component": "Minting Authority", "threat": "Unauthorized CBDC minting by insider", "stride": "Elevation of Privilege", "mitre": "CBDC-NEW-11", "adversary": "Malicious Insider", "likelihood": 2, "impact": 5, "controls": "Multi-party authorization, HSM-backed signing, issuance audit trail", "control_effectiveness": 4},
    {"risk_id": "R-006", "layer": "Asset", "component": "Token/Account State", "threat": "Double-spending via consensus delay exploitation", "stride": "Tampering", "mitre": "CBDC-NEW-10", "adversary": "Organised Crime", "likelihood": 3, "impact": 4, "controls": "Finality guarantees, double-spend proofs, real-time monitoring", "control_effectiveness": 3},
    {"risk_id": "R-007", "layer": "Asset", "component": "UTXO Store", "threat": "UTXO replay attack causing double-spend", "stride": "Tampering", "mitre": "CBDC-NEW-10", "adversary": "Organised Crime", "likelihood": 2, "impact": 4, "controls": "Nonce enforcement, state validation, transaction ordering", "control_effectiveness": 4},
    {"risk_id": "R-008", "layer": "Asset", "component": "Asset Lifecycle Manager", "threat": "Lifecycle bypass allowing counterfeit tokens", "stride": "Elevation of Privilege", "mitre": "CBDC-NEW-11", "adversary": "Malicious Insider", "likelihood": 2, "impact": 5, "controls": "Lifecycle state machine enforcement, separation of duties, code review", "control_effectiveness": 3},
    {"risk_id": "R-009", "layer": "Service", "component": "Payment Processor", "threat": "Transaction routing manipulation for fraud", "stride": "Tampering", "mitre": "T1036", "adversary": "Organised Crime", "likelihood": 3, "impact": 4, "controls": "Transaction integrity checks, behavioral analytics, real-time monitoring", "control_effectiveness": 3},
    {"risk_id": "R-010", "layer": "Service", "component": "AML/CFT Engine", "threat": "AML threshold evasion via transaction structuring", "stride": "Tampering", "mitre": "CBDC-NEW-06", "adversary": "Organised Crime", "likelihood": 4, "impact": 3, "controls": "Aggregate monitoring, graph analytics, adaptive thresholds, velocity checks", "control_effectiveness": 3},
    {"risk_id": "R-011", "layer": "Service", "component": "Smart Contract Runtime", "threat": "Reentrancy exploit draining escrow funds", "stride": "Tampering", "mitre": "CBDC-NEW-02", "adversary": "Organised Crime", "likelihood": 3, "impact": 4, "controls": "Formal verification, reentrancy guards, contract auditing, deployment approvals", "control_effectiveness": 4},
    {"risk_id": "R-012", "layer": "Service", "component": "Interoperability Bridge", "threat": "Cross-chain bridge exploit draining liquidity", "stride": "Tampering", "mitre": "CBDC-NEW-09", "adversary": "Nation-State", "likelihood": 3, "impact": 5, "controls": "Multi-sig validation, rate limiting, circuit breakers, bridge audits", "control_effectiveness": 3},
    {"risk_id": "R-013", "layer": "Service", "component": "Offline Payment Module", "threat": "Offline payment token replay attack", "stride": "Spoofing", "mitre": "CBDC-NEW-12", "adversary": "End-User Fraud", "likelihood": 4, "impact": 2, "controls": "Token uniqueness enforcement, hardware storage, reconciliation on reconnect", "control_effectiveness": 3},
    {"risk_id": "R-014", "layer": "Service", "component": "Oracle Interface", "threat": "Oracle data manipulation affecting smart contracts", "stride": "Tampering", "mitre": "CBDC-NEW-03", "adversary": "Compromised Third-Party", "likelihood": 3, "impact": 4, "controls": "Multiple oracle sources, median aggregation, anomaly detection", "control_effectiveness": 3},
    {"risk_id": "R-015", "layer": "Access", "component": "Wallet Providers", "threat": "Malicious wallet app distributing trojanised software", "stride": "Tampering", "mitre": "CBDC-NEW-01", "adversary": "Compromised Third-Party", "likelihood": 3, "impact": 4, "controls": "App store verification, code signing, wallet attestation, supply chain security", "control_effectiveness": 3},
    {"risk_id": "R-016", "layer": "Access", "component": "API Gateways", "threat": "DDoS attack halting CBDC payment services", "stride": "Denial of Service", "mitre": "T1489", "adversary": "Hacktivist", "likelihood": 4, "impact": 4, "controls": "DDoS mitigation, CDN, auto-scaling, traffic scrubbing, geographic distribution", "control_effectiveness": 4},
    {"risk_id": "R-017", "layer": "Access", "component": "KYC/IAM System", "threat": "Synthetic identity fraud for wallet creation", "stride": "Spoofing", "mitre": "T1136", "adversary": "Organised Crime", "likelihood": 4, "impact": 3, "controls": "Enhanced KYC, biometric verification, identity graph analysis, velocity checks", "control_effectiveness": 3},
    {"risk_id": "R-018", "layer": "Access", "component": "Mobile Apps", "threat": "Credential theft via phishing campaign", "stride": "Spoofing", "mitre": "T1566", "adversary": "Organised Crime", "likelihood": 4, "impact": 3, "controls": "MFA, user awareness training, email filtering, domain monitoring", "control_effectiveness": 3},
    {"risk_id": "R-019", "layer": "Access", "component": "Web Portals", "threat": "Session hijacking of intermediary admin portal", "stride": "Spoofing", "mitre": "T1190", "adversary": "Organised Crime", "likelihood": 3, "impact": 4, "controls": "Secure session management, CSP headers, CSRF protection, short session timeouts", "control_effectiveness": 4},
    {"risk_id": "R-020", "layer": "Platform", "component": "Core Ledger", "threat": "Data destruction wiping ledger records", "stride": "Denial of Service", "mitre": "T1485", "adversary": "Nation-State", "likelihood": 1, "impact": 5, "controls": "Geo-distributed backups, immutable storage, disaster recovery, BFT consensus", "control_effectiveness": 5},
    {"risk_id": "R-021", "layer": "Access", "component": "Wallet Providers", "threat": "Brute force attack on wallet PINs", "stride": "Spoofing", "mitre": "T1110", "adversary": "End-User Fraud", "likelihood": 3, "impact": 2, "controls": "Account lockout, rate limiting, biometric auth, hardware tokens", "control_effectiveness": 4},
    {"risk_id": "R-022", "layer": "Service", "component": "Payment Processor", "threat": "Masquerading attack transactions as legitimate payments", "stride": "Spoofing", "mitre": "T1036", "adversary": "Organised Crime", "likelihood": 3, "impact": 3, "controls": "Behavioral analytics, transaction pattern analysis, anomaly detection", "control_effectiveness": 3},
    {"risk_id": "R-023", "layer": "Access", "component": "KYC/IAM System", "threat": "Bulk extraction of PII from KYC database", "stride": "Information Disclosure", "mitre": "T1119", "adversary": "Nation-State", "likelihood": 2, "impact": 5, "controls": "Data loss prevention, query rate limiting, database activity monitoring, encryption", "control_effectiveness": 4},
    {"risk_id": "R-024", "layer": "Platform", "component": "Consensus Nodes", "threat": "Cross-tier pivot from retail to wholesale network", "stride": "Elevation of Privilege", "mitre": "CBDC-NEW-08", "adversary": "Nation-State", "likelihood": 2, "impact": 5, "controls": "Air-gapped tier separation, strict firewall rules, zero-trust architecture", "control_effectiveness": 4},
    {"risk_id": "R-025", "layer": "Access", "component": "Aggregators", "threat": "Supply chain compromise of aggregator APIs", "stride": "Tampering", "mitre": "T1190", "adversary": "Compromised Third-Party", "likelihood": 3, "impact": 3, "controls": "API security testing, vendor assessments, input validation, WAF", "control_effectiveness": 3},
])
RISK_REGISTER["inherent_risk"] = RISK_REGISTER["likelihood"] * RISK_REGISTER["impact"]
RISK_REGISTER["residual_risk"] = (
    RISK_REGISTER["inherent_risk"]
    * (6 - RISK_REGISTER["control_effectiveness"]) / 5
).round(1)

# Simulation process → ASAP layer mapping
PROCESS_LAYER_MAP = {
    "P1_issuance": ["Asset"],
    "P2_transfer": ["Asset", "Service"],
    "P3_redemption": ["Asset"],
    "P4_kyc": ["Access"],
    "P5_risk_check": ["Service"],
    "L1_ledger": ["Platform"],
    "L2_consensus": ["Platform"],
    "L3_api": ["Access"],
    "L4_wallet": ["Access"],
}

# Agent → Adversary mapping for simulation link
AGENT_ADVERSARY_MAP = {
    "Q-Learning": "End-User Fraud",
    "DQN": "Organised Crime",
    "REINFORCE": "Compromised Third-Party",
    "A2C": "Nation-State",
}

TACTIC_ORDER = [
    "Reconnaissance", "Initial Access", "Execution", "Persistence",
    "Privilege Escalation", "Defense Evasion", "Credential Access",
    "Lateral Movement", "Collection", "Impact",
]

STRIDE_CATEGORIES = [
    "Spoofing", "Tampering", "Repudiation",
    "Information Disclosure", "Denial of Service", "Elevation of Privilege",
]


# ═══════════════════════════════════════════════════════════════════════════
# TAB 8: CBDC Architecture
# ═══════════════════════════════════════════════════════════════════════════
with tab8:
    st.header("CBDC Architecture — IMF ASAP 4-Layer Model")
    st.write(
        "Interactive view of the IMF ASAP reference architecture applied to "
        "a **two-tier CBDC ecosystem** (Central Bank wholesale + Intermediary/Retail)."
    )

    # ── 1. ASAP Layer Diagram ─────────────────────────────────────────────
    st.subheader("ASAP Layer Diagram")

    layer_names = ["Platform", "Asset", "Service", "Access"]
    layer_colors = [ASAP_LAYERS[l]["color"] for l in layer_names]
    layer_funcs = [ASAP_LAYERS[l]["functions"][:80] + "…" for l in layer_names]
    layer_components = [", ".join(ASAP_LAYERS[l]["components"]) for l in layer_names]
    layer_tiers = [ASAP_LAYERS[l]["tier"] for l in layer_names]

    fig = go.Figure()
    for i, layer in enumerate(layer_names):
        fig.add_trace(go.Bar(
            x=[1],
            y=[1],
            name=layer,
            marker_color=ASAP_LAYERS[layer]["color"],
            text=f"<b>{layer} Layer</b><br>{layer_components[i]}",
            textposition="inside",
            insidetextanchor="middle",
            hovertext=(
                f"<b>{layer} Layer</b><br>"
                f"Tier: {layer_tiers[i]}<br>"
                f"Functions: {ASAP_LAYERS[layer]['functions']}<br>"
                f"Components: {layer_components[i]}"
            ),
            hoverinfo="text",
            textfont=dict(size=13, color="white"),
        ))

    fig.update_layout(
        barmode="stack",
        showlegend=False,
        height=450,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(
            showticklabels=True, showgrid=False, zeroline=False,
            tickvals=[0.5, 1.5, 2.5, 3.5],
            ticktext=["Platform (L0)", "Asset (L1)", "Service (L2)", "Access (L3)"],
        ),
        title="IMF ASAP 4-Layer Architecture (Bottom → Top)",
        margin=dict(l=120),
    )

    # Tier annotations
    fig.add_annotation(x=1.15, y=1, text="<b>Central Bank<br>Tier</b>",
                       showarrow=False, xanchor="left", font=dict(size=11, color="#636EFA"))
    fig.add_annotation(x=1.15, y=3.5, text="<b>Intermediary /<br>Retail Tier</b>",
                       showarrow=False, xanchor="left", font=dict(size=11, color="#AB63FA"))
    fig.add_annotation(x=1.15, y=2.25, text="<b>Both Tiers</b>",
                       showarrow=False, xanchor="left", font=dict(size=11, color="#00CC96"))

    st.plotly_chart(fig, use_container_width=True)

    # ── Two-tier model summary ───────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        st.markdown("#### Tier 1 — Central Bank (Wholesale)")
        st.markdown(
            "**Components:** Core Ledger, RTGS Interface, wCBDC DLT, "
            "Central Bank Server, Validator Infrastructure\n\n"
            "**Role:** Issues CBDC, manages lifecycle, runs wholesale settlement"
        )
    with right:
        st.markdown("#### Tier 2 — Intermediaries & End Users (Retail)")
        st.markdown(
            "**Components:** Commercial Bank Servers, Wallet Providers, "
            "KYC Systems, Retail e-Wallet, Merchant Payment Terminals\n\n"
            "**Role:** Handles KYC, manages retail wallets, processes retail transactions"
        )

    # ── 2. Component Explorer ────────────────────────────────────────────
    st.subheader("Component Explorer")
    all_components = sorted(COMPONENT_DETAILS.keys())
    selected_component = st.selectbox("Select a component", all_components, key="arch_component")

    comp = COMPONENT_DETAILS[selected_component]
    c1, c2, c3 = st.columns(3)
    c1.metric("ASAP Layer", comp["layer"])
    c2.metric("Tier", comp["tier"])
    c3.metric("STRIDE Categories", str(len(comp["stride"])))

    st.markdown(f"**Functions:** {comp['functions']}")
    st.markdown(f"**Attack Surface:** {comp['attack_surface']}")
    st.markdown(f"**STRIDE:** {', '.join(comp['stride'])}")
    st.markdown(f"**MITRE Techniques:** {', '.join(comp['mitre'])}")
    st.markdown(f"**Connected Components:** {' → '.join(comp['connected'])}")

    # Show relevant techniques
    comp_techniques = MITRE_TECHNIQUES[
        MITRE_TECHNIQUES["technique_id"].isin(comp["mitre"])
    ][["technique_id", "technique_name", "tactic", "cbdc_application"]]
    if not comp_techniques.empty:
        st.dataframe(comp_techniques, use_container_width=True, hide_index=True)

    # ── 3. Simulation Mapping (CSV only) ─────────────────────────────────
    if has_csv and not csv_empty:
        st.subheader("Simulation → ASAP Layer Mapping")
        sim_layer_counts = {}
        for _, row in filtered.iterrows():
            proc = str(row.get("process", ""))
            layers = PROCESS_LAYER_MAP.get(proc, [])
            for layer in layers:
                sim_layer_counts[layer] = sim_layer_counts.get(layer, 0) + 1

        if sim_layer_counts:
            sim_df = pd.DataFrame(
                list(sim_layer_counts.items()), columns=["ASAP Layer", "Event Count"]
            ).sort_values("Event Count", ascending=False)
            fig = px.bar(
                sim_df, x="ASAP Layer", y="Event Count",
                color="ASAP Layer",
                color_discrete_map={l: ASAP_LAYERS[l]["color"] for l in ASAP_LAYERS},
                title="Simulation Events by ASAP Layer",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No process mappings found in the uploaded data.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 9: Threat Catalogue
# ═══════════════════════════════════════════════════════════════════════════
with tab9:
    st.header("Threat Catalogue — MITRE ATT&CK for CBDC")
    st.write(
        "Structured mapping of **MITRE ATT&CK techniques** adapted for CBDC threat modelling, "
        "including CBDC-specific extensions (CBDC-NEW-xx)."
    )

    # ── 1. Kill Chain View ────────────────────────────────────────────────
    st.subheader("Kill Chain View")
    tactic_counts = MITRE_TECHNIQUES.groupby("tactic").agg(
        total=("technique_id", "size"),
        cbdc_new=("is_cbdc_new", "sum"),
        standard=("is_cbdc_new", lambda x: (~x).sum()),
    ).reindex(TACTIC_ORDER).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tactic_counts["tactic"], y=tactic_counts["standard"],
        name="Standard MITRE", marker_color="#636EFA",
    ))
    fig.add_trace(go.Bar(
        x=tactic_counts["tactic"], y=tactic_counts["cbdc_new"],
        name="CBDC-Specific (NEW)", marker_color="#EF553B",
    ))
    fig.update_layout(
        barmode="stack", title="MITRE ATT&CK Kill Chain — Technique Counts by Tactic",
        xaxis_title="Tactic (Kill Chain Order)", yaxis_title="Technique Count",
        xaxis_tickangle=-35, height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 2. Technique Table ────────────────────────────────────────────────
    st.subheader("Technique Table")
    display_techniques = MITRE_TECHNIQUES[[
        "tactic", "technique_id", "technique_name", "cbdc_application",
        "target_layer", "stride", "adversaries",
    ]].copy()
    display_techniques.columns = [
        "Tactic", "ID", "Name", "CBDC Application",
        "Target Layer", "STRIDE", "Adversary Classes",
    ]

    tactic_filter = st.multiselect(
        "Filter by tactic", TACTIC_ORDER, default=TACTIC_ORDER, key="tc_tactic_filter"
    )
    layer_filter = st.multiselect(
        "Filter by target layer", ["Platform", "Asset", "Service", "Access"],
        default=["Platform", "Asset", "Service", "Access"], key="tc_layer_filter"
    )

    filtered_techniques = display_techniques[
        display_techniques["Tactic"].isin(tactic_filter)
        & display_techniques["Target Layer"].isin(layer_filter)
    ]
    st.dataframe(filtered_techniques, use_container_width=True, hide_index=True, height=400)

    # ── 3. STRIDE × MITRE Cross-Reference Heatmap ────────────────────────
    st.subheader("STRIDE × MITRE Tactic Heatmap")
    cross_ref = MITRE_TECHNIQUES.groupby(["stride", "tactic"]).size().reset_index(name="count")
    cross_pivot = cross_ref.pivot(index="stride", columns="tactic", values="count").fillna(0)
    cross_pivot = cross_pivot.reindex(index=STRIDE_CATEGORIES, columns=TACTIC_ORDER, fill_value=0)

    fig = px.imshow(
        cross_pivot.values,
        x=TACTIC_ORDER, y=STRIDE_CATEGORIES,
        text_auto=True, aspect="auto",
        color_continuous_scale="YlOrRd",
        title="STRIDE Category × MITRE Tactic (Technique Count)",
        labels=dict(color="Count"),
    )
    fig.update_layout(xaxis_tickangle=-35, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ── 4. Technique Detail ──────────────────────────────────────────────
    st.subheader("Technique Detail")
    tech_options = MITRE_TECHNIQUES["technique_id"] + " — " + MITRE_TECHNIQUES["technique_name"]
    selected_tech = st.selectbox("Select technique", tech_options.tolist(), key="tc_detail")
    tech_id = selected_tech.split(" — ")[0]
    tech_row = MITRE_TECHNIQUES[MITRE_TECHNIQUES["technique_id"] == tech_id].iloc[0]

    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Tactic", tech_row["tactic"])
    tc2.metric("Target Layer", tech_row["target_layer"])
    tc3.metric("CBDC-Specific", "Yes" if tech_row["is_cbdc_new"] else "No")

    st.markdown(f"**CBDC Application:** {tech_row['cbdc_application']}")
    st.markdown(f"**STRIDE Category:** {tech_row['stride']}")
    st.markdown(f"**Adversary Classes:** {tech_row['adversaries']}")
    st.markdown(f"**Example Scenario:** {tech_row['example_scenario']}")
    st.markdown(f"**Recommended Controls:** {tech_row['controls']}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 10: Adversary Profiles
# ═══════════════════════════════════════════════════════════════════════════
with tab10:
    st.header("Adversary Profiles — Threat Actor Intelligence")
    st.write(
        "Interactive profiles of **6 adversary classes** targeting CBDC infrastructure, "
        "with capability assessments, target preferences, and STRIDE analysis."
    )

    adv_names = list(ADVERSARIES.keys())
    radar_attrs = ["capability", "resources", "sophistication", "persistence", "motivation_score"]
    radar_labels = ["Capability", "Resources", "Sophistication", "Persistence", "Motivation"]

    # ── 1. Adversary Radar Charts ─────────────────────────────────────────
    st.subheader("Adversary Capability Radar")
    selected_adversaries = st.multiselect(
        "Select adversaries to compare", adv_names, default=adv_names, key="adv_radar_select"
    )

    if selected_adversaries:
        fig = go.Figure()
        adv_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
        for i, adv in enumerate(selected_adversaries):
            vals = [ADVERSARIES[adv][a] for a in radar_attrs]
            vals.append(vals[0])  # close the polygon
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                name=adv,
                line_color=adv_colors[i % len(adv_colors)],
                opacity=0.7,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            title="Adversary Capability Comparison",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── 2. Target Preference Matrix ──────────────────────────────────────
    st.subheader("Target Preference Matrix (Adversary × ASAP Layer)")
    layer_names_list = ["Platform", "Asset", "Service", "Access"]
    target_matrix = []
    for adv in adv_names:
        row = [ADVERSARIES[adv]["layer_scores"][l] for l in layer_names_list]
        target_matrix.append(row)
    target_arr = np.array(target_matrix)

    fig = px.imshow(
        target_arr,
        x=layer_names_list, y=adv_names,
        text_auto=True, aspect="auto",
        color_continuous_scale="YlOrRd",
        title="Adversary × ASAP Layer Target Preference (3=High, 2=Medium, 1=Low, 0=None)",
        labels=dict(color="Score"),
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    # ── 3. STRIDE Preference by Adversary ────────────────────────────────
    st.subheader("STRIDE Preference by Adversary")
    stride_pref_rows = []
    for adv in adv_names:
        for s in ADVERSARIES[adv]["stride_focus"]:
            stride_pref_rows.append({"Adversary": adv, "STRIDE": s, "count": 1})
    stride_pref_df = pd.DataFrame(stride_pref_rows)
    if not stride_pref_df.empty:
        stride_agg = stride_pref_df.groupby(["Adversary", "STRIDE"])["count"].sum().reset_index()
        fig = px.bar(
            stride_agg, x="Adversary", y="count", color="STRIDE",
            barmode="group", title="STRIDE Category Focus by Adversary",
            color_discrete_map=STRIDE_COLOURS,
        )
        fig.update_layout(yaxis_title="Focus Count")
        st.plotly_chart(fig, use_container_width=True)

    # ── 4. Adversary Summary Cards ───────────────────────────────────────
    st.subheader("Adversary Summary Cards")
    card_cols = st.columns(3)
    for idx, adv in enumerate(adv_names):
        data = ADVERSARIES[adv]
        with card_cols[idx % 3]:
            st.markdown(f"#### {adv}")
            st.metric("Motivation", data["motivation"])
            st.markdown(f"**Top Targets:** {', '.join(data['top_components'][:3])}")
            st.markdown(f"**Top Techniques:** {', '.join(data['top_techniques'][:3])}")
            st.markdown(f"**Example:** {data['example']}")
            st.markdown("---")

    # ── 5. Simulation Link (CSV only) ────────────────────────────────────
    if has_csv and not csv_empty:
        st.subheader("Simulation → Adversary Profile Mapping")
        st.write(
            "Maps RL agents to adversary profiles based on sophistication: "
            "Q-Learning → End-User Fraud, DQN → Organised Crime, "
            "REINFORCE → Compromised Third-Party, A2C → Nation-State."
        )

        sim_adv_rows = []
        for agent in RL_AGENTS:
            agent_data = rl_df[rl_df["agent_id"] == agent]
            if agent_data.empty:
                continue
            adv_name = AGENT_ADVERSARY_MAP.get(agent, "Unknown")
            adv_profile = ADVERSARIES.get(adv_name, {})
            sim_adv_rows.append({
                "RL Agent": agent,
                "Mapped Adversary": adv_name,
                "Sim Events": len(agent_data),
                "Avg Amount": round(agent_data["amount"].mean(), 1) if agent_data["amount"].notna().any() else 0,
                "Adversary Capability": adv_profile.get("capability", 0),
                "Adversary Sophistication": adv_profile.get("sophistication", 0),
            })

        if sim_adv_rows:
            sim_adv_df = pd.DataFrame(sim_adv_rows)
            st.dataframe(sim_adv_df, use_container_width=True, hide_index=True)

            fig = px.bar(
                sim_adv_df, x="RL Agent", y="Sim Events",
                color="Mapped Adversary", title="Simulation Events by Agent → Adversary Mapping",
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 11: Risk Register
# ═══════════════════════════════════════════════════════════════════════════
with tab11:
    st.header("Risk Register — Quantitative Risk Assessment")
    st.write(
        "Comprehensive risk register with **25 risk entries** covering all 4 ASAP layers. "
        "Includes inherent/residual risk scoring and interactive risk calculator."
    )

    # Work with a session-state copy for interactive edits
    if "risk_register" not in st.session_state:
        st.session_state["risk_register"] = RISK_REGISTER.copy()
    rr = st.session_state["risk_register"]

    # ── 1. Risk Heatmap (5×5) ─────────────────────────────────────────────
    st.subheader("Risk Heatmap (Likelihood × Impact)")
    heatmap_matrix = np.zeros((5, 5), dtype=int)
    for _, row in rr.iterrows():
        li = int(row["likelihood"]) - 1
        ii = int(row["impact"]) - 1
        heatmap_matrix[li][ii] += 1

    # Color scale: green→yellow→orange→red
    risk_score_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            risk_score_matrix[i][j] = (i + 1) * (j + 1)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_matrix,
        x=["1 - Negligible", "2 - Minor", "3 - Moderate", "4 - Major", "5 - Catastrophic"],
        y=["1 - Rare", "2 - Unlikely", "3 - Possible", "4 - Likely", "5 - Almost Certain"],
        text=heatmap_matrix,
        texttemplate="%{text}",
        textfont=dict(size=16, color="white"),
        colorscale=[
            [0.0, "#2ecc71"], [0.25, "#f1c40f"],
            [0.5, "#e67e22"], [0.75, "#e74c3c"], [1.0, "#c0392b"],
        ],
        showscale=True,
        colorbar=dict(title="Count"),
    ))
    fig.update_layout(
        title="5×5 Risk Heatmap — Number of Risks per Cell",
        xaxis_title="Impact", yaxis_title="Likelihood",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 2. Risk Register Table ───────────────────────────────────────────
    st.subheader("Full Risk Register")
    display_rr = rr[[
        "risk_id", "layer", "component", "threat", "stride", "mitre",
        "adversary", "likelihood", "impact", "inherent_risk",
        "controls", "control_effectiveness", "residual_risk",
    ]].copy()
    display_rr.columns = [
        "Risk ID", "ASAP Layer", "Component", "Threat", "STRIDE", "MITRE",
        "Adversary", "Likelihood", "Impact", "Inherent Risk",
        "Controls", "Control Eff.", "Residual Risk",
    ]

    rr_layer_filter = st.multiselect(
        "Filter by ASAP Layer", ["Platform", "Asset", "Service", "Access"],
        default=["Platform", "Asset", "Service", "Access"], key="rr_layer_filter"
    )
    filtered_rr = display_rr[display_rr["ASAP Layer"].isin(rr_layer_filter)]
    st.dataframe(
        filtered_rr.sort_values("Inherent Risk", ascending=False),
        use_container_width=True, hide_index=True, height=400,
    )

    # ── 3. Risk by Layer ─────────────────────────────────────────────────
    st.subheader("Risk by ASAP Layer")
    layer_risk = rr.groupby("layer").agg(
        inherent=("inherent_risk", "sum"),
        residual=("residual_risk", "sum"),
    ).reindex(["Platform", "Asset", "Service", "Access"]).reset_index()
    layer_risk.columns = ["Layer", "Inherent Risk", "Residual Risk"]

    layer_melt = layer_risk.melt(id_vars="Layer", var_name="Type", value_name="Score")
    fig = px.bar(
        layer_melt, x="Layer", y="Score", color="Type",
        barmode="group", title="Aggregate Risk Score by ASAP Layer",
        color_discrete_map={"Inherent Risk": "#EF553B", "Residual Risk": "#636EFA"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 4. Control Effectiveness Analysis ────────────────────────────────
    st.subheader("Control Effectiveness Analysis")
    fig = px.scatter(
        rr, x="inherent_risk", y="residual_risk",
        color="layer", size=[3] * len(rr),
        color_discrete_map={l: ASAP_LAYERS[l]["color"] for l in ASAP_LAYERS},
        title="Inherent vs Residual Risk (below diagonal = effective controls)",
        labels={"inherent_risk": "Inherent Risk", "residual_risk": "Residual Risk", "layer": "ASAP Layer"},
        hover_data=["risk_id", "component", "threat"],
    )
    # Diagonal reference line
    max_val = max(rr["inherent_risk"].max(), rr["residual_risk"].max()) + 1
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(dash="dash", color="gray"),
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ── 5. Top 10 Risks ─────────────────────────────────────────────────
    st.subheader("Top 10 Highest Residual Risks")
    top10 = rr.nlargest(10, "residual_risk")[
        ["risk_id", "layer", "component", "threat", "inherent_risk",
         "residual_risk", "controls"]
    ].copy()
    top10.columns = [
        "Risk ID", "Layer", "Component", "Threat",
        "Inherent Risk", "Residual Risk", "Current Controls",
    ]
    st.dataframe(top10, use_container_width=True, hide_index=True)

    # ── 6. Interactive Risk Calculator ───────────────────────────────────
    st.subheader("Interactive Risk Calculator")
    st.write("Adjust likelihood and impact for any risk entry to see recalculated scores.")

    calc_risk = st.selectbox(
        "Select risk entry",
        rr["risk_id"].tolist(),
        format_func=lambda x: f"{x} — {rr[rr['risk_id']==x]['threat'].iloc[0][:60]}",
        key="risk_calc_select",
    )
    calc_row = rr[rr["risk_id"] == calc_risk].iloc[0]

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        new_likelihood = st.slider(
            "Likelihood", 1, 5, int(calc_row["likelihood"]),
            key="calc_likelihood",
            help="1=Rare, 2=Unlikely, 3=Possible, 4=Likely, 5=Almost Certain",
        )
    with rc2:
        new_impact = st.slider(
            "Impact", 1, 5, int(calc_row["impact"]),
            key="calc_impact",
            help="1=Negligible, 2=Minor, 3=Moderate, 4=Major, 5=Catastrophic",
        )
    with rc3:
        new_ctrl_eff = st.slider(
            "Control Effectiveness", 1, 5, int(calc_row["control_effectiveness"]),
            key="calc_ctrl_eff",
        )

    new_inherent = new_likelihood * new_impact
    new_residual = round(new_inherent * (6 - new_ctrl_eff) / 5, 1)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("New Inherent Risk", new_inherent, f"{new_inherent - calc_row['inherent_risk']:+.0f}")
    mc2.metric("New Residual Risk", new_residual, f"{new_residual - calc_row['residual_risk']:+.1f}")

    # Risk level label
    if new_inherent <= 4:
        risk_level = "LOW"
        risk_color = "green"
    elif new_inherent <= 9:
        risk_level = "MEDIUM"
        risk_color = "orange"
    elif new_inherent <= 15:
        risk_level = "HIGH"
        risk_color = "red"
    else:
        risk_level = "CRITICAL"
        risk_color = "darkred"
    mc3.metric("Risk Level", risk_level)
    mc4.metric("Original Inherent", int(calc_row["inherent_risk"]))

    # Update session state if values changed
    if (new_likelihood != calc_row["likelihood"] or
            new_impact != calc_row["impact"] or
            new_ctrl_eff != calc_row["control_effectiveness"]):
        idx = rr.index[rr["risk_id"] == calc_risk][0]
        st.session_state["risk_register"].at[idx, "likelihood"] = new_likelihood
        st.session_state["risk_register"].at[idx, "impact"] = new_impact
        st.session_state["risk_register"].at[idx, "control_effectiveness"] = new_ctrl_eff
        st.session_state["risk_register"].at[idx, "inherent_risk"] = new_inherent
        st.session_state["risk_register"].at[idx, "residual_risk"] = new_residual


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "Run with: `streamlit run streamlit_app.py`  |  "
    "Upload the `cbdc_logs.csv` exported from your notebook."
)
