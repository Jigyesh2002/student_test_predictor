"""
app.py — Streamlit frontend for the Student Score Predictor.

5 tabs:
  1. 📊 Data Overview
  2. 🔗 Correlations
  3. 🤖 Model Metrics
  4. 🎯 Predict Score
  5. 📈 What-If Analysis
"""

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "https://student-test-predictor-3.onrender.com"

# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def call_api(endpoint: str, method: str = "GET", payload: dict | None = None):
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=payload, timeout=15)
        else:
            resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
health = call_api("/health")
if health is None:
    st.error(
        "⚠️ Cannot reach the backend at **http://localhost:8000**. "
        "Please start it first:\n\n"
        "```bash\ncd student_score_predictor\n"
        "uvicorn backend.main:app --reload --port 8000\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Cached API fetchers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_stats():
    return call_api("/stats")

@st.cache_data(show_spinner=False)
def fetch_metrics():
    return call_api("/metrics")

@st.cache_data(show_spinner=False)
def fetch_correlation():
    return call_api("/correlation")

@st.cache_data(show_spinner=False)
def fetch_scatter():
    return call_api("/scatter-data")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 Student Score Predictor")
    st.markdown(
        """
        **Predicts a student's final exam score** based on:
        - 📚 Study hours per day
        - 😴 Sleep hours per day

        **Models available:**
        - Linear Regression
        - Random Forest

        **Dataset:** Enhanced Student Habits & Performance Dataset
        """
    )
    st.divider()
    st.caption("Backend: FastAPI  |  Frontend: Streamlit  |  Charts: Plotly")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview",
    "🔗 Correlations",
    "🤖 Model Metrics",
    "🎯 Predict Score",
    "📈 What-If Analysis",
])

# ============================================================
# TAB 1 — Data Overview
# ============================================================
with tab1:
    st.header("📊 Dataset Overview")
    st.markdown("Key statistics for the three variables used in modelling.")

    stats = fetch_stats()
    scatter_rows = fetch_scatter()

    if stats:
        # Descriptive stats table
        cols_display = ["study_hours_per_day", "sleep_hours", "exam_score"]
        stat_labels = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        table = {
            col: {k: round(stats[col][k], 3) for k in stat_labels if k in stats[col]}
            for col in cols_display
        }
        st.dataframe(
            pd.DataFrame(table).T.rename_axis("Variable"),
            use_container_width=True,
        )

    if scatter_rows:
        df = pd.DataFrame(scatter_rows)
        col1, col2, col3 = st.columns(3)

        with col1:
            fig = px.histogram(
                df, x="study_hours", nbins=30,
                title="Study Hours Distribution",
                color_discrete_sequence=["#3b82d4"],
                labels={"study_hours": "Study Hours / Day"},
            )
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                df, x="sleep_hours", nbins=20,
                title="Sleep Hours Distribution",
                color_discrete_sequence=["#7c5cd8"],
                labels={"sleep_hours": "Sleep Hours / Day"},
            )
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = px.histogram(
                df, x="exam_score", nbins=25,
                title="Exam Score Distribution",
                color_discrete_sequence=["#22c55e"],
                labels={"exam_score": "Exam Score"},
            )
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

        # Box plots
        st.subheader("Box Plots")
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.box(df, y="study_hours", title="Study Hours",
                         color_discrete_sequence=["#3b82d4"])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.box(df, y="sleep_hours", title="Sleep Hours",
                         color_discrete_sequence=["#7c5cd8"])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2 — Correlations
# ============================================================
with tab2:
    st.header("🔗 Correlations")

    corr_data = fetch_correlation()
    scatter_rows = fetch_scatter()

    if corr_data:
        col_names = list(corr_data.keys())
        matrix = [[corr_data[r][c] for c in col_names] for r in col_names]

        st.subheader("Pearson Correlation Matrix")
        fig = px.imshow(
            matrix,
            x=col_names, y=col_names,
            text_auto=True,
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            title="Correlation Heatmap",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    if scatter_rows:
        df = pd.DataFrame(scatter_rows)

        st.subheader("3D Scatter — Study Hours × Sleep Hours × Exam Score")
        fig = px.scatter_3d(
            df,
            x="study_hours", y="sleep_hours", z="exam_score",
            color="exam_score",
            color_continuous_scale="Viridis",
            labels={
                "study_hours": "Study Hrs/Day",
                "sleep_hours": "Sleep Hrs/Day",
                "exam_score": "Exam Score",
            },
            opacity=0.65,
            title="3D Scatter Plot",
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

        # 2D scatter panels
        st.subheader("2D Scatter Panels")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(
                df, x="study_hours", y="exam_score",
                trendline="ols",
                color_discrete_sequence=["#3b82d4"],
                labels={"study_hours": "Study Hours/Day", "exam_score": "Exam Score"},
                title="Study Hours vs Exam Score",
                opacity=0.5,
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(
                df, x="sleep_hours", y="exam_score",
                trendline="ols",
                color_discrete_sequence=["#7c5cd8"],
                labels={"sleep_hours": "Sleep Hours/Day", "exam_score": "Exam Score"},
                title="Sleep Hours vs Exam Score",
                opacity=0.5,
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 — Model Metrics
# ============================================================
with tab3:
    st.header("🤖 Model Evaluation Metrics")
    st.markdown("Performance on the **20 % held-out test set** (80/20 split, random_state=42).")

    metrics = fetch_metrics()
    if metrics:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📐 Linear Regression")
            lm = metrics.get("linear_regression", {})
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{lm.get('mae', 'N/A')}")
            m2.metric("RMSE", f"{lm.get('rmse', 'N/A')}")
            m3.metric("R²", f"{lm.get('r2', 'N/A')}")

        with col2:
            st.subheader("🌲 Random Forest")
            rf = metrics.get("random_forest", {})
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{rf.get('mae', 'N/A')}")
            m2.metric("RMSE", f"{rf.get('rmse', 'N/A')}")
            m3.metric("R²", f"{rf.get('r2', 'N/A')}")

        # Bar chart comparison
        st.subheader("Side-by-Side Comparison")
        compare_df = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "R²"],
            "Linear Regression": [lm.get("mae"), lm.get("rmse"), lm.get("r2")],
            "Random Forest": [rf.get("mae"), rf.get("rmse"), rf.get("r2")],
        })
        fig = px.bar(
            compare_df.melt(id_vars="Metric", var_name="Model", value_name="Value"),
            x="Metric", y="Value", color="Model",
            barmode="group",
            color_discrete_map={"Linear Regression": "#3b82d4", "Random Forest": "#22c55e"},
            title="Model Performance Comparison",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        # Metric descriptions
        with st.expander("ℹ️ Metric Definitions"):
            st.markdown("""
| Metric | Description | Lower is better? |
|--------|-------------|-----------------|
| **MAE** | Mean Absolute Error — average absolute difference between predicted and actual score | ✅ Yes |
| **RMSE** | Root Mean Squared Error — penalises large errors more than MAE | ✅ Yes |
| **R²** | Coefficient of determination — proportion of variance explained (1.0 = perfect) | ❌ Higher is better |
            """)

# ============================================================
# TAB 4 — Predict Score
# ============================================================
with tab4:
    st.header("🎯 Predict a Student's Exam Score")

    col_inputs, col_result = st.columns([1, 1])

    with col_inputs:
        st.subheader("Input Parameters")
        study_h = st.slider(
            "📚 Study Hours per Day", min_value=0.0, max_value=12.0,
            value=4.0, step=0.5,
            help="How many hours the student studies per day",
        )
        sleep_h = st.slider(
            "😴 Sleep Hours per Day", min_value=4.0, max_value=12.0,
            value=7.0, step=0.5,
            help="How many hours the student sleeps per day",
        )
        model_choice = st.selectbox(
            "🤖 Model",
            options=["linear", "random_forest"],
            format_func=lambda x: "Linear Regression" if x == "linear" else "Random Forest",
        )
        predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Prediction Result")

        if predict_btn:
            with st.spinner("Predicting…"):
                result = call_api(
                    "/predict", method="POST",
                    payload={"study_hours": study_h, "sleep_hours": sleep_h, "model": model_choice},
                )

            if result:
                score = result["predicted_score"]
                lower = result["lower_bound"]
                upper = result["upper_bound"]
                model_label = "Linear Regression" if model_choice == "linear" else "Random Forest"

                st.success(f"**Predicted Score: {score:.1f} / 100**")
                st.caption(f"95% Prediction Interval: [{lower:.1f}, {upper:.1f}]")
                st.caption(f"Model: {model_label}")

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=score,
                    delta={"reference": 75, "valueformat": ".1f"},
                    title={"text": "Exam Score", "font": {"size": 20}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": "#3b82d4"},
                        "steps": [
                            {"range": [0, 50], "color": "#fee2e2"},
                            {"range": [50, 75], "color": "#fef9c3"},
                            {"range": [75, 100], "color": "#dcfce7"},
                        ],
                        "threshold": {
                            "line": {"color": "#ef4444", "width": 3},
                            "thickness": 0.75,
                            "value": 75,
                        },
                    },
                    number={"suffix": "/100", "valueformat": ".1f"},
                ))
                fig.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)

                # Confidence band bar
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=["Lower (95%)", "Predicted", "Upper (95%)"],
                    y=[lower, score, upper],
                    marker_color=["#93c5fd", "#3b82d4", "#93c5fd"],
                    text=[f"{v:.1f}" for v in [lower, score, upper]],
                    textposition="outside",
                ))
                fig2.update_layout(
                    title="Score with 95% Prediction Interval",
                    yaxis=dict(range=[0, 105], title="Score"),
                    height=300,
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("👈 Adjust the sliders and click **Predict** to see the result.")

# ============================================================
# TAB 5 — What-If Analysis
# ============================================================
with tab5:
    st.header("📈 What-If Analysis")
    st.markdown(
        "Fix one variable and sweep the other to see how the predicted exam score changes."
    )

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Controls")
        sweep_var = st.radio(
            "Sweep variable",
            options=["study_hours", "sleep_hours"],
            format_func=lambda x: "📚 Study Hours" if x == "study_hours" else "😴 Sleep Hours",
        )

        if sweep_var == "study_hours":
            fixed_label = "😴 Fixed Sleep Hours"
            fixed_val = st.slider(fixed_label, 4.0, 12.0, 7.0, 0.5)
            sweep_range = [round(v * 0.5, 1) for v in range(0, 25)]  # 0–12 step 0.5
        else:
            fixed_label = "📚 Fixed Study Hours"
            fixed_val = st.slider(fixed_label, 0.0, 12.0, 4.0, 0.5)
            sweep_range = [round(v * 0.5, 1) for v in range(8, 25)]  # 4–12 step 0.5

        wi_model = st.selectbox(
            "Model",
            options=["linear", "random_forest"],
            format_func=lambda x: "Linear Regression" if x == "linear" else "Random Forest",
            key="wi_model",
        )
        run_wi = st.button("▶ Run Analysis", type="primary", use_container_width=True)

    with col_plot:
        st.subheader("Predicted Score Curve")

        if run_wi:
            with st.spinner("Running sweep…"):
                scores, lowers, uppers = [], [], []
                for val in sweep_range:
                    if sweep_var == "study_hours":
                        payload = {"study_hours": val, "sleep_hours": fixed_val, "model": wi_model}
                    else:
                        payload = {"study_hours": fixed_val, "sleep_hours": val, "model": wi_model}
                    res = call_api("/predict", method="POST", payload=payload)
                    if res:
                        scores.append(res["predicted_score"])
                        lowers.append(res["lower_bound"])
                        uppers.append(res["upper_bound"])
                    else:
                        scores.append(None)
                        lowers.append(None)
                        uppers.append(None)

            x_label = "Study Hours / Day" if sweep_var == "study_hours" else "Sleep Hours / Day"
            plot_df = pd.DataFrame({
                x_label: sweep_range,
                "Predicted Score": scores,
                "Lower (95%)": lowers,
                "Upper (95%)": uppers,
            }).dropna()

            fig = go.Figure()

            # Confidence band
            fig.add_trace(go.Scatter(
                x=list(plot_df[x_label]) + list(plot_df[x_label])[::-1],
                y=list(plot_df["Upper (95%)"]) + list(plot_df["Lower (95%)"])[::-1],
                fill="toself",
                fillcolor="rgba(59,130,212,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% Interval",
            ))

            # Main line
            fig.add_trace(go.Scatter(
                x=plot_df[x_label],
                y=plot_df["Predicted Score"],
                mode="lines+markers",
                line=dict(color="#3b82d4", width=3),
                marker=dict(size=6),
                name="Predicted Score",
            ))

            fixed_label_short = "Sleep" if sweep_var == "study_hours" else "Study"
            model_label = "Linear Regression" if wi_model == "linear" else "Random Forest"

            fig.update_layout(
                title=f"Predicted Score vs {x_label} (Fixed {fixed_label_short} = {fixed_val}h | {model_label})",
                xaxis_title=x_label,
                yaxis_title="Predicted Exam Score",
                yaxis=dict(range=[0, 105]),
                height=480,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            with st.expander("📋 Raw Sweep Data"):
                st.dataframe(plot_df.style.format("{:.2f}"), use_container_width=True)
        else:
            st.info("👈 Set your controls and click **Run Analysis**.")
