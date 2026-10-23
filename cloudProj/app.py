
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Housing AI | Price Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. CUSTOM DESIGN
# =========================================================
st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --bg: #080f1d;
        --panel: #101c2e;
        --border: #263b54;
        --blue: #4da3ff;
        --cyan: #5eead4;
        --text: #edf4ff;
        --muted: #91a4bd;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #142b49 0%, transparent 40%),
            linear-gradient(145deg, #080f1d, #0b1424 60%, #101b2c);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #0b1525;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, p, label {
        color: var(--text);
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 35px;
        border-radius: 24px;
        border: 1px solid #294968;
        background: linear-gradient(115deg, #122843, #163d60 60%, #11506a);
        margin-bottom: 26px;
    }

    .hero:after {
        content: "AI";
        position: absolute;
        right: 28px;
        top: -34px;
        font-size: 180px;
        font-weight: 800;
        color: rgba(255,255,255,.045);
        pointer-events: none;
    }

    .eyebrow {
        color: #76d9ff;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: clamp(30px, 4vw, 48px);
        font-weight: 800;
        letter-spacing: -1.5px;
        line-height: 1.15;
        margin: 12px 0;
        color: #ffffff;
    }

    .hero-description {
        max-width: 670px;
        color: #c1d4e8;
        font-size: 15px;
        line-height: 1.8;
    }

    .section-heading {
        font-size: 21px;
        font-weight: 750;
        margin: 20px 0 5px;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 18px;
    }

    .metric-card {
        background: linear-gradient(145deg, #13243a, #0e1a2b);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 20px;
        min-height: 125px;
    }

    .metric-label {
        color: #9db1ca;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: .5px;
        text-transform: uppercase;
    }

    .metric-value {
        color: #f1f7ff;
        font-size: 25px;
        font-weight: 800;
        margin-top: 12px;
        overflow-wrap: anywhere;
    }

    .metric-caption {
        color: #7f98b5;
        font-size: 12px;
        margin-top: 5px;
    }

    .result-card {
        background: linear-gradient(135deg, #123c50, #102b43);
        border: 1px solid #247d8c;
        border-radius: 22px;
        padding: 28px;
        margin-top: 15px;
    }

    .result-label {
        color: #83e6e2;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.6px;
    }

    .result-value {
        font-size: clamp(34px, 5vw, 54px);
        font-weight: 800;
        color: white;
        margin: 12px 0;
        overflow-wrap: anywhere;
    }

    .result-note {
        color: #c1d9e8;
        font-size: 13px;
        line-height: 1.7;
    }

    .info-card {
        background: #101d30;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
    }

    .info-title {
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .info-text {
        color: #a5b7cd;
        font-size: 13px;
        line-height: 1.8;
    }

    .footer {
        color: #7288a3;
        text-align: center;
        font-size: 12px;
        padding-top: 24px;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        min-height: 46px;
        border-radius: 11px;
        font-weight: 700;
        border: 1px solid #315779;
        transition: all .2s ease;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #1677d2, #147f9c);
        color: white;
        border: none;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        border-color: #69c9ff;
        transform: translateY(-1px);
    }

    div[data-testid="stMetric"] {
        background: #101d30;
        padding: 15px;
        border: 1px solid var(--border);
        border-radius: 13px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a5b7cd;
    }

    .stTextInput input,
    .stNumberInput input {
        border-radius: 9px;
    }

    hr {
        border-color: #263b54;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD TRAINED MODEL
# =========================================================
MODEL_PATH = Path(__file__).parent / "best_random_forest.joblib"


@st.cache_resource
def load_model(path_string):
    saved = joblib.load(path_string)

    if not isinstance(saved, dict):
        raise ValueError(
            "The model file must contain a dictionary "
            "with 'model' and 'features' keys."
        )

    if "model" not in saved or "features" not in saved:
        raise ValueError(
            "The model file must contain 'model' and 'features'."
        )

    loaded_model = saved["model"]
    loaded_features = list(saved["features"])

    return loaded_model, loaded_features


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Put "
        "'best_random_forest.joblib' in the same folder as app.py."
    )
    st.stop()

try:
    model, features = load_model(str(MODEL_PATH))

except Exception as exc:
    st.error("Unable to load the trained model.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info(
        "Check the model file and the scikit-learn version "
        "used when the model was trained."
    )
    st.stop()


# =========================================================
# 4. HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">MACHINE LEARNING • REGRESSION</div>
        <div class="hero-title">Housing Price Intelligence</div>
        <div class="hero-description">
            Explore housing data and generate predictions using
            a trained Random Forest model. Enter property
            characteristics, run the model, and review the result.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 5. MODEL OVERVIEW
# =========================================================
col1, col2, col3 = st.columns(3)

overview = [
    (col1, "MODEL ENGINE", "Random Forest", "Trained regression model"),
    (col2, "INPUT FEATURES", str(len(features)), "Features expected by model"),
    (col3, "MODEL STATUS", "Ready", "Saved model loaded successfully"),
]

for col, label, value, caption in overview:
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-caption">{caption}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 6. SIDEBAR INPUTS
# =========================================================
st.sidebar.markdown(
    """
    <div class="eyebrow">PREDICTION CONTROLS</div>
    <h2 style="margin-top:10px;">Property Details</h2>
    <p style="color:#91a4bd;font-size:13px;line-height:1.7;">
        Enter values for the property and its surrounding area.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    with st.expander("01 • Property characteristics", expanded=True):
        rm = st.number_input(
            "Average rooms (RM)",
            min_value=1.0,
            max_value=20.0,
            value=6.0,
            step=0.1,
        )

        age = st.number_input(
            "Older housing proportion (AGE)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0,
        )

    with st.expander("02 • Neighborhood", expanded=True):
        lstat = st.number_input(
            "LSTAT",
            min_value=0.0,
            max_value=100.0,
            value=12.0,
            step=0.5,
        )

        dis = st.number_input(
            "Distance index (DIS)",
            min_value=0.0,
            max_value=100.0,
            value=4.0,
            step=0.1,
        )

        ptratio = st.number_input(
            "Pupil-teacher ratio (PTRATIO)",
            min_value=0.0,
            max_value=100.0,
            value=18.0,
            step=0.5,
        )

    with st.expander("03 • Environmental & economic data", expanded=True):
        crim = st.number_input(
            "Crime rate (CRIM)",
            min_value=0.0,
            max_value=1000.0,
            value=0.1,
            step=0.01,
            format="%.4f",
        )

        tax = st.number_input(
            "Property tax index (TAX)",
            min_value=0.0,
            max_value=5000.0,
            value=300.0,
            step=10.0,
        )

        nox = st.number_input(
            "Nitric oxide concentration (NOX)",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.01,
            format="%.4f",
        )

    st.divider()

    predict_button = st.button(
        "✦ GENERATE PREDICTION",
        type="primary",
        width="stretch",
    )

    reset_button = st.button(
        "Reset result",
        width="stretch",
    )


# =========================================================
# 7. SESSION STATE
# =========================================================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if reset_button:
    st.session_state.prediction_result = None


# =========================================================
# 8. FEATURE ENGINEERING
# =========================================================
# IMPORTANT:
# These formulas must match the original training notebook.
input_values = {
    "RM": rm,
    "LSTAT": lstat,
    "DIS": dis,
    "PTRATIO": ptratio,
    "CRIM_NOX": crim * nox,
    "RM_CRIM": rm * crim,
    "CRIM": crim,
    "TAX": tax,
    "AGE_DIS": age * dis,
    "NOX": nox,
}

missing_features = [
    name for name in features if name not in input_values
]

if missing_features:
    st.error(
        f"Input values are missing required features: {missing_features}"
    )
    st.stop()

input_df = pd.DataFrame(
    [[input_values[name] for name in features]],
    columns=features,
)


# =========================================================
# 9. MAIN WORKSPACE
# =========================================================
st.markdown(
    '<div class="section-heading">Prediction Workspace</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    'Review your input data and the output generated by the model.'
    '</div>',
    unsafe_allow_html=True,
)

input_tab, result_tab, model_tab = st.tabs(
    ["Input Overview", "Prediction Result", "Model Details"]
)


# ---------------------------------------------------------
# TAB 1: INPUT OVERVIEW
# ---------------------------------------------------------
with input_tab:
    st.markdown("#### Feature values")

    st.caption(
        "The table uses the feature names and order expected "
        "by your saved model."
    )

    display_df = input_df.T.reset_index()
    display_df.columns = ["Feature", "Input Value"]

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Input summary")

    a, b, c = st.columns(3)

    a.metric("Input fields", len(features))
    b.metric("Original input values", 8)
    c.metric("Derived features", 3)

    st.info(
        "Derived features must use the exact formulas from "
        "the original model training process."
    )


# ---------------------------------------------------------
# TAB 2: PREDICTION RESULT
# ---------------------------------------------------------
with result_tab:
    if predict_button:
        try:
            prediction = float(model.predict(input_df)[0])

            st.session_state.prediction_result = {
                "value": prediction,
                "features": input_df.copy(),
            }

        except Exception as exc:
            st.session_state.prediction_result = None

            st.error("Prediction failed.")
            st.code(f"{type(exc).__name__}: {exc}")

            st.warning(
                "If the error mentions SimpleImputer or a missing "
                "internal attribute, the saved model may be "
                "incompatible with the installed scikit-learn version."
            )

    result = st.session_state.prediction_result

    if result is None:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">Your prediction will appear here</div>
                <div class="info-text">
                    Enter the property details in the sidebar and click
                    GENERATE PREDICTION to run your trained model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        prediction = result["value"]

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">MODEL PREDICTION</div>
                <div class="result-value">{prediction:,.4f}</div>
                <div class="result-note">
                    Predicted target value from your saved Random Forest
                    model. Interpret its units according to the target
                    variable used during training.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        csv_df = pd.DataFrame(
            {"Predicted_Value": [prediction]}
        )

        st.download_button(
            "Download prediction report (CSV)",
            data=csv_df.to_csv(index=False).encode("utf-8"),
            file_name="housing_prediction.csv",
            mime="text/csv",
            width="stretch",
        )

        with st.expander("View exact input used for prediction"):
            st.dataframe(
                result["features"],
                width="stretch",
            )


# ---------------------------------------------------------
# TAB 3: MODEL DETAILS
# ---------------------------------------------------------
with model_tab:
    st.markdown("#### Model information")

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">Algorithm</div>
                <div class="info-text">
                    Random Forest Regression<br>
                    Ensemble of decision trees
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">Model artifact</div>
                <div class="info-text">
                    best_random_forest.joblib
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">Expected features</div>
                <div class="info-text">
                    {len(features)} features loaded from the saved artifact.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">Evaluation metrics</div>
                <div class="info-text">
                    R², MAE and RMSE must come from the held-out
                    test evaluation. They are not recalculated here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Inspect feature names"):
        st.write(features)


# =========================================================
# 10. FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        HOUSING AI • MACHINE LEARNING PROJECT<br>
        Powered by Python, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
