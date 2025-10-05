import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import xgboost as xgb
import requests
from typing import List, Dict, Any, Iterator

MODEL_PATH = Path("xgb_best_model.pkl")
DATA_PATH = Path("kepler_exoplanet_data_cleaned.csv")
TARGET_COLUMN = "koi_disposition"
BEST_PARAMS_PATH = Path("best_params.txt")
ACCURACY_PATH = Path("accuracy.txt")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
DISPOSITION_LABELS = {
    0: "False Positive",
    1: "Candidate",
    2: "Confirmed"
}


def _configure_page():
    try:
        st.set_page_config(
            page_title="Kepler Exoplanet Disposition Predictor",
            layout="wide"
        )
    except Exception:
        pass


_configure_page()

st.markdown(
    """
    <style>
        .main {
            background-image: linear-gradient(135deg, #050508 0%, #1a1a20 50%, #050508 100%);
            color: #e4e4ea;
        }
        section[data-testid="stSidebar"] {
            background-color: #08080d !important;
            color: #d4d4dc !important;
        }
        .stAppHeader, .st-emotion-cache-6qob1r {
            background-color: transparent !important;
        }
        .hero-card {
            background: rgba(12, 14, 22, 0.85);
            border: 1px solid rgba(0, 255, 229, 0.4);
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 24px;
            box-shadow: 0 20px 40px -24px rgba(0, 255, 204, 0.55);
        }
        .hero-title {
            font-size: 2.6rem;
            font-weight: 700;
            margin-bottom: 4px;
            color: #f5f5f7;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #c9c9d1;
            line-height: 1.6rem;
        }
        .stButton>button {
            border-radius: 999px;
            padding: 0.55rem 1.4rem;
            font-weight: 600;
            background: linear-gradient(135deg, #06f5ff, #00ffc6);
            border: none;
            color: #050505;
            box-shadow: 0 18px 30px -18px rgba(0, 255, 210, 0.72);
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 20px 36px -18px rgba(0, 255, 234, 0.85);
        }
        .metric-card {
            background: rgba(0, 255, 204, 0.08);
            border-radius: 14px;
            padding: 12px 16px;
            border: 1px solid rgba(0, 255, 229, 0.2);
            backdrop-filter: blur(9px);
        }
        .st-expander {
            border-radius: 14px !important;
            border: 1px solid rgba(0, 255, 229, 0.25) !important;
            background: rgba(14, 16, 24, 0.7) !important;
        }
        textarea, .stTextInput>div>div>input {
            border-radius: 12px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Kepler Exoplanet Wiki</div>
        <div class="hero-subtitle">
            Explore, tune, and converse with your models in a single interface. Predict dispositions, experiment with
            new training data, and collaborate with Ollama assistants, all without leaving this dashboard.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_reference_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find the cleaned dataset at `{DATA_PATH}`.")

    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column `{TARGET_COLUMN}` is missing from the dataset.")

    feature_df = df.drop(columns=[TARGET_COLUMN])
    target_series = df[TARGET_COLUMN]
    return feature_df, target_series


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find the trained model at `{MODEL_PATH}`.")
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def build_imputer(reference_features: pd.DataFrame):
    imputer = KNNImputer(n_neighbors=7)
    imputer.fit(reference_features)
    return imputer


def get_initial_row(option: str, features: pd.DataFrame) -> pd.DataFrame:
    if option == "Dataset index":
        index = st.sidebar.slider("Select dataset index", 0, len(features) - 1, 0)
        row = features.iloc[[index]].copy()
    elif option == "Manual entry":
        row = pd.DataFrame([{col: np.nan for col in features.columns}])
    else:
        row = features.head(1).copy()

    row = row.astype(float)
    row.reset_index(drop=True, inplace=True)
    return row


def prepare_user_input(edited_row: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    cleaned = edited_row.reindex(columns=columns)
    for col in cleaned.columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    return cleaned


def _coerce_param_value(raw: str):
    raw = raw.strip().rstrip(",")
    try:
        value = float(raw)
        if value.is_integer():
            return int(value)
        return value
    except ValueError:
        return raw


@st.cache_data(show_spinner=False)
def load_best_params() -> dict:
    if not BEST_PARAMS_PATH.exists():
        return {}

    params = {}
    with BEST_PARAMS_PATH.open() as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            params[key.strip()] = _coerce_param_value(value)
    return params


def get_hyperparameter_defaults(best_params: dict) -> dict:
    defaults = {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
    }
    merged = defaults.copy()
    for key, value in best_params.items():
        if key in merged and isinstance(value, (int, float)):
            merged[key] = value
    return merged


def _convert_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    for column in converted.columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    return converted


def prepare_training_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Uploaded data must include the `{TARGET_COLUMN}` column.")

    numeric_df = _convert_numeric_dataframe(df)
    numeric_df = numeric_df.dropna(axis=0, subset=[TARGET_COLUMN])

    target = numeric_df[TARGET_COLUMN].round().astype(int)
    features = numeric_df.drop(columns=[TARGET_COLUMN])
    features = features.select_dtypes(include="number").astype(float)
    features = features.dropna(axis=1, how="all")

    if features.empty:
        raise ValueError("No numeric feature columns available after preprocessing.")

    return features, target


def train_model_with_params(features: pd.DataFrame, target: pd.Series, params: dict):
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=23,
        stratify=target if len(target.unique()) > 1 else None,
    )

    imputer = KNNImputer(n_neighbors=7)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        gamma=float(params["gamma"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_weight=int(params["min_child_weight"]),
        eval_metric="mlogloss",
        objective="multi:softprob",
        n_jobs=-1,
        verbosity=0,
    )

    eval_set = [(X_train_imputed, y_train), (X_test_imputed, y_test)]
    model.fit(X_train_imputed, y_train, eval_set=eval_set, verbose=False)

    metrics = {
        "accuracy": float(accuracy_score(y_test, model.predict(X_test_imputed))),
    }

    try:
        metrics["roc_auc"] = float(
            roc_auc_score(y_test, model.predict_proba(X_test_imputed), multi_class="ovr")
        )
    except ValueError:
        metrics["roc_auc"] = None

    return model, imputer, metrics


def evaluate_model_performance(model, imputer, features: pd.DataFrame, target: pd.Series) -> dict:
    imputed = imputer.transform(features)
    predictions = model.predict(imputed)

    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
    }

    try:
        probabilities = model.predict_proba(imputed)
        metrics["roc_auc"] = float(roc_auc_score(target, probabilities, multi_class="ovr"))
    except ValueError:
        metrics["roc_auc"] = None

    return metrics


def read_logged_metrics(path: Path = ACCURACY_PATH) -> dict:
    if not path.exists():
        return {}

    metrics = {}
    with path.open() as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip().rstrip("%")
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
    return metrics


def list_ollama_models() -> list[str]:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        models = [item["name"] for item in data.get("models", []) if "name" in item]
        return sorted(models)
    except Exception:
        return []


def render_probability_breakdown(probabilities: np.ndarray):
    chart_df = pd.DataFrame({
        "Disposition": [DISPOSITION_LABELS[i] for i in range(len(probabilities))],
        "Probability": probabilities
    }).set_index("Disposition")
    st.bar_chart(chart_df)


def call_ollama_chat(model_name: str, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": stream,
    }
    response = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def stream_ollama_chat(model_name: str, messages: List[Dict[str, str]]) -> Iterator[Dict[str, Any]]:
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
    }
    with requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, stream=True, timeout=60) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            yield chunk


def main():
    try:
        features, target = load_reference_data()
        model = load_model()
        imputer = build_imputer(features)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    best_params = load_best_params()
    hyper_defaults = get_hyperparameter_defaults(best_params)

    if "active_model" not in st.session_state:
        st.session_state.active_model = model
    if "active_imputer" not in st.session_state:
        st.session_state.active_imputer = imputer
    if "reference_features" not in st.session_state:
        st.session_state.reference_features = features
    if "reference_target" not in st.session_state:
        st.session_state.reference_target = target
    if "base_features" not in st.session_state:
        st.session_state.base_features = features.copy()
    if "base_target" not in st.session_state:
        st.session_state.base_target = target.copy()
    if "best_params" not in st.session_state:
        st.session_state.best_params = best_params
    if "latest_training_metrics" not in st.session_state:
        st.session_state.latest_training_metrics = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_token_usage" not in st.session_state:
        st.session_state.chat_token_usage = None
    if "ollama_models" not in st.session_state:
        st.session_state.ollama_models = list_ollama_models()

    st.sidebar.header("Input configuration")
    init_option = st.sidebar.radio(
        "Start from",
        options=["Dataset index", "Manual entry"],
        help="Pick how to initialize the feature values before making adjustments. "
             "Choose Manual entry to start from a blank slate."
    )

    if st.sidebar.button("Refresh template"):
        st.rerun()

    predict_tab, train_tab, chat_tab = st.tabs([
        "Predict disposition",
        "Train or fine-tune model",
        "Chat with Ollama",
    ])

    active_model = st.session_state.active_model
    active_imputer = st.session_state.active_imputer
    reference_features = st.session_state.reference_features
    reference_target = st.session_state.reference_target

    try:
        current_metrics = evaluate_model_performance(active_model, active_imputer, reference_features, reference_target)
    except Exception:
        current_metrics = {}
    st.session_state.current_model_metrics = current_metrics

    with predict_tab:
        starter_row = get_initial_row(init_option, reference_features)

        st.markdown("### Configure candidate features")
        st.caption(
            "Tip: Start from a known profile, the dataset index, or a blank slate. When you're ready, hit *Predict* to\n"
            " generate the disposition estimate."
        )

        parameter_card = st.container()
        with parameter_card:
            st.markdown("---")
            st.markdown("#### Editable observation")
            st.caption("All numeric fields will be auto-imputed if left blank.")

        edited_row = st.data_editor(
            starter_row,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True
        )

        prepared_row = prepare_user_input(edited_row, reference_features.columns)

        predict_col, info_col = st.columns([1, 1])

        with predict_col:
            st.markdown("#### Prediction console")
            if st.button("Predict disposition", type="primary"):
                with st.spinner("Estimating disposition..."):
                    imputed = active_imputer.transform(prepared_row)
                    prediction = active_model.predict(imputed)[0]
                    probabilities = active_model.predict_proba(imputed)[0]

                predicted_label = DISPOSITION_LABELS.get(int(prediction), str(prediction))
                st.success(f"Predicted disposition: **{predicted_label}**")
                st.caption("Class probabilities")
                render_probability_breakdown(probabilities)

        with info_col:
            st.markdown("#### Model snapshot")
            if current_metrics:
                # info_col.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Active model accuracy", f"{current_metrics['accuracy'] * 100:.2f}%")
                if current_metrics.get("roc_auc") is not None:
                    st.caption(f"ROC AUC: {current_metrics['roc_auc']:.2f}")
                info_col.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Accuracy metrics will appear once the model has been evaluated.")

            st.markdown("#### Reference breakdown")
            st.caption("Distribution of target classes in the currently loaded dataset.")
            st.metric("Total observations", len(reference_target))
            class_counts = reference_target.map(DISPOSITION_LABELS).value_counts()
            st.dataframe(class_counts.rename("Count"), use_container_width=True)

        with st.expander("Model details"):
            st.markdown(
                """
                - **Model**: XGBoost multi-class classifier (`xgb_best_model.pkl`)
                - **Imputation**: KNNImputer with 7 neighbors fitted on the cleaned dataset
                - **Classes**: 0 = False Positive, 1 = Candidate, 2 = Confirmed
                - **Features**: {} columns from `kepler_exoplanet_data_cleaned.csv`
                """.format(len(reference_features.columns))
            )

    with train_tab:
        st.markdown("### Training")
        st.caption(
            "Bring your own labeled CSV or lean on the mission dataset to spin up a fresh XGBoost model."
            " Tweak hyperparameters, merge datasets, and decide whether to promote the result."
        )

        st.markdown("---")
        st.markdown("#### Best hyperparameters in use")
        best_display = st.session_state.get("best_params") or best_params
        if best_display:
            st.json(best_display)
        else:
            st.info("Best hyperparameters file not found. Defaults will be used unless you override them.")

        with st.form("training_form"):
            st.markdown("#### Training hyperparameters")
            st.caption("Adjust any values before launching a new training run.")

            hp_cols = st.columns(3)
            hyperparam_inputs = {
                "n_estimators": hp_cols[0].number_input(
                    "n_estimators",
                    min_value=50,
                    max_value=1000,
                    value=int(hyper_defaults["n_estimators"]),
                    step=25,
                ),
                "max_depth": hp_cols[1].number_input(
                    "max_depth",
                    min_value=1,
                    max_value=20,
                    value=int(hyper_defaults["max_depth"]),
                    step=1,
                ),
                "learning_rate": hp_cols[2].number_input(
                    "learning_rate",
                    min_value=0.001,
                    max_value=0.5,
                    value=float(hyper_defaults["learning_rate"]),
                    step=0.001,
                    format="%.3f",
                ),
            }

            hp_cols_2 = st.columns(3)
            hyperparam_inputs.update({
                "gamma": hp_cols_2[0].number_input(
                    "gamma",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(hyper_defaults["gamma"]),
                    step=0.05,
                    format="%.2f",
                ),
                "reg_alpha": hp_cols_2[1].number_input(
                    "reg_alpha",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(hyper_defaults["reg_alpha"]),
                    step=0.01,
                    format="%.2f",
                ),
                "reg_lambda": hp_cols_2[2].number_input(
                    "reg_lambda",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(hyper_defaults["reg_lambda"]),
                    step=0.01,
                    format="%.2f",
                ),
            })

            hp_cols_3 = st.columns(3)
            hyperparam_inputs.update({
                "subsample": hp_cols_3[0].number_input(
                    "subsample",
                    min_value=0.1,
                    max_value=1.0,
                    value=float(hyper_defaults["subsample"]),
                    step=0.05,
                    format="%.2f",
                ),
                "colsample_bytree": hp_cols_3[1].number_input(
                    "colsample_bytree",
                    min_value=0.1,
                    max_value=1.0,
                    value=float(hyper_defaults["colsample_bytree"]),
                    step=0.05,
                    format="%.2f",
                ),
                "min_child_weight": hp_cols_3[2].number_input(
                    "min_child_weight",
                    min_value=1,
                    max_value=20,
                    value=int(hyper_defaults["min_child_weight"]),
                    step=1,
                ),
            })

            uploaded_file = st.file_uploader(
                "Upload CSV for training (optional)",
                type="csv",
                help="Provide additional labeled data matching the Kepler schema to fine-tune or retrain the model.",
            )

            combine_with_base = False
            if uploaded_file is not None:
                combine_with_base = st.checkbox(
                    "Append uploaded data to the baseline Kepler dataset",
                    value=False,
                    help="Enable to merge both datasets before training. Leave unchecked to train only on the uploaded file.",
                )
            else:
                combine_with_base = True
                st.info("No file uploaded. The baseline cleaned dataset will be used for training.")

            use_in_session = st.checkbox(
                "Use the newly trained model for predictions in this session",
                value=True,
            )
            persist_to_disk = st.checkbox(
                "Overwrite the saved production model and hyperparameters",
                value=False,
                help="When enabled, the trained model replaces `xgb_best_model.pkl` and updates `best_params.txt`.",
            )

            submit_training = st.form_submit_button("Train new model", type="primary")

        if submit_training:
            with st.spinner("Training XGBoost model..."):
                try:
                    data_frames = []
                    if combine_with_base:
                        base_df = st.session_state.base_features.copy()
                        base_df[TARGET_COLUMN] = st.session_state.base_target.values
                        data_frames.append(base_df)

                    if uploaded_file is not None:
                        uploaded_file.seek(0)
                        uploaded_df = pd.read_csv(uploaded_file)
                        data_frames.append(uploaded_df)

                    if not data_frames:
                        st.error("Please upload a dataset or include the baseline data before training.")
                        raise ValueError("No data provided for training.")

                    training_df = pd.concat(data_frames, ignore_index=True)
                    training_features, training_target = prepare_training_dataset(training_df)

                    model_out, imputer_out, metrics = train_model_with_params(
                        training_features,
                        training_target,
                        hyperparam_inputs,
                    )

                    st.session_state.latest_training_metrics = metrics
                    st.session_state.latest_training_params = hyperparam_inputs.copy()

                    if use_in_session:
                        st.session_state.active_model = model_out
                        st.session_state.active_imputer = imputer_out
                        st.session_state.reference_features = training_features.reset_index(drop=True)
                        st.session_state.reference_target = training_target.reset_index(drop=True)
                        current_metrics = evaluate_model_performance(
                            st.session_state.active_model,
                            st.session_state.active_imputer,
                            st.session_state.reference_features,
                            st.session_state.reference_target,
                        )
                        st.session_state.current_model_metrics = current_metrics

                    if persist_to_disk:
                        joblib.dump(model_out, MODEL_PATH)
                        with BEST_PARAMS_PATH.open("w") as handle:
                            for key, value in hyperparam_inputs.items():
                                handle.write(f"{key}: {value}\n")
                        load_best_params.clear()
                        st.session_state.best_params = hyperparam_inputs.copy()
                        st.success("Trained model saved to disk and hyperparameters updated.")

                    st.success("Training complete. Review the updated metrics below.")
                except Exception as train_error:
                    st.error(f"Training failed: {train_error}")

        st.markdown("---")
        st.markdown("#### Performance summary")
        perf_cols = st.columns(2)

        active_metrics = st.session_state.get("current_model_metrics") or current_metrics
        with perf_cols[0]:
            st.markdown("**Active model**")
            if active_metrics:
                st.metric("Accuracy", f"{active_metrics['accuracy'] * 100:.2f}%")
                if active_metrics.get("roc_auc") is not None:
                    st.caption(f"ROC AUC: {active_metrics['roc_auc']:.2f}")
            else:
                st.caption("Accuracy metrics will appear once the model has been evaluated.")

        with perf_cols[1]:
            st.markdown("**Last training run**")
            latest_metrics = st.session_state.latest_training_metrics
            if latest_metrics:
                st.metric("Accuracy", f"{latest_metrics['accuracy'] * 100:.2f}%")
                if latest_metrics.get("roc_auc") is not None:
                    st.caption(f"ROC AUC: {latest_metrics['roc_auc']:.2f}")
            else:
                logged_metrics = read_logged_metrics()
                if logged_metrics:
                    acc = logged_metrics.get("test set accuracy")
                    roc = logged_metrics.get("test set roc auc")
                    if acc is not None:
                        st.metric("Baseline accuracy", f"{acc:.2f}%")
                    if roc is not None:
                        st.caption(f"Baseline ROC AUC: {roc:.2f}")
                else:
                    st.caption("Train a model to see recorded metrics.")

    with chat_tab:
        st.markdown("### Chat with Ollama")
        st.caption(
            "Fire off questions and watch responses stream live in markdown."
            " Thinking tokens will surface when the model shares its intermediate reasoning."
        )
        if not st.session_state.ollama_models:
            st.warning("No Ollama models detected. Ensure Ollama is running locally and models are available.")
        selected_model = st.selectbox(
            "Select Ollama model",
            options=st.session_state.ollama_models,
            help="Models are discovered via the local Ollama server `GET /api/tags`.",
        )

        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_history:
                for entry in st.session_state.chat_history:
                    role = entry.get("role", "assistant")
                    message = entry.get("content", "")
                    model_name = entry.get("model", selected_model)
                    if role == "user":
                        st.markdown(f"**You:** {message}")
                    else:
                        st.markdown(f"**{model_name}:**\n\n{message}")
                        if "thinking" in entry and entry["thinking"]:
                            with st.expander("Thinking tokens"):
                                st.markdown(entry["thinking"])

        with st.form("ollama_chat_form", clear_on_submit=True):
            user_message = st.text_area("Message", placeholder="Ask the Ollama model anything…")
            submitted = st.form_submit_button("Send", type="primary")

        if submitted and user_message.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_message})
            try:
                api_messages = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ] + [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.chat_history
                ]

                stream_container = chat_container.container()
                assistant_placeholder = stream_container.empty()
                thinking_placeholder = stream_container.empty()

                content_parts: List[str] = []
                thinking_parts: List[str] = []
                latest_usage: Dict[str, Any] = {}

                for chunk in stream_ollama_chat(selected_model, api_messages):
                    message_chunk = chunk.get("message", {})
                    if message_chunk.get("role") == "assistant":
                        delta = message_chunk.get("content", "")
                        if delta:
                            content_parts.append(delta)
                            assistant_placeholder.markdown(
                                f"**{selected_model}:**\n\n{''.join(content_parts)}"
                            )
                        thinking_delta = message_chunk.get("thinking")
                        if thinking_delta:
                            thinking_parts.append(thinking_delta)
                            thinking_placeholder.markdown(
                                "**Thinking tokens (streaming):**\n\n" + "".join(thinking_parts)
                            )

                    if chunk.get("usage"):
                        latest_usage = chunk["usage"]

                final_message = "".join(content_parts)
                final_thinking = "".join(thinking_parts)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_message,
                    "thinking": final_thinking,
                    "model": selected_model,
                })
                if latest_usage:
                    st.session_state.chat_token_usage = latest_usage

            except requests.exceptions.RequestException as err:
                st.error(f"Failed to reach Ollama: {err}")
            except Exception as exc:
                st.error(f"Unexpected error while chatting with Ollama: {exc}")

        if st.session_state.chat_token_usage:
            usage = st.session_state.chat_token_usage
            st.markdown("#### Token usage")
            usage_cols = st.columns(3)
            usage_cols[0].metric("Prompt tokens", usage.get("prompt_tokens", 0))
            usage_cols[1].metric("Completion tokens", usage.get("completion_tokens", 0))
            usage_cols[2].metric("Total tokens", usage.get("total_tokens", 0))
            if usage.get("thinking_tokens") is not None:
                st.caption(f"Thinking tokens: {usage['thinking_tokens']}")


if __name__ == "__main__":
    main()
