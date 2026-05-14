"""
Streamlit Web Application for AI Model Serving

Provides an interactive dashboard for model inference, file upload,
parameter tuning, and performance visualization.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import streamlit as st

# Ensure project root is on the import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Advanced – Prediction Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar – configuration panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    model_name = st.selectbox(
        "Model Architecture",
        options=["simple_nn", "deep_nn", "wide_nn"],
        index=0,
        help="Select the model architecture to use for predictions.",
    )

    input_dim = st.number_input("Input Features", value=128, min_value=1, max_value=2048, step=1)
    hidden_dims_str = st.text_input(
        "Hidden Layers (comma-separated)",
        value="256, 128, 64",
        help="Sizes of hidden layers, e.g. 256, 128, 64",
    )
    hidden_dims = [int(x.strip()) for x in hidden_dims_str.split(",") if x.strip()]
    output_dim = st.number_input("Output Classes", value=1, min_value=1, max_value=1000, step=1)
    dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.8, value=0.3, step=0.05)
    learning_rate = st.number_input("Learning Rate", value=1e-3, format="%e", step=1e-4)
    batch_size = st.number_input("Batch Size", value=64, min_value=1, max_value=1024, step=1)
    num_epochs = st.number_input("Training Epochs", value=10, min_value=1, max_value=500, step=1)

    st.divider()
    st.header("📁 File Upload")
    uploaded_file = st.file_uploader(
        "Upload a CSV file for batch prediction",
        type=["csv"],
        help="The CSV should contain numeric feature columns matching the model's input dimension.",
    )

    st.divider()
    st.header("📝 Notes")
    st.info(
        "Configure the model and upload data on the left, then click "
        "'Run Prediction' in the main panel to see results."
    )

# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

st.title("🧠 AI Advanced – Prediction Dashboard")
st.markdown(
    "This dashboard lets you configure a neural network model, upload data, "
    "and run predictions interactively."
)

# ---------------------------------------------------------------------------
# Model loading (cached so it is not rebuilt on every interaction)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model...")
def load_model(_input_dim, _hidden_dims, _output_dim, _dropout_rate):
    """Build and optionally load a pre-trained model checkpoint."""
    import torch
    from models.model import AIModel

    m = AIModel(
        input_dim=_input_dim,
        hidden_dims=_hidden_dims,
        output_dim=_output_dim,
        dropout_rate=_dropout_rate,
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(dev)
    m.eval()

    ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=dev)
        m.load_state_dict(state["model_state_dict"])
    return m, dev


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------


def run_prediction(model, device, features_array: np.ndarray) -> np.ndarray:
    """Run model inference on a 2-D numpy array of shape (n_samples, input_dim)."""
    import torch

    model.eval()
    x = torch.tensor(features_array, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
    return logits.cpu().numpy()


# ---------------------------------------------------------------------------
# Build layout
# ---------------------------------------------------------------------------

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📥 Input")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns from uploaded file.")
            st.dataframe(df.head(10), use_container_width=True)

            # Try to extract numeric columns for prediction
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) == 0:
                st.error("No numeric columns found in the uploaded file.")
                df_features = None
            else:
                if len(numeric_cols) > input_dim:
                    st.warning(
                        f"The file has {len(numeric_cols)} numeric columns but the model expects "
                        f"{input_dim}. Truncating to the first {input_dim} columns."
                    )
                    numeric_cols = numeric_cols[:input_dim]
                elif len(numeric_cols) < input_dim:
                    st.warning(
                        f"The file has {len(numeric_cols)} numeric columns but the model expects "
                        f"{input_dim}. Padding remaining columns with zeros."
                    )
                    padding = pd.DataFrame(
                        np.zeros((df.shape[0], input_dim - len(numeric_cols))),
                        columns=[f"pad_{i}" for i in range(input_dim - len(numeric_cols))],
                    )
                    df = pd.concat([df[numeric_cols], padding], axis=1)
                    numeric_cols = list(df.columns)

                df_features = df[numeric_cols[:input_dim]].values.astype(np.float32)
        except Exception as exc:
            st.error(f"Failed to read uploaded file: {exc}")
            df = None
            df_features = None
    else:
        # Manual feature entry
        st.markdown("Enter a comma-separated feature vector:")
        manual_input = st.text_area(
            "Feature Vector",
            value=",".join(["0.0"] * min(input_dim, 10)),
            height=80,
            help=f"Provide {input_dim} comma-separated float values.",
        )
        try:
            values = [float(x.strip()) for x in manual_input.split(",") if x.strip()]
            if len(values) != input_dim:
                st.warning(f"Expected {input_dim} values, got {len(values)}. Results may be unreliable.")
            # Pad or truncate
            if len(values) < input_dim:
                values.extend([0.0] * (input_dim - len(values)))
            else:
                values = values[:input_dim]
            df_features = np.array(values, dtype=np.float32).reshape(1, -1)
        except ValueError:
            st.error("Please enter valid comma-separated float values.")
            df_features = None
        df = None

with col_output:
    st.subheader("📤 Output")

    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
        if df_features is not None:
            with st.spinner("Running inference..."):
                model, device = load_model(input_dim, hidden_dims, output_dim, dropout_rate)
                logits = run_prediction(model, device, df_features)

                if output_dim == 1:
                    import torch
                    probs = torch.sigmoid(torch.tensor(logits)).numpy()
                    labels = (probs >= 0.5).astype(int)
                    results = pd.DataFrame(
                        {"Logit": logits.flatten(), "Probability": probs.flatten(), "Predicted Class": labels.flatten()}
                    )
                else:
                    import torch
                    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
                    labels = probs.argmax(axis=1)
                    results = pd.DataFrame(probs, columns=[f"Class_{i}" for i in range(output_dim)])
                    results.insert(0, "Predicted Class", labels)

                st.success("Prediction complete!")
                st.dataframe(results, use_container_width=True)

                # Show a bar chart of the first sample's probabilities
                if df_features.shape[0] >= 1:
                    st.markdown("**First Sample – Probability Distribution**")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    if output_dim == 1:
                        ax.bar(["Class 0", "Class 1"], [1 - probs[0, 0], probs[0, 0]], color=["#ef4444", "#22c55e"])
                        ax.set_ylabel("Probability")
                        ax.set_ylim(0, 1)
                    else:
                        ax.bar(range(output_dim), probs[0], color="#3b82f6")
                        ax.set_xlabel("Class")
                        ax.set_ylabel("Probability")
                    ax.set_title("Prediction Probabilities")
                    st.pyplot(fig, use_container_width=True)
        else:
            st.warning("No input data available. Upload a file or enter features manually.")

# ---------------------------------------------------------------------------
# Performance metrics section
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📊 Model Summary")

try:
    model, device = load_model(input_dim, hidden_dims, output_dim, dropout_rate)
    params = model.get_num_parameters()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Parameters", f"{params['total']:,}")
    with col2:
        st.metric("Trainable Parameters", f"{params['trainable']:,}")
    with col3:
        st.metric("Device", str(device))
except Exception as exc:
    st.error(f"Could not load model summary: {exc}")

# ---------------------------------------------------------------------------
# Training simulation / metrics visualization
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📈 Training Metrics Visualization")

metric_option = st.radio(
    "Select a metric to visualize",
    options=["Loss Curve", "Accuracy Curve", "Synthetic Demo"],
    horizontal=True,
)

if metric_option == "Synthetic Demo":
    # Generate and display a synthetic training curve for demonstration
    epochs = np.arange(1, num_epochs + 1)
    np.random.seed(42)
    train_loss = 2.0 * np.exp(-0.15 * epochs) + np.random.normal(0, 0.02, num_epochs)
    val_loss = 2.0 * np.exp(-0.12 * epochs) + np.random.normal(0, 0.04, num_epochs)
    train_acc = 1.0 - np.exp(-0.18 * epochs) + np.random.normal(0, 0.01, num_epochs)
    val_acc = 1.0 - np.exp(-0.14 * epochs) + np.random.normal(0, 0.02, num_epochs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="Train Loss", color="#2563eb")
    axes[0].plot(epochs, val_loss, label="Val Loss", color="#dc2626")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, label="Train Acc", color="#2563eb")
    axes[1].plot(epochs, val_acc, label="Val Acc", color="#16a34a")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

else:
    # Try loading a training history JSON file if it exists
    history_path = PROJECT_ROOT / "outputs" / "training_history.json"
    if history_path.is_file():
        with open(history_path) as f:
            history = json.load(f)
        fig, ax = plt.subplots(figsize=(8, 4))
        key = "train_loss" if metric_option == "Loss Curve" else "train_acc"
        val_key = "val_loss" if metric_option == "Loss Curve" else "val_acc"
        ax.plot(history.get(key, []), label=f"Train {key.split('_')[1].title()}")
        ax.plot(history.get(val_key, []), label=f"Val {val_key.split('_')[1].title()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.split("_")[1].title())
        ax.set_title(metric_option)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info(
            "No training history file found at `outputs/training_history.json`. "
            "Select 'Synthetic Demo' to see a demonstration plot."
        )
