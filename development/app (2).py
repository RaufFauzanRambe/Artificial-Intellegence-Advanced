"""
Flask Web Application for AI Model Serving

Provides a simple web interface for interacting with the trained AI model.
Includes routes for the index page, prediction endpoint, and an about page.
HTML templates are rendered inline using render_template_string.
"""

import sys
import os
import numpy as np

from flask import Flask, request, render_template_string

# Ensure project root is on the path so we can import src modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model import AIModel, build_model

# ---------------------------------------------------------------------------
# Global model & preprocessor (loaded once at startup)
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Default model parameters (can be overridden via environment variables)
INPUT_DIM = int(os.environ.get("AI_INPUT_DIM", "128"))
HIDDEN_DIMS = [int(x) for x in os.environ.get("AI_HIDDEN_DIMS", "256,128,64").split(",")]
OUTPUT_DIM = int(os.environ.get("AI_OUTPUT_DIM", "1"))
DROPOUT_RATE = float(os.environ.get("AI_DROPOUT", "0.3"))

model = None
device = None


def load_model() -> AIModel:
    """Load or build the model for inference.

    If a checkpoint file exists at ``checkpoints/best_model.pt``, its weights
    are loaded.  Otherwise a fresh model is created and used in eval mode.
    """
    m = AIModel(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        output_dim=OUTPUT_DIM,
        dropout_rate=DROPOUT_RATE,
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(dev)
    m.eval()

    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=dev)
        m.load_state_dict(state["model_state_dict"])
        print(f"[app] Loaded checkpoint from {ckpt_path} (epoch {state.get('epoch', '?')})")
    else:
        print("[app] No checkpoint found – using freshly initialized model weights")

    return m, dev


# ---------------------------------------------------------------------------
# HTML Templates (inline)
# ---------------------------------------------------------------------------

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Model – Prediction Service</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f6f9; color: #333; }
        .container { max-width: 720px; margin: 40px auto; background: #fff; border-radius: 8px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        h1 { font-size: 1.8rem; margin-bottom: 8px; }
        p.subtitle { color: #666; margin-bottom: 24px; }
        label { display: block; font-weight: 600; margin-top: 16px; }
        textarea { width: 100%; min-height: 100px; font-family: monospace; font-size: 0.9rem; padding: 8px; border: 1px solid #ccc; border-radius: 4px; resize: vertical; }
        button { margin-top: 20px; padding: 10px 28px; font-size: 1rem; background: #2563eb; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        .result { margin-top: 24px; padding: 16px; background: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 4px; }
        .result h3 { margin-bottom: 8px; color: #166534; }
        .result pre { white-space: pre-wrap; word-break: break-word; }
        .error { background: #fef2f2; border-left-color: #ef4444; }
        .error h3 { color: #991b1b; }
        nav { margin-bottom: 24px; }
        nav a { color: #2563eb; text-decoration: none; margin-right: 16px; }
        nav a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Model – Prediction Service</h1>
        <p class="subtitle">Enter feature values as a JSON list of floats to get a prediction.</p>
        <nav>
            <a href="/">Home</a>
            <a href="/about">About</a>
        </nav>
        <form method="POST" action="/predict">
            <label for="features">Feature Vector (JSON list of {{ input_dim }} floats)</label>
            <textarea name="features" id="features" placeholder='[0.1, 0.5, -0.3, 0.8, ...]'>{{ default_features }}</textarea>
            <button type="submit">Run Prediction</button>
        </form>

        {% if result is not none %}
        <div class="result {% if error %}error{% endif %}">
            <h3>{% if error %}Error{% else %}Prediction Result{% endif %}</h3>
            <pre>{{ result }}</pre>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

ABOUT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About – AI Model Service</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f6f9; color: #333; }
        .container { max-width: 720px; margin: 40px auto; background: #fff; border-radius: 8px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        h1 { font-size: 1.8rem; margin-bottom: 16px; }
        h2 { margin-top: 20px; font-size: 1.2rem; color: #2563eb; }
        p, li { line-height: 1.7; }
        ul { padding-left: 20px; margin-top: 8px; }
        nav { margin-bottom: 24px; }
        nav a { color: #2563eb; text-decoration: none; margin-right: 16px; }
        nav a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>About This Service</h1>
        <nav>
            <a href="/">Home</a>
            <a href="/about">About</a>
        </nav>
        <p>
            This is a Flask-based web interface for the <strong>Artificial Intelligence Advanced</strong> project.
            It serves a trained neural network model and provides a simple form for submitting feature vectors
            to receive real-time predictions.
        </p>
        <h2>Model Architecture</h2>
        <ul>
            <li><strong>Type:</strong> Fully-connected feedforward neural network</li>
            <li><strong>Input dim:</strong> {{ input_dim }}</li>
            <li><strong>Hidden dims:</strong> {{ hidden_dims | join(', ') }}</li>
            <li><strong>Output dim:</strong> {{ output_dim }}</li>
            <li><strong>Dropout:</strong> {{ dropout_rate }}</li>
        </ul>
        <h2>Parameters</h2>
        <ul>
            <li>Total parameters: {{ params.total | comma }}</li>
            <li>Trainable parameters: {{ params.trainable | comma }}</li>
        </ul>
        <h2>API Endpoints</h2>
        <ul>
            <li><code>GET /</code> – Home page with prediction form</li>
            <li><code>POST /predict</code> – Submit features and receive a prediction</li>
            <li><code>GET /about</code> – This page</li>
        </ul>
    </div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Render the home page with the prediction form."""
    default_features = ", ".join(["0.0"] * INPUT_DIM)
    return render_template_string(
        INDEX_TEMPLATE,
        input_dim=INPUT_DIM,
        default_features=default_features,
        result=None,
        error=False,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Parse the submitted feature vector, run inference, and display the result."""
    global model, device

    features_raw = request.form.get("features", "")
    try:
        features = json.loads(features_raw)
    except (json.JSONDecodeError, TypeError):
        return render_template_string(
            INDEX_TEMPLATE,
            input_dim=INPUT_DIM,
            default_features=features_raw,
            result="Invalid JSON. Please provide a list of floats, e.g. [0.1, 0.5, -0.3, ...]",
            error=True,
        )

    if not isinstance(features, list) or len(features) != INPUT_DIM:
        return render_template_string(
            INDEX_TEMPLATE,
            input_dim=INPUT_DIM,
            default_features=features_raw,
            result=f"Expected a JSON list of exactly {INPUT_DIM} floats, got {type(features).__name__}"
                   + (f" of length {len(features)}" if isinstance(features, list) else ""),
            error=True,
        )

    try:
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    except ValueError as exc:
        return render_template_string(
            INDEX_TEMPLATE,
            input_dim=INPUT_DIM,
            default_features=features_raw,
            result=f"Could not convert to tensor: {exc}",
            error=True,
        )

    with torch.no_grad():
        logits = model(x)

    # Sigmoid for binary classification
    if OUTPUT_DIM == 1:
        probability = torch.sigmoid(logits).item()
        label = 1 if probability >= 0.5 else 0
        result_text = (
            f"Logits    : {logits.item():.6f}\n"
            f"Probability (sigmoid): {probability:.6f}\n"
            f"Predicted class : {label}"
        )
    else:
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        label = int(probs.argmax())
        result_text = f"Predicted class: {label}\nClass probabilities:\n"
        for cls, prob in enumerate(probs):
            result_text += f"  Class {cls}: {prob:.6f}\n"

    return render_template_string(
        INDEX_TEMPLATE,
        input_dim=INPUT_DIM,
        default_features=features_raw,
        result=result_text,
        error=False,
    )


@app.route("/about", methods=["GET"])
def about():
    """Display information about the model and this service."""
    import json as _json
    params = model.get_num_parameters() if model else {"total": 0, "trainable": 0}
    return render_template_string(
        ABOUT_TEMPLATE,
        input_dim=INPUT_DIM,
        hidden_dims=[str(d) for d in HIDDEN_DIMS],
        output_dim=OUTPUT_DIM,
        dropout_rate=DROPOUT_RATE,
        params=params,
    )


# ---------------------------------------------------------------------------
# Template filter for comma-separated numbers
# ---------------------------------------------------------------------------

@app.template_filter("comma")
def comma_filter(value):
    """Format an integer with comma separators."""
    return f"{value:,}"


# ---------------------------------------------------------------------------
# Application startup
# ---------------------------------------------------------------------------

import json
import torch  # imported here so Flask can load first


@app.before_request
def _ensure_model():
    """Lazy-load the model on the first request if not already loaded."""
    global model, device
    if model is None:
        model, device = load_model()


if __name__ == "__main__":
    print("[app] Loading model...")
    model, device = load_model()
    print(f"[app] Model ready – {model.get_num_parameters()['trainable']:,} trainable parameters")
    app.run(host="0.0.0.0", port=5000, debug=False)
