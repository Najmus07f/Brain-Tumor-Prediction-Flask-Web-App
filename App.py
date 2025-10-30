import os, json, numpy as np
from PIL import Image
import torch
from torchvision import transforms
import timm
import joblib
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# --------- paths ----------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------- load configs & meta ----------
with open(os.path.join(MODEL_DIR, "labels.json"), "r") as f:
    LABELS = json.load(f)  # 4 tumor labels + "Irrelevant Image" (for UI list only)

with open(os.path.join(MODEL_DIR, "preprocess.json"), "r") as f:
    PP = json.load(f)
MEAN, STD = PP["mean"], PP["std"]
GRAYSCALE_TO_RGB = PP.get("grayscale_to_rgb", True)

with open(os.path.join(MODEL_DIR, "ensemble_config.json"), "r") as f:
    ENS = json.load(f)
THR = ENS["thresholds"]  # {maxprob, entropy, color_saturation}

META_BUNDLE = joblib.load(os.path.join(MODEL_DIR, ENS["meta_path"]))
meta = META_BUNDLE["meta"]  # LogisticRegression over 4 classes

device = torch.device("cpu")

def make_tfms(img_size: int):
    ops = [transforms.Resize((img_size, img_size))]
    if GRAYSCALE_TO_RGB:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)

# --------- load base models ----------
MODEL_OBJS = []
NUM_CLASSES = len(LABELS) - 1  # first 4 are tumors; last is "Irrelevant Image" used only for UI
for bm in ENS["base_models"]:
    name, ckpt, img_size = bm["name"], bm["ckpt"], bm["img_size"]
    model = timm.create_model(name, pretrained=False, num_classes=NUM_CLASSES)
    state = torch.load(os.path.join(MODEL_DIR, ckpt), map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    MODEL_OBJS.append({
        "name": name,
        "img_size": img_size,
        "model": model,
        "tfm": make_tfms(img_size)
    })

def entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector."""
    p = np.clip(p, 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())

# --------- flask app ----------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    # Render upload screen (no prediction yet)
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if not f or f.filename == "":
        return redirect(url_for("home"))

    filename = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    f.save(path)

    # quick colorfulness gate (MRIs are typically low-saturation)
    pil = Image.open(path).convert("RGB")
    arr = np.array(pil, dtype=np.float32) / 255.0
    colorfulness = float(np.std(arr, axis=2).mean())
    too_colorful = colorfulness > THR["color_saturation"]

    # run all base models -> concat probs
    parts = []
    for b in MODEL_OBJS:
        x = b["tfm"](pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(b["model"](x), dim=1).cpu().numpy()[0]  # (4,)
        parts.append(probs)
    X = np.concatenate(parts, axis=0).reshape(1, -1)  # (1, 4*#models)

    # meta prediction over 4 tumor classes
    meta_proba = meta.predict_proba(X)[0]
    meta_top   = float(meta_proba.max())
    H          = entropy(meta_proba)
    meta_idx   = int(meta_proba.argmax())
    tumor_label = LABELS[meta_idx]  # one of 4 tumor labels

    # OOD / irrelevant gate
    if too_colorful or (meta_top < THR["maxprob"]) or (H > THR["entropy"]):
        pred_label = "Irrelevant Image"
    else:
        pred_label = tumor_label

    return render_template("index.html", prediction=pred_label)

if __name__ == "__main__":
    # debug=True for local dev; change to False for production
    app.run(debug=True)
