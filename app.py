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
with open(os.path.join(MODEL_DIR, "labels.json"), "r", encoding="utf-8") as f:
    LABELS = json.load(f)  # 4 tumor labels + "Irrelevant Image" at the end

with open(os.path.join(MODEL_DIR, "preprocess.json"), "r", encoding="utf-8") as f:
    PP = json.load(f)
MEAN = PP["mean"]
STD  = PP["std"]
GRAYSCALE_TO_RGB = PP.get("grayscale_to_rgb", True)
PER_MODEL_SIZE = PP.get("per_model_img_size", {})

with open(os.path.join(MODEL_DIR, "ensemble_config.json"), "r", encoding="utf-8") as f:
    ENS = json.load(f)
THR = ENS["thresholds"]
CLASS_ORDER = ENS.get("class_order", LABELS[:4])   # enforced training order for the 4 tumor classes

# meta bundle (LogisticRegression trained on the 4 tumor classes)
META_BUNDLE = joblib.load(os.path.join(MODEL_DIR, ENS["meta_path"]))
meta = META_BUNDLE["meta"]

device = torch.device("cpu")

def make_tfms(img_size: int):
    # Mirror training: plain Resize -> (optional) Grayscale->3ch -> ToTensor -> Normalize.
    ops = [transforms.Resize((img_size, img_size))]
    if GRAYSCALE_TO_RGB:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)

# --------- load base models ----------
MODEL_OBJS = []
NUM_CLASSES = len(LABELS) - 1  # exclude "Irrelevant Image"

for bm in ENS["base_models"]:
    name, ckpt = bm["name"], bm["ckpt"]
    img_size = int(PER_MODEL_SIZE.get(name, bm.get("img_size", 224)))

    model = timm.create_model(name, pretrained=False, num_classes=NUM_CLASSES)
    state = torch.load(os.path.join(MODEL_DIR, ckpt), map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval().to(device)

    MODEL_OBJS.append({
        "name": name,
        "img_size": img_size,
        "model": model,
        "tfm": make_tfms(img_size)
    })

def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())

# --------- flask app ----------
app = Flask(__name__)

app.logger.info("LABELS.json: %s", LABELS)
app.logger.info("Meta classes (as stored in joblib): %s", getattr(meta, "classes_", None))
app.logger.info("CLASS_ORDER (training order): %s", CLASS_ORDER)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if not f or f.filename == "":
        return redirect(url_for("home"))

    filename = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    f.save(path)

    # open image
    pil = Image.open(path).convert("RGB")

    # quick colorfulness gate (grayscale MRI should be low)
    arr = np.array(pil, dtype=np.float32) / 255.0
    colorfulness = float(np.std(arr, axis=2).mean())
    too_colorful = colorfulness > THR["color_saturation"]

    # base models -> concat probs
    parts = []
    with torch.no_grad():
        for b in MODEL_OBJS:
            x = b["tfm"](pil).unsqueeze(0).to(device)
            probs = torch.softmax(b["model"](x), dim=1).cpu().numpy()[0]  # (4,)
            # IMPORTANT: interpret these probs in the TRAINING class order (CLASS_ORDER)
            # If your backbones already used CLASS_ORDER, this is a no-op. Otherwise, set
            # per-model orders in ensemble_config and reindex here. (kept simple for now)
            parts.append(probs)

    X = np.concatenate(parts, axis=0).reshape(1, -1)  # (1, 4 * num_models)

    # meta prediction (columns follow meta.classes_)
    meta_proba = meta.predict_proba(X)[0]
    meta_top   = float(meta_proba.max())
    H          = entropy(meta_proba)

    # Map meta column -> training class NAME
    col_idx = int(meta_proba.argmax())
    cls_val = meta.classes_[col_idx]
    if isinstance(cls_val, (int, np.integer)):
        class_name = CLASS_ORDER[int(cls_val)]
    else:
        class_name = str(cls_val)

    # Final gate
    if too_colorful or (meta_top < THR["maxprob"]) or (H > THR["entropy"]):
        pred_label = "Irrelevant Image"
    else:
        pred_label = class_name

    # robust logging (no crash on Windows console)
    try:
        order = meta_proba.argsort()[-2:][::-1]
        def name_at(i):
            v = meta.classes_[i]
            return CLASS_ORDER[int(v)] if isinstance(v, (int, np.integer)) else str(v)
        top2 = [(name_at(i), float(meta_proba[i])) for i in order]
        app.logger.info("file=%s color=%.4f top2=%s maxp=%.4f H=%.4f final='%s'",
                        filename, colorfulness, top2, meta_top, H, pred_label)
    except Exception as e:
        app.logger.exception("debug logging failed: %s", e)

    return render_template("index.html", prediction=pred_label)

if __name__ == "__main__":
    app.run(debug=True)
