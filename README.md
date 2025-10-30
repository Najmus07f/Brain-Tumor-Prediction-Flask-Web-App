Brain Tumor Prediction (Flask)

A clean, one-page web app that predicts the tumor type from a brain MRI.
Possible results: Glioma, Meningioma, Pituitary, No Tumor, or Irrelevant Image (when the upload is not a brain MRI or the model is unsure).
Why it’s different: the app refuses to guess on bad inputs. If confidence is low or the image looks non-MRI, it returns Irrelevant Image instead of a wrong label.

Demo

Home: upload an image and click Predict
Result: predicted class + confidence

Highlights

Simple UI: drag-and-drop style flow—upload → predict → result.
Stronger predictions: three CNN backbones combined with a small meta-model.
Safe output: a built-in rejection rule returns Irrelevant Image for non-MRI photos or uncertain cases (no retraining needed).
CPU-friendly: runs locally without a GPU.
Config-driven: small JSON files control model names, image size, and thresholds.

How to run

Create and activate a Python virtual environment.
Install dependencies from requirements.txt.

Put model files in models/:
three backbone weights (*.pth)
stacking_meta_lr.joblib
config JSONs: labels.json, preprocess.json, ensemble_config.json
Start the app (python app.py) and open the local URL shown in the terminal.
Upload an MRI image and press Predict.
Tip: If PyTorch wheels fail on Windows, install the CPU wheels (no GPU required).

What “Irrelevant Image” really means

We don’t train a 5th class. Instead, we reject when the input looks risky:
Low confidence: top class probability is below a threshold.
High uncertainty: probability distribution is too spread out (high entropy).
Not MRI-like: colorfulness is unusually high for MRI images.


What reviewers can check quickly
UX: minimal, fast, and easy to test.
Engineering: ensemble + meta-learner for stability; thresholds in configs, not hard-coded.
Safety: graceful handling of wrong uploads is built-in.
Maintainability: small, readable files; no heavy framework.

Notes & limits

Works on standard 2D MRI slices; unusual formats may need preprocessing.
Confidence is provided for transparency, not clinical decision-making.
If many valid MRIs are rejected, loosen thresholds. If non-MRIs slip through, tighten them.

Credits / Contact
Built with Flask, PyTorch (timm), scikit-learn, Pillow, and Numpy.
Happy to share the model bundle via a GitHub Release and walk through the design if needed.
