Brain Tumor Prediction — Flask Web App

A simple, single-page web app that predicts brain tumor type from an MRI image.
Results shown: Glioma, Meningioma, Pituitary, No Tumor, or Irrelevant Image (when the photo isn’t a brain MRI or the model is unsure).

Why this matters: real users often upload the wrong image. Instead of giving a confident but wrong label, the app safely says Irrelevant Image. That’s responsible ML.

What’s inside (in plain English)

A clean Flask UI: Upload an image → Predict → See result.

An ensemble of three CNN backbones with a lightweight meta-model for stable output.

A no-retraining rejection rule: if confidence is low, uncertainty is high, or the picture is too colorful for an MRI, we return Irrelevant Image.

Thresholds are editable in a small config file so strictness can be tuned easily.

How to run (short)

Create a Python virtual environment and activate it.

Install requirements from requirements.txt.

Put the model files (three .pth weights + one .joblib) and the three tiny JSON configs into the models/ folder.

Run the app and open the local URL shown in the terminal.

Upload an image, press Predict, and check the result.

(If PyTorch gives wheel issues on your machine, install CPU wheels. No GPU required.)

How “Irrelevant Image” works (brief)

We don’t train a 5th class. We decide to reject when:

the top class probability is too low (low confidence),

the probability distribution is too spread out (high entropy),

the image looks unusually colorful for an MRI (colorfulness check).

All three cut-offs live in the config and can be adjusted without touching training code.

What reviewers should look at

User experience: focused, minimal, and easy to demo.

Reliability: ensemble + meta-learner gives steadier predictions than a single model.

Safety: built-in rejection avoids misleading outputs on non-MRI photos.

Maintainability: thresholds and model file names are driven by small JSONs, not hard-coded.

Tech snapshot

Flask • PyTorch (timm backbones) • scikit-learn (meta-learner) • Pillow/Numpy
Runs on CPU. Works locally. Screenshots are included for a quick glance at the UI.
