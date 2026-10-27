# ASL Alphabet Recognition — Public Web Demo

A Streamlit web application for recognizing static ASL alphabet signs with a trained MobileNetV2 model.

## Files
- `streamlit_app.py` — browser-based app
- `models/asl_mobilenetv2.keras` — trained model
- `models/class_names.json` — output class labels
- `requirements.txt` — deployment dependencies

The training dataset is not required to run the app.

## Deploy for free using Streamlit Community Cloud
1. Create/sign in to a GitHub account at https://github.com/.
2. Create a **public** repository, for example `asl-alphabet-recognition`.
3. Upload the contents of this folder to the repository root. Ensure `streamlit_app.py`, `requirements.txt`, and the `models/` folder are at the root level.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Select **Create app**, choose the repository and branch, and set the main file path to `streamlit_app.py`.
6. Deploy. Streamlit will install dependencies and provide a public `https://....streamlit.app` link.
7. Open the link and test camera access. The visitor's browser will ask for camera permission; camera access generally requires HTTPS and user permission.

## Run locally
Use Python 3.11 (64-bit) if available, create a virtual environment, then run:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Important limitations
- This app recognizes static alphabet signs from a captured image; it does not currently provide continuous live-video prediction or sentence recognition.
- The original desktop `app.py` is not used by this web demo.
- This web app uses the original model's RGB 160×160 preprocessing and 0–255 float input scale.
- Free hosting can sleep when idle, and the first visit may take time while the app wakes up or loads its model.
