# 🌸 Iris Species Interactive Classifier (Streamlit ML Web App)

An interactive Machine Learning web app built using **Streamlit**, **Scikit-learn**, and **Joblib**. Deployed with real-time prediction and probability visualizations.

## 🚀 Features
- **Joblib Model Persistence**: Loads serialized Random Forest artifacts seamlessly.
- **Cached Loading**: Uses `@st.cache_resource` for zero-latency slider interactions.
- **Strict Feature Alignment**: Enforces exact feature column orders between training and inference to avoid data drift/errors.
- **Real-Time Prediction**: Instant live updates showing predicted class, confidence score, and distribution bar chart.

---

## 🛠️ Local Setup & Running in VS Code

### 1. Open the project in VS Code
Open VS Code, press `Ctrl + O` (or `File` -> `Open Folder...`), and select the `streamlit-ml-app` directory.

### 2. Create and activate a Virtual Environment
Open the VS Code Terminal (``Ctrl + ` `` or `Terminal -> New Terminal`):

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python train_model.py
```
This generates `model.joblib`.

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Your default browser will automatically open: `http://localhost:8501`.

---

## 🌐 Free Cloud Deployment (Streamlit Community Cloud)

1. Push this directory to a public GitHub repository.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **"New app"**.
4. Select your repository, branch (`main`), and set Main file path to `app.py`.
5. Click **"Deploy!"**. You will get a live public URL.
