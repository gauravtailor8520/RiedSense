---
title: "RideSense"
emoji: "🚖"
colorFrom: "purple"
colorTo: "orange"
sdk: streamlit
sdk_version: "1.33.0"
app_file: streamlit_app/app.py
pinned: false
---

# 🚖 RideSense — Smart Fare & ETA Prediction Engine

[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?logo=streamlit)](https://streamlit.io)  
[![ML Stack: XGBoost + Sklearn](https://img.shields.io/badge/ML-XGBoost%2C%20Sklearn-blue?logo=scikit-learn)](https://scikit-learn.org)  
[![Deployment: Hugging Face Spaces](https://img.shields.io/badge/Deployed%20on-HuggingFace-orange?logo=huggingface)](https://huggingface.co/spaces/rajesh1804/RideSense)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

> 🚀 **RideSense** is an intelligent ride-hailing prediction system that estimates fares and arrival times using real NYC taxi data. Built with XGBoost and quantile regression, it provides uncertainty-aware predictions through an interactive Streamlit interface.

---

## 🎯 What It Does

**RideSense** answers the critical question: *"What will this ride cost and how long will it take?"*

### Key Capabilities:
- **🕐 ETA Prediction**: XGBoost-powered time estimation
- **💰 Fare Estimation**: Quantile regression with confidence bands (10th, 50th, 90th percentiles)
- **🗺️ Demand Visualization**: Interactive NYC heatmap
- **⚡ Real-time Inference**: Sub-50ms prediction latency
- **📊 Performance Monitoring**: Built-in model metrics and evaluation

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** (recommended)
- **pip** package manager
- **Git** for cloning

### 1. Clone & Setup
```bash
# Clone the repository
git clone https://github.com/rajesh1804/RideSense.git
cd RideSense

# Create virtual environment (recommended)
python -m venv env

# Activate virtual environment
# Windows:
env\Scripts\activate
# Linux/Mac:
source env/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download & Prepare Data
```bash
# Download NYC taxi dataset (~100MB)
python src/download_dataset.py

# Process and clean the data
python src/preprocess.py
```

### 4. Train ML Models
```bash
# Train ETA prediction model (~5 seconds)
python src/train_eta_model.py

# Train fare prediction models (~20 minutes for all quantiles)
python src/train_fare_model.py
```

### 5. Launch Application
```bash
# Start Streamlit app
streamlit run streamlit_app/app.py
```

🎉 **Access the app at:** `http://localhost:8501`

---

## 📋 Detailed Setup Instructions

### For Windows Users:
```powershell
# Clone repository
git clone https://github.com/rajesh1804/RideSense.git
cd RideSense

# Create and activate virtual environment
python -m venv env
env\Scripts\activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download and process data
python src\download_dataset.py
python src\preprocess.py

# Train models
python src\train_eta_model.py
python src\train_fare_model.py

# Run application
streamlit run streamlit_app\app.py
```

### For Linux/Mac Users:
```bash
# Clone repository
git clone https://github.com/rajesh1804/RideSense.git
cd RideSense

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download and process data
python src/download_dataset.py
python src/preprocess.py

# Train models
python src/train_eta_model.py
python src/train_fare_model.py

# Run application
streamlit run streamlit_app/app.py
```

---

## � Training Output Examples

```scss
                        ┌────────────────────────────┐
                        │        User Input          | 
                        │ (Time, Day, Distance)      │
                        └────────────┬───────────────┘
                                     │
                                     ▼
                        ┌────────────────────────────┐
                        │    Feature Engineering     │
                        │ (rush hour, sin/cos, etc.) │
                        └────────────┬───────────────┘
                                     │
                                     ▼
                   ┌────────────────────────────────────────┐
                   │          ML Models (XGBoost, GBR)       │
                   │   ETA Model + Fare Models (Q10/Q50/Q90) │
                   └────────────┬───────────────┬────────────┘
                                │               │
                          ETA Prediction   Fare Prediction
                                │               │
                                ▼               ▼
                     ┌────────────────┐  ┌────────────────────┐
                     │   ETA Output   │  │ Fare Estimate (+/-)│
                     └────────────────┘  └────────────────────┘
                                │               │
                                └──────┬────────┘
                                       ▼
                          ┌───────────────────────────┐
                          │     Streamlit UI (App)     │
                          │ Predict Button triggers ML │
                          │ Heatmap visual via Plotly  │
                          └────────────┬───────────────┘
                                       │
                                       ▼
                   ┌──────────────────────────────────────┐
                   │   🔥 NYC Demand Heatmap              |
                   │       Real-time Plotly Density Map   │
                   └──────────────────────────────────────┘

```

---

## 🔍 Feature Engineering Highlights

- `hour_sin`, `hour_cos`: Circular encoding of time  
- `is_rush_hour`: Encodes commute stress  
- `is_weekend`: Affects ride supply/demand  
- `Quantile regression`: Predicts fare **range** not just point estimate

> ✅ Makes the model explainable and closer to real-world behavior

---

## 📊 Evaluation

> ⚡ Prediction latency (in ms) is logged and displayed in the app UI on each request.

| Model       | Metric    | Value       |
|-------------|-----------|-------------|
| ETA Model   | MAE       | ~3.9 mins   |
| Fare Model  | MAE (Q50) | ~2.3 USD    |

> Metrics auto-logged and displayed in app sidebar.

---


## 🧠 How It Works — Under the Hood of RideSense

RideSense isn't just a demo — it's a modular, production-grade ML platform designed to mirror real-world ride-hailing pricing engines. Here's how it works:

### 🚖 1. Feature Engineering from Raw Taxi Data

Raw trip data from NYC Yellow Taxi dataset is processed to extract meaningful features:

- Distance (in miles)
- Pickup hour (0-23)
- Day of week (0-6)
- Is weekend / Is rush hour
- Time cyclic encodings (sin/cos of hour)

> ✅ Why it matters: Features mimic how actual ride-hailing engines reason about traffic, demand, and surge pricing.

### 🧠 2. ETA & Fare Estimation Models

Two types of models are trained:

- ETA Model: `XGBoost Regressor`
- Fare Models: 3 separate `GradientBoostingRegressor` models for **10th, 50th, and 90th percentile** fare estimates.

This helps simulate a **fare range** with uncertainty — like Uber's real-time quote system.

> ✅ Why it matters: Demonstrates quantile regression and uncertainty estimation for real-world predictions.

### 🧪 3. Streamlit App for Real-Time Inference

The frontend UI lets users input:

- Pickup hour
- Day of week
- Trip distance

And returns:

- ETA prediction  
- Fare prediction (Low / Median / High)  
- **Prediction latency in ms**  
- Dynamic demand heatmap

> ✅ Why it matters: Showcases full-stack ML skill from model to UI deployment with latency logging.

---

## 🧪 Example Flow

1. User selects trip time and distance via Streamlit UI  
2. App predicts:
   - ⏱️ ETA (in minutes)
   - 💸 Fare range (low–mid–high estimate)
3. 🔥 Demand heatmap of NYC shown for visual context  
4. 📊 Sidebar displays live model performance (e.g., MAE)

---

## 📽️ Live Demo

👉 Try it on [Hugging Face](https://huggingface.co/spaces/rajesh1804/RideSense)

<p align="center">
  <img src="assets/ridesense-demo.gif" alt="Demo" width="800"/>
</p>

---

## 🚀 Getting Started Locally

### 1. Clone the repo

```bash
git clone https://github.com/rajesh1804/RideSense.git
cd RideSense
```

### 2. Setup Python 3.10 (Recommended)
Use Python 3.10 for full compatibility.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset and Preprocess
```bash
python src/download_dataset.py  
python src/preprocess.py
```

### 5. Train Models Locally
```bash
(venv) RideSense>python src\train_eta_model.py 
📥 Loading processed data...
🔀 Splitting into train/test...
🚀 Starting ETA model training...
✅ Training completed in 3.75 seconds   
📈 Evaluating model on test set...
📊 ETA MAE: 3.11 minutes
💾 ETA model saved to src/models/eta_model.pkl

(venv) RideSense>python src\train_fare_model.py
🔄 Loading and preprocessing data...
✅ Data loaded and split. Time taken: 5.58s

🚀 Training quantile model for q=0.1...
✅ Done. MAE @ quantile 0.1: 2.72 | Time taken: 441.53s
💾 Model saved to src/models/fare_model_q10.pkl

🚀 Training quantile model for q=0.5...
✅ Done. MAE @ quantile 0.5: 1.85 | Time taken: 483.59s
💾 Model saved to src/models/fare_model_q50.pkl

🚀 Training quantile model for q=0.9...
✅ Done. MAE @ quantile 0.9: 3.44 | Time taken: 442.07s
💾 Model saved to src/models/fare_model_q90.pkl

🎉 All quantile models trained and saved successfully.
```

### 5. Run Streamlit App
```bash
streamlit run streamlit_app/app.py
```

---

## 📁 Project Structure

```css
RideSense/
├── assets/
│   └── ridesense-architecture.png
├── data/
│   └── yellow_tripdata_2023-01.parquet
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train_eta_model.py
│   ├── train_fare_model.py
│   └── models/
│       └── *.pkl
├── streamlit_app/
│   ├── app.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## 🧪 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) for interactive UI
- **ML Models**: XGBoost, GradientBoostingRegressor
- **Metrics**: Scikit-learn (MAE, metrics logging)
- **Feature Engg.**: NumPy, Pandas
- **Deployment**: Hugging Face Spaces

---

## 📍 Heatmap Note

The NYC demand heatmap is currently simulated using pseudo-random normalized ride volumes per zone. This can be replaced with real pickup zone distributions using TLC GeoJSONs.

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

---

## 👋 Author

Built by **Gaurav Tailor**  
🏆 Uber, Lyft, Bolt, Grab  
📧 Say hi at [Gmail](gauravtailor43@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/gaurav-tailor-bb4924223/)
