# 🎓 Student Score Predictor

A full-stack Python application that predicts a student's **final exam score** based on:
- 📚 **Study hours per day**
- 😴 **Sleep hours per day**

Built with **FastAPI** (backend) + **Streamlit** (frontend) + **scikit-learn** (ML).

---

## Features

| Tab | Description |
|-----|-------------|
| 📊 Data Overview | Histograms and statistics for study hours, sleep hours, and exam scores |
| 🔗 Correlations | Heatmap and 3D scatter of the three variables |
| 🤖 Model Metrics | MAE, RMSE, R² for Linear Regression and Random Forest |
| 🎯 Predict Score | Sliders → predicted score with gauge chart and 95% confidence band |
| 📈 What-If Analysis | Sweep one variable to see how predicted score changes |

---

## Prerequisites

- Python **3.9+**
- pip

---

## Setup

```bash
# 1. Navigate to the project folder
cd student_score_predictor

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the App

You need **two terminals** — one for the backend, one for the frontend.

### Terminal 1 — Backend (FastAPI)

```bash
cd student_score_predictor
uvicorn backend.main:app --reload --port 8000
```

The API will be available at: http://localhost:8000  
Interactive docs: http://localhost:8000/docs

### Terminal 2 — Frontend (Streamlit)

```bash
cd student_score_predictor
streamlit run frontend/app.py
```

The UI will open automatically at: http://localhost:8501

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/stats` | Descriptive statistics |
| GET | `/metrics` | Model evaluation metrics |
| GET | `/correlation` | Pearson correlation matrix |
| GET | `/scatter-data` | Sampled dataset rows for charts |
| POST | `/predict` | Predict exam score |

### Example prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours": 6.0, "sleep_hours": 7.5, "model": "random_forest"}'
```

---

## Project Structure

```
student_score_predictor/
├── data/
│   └── enhanced_student_habits_performance_dataset.csv
├── backend/
│   ├── __init__.py
│   ├── main.py        # FastAPI app
│   ├── model.py       # ML logic (Linear Regression + Random Forest)
│   └── schemas.py     # Pydantic models
├── frontend/
│   ├── __init__.py
│   └── app.py         # Streamlit 5-tab UI
├── requirements.txt
└── README.md
```

---

## Dataset

**Enhanced Student Habits & Performance Dataset**  
Key columns used: `study_hours_per_day`, `sleep_hours`, `exam_score`  
80 / 20 train-test split with `random_state=42`.
