❤️ Heart Disease Prediction using Machine Learning

A complete Python + Flask project featuring RandomForest, XGBoost, Model Tuning, and a Local Interactive Website.

📌 Overview

This project predicts whether a patient has heart disease using machine learning models trained on the Cleveland Heart Disease dataset.
It includes:

✔ A full ML pipeline (data preprocessing → training → evaluation → inference)
✔ Tuned RandomForest achieving 98%+ accuracy
✔ XGBoost model for comparison
✔ A beautiful Flask-based local website for user interaction
✔ Custom UI with images, probability gauge, explanations, and health advice
✔ Visualizations for model evaluation
✔ Easy-to-follow folder structure

📁 Project Directory Structure
HeartDisease_Prediction/
│
├── data/
│   ├── raw/
│   │   └── heart.csv               # original dataset
│   └── processed/                  # (optional) cleaned files
│
├── models/
│   ├── heart_disease_model.joblib  # tuned RandomForest model
│   └── heart_disease_xgb.joblib    # XGBoost model (optional)
│
├── reports/
│   ├── figures/                    # saved evaluation charts
│   └── metrics/                    # classification reports, JSON logs
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py                    # train RandomForest
│   ├── train_xgboost.py            # train XGBoost
│   ├── tune_random_forest.py       # hyperparameter tuning
│   ├── evaluate.py                 # evaluate saved model
│   └── inference.py                # run ML inference on sample patient
│
├── web/
│   ├── app.py                      # Flask server
│   ├── static/
│   │   ├── heart_bg.jpg
│   │   ├── logo.png
│   │   ├── healthy.png
│   │   └── unhealthy.png
│   └── templates/
│       └── index.html              # website UI
│
├── venv/                           # virtual environment
├── requirements.txt
└── README.md

🛠️ Installation Instructions
1️⃣ Clone or download the project

Place the project folder anywhere you like.

2️⃣ Create a Python virtual environment
python -m venv venv


Activate it:

Windows

venv\Scripts\activate


Mac / Linux

source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

💡 Machine Learning Pipeline Usage
4️⃣ Train RandomForest Model
python -m src.train


This:

Loads dataset

Preprocesses data

Trains RF model

Prints accuracy, ROC-AUC

Saves model → models/heart_disease_model.joblib

5️⃣ Hyperparameter Tuning (Recommended)
python -m src.tune_random_forest


This uses cross-validation + RandomizedSearchCV to build a superior model.

Outputs:

New optimized model saved

Best parameters

Improved metrics

6️⃣ Train XGBoost model
python -m src.train_xgboost


Saves:

models/heart_disease_xgb_model.joblib

7️⃣ Evaluate saved model
python -m src.evaluate


Prints:

Accuracy

ROC-AUC

Confusion Matrix

Classification Report

8️⃣ Run inference on a custom patient
python -m src.inference


You can modify the example_patient dictionary inside the file to test custom values.

🌐 Running the Local Website

The interactive UI is built with Flask.

1️⃣ Navigate to project folder
cd HeartDisease_Prediction

2️⃣ Activate venv
venv\Scripts\activate

3️⃣ Run the Flask app
python web/app.py


Then open your browser:

http://127.0.0.1:5000

🎨 Website Features
✔ Beautiful UI

Background hero image (heart_bg)

Website logo

Healthy/unhealthy result images

Clean card layout

Semi-circular probability gauge

Inline hints for each medical field

✔ Interactive prediction

Enter patient details

Model predicts: Heart Disease / No Heart Disease

Shows probability

Personalized feedback message

Healthy advice section

✔ Image handling

Place all UI images here:

web/static/
    heart_bg.jpg
    logo.png
    healthy.png
    unhealthy.png

📊 Model Visualization Options

You can generate charts such as:

1. Confusion Matrix Heatmap
2. ROC Curve
3. Precision–Recall Curve
4. Feature Importance (RandomForest + XGBoost)

All charts should be stored inside:

reports/figures/


We use:

matplotlib

seaborn

scikit-learn metrics

🧠 Models Used
RandomForestClassifier

Tuned version reaches ~98% accuracy

Stable on structured tabular data

Easy to interpret with feature importances

XGBoost

Gradient boosting approach

Performs well with optimized parameters

Great for competition-level performance

👨‍⚕️ Health Disclaimer

This model is trained on a publicly available clinical dataset and is for:

Educational

Demonstration

Portfolio

purposes only.

It is NOT intended for real medical diagnosis or decision-making.

🧩 Future Enhancements

Deploy Flask app to a cloud service

Add SHAP explanations for deep interpretability

Add user authentication

Build full React frontend

🏁 Conclusion

This project demonstrates:

End-to-end ML engineering

Clean project architecture

Real-time inference through a web interface

Strong predictive performance

Professional UI + model interpretability