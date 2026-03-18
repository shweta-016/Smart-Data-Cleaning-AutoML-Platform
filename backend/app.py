from flask import Flask, render_template, request, send_file
import pandas as pd
import os

from backend.data_cleaning import clean_data
from backend.preprocessing import preprocess_data
from backend.automl_engine import run_automl
from backend.model_evaluator import evaluate_model
from backend.utils import save_model,generate_plot

# Base directory of project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Absolute paths
TEMPLATE_DIR = os.path.join(BASE_DIR, "frontend", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


# Home page route
@app.route("/")
def home():
    return render_template("index.html")


# Route for dataset upload
@app.route("/upload", methods=["POST"])
def upload():

    # Get uploaded CSV file
    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    # Save file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Load dataset using pandas
    df = pd.read_csv(filepath)

    # ---------- STEP 1: DATA CLEANING ----------
    df = clean_data(df)
    # Save cleaned dataset
    cleaned_path = os.path.join("uploads", "cleaned_data.csv")
    df.to_csv(cleaned_path, index=False)

    # ---------- STEP 2: PREPROCESSING ----------
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # ---------- STEP 3: AUTOML MODEL SELECTION ----------
    best_model, best_name, scores = run_automl(X_train, X_test, y_train, y_test)

    # ---------- STEP 4: MODEL EVALUATION ----------
    accuracy = evaluate_model(best_model, X_test, y_test)

    # ---------- STEP 5: SAVE MODEL ----------
    save_model(best_model)

    # ---------- STEP 6: CREATE VISUALIZATION ----------
    generate_plot(scores)

    return render_template(
        "result.html",
        model=best_name,
        accuracy=round(accuracy * 100, 2),
        scores=scores
)
from flask import send_file

@app.route("/download_cleaned")
def download_cleaned():

    path = os.path.join("C:/Users/Admin/OneDrive/Desktop/Smart-data-cleaning/uploads", "cleaned_data.csv")

    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
