import joblib
import os
import matplotlib.pyplot as plt


def save_model(model):
    """
    Save trained model
    """

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/best_model.pkl")


def generate_plot(scores):
    """
    Generate model comparison chart
    """

    models = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(8, 5))

    plt.bar(models, values)

    plt.title("Model Comparison")

    plt.ylabel("Training Score")

    plt.xticks(rotation=45)

    os.makedirs("static", exist_ok=True)

    plt.savefig("frontend/static/model_comparison.png")