from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "iris_model.joblib"


def train_and_save_model() -> None:
    iris = load_iris()

    model = LogisticRegression(max_iter=200)
    model.fit(iris.data, iris.target)

    saved_data = {
        "model": model,
        "target_names": iris.target_names.tolist(),
    }

    joblib.dump(saved_data, MODEL_PATH)
    print(f"Model trained and saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
