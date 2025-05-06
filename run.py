from zenml import step, pipeline, log_metadata
import numpy as np
from typing import Annotated, Tuple
from sklearn.linear_model import LinearRegression


@step
def load_data() -> Tuple[
    Annotated[np.ndarray, "training_data"],
    Annotated[np.ndarray, "training_labels"]
]:
    data = np.random.rand(100, 2)
    labels = np.random.rand(100)
    return data, labels


@step
def train_model(
    data: np.ndarray,
    labels: np.ndarray,
) -> Annotated[LinearRegression, "trained_model"]:
    model = LinearRegression().fit(data, labels)
    print(f"Model coefficients: {model.coef_}, intercept: {model.intercept_}")
    log_metadata(
        metadata={
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
        }
    )
    return model


@pipeline
def basic_pipeline():
    train_model(*load_data())


if __name__ == "__main__":
    basic_pipeline()
