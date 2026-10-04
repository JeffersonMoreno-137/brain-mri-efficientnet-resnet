"""
Evaluation metrics, confusion matrices, and qualitative visualization.
"""

from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import CLASS_NAMES, FIGURES_DIR


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    test_targets: list,
    model_name: str,
    save_figures: bool = True,
) -> Dict[str, Any]:
    """Evaluates model performance on independent test partition."""
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    probabilities = model.predict(test_ds, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    report = classification_report(
        test_targets,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    cm = confusion_matrix(test_targets, predictions)

    if save_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(cm, cmap="Blues")
        plt.colorbar(cax)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicción", fontsize=11)
        ax.set_ylabel("Real", fontsize=11)
        ax.set_title(f"Matriz de Confusión - {model_name}", fontsize=12, fontweight="bold")

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")

        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_").replace("+", "ft")
        plt.savefig(FIGURES_DIR / f"cm_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    return {
        "model_name": model_name,
        "loss": loss,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "predictions": predictions,
    }
