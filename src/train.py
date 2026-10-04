"""
Training routines and checkpoint callbacks.
"""

from pathlib import Path
import tensorflow as tf

from src.config import (
    CHECKPOINTS_DIR,
    BEST_EFFICIENTNET_PATH,
    BEST_RESNET_PATH,
    BEST_RESNET_FT_PATH,
    NUM_EPOCHS,
    NUM_FINE_TUNE_EPOCHS,
    LEARNING_RATE,
    RESNET_FINE_TUNE_LR,
)
from src.models import (
    build_efficientnet_b0,
    build_resnet50,
    setup_resnet50_fine_tuning,
)


def train_efficientnet(train_ds, val_ds):
    """Compiles and trains EfficientNet-B0 transfer learning model."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    model, _ = build_efficientnet_b0()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=criterion, metrics=["accuracy"])

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_EFFICIENTNET_PATH),
        monitor="val_loss",
        save_best_only=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS,
        callbacks=[checkpoint_cb],
        verbose=1,
    )
    return model, history


def train_resnet50(train_ds, val_ds, run_fine_tuning: bool = True):
    """Compiles and trains ResNet50 with optional conv5 fine-tuning."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    model, base_model = build_resnet50()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=criterion, metrics=["accuracy"])

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_RESNET_PATH),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS,
        callbacks=[checkpoint_cb],
        verbose=1,
    )

    ft_history = None
    if run_fine_tuning:
        model.load_weights(str(BEST_RESNET_PATH))
        setup_resnet50_fine_tuning(base_model)

        ft_optimizer = tf.keras.optimizers.Adam(learning_rate=RESNET_FINE_TUNE_LR)
        model.compile(optimizer=ft_optimizer, loss=criterion, metrics=["accuracy"])

        ft_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_RESNET_FT_PATH),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )

        ft_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=NUM_FINE_TUNE_EPOCHS,
            callbacks=[ft_checkpoint_cb],
            verbose=1,
        )

    return model, history, ft_history
