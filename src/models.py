"""
Model architectures and Transfer Learning definitions.
"""

import tensorflow as tf
from src.config import NUM_CLASSES


def get_data_augmentation() -> tf.keras.Sequential:
    """Standardized augmentation pipeline applied during training."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomContrast(0.2),
        ],
        name="data_augmentation",
    )


def build_efficientnet_b0(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Constructs an EfficientNet-B0 model with frozen ImageNet feature extractor.
    """
    augmentation = get_data_augmentation()
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")
    x = augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="predictions"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_MRI")
    return model, base_model


def build_resnet50(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Constructs a ResNet50 model with frozen ImageNet feature extractor.
    """
    augmentation = get_data_augmentation()
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")
    x = augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="predictions"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_MRI")
    return model, base_model


def setup_resnet50_fine_tuning(base_model: tf.keras.Model) -> None:
    """
    Unfreezes 'conv5_' layers of ResNet50 while preserving BatchNorm frozen for stability.
    """
    base_model.trainable = True
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = layer.name.startswith("conv5_")
