"""Fine-tune a trained character recognizer with user-labelled examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = (64, 64)
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


def _load_image(path: tf.Tensor) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32) / 255.0


def _find_examples(
    examples_dir: Path, labels: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    label_indices = {label: index for index, label in enumerate(labels)}
    paths: list[str] = []
    targets: list[int] = []
    unknown_labels: set[str] = set()

    for label_dir in sorted(path for path in examples_dir.iterdir() if path.is_dir()):
        label_index = label_indices.get(label_dir.name)
        if label_index is None:
            unknown_labels.add(label_dir.name)
            continue
        for image in sorted(label_dir.iterdir()):
            if image.is_file() and image.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(str(image))
                targets.append(label_index)

    if unknown_labels:
        names = ", ".join(sorted(unknown_labels))
        raise ValueError(f"Examples contain labels absent from the model: {names}")
    if not paths:
        raise ValueError(f"No labelled images found in {examples_dir}")
    return np.asarray(paths, dtype=str), np.asarray(targets, dtype=np.int32)


def fine_tune(
    model_path: Path,
    labels_path: Path,
    examples_dir: Path,
    epochs: int = 3,
    batch_size: int = 8,
) -> None:
    """Fine-tune the model using every saved, labelled user example."""
    labels = tuple(labels_path.read_text(encoding="utf-8").splitlines())
    paths, targets = _find_examples(examples_dir, labels)
    dataset = tf.data.Dataset.from_tensor_slices((paths, targets)).shuffle(
        len(paths), seed=42, reshuffle_each_iteration=True
    )
    augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.04, fill_mode="constant", fill_value=1.0),
            layers.RandomTranslation(
                0.04, 0.04, fill_mode="constant", fill_value=1.0
            ),
            layers.RandomZoom(
                (-0.08, 0.08), fill_mode="constant", fill_value=1.0
            ),
        ]
    )

    def prepare(path: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = augmentation(_load_image(path), training=True)
        return image, target

    dataset = dataset.map(prepare, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size
    )
    model = keras.models.load_model(model_path)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.fit(dataset, epochs=epochs, verbose=0)

    temporary_path = model_path.with_name(f"{model_path.stem}.updated.keras")
    model.save(temporary_path)
    temporary_path.replace(model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/best_model.keras"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/class_names.txt"))
    parser.add_argument("--examples", type=Path, default=Path("user_samples"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fine_tune(args.model, args.labels, args.examples, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
