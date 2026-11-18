"""Train and compare TensorFlow CNNs for Chinese handwritten character recognition.

The dataset is expected to contain one directory per character. Files are sorted
before the first 40 samples are used for training and the remaining samples are
used for testing.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMAGE_SIZE = (64, 64)
TRAIN_SAMPLES_PER_CHARACTER = 40
TARGET_TRAIN_SAMPLES_PER_CHARACTER = 200
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class DatasetSplit:
    class_names: tuple[str, ...]
    train_files: tuple[tuple[Path, ...], ...]
    test_files: tuple[tuple[Path, ...], ...]


def discover_split(
    dataset_dir: Path,
    train_samples_per_character: int = TRAIN_SAMPLES_PER_CHARACTER,
) -> DatasetSplit:
    """Find character folders and split each folder without mixing train/test data."""
    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())

    class_files: list[tuple[str, tuple[Path, ...]]] = []
    has_directory_layout = any(
        any(
            path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            for path in class_dir.iterdir()
        )
        for class_dir in class_dirs
    )
    if has_directory_layout:
        class_files = []
        for class_dir in class_dirs:
            files = tuple(
                sorted(
                    path
                    for path in class_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in SUPPORTED_EXTENSIONS
                )
            )
            if not files:
                class_files.append((class_dir.name, files))
                continue
            class_name = class_dir.name
            if class_dir.name.isdecimal():
                filename_label, separator, _sample_number = files[0].stem.rpartition("_")
                if separator and filename_label:
                    class_name = filename_label
            class_files.append((class_name, files))
    else:
        files_by_class: dict[str, list[Path]] = {}
        for image_path in sorted(
            path
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            class_name, separator, _sample_number = image_path.stem.rpartition("_")
            if not separator or not class_name:
                continue
            files_by_class.setdefault(class_name, []).append(image_path)
        class_files = [
            (class_name, tuple(files))
            for class_name, files in sorted(files_by_class.items())
        ]

    train_files: list[tuple[Path, ...]] = []
    test_files: list[tuple[Path, ...]] = []
    class_names: list[str] = []
    skipped_classes: list[str] = []
    for class_name, files in class_files:
        if len(files) <= train_samples_per_character:
            skipped_classes.append(f"{class_name} ({len(files)} 张)")
            continue
        class_names.append(class_name)
        train_files.append(files[:train_samples_per_character])
        test_files.append(files[train_samples_per_character:])

    if skipped_classes:
        print(
            f"Skipped {len(skipped_classes)} classes with fewer than "
            f"{train_samples_per_character + 1} images"
        )
    if not class_names:
        raise ValueError("没有类别满足训练集和测试集划分要求")
    return DatasetSplit(tuple(class_names), tuple(train_files), tuple(test_files))


def _sample_records(
    files_by_class: Sequence[Sequence[Path]], repeats: int
) -> list[tuple[str, int]]:
    return [
        (str(image_path), class_index)
        for class_index, class_files in enumerate(files_by_class)
        for _ in range(repeats)
        for image_path in class_files
    ]


def _load_image(path: tf.Tensor) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32) / 255.0


def _augmentation_pipeline() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomRotation(0.08, fill_mode="constant", fill_value=1.0),
            layers.RandomTranslation(
                0.10, 0.10, fill_mode="constant", fill_value=1.0
            ),
            layers.RandomZoom(
                (-0.15, 0.15), fill_mode="constant", fill_value=1.0
            ),
        ],
        name="offline_style_augmentation",
    )


def make_dataset(
    files_by_class: Sequence[Sequence[Path]],
    batch_size: int,
    training: bool,
    target_samples_per_class: int = TARGET_TRAIN_SAMPLES_PER_CHARACTER,
) -> tf.data.Dataset:
    """Create a dataset with exactly at least the requested samples per class.

    Training samples are repeated and independently transformed, so every
    character contributes at least 200 examples while the test set is untouched.
    """
    repeats = (
        max(1, math.ceil(target_samples_per_class / len(files_by_class[0])))
        if training
        else 1
    )
    records = _sample_records(files_by_class, repeats if training else 1)
    paths = np.array([record[0] for record in records], dtype=str)
    labels = np.array([record[1] for record in records], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(min(len(records), 10000), seed=42)
    augmentation = _augmentation_pipeline() if training else None

    def load_and_augment(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = _load_image(path)
        if augmentation is not None:
            image = augmentation(image, training=True)
        return image, label

    return (
        dataset.map(load_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_compact_cnn(num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(*IMAGE_SIZE, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="compact_cnn")


def build_deep_cnn(num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(*IMAGE_SIZE, 1))
    x = inputs
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.15)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="deep_cnn")


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def _evaluate_model(
    model: keras.Model,
    test_dataset: tf.data.Dataset,
    class_count: int,
) -> float:
    result = model.evaluate(test_dataset, verbose=0, return_dict=True)
    accuracy = float(result["accuracy"])
    if not 0.0 <= accuracy <= 1.0 or class_count < 2:
        raise RuntimeError("TensorFlow returned an invalid evaluation result")
    return accuracy


def train_and_compare(
    split: DatasetSplit,
    output_dir: Path,
    batch_size: int,
    epochs: int,
) -> list[tuple[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "class_names.txt").write_text(
        "\n".join(split.class_names) + "\n", encoding="utf-8"
    )
    train_dataset = make_dataset(split.train_files, batch_size, training=True)
    test_dataset = make_dataset(split.test_files, batch_size, training=False)
    models = (build_compact_cnn, build_deep_cnn)
    scores: list[tuple[str, float]] = []

    for build_model in models:
        model = build_model(len(split.class_names))
        compile_model(model, learning_rate=1e-3)
        model.fit(train_dataset, epochs=epochs, verbose=1)
        accuracy = _evaluate_model(model, test_dataset, len(split.class_names))
        scores.append((model.name, accuracy))
        model.save(output_dir / f"{model.name}.keras")
        print(f"{model.name}: test accuracy={accuracy:.4%}")

    scores.sort(key=lambda item: item[1], reverse=True)
    with (output_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "test_accuracy"])
        writer.writerows(scores)
    best_model, best_accuracy = scores[0]
    shutil.copy2(
        output_dir / f"{best_model}.keras",
        output_dir / "best_model.keras",
    )
    print(f"Best model: {best_model} ({best_accuracy:.4%})")
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--expected-classes", type=int, default=13065)
    parser.add_argument(
        "--allow-different-class-count",
        action="store_true",
        help="Permit a small sample dataset during development.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = discover_split(args.dataset)
    if not args.allow_different_class_count and len(split.class_names) != args.expected_classes:
        raise ValueError(
            f"Expected {args.expected_classes} character classes, "
            f"found {len(split.class_names)}. Use --allow-different-class-count "
            "only for development subsets."
        )
    train_and_compare(split, args.output, args.batch_size, args.epochs)


if __name__ == "__main__":
    main()
