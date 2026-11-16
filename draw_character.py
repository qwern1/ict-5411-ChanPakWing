"""Draw one Chinese character and classify it with a trained TensorFlow model."""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps

CANVAS_SIZE = 400
IMAGE_SIZE = (64, 64)
BRUSH_WIDTH = 28
MIN_CONFIDENCE = 0.0


class CharacterDrawer:
    def __init__(
        self, model_path: Path, labels_path: Path, examples_dir: Path
    ) -> None:
        self.model = tf.keras.models.load_model(model_path)
        self.model_path = model_path
        self.labels_path = labels_path
        self.examples_dir = examples_dir
        self.labels = labels_path.read_text(encoding="utf-8").splitlines()
        if self.model.output_shape[-1] != len(self.labels):
            raise ValueError(
                f"模型有 {self.model.output_shape[-1]} 個輸出，但標籤檔案有 "
                f"{len(self.labels)} 個類別"
            )

        self.root = tk.Tk()
        self.root.title("繁體中文手寫字元辨識")
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            cursor="crosshair",
        )
        self.canvas.pack(padx=12, pady=12)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.drawer = ImageDraw.Draw(self.image)
        self.last_point: tuple[int, int] | None = None
        self.result = tk.StringVar(value="請在上方寫一個漢字")

        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        controls = tk.Frame(self.root)
        controls.pack(pady=(0, 12))
        tk.Button(controls, text="辨識", command=self.predict).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(controls, text="清除", command=self.clear).pack(side=tk.LEFT, padx=4)
        tk.Label(
            self.root, textvariable=self.result, font=("Microsoft YaHei", 16)
        ).pack(pady=(0, 12))

    def start_stroke(self, event: tk.Event) -> None:
        self.last_point = (event.x, event.y)

    def draw_stroke(self, event: tk.Event) -> None:
        if self.last_point is None:
            self.start_stroke(event)
            return
        point = (event.x, event.y)
        self.canvas.create_line(
            *self.last_point,
            *point,
            fill="black",
            width=BRUSH_WIDTH,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
        )
        self.drawer.line(
            [self.last_point, point],
            fill=0,
            width=BRUSH_WIDTH,
            joint="curve",
        )
        self.last_point = point

    def end_stroke(self, _event: tk.Event) -> None:
        self.last_point = None

    def clear(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.drawer = ImageDraw.Draw(self.image)
        self.result.set("請在上方寫一個漢字")

    def _prepare_input(self) -> np.ndarray:
        ink = ImageOps.invert(self.image)
        bounding_box = ink.getbbox()
        if bounding_box is None:
            raise ValueError("請先在畫布上寫一個漢字")

        character = self.image.crop(bounding_box)
        side = max(character.size)
        padding = max(4, int(side * 0.15))
        square = Image.new("L", (side + padding * 2, side + padding * 2), 255)
        offset = ((square.width - character.width) // 2, (square.height - character.height) // 2)
        square.paste(character, offset)
        square.thumbnail((56, 56), Image.Resampling.LANCZOS)

        centered = Image.new("L", IMAGE_SIZE, 255)
        centered.paste(
            square,
            ((IMAGE_SIZE[0] - square.width) // 2, (IMAGE_SIZE[1] - square.height) // 2),
        )
        return np.asarray(centered, dtype=np.float32) / 255.0

    def predict(self) -> None:
        try:
            input_image = self._prepare_input()
        except ValueError as error:
            messagebox.showwarning("提示", str(error))
            return
        probabilities = self.model.predict(input_image[None, ..., None], verbose=0)[0]
        top_indices = np.argsort(probabilities)[-5:][::-1]
        best_index = int(top_indices[0])
        top_results = "，".join(
            f"{self.labels[int(index)]} {probabilities[int(index)]:.1%}"
            for index in top_indices
        )
        confidence = float(probabilities[best_index])
        if confidence < MIN_CONFIDENCE:
            result = f"無法可靠辨識（最高信心度 {confidence:.1%}）"
            message = f"目前模型可能尚未訓練此字元。\n\n候選結果：{top_results}"
        else:
            result = f"辨識結果：{self.labels[best_index]}（{confidence:.1%}）"
            message = f"最可能的字元：{self.labels[best_index]}\n\n{top_results}"
        self.result.set(result)
        messagebox.showinfo("辨識結果", message)

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("full_trial_artifacts/compact_cnn.keras")
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("full_trial_artifacts/class_names.txt")
    )
    parser.add_argument("--examples", type=Path, default=Path("user_samples"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"找不到模型檔案：{args.model}")
    if not args.labels.is_file():
        raise FileNotFoundError(f"找不到類別標籤檔案：{args.labels}")
    CharacterDrawer(args.model, args.labels, args.examples).run()


if __name__ == "__main__":
    main()
