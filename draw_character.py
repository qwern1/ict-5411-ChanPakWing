"""Draw one Chinese character and classify it with a trained TensorFlow model."""

from __future__ import annotations

import argparse
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox, simpledialog

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps

from learn_from_examples import fine_tune

CANVAS_SIZE = 400
IMAGE_SIZE = (64, 64)
BRUSH_WIDTH = 28
MIN_CONFIDENCE = 0.5


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
                f"模型有 {self.model.output_shape[-1]} 个输出，但标签文件有 "
                f"{len(self.labels)} 个类别"
            )

        self.root = tk.Tk()
        self.root.title("中文手写字符识别")
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
        self.result = tk.StringVar(value="请在上方写一个汉字")

        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        controls = tk.Frame(self.root)
        controls.pack(pady=(0, 12))
        tk.Button(controls, text="识别", command=self.predict).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(controls, text="学习", command=self.learn).pack(
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
        self.result.set("请在上方写一个汉字")

    def _prepare_input(self) -> np.ndarray:
        ink = ImageOps.invert(self.image)
        bounding_box = ink.getbbox()
        if bounding_box is None:
            raise ValueError("请先在画布上写一个汉字")

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
            result = f"无法可靠识别（最高置信度 {confidence:.1%}）"
            message = f"当前模型可能没有训练这个字符。\n\n候选结果：{top_results}"
        else:
            result = f"识别结果：{self.labels[best_index]}（{confidence:.1%}）"
            message = f"最可能的字符：{self.labels[best_index]}\n\n{top_results}"
        self.result.set(result)
        messagebox.showinfo("识别结果", message)

    def learn(self) -> None:
        try:
            input_image = self._prepare_input()
        except ValueError as error:
            messagebox.showwarning("提示", str(error))
            return

        label = simpledialog.askstring(
            "添加学习样本",
            "请输入这幅字的正确汉字：",
            parent=self.root,
        )
        if label is None:
            return
        label = label.strip()
        if label not in self.labels:
            messagebox.showerror(
                "无法学习",
                "这个汉字不在当前模型的类别中，不能新增类别。"
                "请使用训练数据中已有的汉字。",
            )
            return

        label_dir = self.examples_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        image_path = label_dir / f"{timestamp}.png"
        Image.fromarray((input_image * 255).astype(np.uint8), mode="L").save(
            image_path
        )
        try:
            fine_tune(self.model_path, self.labels_path, self.examples_dir)
            self.model = tf.keras.models.load_model(self.model_path)
        except (OSError, ValueError, RuntimeError) as error:
            messagebox.showerror("学习失败", str(error))
            return
        self.result.set(f"已学习：{label}")
        messagebox.showinfo(
            "学习完成",
            f"已保存样本并更新模型：{label}\n"
            f"累计样本目录：{self.examples_dir}",
        )

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/best_model.keras"))
    parser.add_argument(
        "--labels", type=Path, default=Path("artifacts/class_names.txt")
    )
    parser.add_argument("--examples", type=Path, default=Path("user_samples"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"找不到模型文件：{args.model}")
    if not args.labels.is_file():
        raise FileNotFoundError(f"找不到类别标签文件：{args.labels}")
    CharacterDrawer(args.model, args.labels, args.examples).run()


if __name__ == "__main__":
    main()
