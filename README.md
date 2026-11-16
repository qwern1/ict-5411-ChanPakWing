# DIT5411 Chinese Handwritten Character Recognition

This project implements the DIT5411 machine-learning assignment with
TensorFlow. It recognises the 13,065 Chinese character classes in the
[Traditional Chinese Handwriting Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset).

## Dataset layout

Extract the dataset so that each character has its own directory:

```text
dataset/
  class_00000/
    000.png
    001.png
    ...
  class_00001/
    000.png
    ...
```

The loader sorts each directory's image names, uses the first 40 images for
training, and reserves all remaining images for testing. It refuses to train
when a class has no held-out image.

## Augmentation and models

Each class starts with 40 training images. The training pipeline repeats those
images enough times to provide at least 200 examples per class and applies a
new random rotation, translation, and scale change to every repeated image.
Test images are never augmented.

Two models are trained and compared:

* `compact_cnn`: three convolution blocks and global average pooling.
* `deep_cnn`: four batch-normalised convolution blocks, dropout, and a dense
  layer.

Both models finish with a softmax layer whose width is the number of dataset
classes (13,065 for the full assignment).

## Run

```powershell
python -m venv C:\tfenv
C:\tfenv\Scripts\python.exe -m pip install -r requirements.txt
C:\tfenv\Scripts\python.exe train_character_recognizer.py --dataset .\dataset --expected-classes 4787 --epochs 10
```

You can also run it through the project launcher:

```powershell
.\run_training.ps1 -Dataset C:\path\to\dataset -Epochs 10
```

On Windows, use a short virtual-environment path such as `C:\tfenv`; TensorFlow
contains deeply nested package paths that can exceed Windows' default path
limit when installed inside a long project path.

The script writes both `.keras` models and `model_comparison.csv` to
`artifacts/`, and saves the class-to-index mapping in `class_names.txt`.
For a small development subset, use
`--allow-different-class-count`; the submitted full run must use all 13,065
classes. Record the resulting test accuracies in the final report.

The downloaded `Traditional-Chinese-Handwriting-Dataset-master` archive is the
4,803-character common-word dataset. Sixteen folders contain too few images for
the required first-40/remaining-test split, so the loader skips them and trains
on the 4,787 eligible classes. The 13,065-class assignment dataset is a
different, larger download.

## Write a character and recognise it

After training, start the drawing window:

```powershell
C:\tfenv\Scripts\python.exe draw_character.py
```

Write one Chinese character with the mouse, click **识别**, and the program
will show the most likely character and the top five probabilities. The model
must be trained first because the drawing program loads
`artifacts/best_model.keras` and `artifacts/class_names.txt`.
