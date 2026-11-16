# DIT5411 Chinese Handwritten Character Recognition

This project implements the DIT5411 machine-learning assignment with
TensorFlow. It supports the 13,065-class full Chinese handwriting dataset
and its smaller 4,803-class common-word version.

The GitHub repository contains the source code and documentation only.
The image dataset, virtual environment, user samples, and trained `.keras`
models are not committed because of their size. Download the dataset and
train the models locally as described below.

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

The loader also accepts the downloaded archive layout, where images are
stored in nested directories and named like `漢字_20.png`. In that case, pass
the archive's `data` directory as `--dataset`; the filename prefix before the
last underscore is used as the character label.

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

## Download the dataset

The common-word dataset is available from the
[Traditional Chinese Handwriting Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)
repository. It contains 4,803 character classes.

The full assignment dataset contains 13,065 classes and 684,677 images. It
is distributed separately in the
[`handwritting_data_all`](https://github.com/chenkenanalytic/handwritting_data_all)
repository. On Windows, clone it into `Downloads` and extract the split
archive so that the images are available at:

```text
C:\Users\<your-name>\Downloads\handwritting_data_all\all_data\cleaned_data
```

The full archive has 13,065 classes, but 43 classes contain fewer than 41
images. Because this project requires 40 training images plus at least one
test image per class, the loader skips those classes and trains on 13,022
eligible classes. This limitation must be reported with the experimental
results; the output layer therefore has 13,022 neurons for this archive.

## Run

```powershell
uv run --python 3.11 --with-requirements requirements.txt train_character_recognizer.py --dataset .\dataset --expected-classes 13022 --epochs 10 --allow-different-class-count
```

You can also run it through the project launcher:

```powershell
.\run_training.ps1 -Dataset C:\path\to\dataset -Epochs 10
```

The launcher uses `uv` to select Python 3.11 and create/use an isolated
environment. This avoids Windows' `python.exe` application alias and keeps the
TensorFlow environment independent from the project directory. If `uv` is not
on `PATH`, the launcher also checks `%USERPROFILE%\.local\bin\uv.exe`.

The script writes both `.keras` models and `model_comparison.csv` to
`artifacts/`, and saves the class-to-index mapping in `class_names.txt`.
These generated files remain local and are intentionally excluded from
GitHub. Record the resulting test accuracies in the final report.

The downloaded `Traditional-Chinese-Handwriting-Dataset-master` archive is the
4,803-character common-word dataset. Sixteen folders contain too few images for
the required first-40/remaining-test split, so the loader skips them and trains
on the 4,787 eligible classes. Use `--expected-classes 4787
--allow-different-class-count` when experimenting with that smaller dataset.

## Write a character and recognise it

After training, start the drawing window:

```powershell
uv run --python 3.11 --with-requirements requirements.txt draw_character.py
```

Write one Chinese character with the mouse, click **识别**, and the program
will show the most likely character and the top five probabilities. The model
must be trained first because the drawing program loads
`artifacts/best_model.keras` and `artifacts/class_names.txt`.

To let the model learn from your own handwriting, click **学习**, enter the
correct character, and wait for fine-tuning to finish. The labelled drawing is
saved under `user_samples/<character>/` and all saved examples are used when
the model is updated. Only characters already present in
`artifacts/class_names.txt` can be learned; this prevents the model output
shape and label mapping from becoming inconsistent. Keep a backup of
`artifacts/best_model.keras` if you want to restore the original model.
