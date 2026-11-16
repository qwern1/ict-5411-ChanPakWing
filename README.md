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

Download the full dataset from the
[`handwritting_data_all`](https://github.com/chenkenanalytic/handwritting_data_all)
repository. On Windows, clone it into your `Downloads` folder and extract the
archive. Set `--dataset` to the extracted `cleaned_data` folder:

```text
C:\Users\<your-name>\Downloads\handwritting_data_all\all_data\cleaned_data
```

## Run

From the project directory, run the full dataset training command:

```powershell
cd C:\Users\<your-name>\Desktop\assment\ict-5412-
.\.venv\Scripts\python.exe train_character_recognizer.py `
  --dataset "C:\Users\<your-name>\Downloads\handwritting_data_all\all_data\cleaned_data" `
  --output .\artifacts `
  --epochs 10 `
  --batch-size 128 `
  --expected-classes 13022
```

The training compares two CNN models and saves the results in `artifacts\`:

```powershell
artifacts\compact_cnn.keras
artifacts\deep_cnn.keras
artifacts\best_model.keras
artifacts\class_names.txt
artifacts\model_comparison.csv
```

`best_model.keras` is copied from the model with the highest held-out test
accuracy. Keep the generated files local; they are not required in GitHub.
Copy the two model accuracies from `model_comparison.csv` into the final
report.

The downloaded `Traditional-Chinese-Handwriting-Dataset-master` archive is the
4,803-character common-word dataset. Sixteen folders contain too few images for
the required first-40/remaining-test split, so the loader skips them and trains
on the 4,787 eligible classes. Use `--expected-classes 4787
--allow-different-class-count` when experimenting with that smaller dataset.

## Write a character and recognise it

After training, start the drawing window:

```powershell
.\.venv\Scripts\python.exe draw_character.py `
  --model .\artifacts\best_model.keras `
  --labels .\artifacts\class_names.txt
```

Write one Chinese character on the canvas and click **辨識**. The program
shows the most likely character and the top five probabilities. Click **清除**
to start another test. The interface uses mouse handwriting; it does not
upload a photograph.

## Accuracy and result interpretation

Test accuracy is calculated using the held-out images that were not used for
training:

```text
test accuracy = correctly classified test images / all test images
```

The percentage printed after training is the measured accuracy of the selected
model on the test set. It is different from the confidence percentage shown
for one handwritten prediction. Confidence only describes that single
prediction.

The earlier five-class development experiment achieved approximately **80%**
test accuracy after 10 epochs. This is only a small-scale demonstration and
must not be reported as the accuracy of the full assignment. The final result
for the full dataset must be copied from `model_comparison.csv` after training
finishes. If classes with fewer than 41 images are skipped, report the actual
number of eligible classes and output neurons.

## What is uploaded to GitHub

This repository contains the Python source code, README, and dependency files.
The large image dataset, virtual environment, user samples, and generated
`.keras` models are intentionally not uploaded. Download the dataset and run
the training commands locally before starting the recognition program.
