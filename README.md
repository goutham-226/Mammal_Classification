# Mammal Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

A convolutional neural network built with TensorFlow/Keras that classifies images of mammals into 45 species categories.

## Overview

`Mammal_Classification.ipynb` trains a CNN from scratch on a folder of labelled mammal images. It covers the full loop: loading and augmenting images, defining the model, training with early stopping, plotting accuracy and AUC curves, and evaluating with a classification report and confusion matrix. The last two cells run a prediction on a single image file.

## Requirements

- Python 3.10+
- TensorFlow 2.16 or newer

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install tensorflow scikit-learn matplotlib numpy pillow scipy jupyter
```

`pillow` and `scipy` aren't imported directly but are needed anyway — Pillow decodes the JPEGs, SciPy powers the rotation and zoom augmentations.

On Apple Silicon, `pip install tensorflow-metal` adds GPU acceleration. It's optional and occasionally fussy about versions, so add it only if CPU training is too slow.

## Dataset layout

The notebook expects an image directory with one subfolder per class. Keras infers the labels from the folder names, and the folder count must match the size of the final `Dense` layer (currently 45).

```
.
├── Mammal_Classification.ipynb
├── mammals/
│   ├── african_elephant/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── bear/
│   ├── bobcat/
│   └── ...            (45 class folders total)
└── test/
    └── bear.jpg       used by the single-image prediction cell
```

Paths in the notebook are relative, and they resolve from the directory Jupyter was launched in — not from the notebook file. If `flow_from_directory` reports "Found 0 images belonging to 0 classes", that mismatch is usually the cause.

## Running it

```bash
jupyter notebook Mammal_Classification.ipynb
```

Run the cells top to bottom. Training is 25 epochs at batch size 20 with early stopping on validation AUC (patience 20, best weights restored).

## Model architecture

Input is 256×256 RGB. Three convolutional blocks, each `Conv2D → MaxPooling2D → Dropout`:

| Block | Filters | Kernel | Stride | Dropout |
|-------|---------|--------|--------|---------|
| 1     | 32      | 5×5    | 3      | 0.2     |
| 2     | 64      | 3×3    | 1      | 0.1     |
| 3     | 128     | 5×5    | 1      | 0.2     |

Followed by `Flatten` and a 45-unit softmax layer. Trained with Adam (lr = 0.001) and categorical cross-entropy, tracking categorical accuracy and AUC.

Training images are augmented on the fly: rescaled to [0,1], zoom ±10%, rotation ±25°, width/height shifts of 5%, and random horizontal flips. Validation images are only rescaled.

## Known issues

**Validation set is the training set.** Both `ImageDataGenerator`s call `flow_from_directory` on the same `mammals/` directory, so the validation metrics are measured on data the model has already seen and will look better than they are. To fix, add a split:

```python
training_data = ImageDataGenerator(rescale=1.0/255, validation_split=0.2, ...)
validation_data = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

training_iterator = training_data.flow_from_directory(
    directory, ..., subset='training', shuffle=True, seed=42)
validation_iterator = validation_data.flow_from_directory(
    directory, ..., subset='validation', shuffle=False, seed=42)
```

**Classification report compares mismatched arrays.** The report cell predicts on `training_iterator` but takes `true_classes` from `validation_iterator.classes`, so the predictions and labels describe different images. Predict on `validation_iterator` instead, and make sure it was created with `shuffle=False` — otherwise `.classes` won't line up with the prediction order regardless.

**PyDataset warning on `fit()`.** A `UserWarning` about `super().__init__(**kwargs)` comes from Keras's own legacy `DirectoryIterator`, not from this notebook. It's harmless; the practical effect is that `workers` and `use_multiprocessing` have no effect, so data loading stays single-threaded.

**`ImageDataGenerator` is deprecated.** It still works in Keras 3 as legacy code but is no longer maintained. `tf.keras.utils.image_dataset_from_directory` combined with augmentation layers (`RandomFlip`, `RandomRotation`, `RandomZoom`) is the current approach and is noticeably faster.
