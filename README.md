# Bird Classification Program

## Introduction

This program is designed for bird classification using deep learning models. It includes functionality for training and evaluating models on the CUB-200-2011 dataset.

You can either manually download the datafile or download it using `download.py`.

## Requirements

- Python 3.x
- Pipenv

## Installation

1. Clone this repository to your local machine.
2. Navigate to the project directory in your terminal.
3. Install dependencies using Pipenv:

```bash
pipenv install
```

## Usage

### Enable Pipenv Shell Environment

```bash
pipenv shell
```

### Downloading the Dataset

Before training the models, it's necessary to download the CUB-200-2011 dataset. Run the following command:

```bash
python download.py
```

### Training and Evaluation

To train and evaluate a model, use the bird-classification-training script with the following arguments:

```bash
python train.py <model> [-n NUM_EPOCHS] [-e]
```

- `<model>`: Specify the model to use for training and evaluation. Choose from the following options:
	- `transfer_cnn`: Transfer learning-based CNN model.
	- `attention_net`: NTS-Net model based of this repo [yangze0930/NTS-Net](https://github.com/yangze0930/NTS-Net).
- `-n NUM_EPOCHS`, `--num-epochs NUM_EPOCHS` (optional): Number of epochs for training (default is 10).
- `-e, --evaluate-only` (optional): If provided, the model will only be evaluated without further training.

**Example: **

Train with different epochs

```bash
python train.py transfer_cnn -n 15
```

Only evaluate model

```bash
python train.py transfer_cnn -e
```

## Models

### Transfer Learning (Resnet50)

Basic model using transfer learning from resnet50 as base model.

## Dataset

The CUB-200-2011 dataset is used for training and evaluation. It contains images of 200 bird species.
