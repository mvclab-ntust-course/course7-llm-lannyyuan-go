# Homework 6 : Learn basic pytorch
* The ***pytorch tutorial notebook*** provide several materials to learn Pytorch, Google colab and ipd.
* In the last part of the notebook, there's a simple example to train MLP on CPU. 
* There's the screenshot of my week 7 colab training result.
<img width="1208" alt="screenshots_1" src="https://github.com/mvclab-ntust-course/course7-llm-lannyyuan-go/assets/122262894/31ef178e-e3fa-4b98-9df6-c6cdd3e08149">


# Homework 7 : Learn basic pytorch
### README

## Overview

This homework involves fine-tuning a DistilBERT model for sequence classification on the IMDB dataset using Hugging Face Transformers library. The training process incorporates LoRA (Low-Rank Adaptation) to enhance the model's performance. The model is trained to classify movie reviews as either positive or negative.
Because My Colab GPU quota was used up and I think it's more convenient to run on ipynb on my server, so I don't use Colab to do this part.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- PEFT (Parameter Efficient Fine-Tuning)

## Setup

1. **Install dependencies:**

   ```bash
   pip install torch transformers datasets peft
   ```

## Dataset

The IMDB dataset is used for training and validation. The dataset is loaded and processed using the `datasets` library.

## Training

### Model and Tokenizer

The base model used is `distilbert-base-cased`. The tokenizer is initialized using the same model checkpoint.

### Data Preparation

The dataset is truncated to the first 50 tokens for each review to speed up processing. The dataset is then tokenized and prepared in batches of 16 examples.

### DataLoader

DataLoaders are created for training and evaluation datasets.

### LoRA Configuration

A LoRA configuration is applied to the model to optimize performance. The configuration includes parameters like rank number, alpha (scaling factor), dropout probability, and target modules.

### Training Arguments

The training arguments are defined to control various aspects of the training process such as batch size, learning rate, evaluation strategy, and more.

### Trainer

The Hugging Face `Trainer` is used to manage the training process. It handles the training loop, evaluation, and other functionalities.

### Compute Metrics

A custom function `compute_metrics` is defined to compute the accuracy of the model during evaluation.

### Training the Model

The training process is initiated using the `trainer.train()` method.


## Usage

To run the training process, execute the script:

```bash
python train.py
```
## Result
* Config 1
  <img width="452" alt="截圖 2024-06-23 凌晨2 40 14" src="https://github.com/mvclab-ntust-course/course7-llm-lannyyuan-go/assets/122262894/01523f67-a86e-44cf-921d-4bf28e79d1f7">

* Config 2
  <img width="550" alt="截圖 2024-06-23 凌晨2 41 37" src="https://github.com/mvclab-ntust-course/course7-llm-lannyyuan-go/assets/122262894/0ca3e4b3-b695-4591-8446-822e998712fd">

* Config 3
  <img width="547" alt="截圖 2024-06-23 凌晨2 42 55" src="https://github.com/mvclab-ntust-course/course7-llm-lannyyuan-go/assets/122262894/daf5fe91-1508-4b71-9831-bda77fbbf3ee">

# Latex version
* The pdf is named as Homework_Latex.pdf, please check.
