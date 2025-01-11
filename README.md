
# BERT Fine-Tuning for Multi-label Classification

This project fine-tunes a DistilBERT model for multi-label classification based on input data in JSON format.

## Setup

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

### Preprocessing:
Run the preprocessing script to load and tokenize your data:

```bash
python training/preprocess.py
```

### Training:
Train the model with the following command:

```bash
python training/train.py
```

### Testing:
Evaluate the model performance using the test script:

```bash
python testing/test_model.py
```
