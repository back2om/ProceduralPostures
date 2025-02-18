
# BERT Fine-Tuning for Multi-label Classification For Procedural Postures

Procedural postures are summary of how cases arrive before the court. It describes the procedural history along with the prior decisions.  The task is to automate the labeling of judicial opinions  with procedural postures. This project fine-tunes a DistilBERT model for multi-label classification based on input data in JSON format.

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

## Training and Validation Loss

The training and validation loss for the model is shown below:

![Training and Validation Loss](training_validation_loss.png)
