
from sklearn.metrics import classification_report
import torch
from train import trainer, test_dataset, label2id

# Get predictions
predictions_output = trainer.predict(test_dataset)
logits = predictions_output.predictions
true_labels = predictions_output.label_ids

# Convert logits to binary predictions
threshold = 0.5
binary_predictions = (logits > threshold).astype(int)

# Generate per-label metrics
label_names = list(label2id.keys())
report = classification_report(true_labels, binary_predictions, target_names=label_names, zero_division=0)

print(report)
