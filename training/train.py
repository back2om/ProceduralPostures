
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import BertTokenizer
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from preprocess import load_data, preprocess_data

# Load and preprocess data
input_data = load_data('./data/TRDataChallenge2023.txt')
label2id = {'label1': 0, 'label2': 1}  # Example, modify as needed
dataset = preprocess_data(input_data, label2id)

# Tokenize data
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256, return_tensors='pt')
    tokenized_inputs['labels'] = torch.tensor(examples['labels'], dtype=torch.float32)
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the dataset
train_dataset = tokenized_dataset.train_test_split(test_size=0.1)['train']
test_dataset = tokenized_dataset.train_test_split(test_size=0.1)['test']

# Load pre-trained model
num_labels = len(label2id)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels, problem_type="multi_label_classification")

# Define compute_metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits > 0).astype(int)
    macro_f1 = f1_score(labels, predictions, average='macro')
    macro_precision = precision_score(labels, predictions, average='macro')
    macro_recall = recall_score(labels, predictions, average='macro')
    return {'macro_f1': macro_f1, 'macro_precision': macro_precision, 'macro_recall': macro_recall}

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',         # Directory for saving model checkpoints
    evaluation_strategy="epoch",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    weight_decay=0.01,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
