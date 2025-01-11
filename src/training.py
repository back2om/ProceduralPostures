########################### Cell 1 ################################################
####### Install necessary libraries/dependencies and import them ##############
####### Information: his cell utilizes the Hugging Face Transformers library for fine-tuning BERT,
####### along with NumPy for numerical operations, and PyTorch for model training.

pip install datasets 'accelerate>=0.26.0' scikit-learn

from transformers import BertForSequenceClassification, TrainingArguments, Trainer, BertTokenizer,DistilBertForSequenceClassification
import numpy as np
from datasets import Dataset
import torch
from sklearn.metrics import f1_score, precision_score, recall_score,classification_report
import matplotlib.pyplot as plt

########################### Cell 2 ################################################
####### Load input data from a JSON file ##########################################
####### Information: The data is expected to be in JSON Lines format, where each line is a separate JSON object.
####### This cell reads the data into a Python list called "input_data" ###########

import json
from datasets import Dataset

input_data = []
with open('./TRDataChallenge2023.txt', 'r') as file:
    for line in file:
        input_data.append(json.loads(line))


########################### Cell 3 ################################################
####### EDA (Exploratory Data Analysis) ##########################################

####### Lets take a sneakpeak at the first datapoint and number of datapoints
print(input_data[0],len(input_data))
####### Results: We see a total of 18,000 datapoints each containing "documentId", "postures" and "sections". 
####### Lets check if the entry fields are consistent in input data #### 

all_lens = []
for x in input_data:
    all_lens.append(len(x))

print(list(set(all_lens)))
###### We see three unique values here as expected
###### That is, documentId, postures and sections are availble for each entry

###### We count number of non-empty values in these entries
doc_info =    [x['documentId'] for x in input_data]
postures_info =    [x['postures'] for x in input_data]
sec_info =    [x['sections'] for x in input_data]

count_doc = sum(1 for item in doc_info if item)
count_post = sum(1 for item in postures_info if item)
count_sec = sum(1 for item in sec_info if item)

print("Number of docs are:", count_doc)
print("Number of postures are:", count_post)
print("Number of sec are:", count_sec)

###### We see postures are missing some values but documentId and sections are complete
###### Number of docs are: 18000
###### Number of postures are: 17077
###### Number of sec are: 18000

####### Lets look at the distribution of postures or labels
postures_allinfo = []

for sublist in postures_info:
    if sublist:
        postures_allinfo.append(sublist[0])
    else:
        pass

from collections import Counter
Counter(postures_allinfo)
###############################################################
###### Sample Results:
###### 'On Appeal': 4942,
###### 'Motion to Compel Arbitration': 244,
###### 'Motion for Protective Order': 100,
###### 'Motion to Reargue': 25,
###### We see that prcedural labels are highly imbalanced. Some labels occur more frequently than others
###### These high frequency labels can be more easily predicted than lower frequency labels.
###### We are also facing a multi-label classification, where each text can belong to one or more labels. 
###### This requires adjustments in both the model and the loss function to handle multiple labels per example.


########################### Cell 4 ################################################
####### Prepare multi-label input dataset ##########################################

###### Extract all unique labels from the 'postures' field in input_data
all_labels = set()

###### Loop through the input data and collect all the unique labels from 'postures'
for entry in input_data:
    all_labels.update(entry['postures'])

###### Convert the set to a list (if you want a sorted list, use sorted())
all_labels = sorted(list(all_labels))

###### Print the all_labels to verify
print("Total Unique labels are:", len(all_labels))
print("All labels:", all_labels)
###### Result: Total Unique labels are: 224

###### Create a Label-to-Index Mapping
###### First, create a mapping from the label strings to integer indices. 
###### This is necessary for transforming the labels into a format the model 
###### can understand (i.e., binary vectors).
    
label2id = {label: idx for idx, label in enumerate(all_labels)}  #dictionary to map each label to a unique index
print(label2id)                                                  # Print the label to index mapping


### Function to convert the labels to multi-hot encoding
def labels_to_multihot(labels, label2id):
    """
    Convert a list of labels to a binary vector (multi-hot encoding).
    """
    # Create a vector of zeros with length equal to the number of unique labels
    multihot_vector = np.zeros(len(label2id), dtype=int)
    
    # Set the corresponding positions to 1 for each label present in the current example
    for label in labels:
        if label in label2id:
            multihot_vector[label2id[label]] = 1
    
    return multihot_vector

####### The "sections" or "text" is composed of multiple sections (like 'headtext' and 'paragraphs'), 
####### we need to process it in a way that concatenates or combines these sections before 
####### tokenizing the data. The text field must be either a string or a list of strings

# Step 1: Concatenate text and labels
data = []
for entry in input_data:
    # Concatenate all sections' text
    text = ''
    for section in entry['sections']:
        text += section['headtext'] + ' ' + ' '.join(section['paragraphs']) + ' '
    
    # Extract postures (labels)
    labels = entry['postures']  # This is already in list format

    documentId = entry['documentId']  # This is already in list format

    # Convert the labels to multi-hot encoding
    multi_hot_labels = labels_to_multihot(labels, label2id).astype(np.float32)
    
    # Append the processed data
    data.append({
        'text': text.strip(),  # Strip to remove any leading/trailing spaces
        'labels': multi_hot_labels,
        'documentId':documentId
    })

    # Now you can proceed to create the dataset
from datasets import Dataset
dataset = Dataset.from_dict({
    'text': [entry['text'] for entry in data],
    'labels': [entry['labels'] for entry in data],
    'documentId': [entry['documentId'] for entry in data]
})

# Print out the dataset
print(dataset)

########################### Cell 5 #########################################
####### Prepare tokenized dataset ##########################################

from transformers import BertTokenizer

# Tokenizer initialization
# Load the BERT tokenizer. The tokenizer splits text into subwords, adds special tokens,
# and converts text to token IDs for input to the model.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize_function(examples):
    # Tokenize text
    tokenized_inputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256, return_tensors='pt')
    
    # Ensure labels are in the correct dtype (torch.float32)
    tokenized_inputs['labels'] = torch.tensor(examples['labels'], dtype=torch.float32)
    
    return tokenized_inputs

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Check the tokenized dataset
print(tokenized_dataset)

############################################################################
########################### Cell 6 #########################################
####### Model Training and Evaluation ######################################

# Split the dataset into train and test splits (80/20 or 90/10 depending on your preference)
train_dataset = tokenized_dataset.train_test_split(test_size=0.1)['train']
test_dataset = tokenized_dataset.train_test_split(test_size=0.1)['test']


### Load pre-trained BERT model for sequence classification with the appropriate number of labels.The model is configured for multi-label classification.
### BERT is powerful, it’s also computationally expensive. Consider using smaller models if the task doesn't require the full capacity of BERT. 
### The DistilBERT model is a smaller, faster version of BERT that maintains good performance on many tasks.


# Number of labels is equal to the length of the multi-hot encoded vector
num_labels = len(label2id)
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels,problem_type="multi_label_classification")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels,problem_type="multi_label_classification")


# Define compute_metrics function
def compute_metrics(eval_pred):
    """
    Computes macro F1, precision, and recall scores for the model's predictions.
    """
    logits, labels = eval_pred
    # Convert logits to probabilities and then to binary predictions
    predictions = (logits > 0).astype(int)
    sample_accuracy = np.mean([np.array_equal(pred, true) for pred, true in zip(predictions, labels)])


    # Macro F1, Precision, and Recall
    macro_f1 = f1_score(labels, predictions, average='macro')
    macro_precision = precision_score(labels, predictions, average='macro')
    macro_recall = recall_score(labels, predictions, average='macro')

    return {
        'accuracy': sample_accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall
    }

### Define training arguments
### These settings specify hyperparameters like batch size, learning rate, and number of training epochs.
from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',         # Directory for saving model checkpoints
    evaluation_strategy="epoch",    # Evaluate the model after each epoch
    save_strategy="epoch",
    learning_rate=2e-5,             # Learning rate for the optimizer
    per_device_train_batch_size=4,  # Adjust based on available memory
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.01,              # Weight decay for regularization
    logging_dir='./logs',           # Directory for logging training progress
    logging_steps=15,               # Log progress every 15 steps
    save_steps=500,                 # Save checkpoint every 500 steps
    load_best_model_at_end=True,
)

### Define the Trainer
### The Trainer handles training, evaluation, and prediction
trainer = Trainer(
    model=model,                  # The model to train
    args=training_args,
    train_dataset=train_dataset,  # Training dataset
    eval_dataset=test_dataset,    # Evaluation dataset 
    compute_metrics=compute_metrics          # You can add your own evaluation metric function
)

### Train the model
### This step begins the fine-tuning process of above declared model
trainer.train()


# Evaluate the model on the test dataset
# Evaluate the model on the test set and print the results
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)


# Get predictions from the model
predictions_output = trainer.predict(test_dataset)
logits = predictions_output.predictions
true_labels = predictions_output.label_ids

# Convert logits to binary predictions
threshold = 0.5
binary_predictions = (logits > threshold).astype(int)

# Generate per-label metrics
label_names = list(label2id.keys())  # Retrieve label names from label2id
report = classification_report(true_labels, binary_predictions, target_names=label_names, zero_division=0)

# Print the detailed classification report
print(report)

############ Plot Training and Testing
# Extract logs
train_logs = trainer.state.log_history

# Filter losses for each epoch
train_loss = [log['loss'] for log in train_logs if 'loss' in log and 'epoch' in log]
eval_loss = [log['eval_loss'] for log in train_logs if 'eval_loss' in log]

# Get the corresponding epochs
epochs_train = [log['epoch'] for log in train_logs if 'loss' in log and 'epoch' in log]
epochs_eval = [log['epoch'] for log in train_logs if 'eval_loss' in log]

# Ensure dimensions match
assert len(eval_loss) == len(epochs_eval), "Mismatch between evaluation loss and epochs."

# Plot training and evaluation loss
plt.figure(figsize=(10, 6))
plt.plot(epochs_train, train_loss, label="Training Loss", marker='o')
plt.plot(epochs_eval, eval_loss, label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss vs. Epoch")
plt.legend()
plt.grid(True)
plt.show()

############################################################################
########################### Cell 7 #########################################
####### Save the model #####################################################

# After training, the fine-tuned model is saved to the specified directory.
model.save_pretrained('./final_model_ChoudharyOm')
tokenizer.save_pretrained('./final_model_ChoudharyOm')