
from datasets import Dataset
import json
import numpy as np

def load_data(file_path):
    input_data = []
    with open(file_path, 'r') as file:
        for line in file:
            input_data.append(json.loads(line))
    return input_data

def labels_to_multihot(labels, label2id):
    multihot_vector = np.zeros(len(label2id), dtype=int)
    for label in labels:
        if label in label2id:
            multihot_vector[label2id[label]] = 1
    return multihot_vector

def preprocess_data(input_data, label2id):
    data = []
    for entry in input_data:
        text = ' '.join([section['headtext'] + ' ' + ' '.join(section['paragraphs']) for section in entry['sections']])
        labels = entry['postures']
        multi_hot_labels = labels_to_multihot(labels, label2id).astype(np.float32)
        data.append({'text': text.strip(), 'labels': multi_hot_labels, 'documentId': entry['documentId']})
    return Dataset.from_dict({'text': [entry['text'] for entry in data], 'labels': [entry['labels'] for entry in data], 'documentId': [entry['documentId'] for entry in data]})
