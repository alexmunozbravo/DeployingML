from __future__ import absolute_import, division, print_function

import os
import random
import pickle

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import (DataLoader, Dataset)
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Seeds to obtain the same results
random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)

if torch.cuda.is_available():
    torch.cuda.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)

# Change the working directory to the root of the project
os.chdir(os.path.join(os.getcwd(), os.pardir, os.pardir))

# Model definition
class RoBERTaEncoder(nn.Module):
    def __init__(self, num_labels1):
        super().__init__()
        self.encoder = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest').roberta
        
        # This classifier will be trained to output the Toxicity of SBIC dataset
        self.classifier1 = nn.Linear(self.encoder.config.hidden_size, num_labels1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
                                                       
        logits1 = self.classifier1(pooled_output)
                                                       
        return logits1
    
# Classes and functions to load and manage the dataset
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, note="", community=tuple()):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.note = note
        self.community = community

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid
        

class MAProcessor(object):
    
    def get_direct_control_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")), "missed_micro") + self._create_examples(self._read_tsv(os.path.join(data_dir, "nonmicro_large_train.pkl")), "clean") + self._create_examples(self._read_tsv(os.path.join(data_dir, "hs_train.pkl")), "hateful")
    
    def get_dirctr_corrected_train_examples(self, data_dir, correction_dir, correction_size):
        """See base class."""
        sorted_ex_for_correction = self._read_tsv(os.path.join(correction_dir, "sorted_ex_for_correction.pkl"))
        correct_idx_list = sorted_ex_for_correction[:correction_size]
        examples = self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")), "missed_micro") + self._create_examples(self._read_tsv(os.path.join(data_dir, "nonmicro_large_train.pkl")), "clean") + self._create_examples(self._read_tsv(os.path.join(data_dir, "hs_train.pkl")), "hateful")
        new_examples = []
        for i, example in enumerate(examples):
            if i in correct_idx_list:
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label="1")) # flip the label
            else:
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label=example.label))
        return new_examples
    
    def get_dirctr_checked_train_examples(self, data_dir, correction_dir, correction_size):
        """See base class."""
        sorted_ex_for_correction = self._read_tsv(os.path.join(correction_dir, "sorted_ex_for_correction.pkl"))
        correct_idx_list = sorted_ex_for_correction[:correction_size]
        len_micro_train = len(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")))
        examples = self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")), "missed_micro") + self._create_examples(self._read_tsv(os.path.join(data_dir, "nonmicro_large_train.pkl")), "clean") + self._create_examples(self._read_tsv(os.path.join(data_dir, "hs_train.pkl")), "hateful")
        new_examples = []
        for i, example in enumerate(examples):
            if i in correct_idx_list and i < len_micro_train: # only flip the true microaggressions
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label="1")) # flip the label
            else:
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label=example.label))
        return new_examples
    
    def get_dirctr_gold_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")), "gold_micro") + self._create_examples(self._read_tsv(os.path.join(data_dir, "nonmicro_large_train.pkl")), "clean") + self._create_examples(self._read_tsv(os.path.join(data_dir, "hs_train.pkl")), "hateful")
    
    def get_direct_control_test_examples(self, data_dir):
        """See base class."""
        micro_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_test.pkl")), "gold_micro")
        clean_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "nonmicro_large_test.pkl")), "clean")
        hs_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "hs_test.pkl")), "hateful")
        return (micro_test_ex, clean_test_ex, hs_test_ex)
    
    def get_adv_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_adv.pkl")), "missed_micro")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line[0]
            community = line[1]
            
            if set_type == "hateful":
                label = "1"
            elif set_type == "missed_micro":
                label = "0"
            elif set_type == "gold_micro":
                label = "1"
            elif set_type == "clean":
                label = "0"
            else:
                raise ValueError("Check your set type")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, community=community))
        return examples
    
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "rb") as f:
            pairs = pickle.load(f)
            return pairs

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              guid=example.guid))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# Accuracy metric for binary classification (we'll use it by class)
def accuracy(logits, labels):
    predictions = torch.max(logits, dim=1)[1]
    correct_predictions = torch.eq(predictions, labels).sum().item()
    accuracy = correct_predictions / len(labels)
    return accuracy

def get_dict_comms(dd, communities, counter):
    for text, label, com in dd:
        for comm in com:
            if comm not in communities.keys():
                communities[comm] = counter
                counter += 1

    return communities, counter

# Class for the dataset, inherited from Dataset class from PyTorch
class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])
        label = int(self.df.iloc[idx]['label'])
        task = int(self.df.iloc[idx]['Task'])
        comms = list(self.df.iloc[idx]['communities'])
        
        # encode the text using the tokenizer
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label),
            'task': torch.tensor(task),
            'communities': torch.tensor(comms)
        }
    
def get_datasets():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    top_classes = 15


    # create the dataset
    ma_processor = MAProcessor()
    train_examples = ma_processor.get_dirctr_gold_train_examples("./data/processed/")
    micro_test_examples, clean_test_examples, hs_test_examples = ma_processor.get_direct_control_test_examples("./data/processed/")

    # Get communities
    communities = dict()
    counter = 0

    train_dataset = [[ex.text_a, ex.label, [item.lower() if item is not None else "None" for item in ex.community]] for ex in train_examples]
    train_dataset = [[text, label, list(set(communities))] for text, label, communities in train_dataset]

    micro_test_examples = [[ex.text_a, ex.label, [item.lower() if item is not None else "None" for item in ex.community]] for ex in micro_test_examples]
    clean_test_examples = [[ex.text_a, ex.label, [item.lower() if item is not None else "None" for item in ex.community]] for ex in clean_test_examples]
    hs_test_examples = [[ex.text_a, ex.label, [item.lower() if item is not None else "None" for item in ex.community]] for ex in hs_test_examples]

    micro_test_examples = [[text, label, list(set(communities))] for text, label, communities in micro_test_examples]
    clean_test_examples = [[text, label, list(set(communities))] for text, label, communities in clean_test_examples]
    hs_test_examples = [[text, label, list(set(communities))] for text, label, communities in hs_test_examples]

    communities, counter = get_dict_comms(train_dataset,communities, counter)
    communities, counter = get_dict_comms(micro_test_examples,communities, counter)
    communities, counter = get_dict_comms(clean_test_examples,communities, counter)
    communities, counter = get_dict_comms(hs_test_examples,communities, counter)

    dataset = pd.concat([pd.DataFrame(train_dataset), pd.DataFrame(micro_test_examples), pd.DataFrame(clean_test_examples), pd.DataFrame(hs_test_examples)])
    dataset.columns = ["Text", "Label", "Communities"]
    dataset["Communities"] = dataset["Communities"].apply(lambda row: row[0])
    groups = dataset.groupby("Communities").groups

    for idx, index in groups.items():
        groups[idx] = len(index)

    res = dict(sorted(groups.items(), key = lambda x: x[1], reverse = True)[:top_classes])
    dataset["Communities"] = dataset["Communities"].apply(lambda row: row if row in res else "Other")

    train_dataset = dataset.iloc[:12000,]
    micro_test_examples = dataset.iloc[12000:13000,]
    clean_test_examples = dataset.iloc[13000:14000,]
    hs_test_examples = dataset.iloc[14000:,]

    mapping = {}
    for idx, key in enumerate(res.keys()):
        mapping[key] = idx

    mapping['Other'] = len(mapping)

    train_dataset = [[text, label, torch.nn.functional.one_hot(torch.tensor(mapping[com]), num_classes=len(mapping))] for text,label,com in train_dataset.to_numpy()]
    micro_test_examples = [[text, label, torch.nn.functional.one_hot(torch.tensor(mapping[com]), num_classes=len(mapping))] for text,label,com in micro_test_examples.to_numpy()]
    clean_test_examples = [[text, label, torch.nn.functional.one_hot(torch.tensor(mapping[com]), num_classes=len(mapping))] for text,label,com in clean_test_examples.to_numpy()]
    hs_test_examples = [[text, label, torch.nn.functional.one_hot(torch.tensor(mapping[com]), num_classes=len(mapping))] for text,label,com in hs_test_examples.to_numpy()]

    dd = pd.DataFrame(train_dataset)
    dd.columns = ["text", "label", "communities"]
    dd["Task"] = 0

    micro_test_dd = pd.DataFrame(micro_test_examples)
    micro_test_dd.columns = ["text", "label", "communities"]
    micro_test_dd["Task"] = 0

    clean_test_dd = pd.DataFrame(clean_test_examples)
    clean_test_dd.columns = ["text", "label", "communities"]
    clean_test_dd["Task"] = 0

    hs_test_dd = pd.DataFrame(hs_test_examples)
    hs_test_dd.columns = ["text", "label", "communities"]
    hs_test_dd["Task"] = 0

    dd = CustomDataset(dd, tokenizer, max_length = 128)
    micro_test_dd = CustomDataset(micro_test_dd, tokenizer, max_length = 128)
    clean_test_dd = CustomDataset(clean_test_dd, tokenizer, max_length = 128)
    hs_test_dd = CustomDataset(hs_test_dd, tokenizer, max_length = 128)
    
    return dd, micro_test_dd, clean_test_dd, hs_test_dd

def main():
    # Get the datasets
    train_dataset, micro_test_dd, clean_test_dd, hs_test_dd = get_datasets()

    # Define device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoBERTaEncoder(2)
    model = model.to(device)

    # define the optimizer and learning rate
    num_epochs = 2
    lr = 5e-5
    loss = nn.CrossEntropyLoss()

    # create the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader))

    # train the model
    model.train()
    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader):
            # set the gradients to zero
            optimizer.zero_grad()

            # compute the outputs and targets for each task
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputsSBIC = model(input_ids, attention_mask)

            # Compute loss
            loss_SBIC = loss(outputsSBIC, labels)

            # backpropagate and update the parameters
            loss_SBIC.backward()
            optimizer.step()

        print(f"epoch {epoch}: {loss_SBIC}")

        print("TEST phase")

        model.eval()    
        for idx, dd in enumerate([micro_test_dd, clean_test_dd, hs_test_dd]):
            test_dataloader = DataLoader(dd, batch_size=32, shuffle=True)

            total_test_accuracy_SBIC = 0
            total_num_examples = 0

            with torch.no_grad():
                for batch in test_dataloader:
                    # unpack and move to device
                    input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']

                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.cpu()

                    outputSBIC = model(input_ids, attention_mask)

                    logits_SBIC = F.softmax(outputSBIC, dim=1).detach().cpu()

                    tmp_test_correct_SBIC = accuracy(logits_SBIC, labels)

                    total_test_accuracy_SBIC += tmp_test_correct_SBIC

                    total_num_examples += 1

            print(f"Test mean accuracy with SBIC: {total_test_accuracy_SBIC/total_num_examples*100} %")
    
    # save the model
    torch.save(model.state_dict(), "./models/model.pt")


if __name__ == "__main__":
    main()