import pytest
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

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

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = RoBERTaEncoder(2)
model.load_state_dict(torch.load("./models/model_state_dict.pt", map_location=torch.device('cpu')))
model = model.to(device)

def predict(text):
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(text, padding=True, max_length = 128, truncation=True, return_tensors='pt')
        tokens = tokens.to(device)

        # Perform inference and get the predicted class
        output = model(tokens['input_ids'], tokens['attention_mask'])
        predicted_class = int(torch.argmax(output))
        
        # Print the predicted class
        predicted_class = "Positive" if predicted_class == 0 else "Negative"
        return predicted_class
    

def test_minimum_functionality():
  pos = [
  "good", "realistic", "healthy", "attractive", "appealing", "acceptable", 
  "best", "feasible", "easy", "ideal", "affordable", "economical", "recommended", 
  "exciting", "inexpensive", "obvious", "great", "appropriate", "effective", "excellent",
  ]

  neg = [
    "bitch", "nigger", "idiot", "dickhead", "asshole", "bastard", "motherfucker", 
    "dork", "freak", "jerk", "meathead", "scum", 
  ]

  positive_texts = [f'Exercise is {pos} for your health' for pos_word in pos]
  negative_texts = [f'Exercise is {neg} for your health' for neg_word in neg]
  
  pos_results = [predict(text) for text in positive_texts]
  neg_results = [predict(text) for text in negative_texts]

  assert all('Positive' == sentiment for sentiment in pos_results), "Sentiment positive minimum functionality failed"
  assert all('Negative' == sentiment for sentiment in neg_results), "Sentiment negative minimum functionality failed"

def test_invariance():
  locations = ["Mexico", "Afghanistan", "Kenya", "Peru",
               "Morocco", "Canada", "Germany", "Algeria", "Indonesia"]
  names = ["Michael","Jackson","Jose","Shawn","Steven","Jackson","Noah","Chad","Bryan","Stephen","Derek"]
  
  location_variance = [f"Michael had a blast traveling to {location}" for location in locations]
  name_variance = [f"{name} had a blast traveling to Mexico" for name in names ]

  location_variance_sentiments = [predict(text) for text in location_variance]
  name_variance_sentiments = [predict(text) for text in name_variance]
  # print( predict('Michael had fun traveling to Mexico'))

  print(location_variance_sentiments)
  assert all(sentiment == 'Positive' for sentiment in location_variance_sentiments), "Sentiment invariance failed changing locations"
  assert all(sentiment == 'Negative' for sentiment in name_variance_sentiments), "Sentiment invariance failed changing names"

def test_directional():
  pos_examples = ['The cake is great', 'Anna is so smart.', 'This laptop is very good']
  neg_examples = ['The cake is dogshit', 'Anna is so retarded.', 'This laptop is shit']
  pos_examples_sent = [predict(text) for text in pos_examples]
  neg_examples_sent = [predict(text) for text in neg_examples]

  assert not any([pos_examples_sent[i] == neg_examples_sent[i] for i in range(len(pos_examples))]), "Sentiment directionality failed"