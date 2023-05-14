import argparse
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

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Process the input sentence.')

    # Add arguments to the parser
    parser.add_argument('--input', type=str, help='Input sentence')

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    input = args.input
    
    # Load the model and tokenizer
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = RoBERTaEncoder(2)
    model.load_state_dict(torch.load("./models/model.pt", map_location=torch.device('cpu')))
    model = model.to(device)

    # Transform the input sequence into tokens
    tokens = tokenizer(input, padding=True, max_length = 128, truncation=True, return_tensors='pt')
    tokens = tokens.to(device)

    # Perform inference and get the predicted class
    output = model(tokens['input_ids'], tokens['attention_mask'])
    predicted_class = int(torch.argmax(output))
    
    # Print the predicted class
    predicted_class = "Positive" if predicted_class == 0 else "Negative"
    print(f"The predicted class for the given sentence is {predicted_class}.")

if __name__ == '__main__':
    main()