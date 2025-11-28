import random
import torch
from transformers import AutoModel, AutoTokenizer

random.seed(54321)

BASE_MODEL_NAME = "roberta-base"
EMBEDDING_DIM = 768

class SentimentModel(torch.nn.Module):

    def __init__(self, base_model):
        super(SentimentModel, self).__init__()
        self.base_model = base_model
        self.dropout = torch.nn.Dropout(p=0.5)
        self.dense = torch.nn.Linear(EMBEDDING_DIM, 1)
    
    def forward(self, input_ids, attention_mask):
        output = self.base_model(input_ids, attention_mask)[0]
        output = output[:, 0, :]
        output = self.dropout(output)
        output = self.dense(output)
        return output

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model = SentimentModel(AutoModel.from_pretrained(BASE_MODEL_NAME))
model.load_state_dict(torch.load("./imdb-sentiment-model.pt", weights_only=True))
model.eval()

texts = [
    "This movie was awful. What a waste of time.",
    "I love this film so much! Best one I've seen this year by far."
]

for text in texts:
    encoding = tokenizer(text, return_tensors="pt")

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    output = model(input_ids, attention_mask)
    score = torch.sigmoid(output).item()

    print(text)
    print("P(positive) = {:.3f}\n".format(score))
