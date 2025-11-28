import os

# ------- POWER8 SAFETY SWITCHES -------
# These MUST be set before importing torch / transformers
os.environ["XNNPACK_GLOBAL_DISABLE"] = "1"
os.environ["ATEN_CPU_CAPABILITY"] = "default"
os.environ["PYTORCH_ENABLE_MKLDNN"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"   # avoid weird thread spawning too
# --------------------------------------


import json
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

random.seed(54321)

# Run on the GPU
DEVICE = "cuda"

# Use as our pre-trained foundation model
BASE_MODEL_NAME = "roberta-base"

# Data/model paths
MODEL_OUTPUT = "./imdb-sentiment-model.pt"
TRAIN_FILE = "./imdb-train.jsonl"
VAL_FILE = "./imdb-test.jsonl"

### Model hyperparameters and configuration

# Optimizer learning rate
LEARN_RATE = 5e-5

# Percent of activations to mask in dropout layer
DROPOUT_RATIO = 0.5

# Data batch size
BATCH_SIZE = 32

# roberta-base produces an embedding size of 768
EMBEDDING_DIM = 768

# Number of training epochs
EPOCHS = 3

# Label set
LABELS = [
    "positive",
    "negative"
]

class SentimentDataset(torch.utils.data.Dataset):

    def __init__(self, records, tokenizer):
        texts = []
        labels = []
        for record in records:
            texts.append(record["text"])
            sentiment = record["label"]
            # Binary classification => code "positive" as 1 and "negative" as 0
            label = [1] if sentiment == "positive" else [0]
            labels.append(label)
        
        # Truncate the text to 256 tokens to speed things up
        self.x = tokenizer(texts, truncation=True, padding=True, max_length=256)
        self.y = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.x.items()}
        item['labels'] = torch.tensor(self.y[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.y)

class SentimentModel(torch.nn.Module):

    def __init__(self, base_model):
        super(SentimentModel, self).__init__()
        # Our "base" or "foundation" model
        self.base_model = base_model
        # Dropout layer for regularization
        self.dropout = torch.nn.Dropout(p=DROPOUT_RATIO)
        # Dense layer for classification head
        self.dense = torch.nn.Linear(EMBEDDING_DIM, 1)
    
    def forward(self, input_ids, attention_mask):
        # B x N x K tensor of embeddings
        output = self.base_model(input_ids, attention_mask)[0]
        # Take the first token (i.e. the [CLS] token) embedding
        # Now we have a B x K matrix
        output = output[:, 0, :]
        # Apply dropout regularization
        output = self.dropout(output)
        # Run through classification head
        # Now we have a B x 1 matrix
        output = self.dense(output)
        return output # single logit scalar value
        

def load_records(path):
    with open(path) as file:
        records = []
        for line in list(file):
            record = json.loads(line)
            records.append(record)
        return records

def evaluate(model, data_loader):
    with torch.no_grad():
        model.eval()
        loss_sum = 0
        tpc = 0 # true positive count
        fpc = 0 # false positive count
        fnc = 0 # false negative count
        tnc = 0 # true negative count
        print("EVALUATING...")
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            output = model.forward(
                input_ids,
                attention_mask
            )
            loss = loss_fn(output, labels)
            loss_sum += loss.item()
            scores = torch.sigmoid(output).tolist()
            labels = labels.tolist()
            for score, label in zip(scores, labels):
                prediction = round(score[0])
                label = label[0]
                if prediction == 1:
                    if label == 1:
                        tpc += 1
                    else:
                        fpc += 1
                else:
                    if label == 1:
                        fnc += 1
                    else:
                        tnc += 1
        acc = (tpc + tnc) / (tpc + tnc + fpc + fnc) # accuracy
        pre = tpc / (tpc + fpc) # precision
        rec = tpc / (tpc + fnc) # recall
        f_1 = 2 * (pre * rec) / (pre + rec)
        print("VALIDATION LOSS: {:.5f}\n".format(loss_sum / len(data_loader)))
        print("ACC: {:.3f}".format(acc))
        print("PRE: {:.3f}".format(pre))
        print("REC: {:.3f}".format(rec))
        print("F-1: {:.3f}".format(f_1))
        print("\n")

# Load our base foundation model and associated tokenizer
model = SentimentModel(base_model=AutoModel.from_pretrained(BASE_MODEL_NAME))
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Load the training records from the JSONL files
# Only use the first 3,000 records to speed up training/validation
records_train = load_records(TRAIN_FILE)[:3000]
print("Loaded {} training records".format(len(records_train)))
records_val = load_records(VAL_FILE)[:2000]
print("Loaded {} validation records\n\n".format(len(records_val)))

# Process data into our datasets
data_train = SentimentDataset(records_train, tokenizer)
data_val = SentimentDataset(records_val, tokenizer)

# Batch up our data in data loaders
data_loader_train = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
data_loader_val = DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model with a single forward pass with dummy data
model.forward(
    torch.randint(low=0, high=10_000, size=(BATCH_SIZE, 16), dtype=torch.int32),
    torch.ones(size=(BATCH_SIZE, 16), dtype=torch.int32)
)

# Parallelize
model = torch.nn.DataParallel(model)
model.to(DEVICE)

# Configure our optimizer for updating model parameters
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
# Configure loss function to be binary cross-entropy loss with direct logit input (i.e. omitting the sigmoid activation)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Main training loop
for epoch in range(0, EPOCHS):
    train_loss_sum = 0
    model.train()
    # Iterate through our batched data
    for batch_index, batch in enumerate(data_loader_train):
        # Grab inputs/labels and load onto the GPU device
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        label = batch["labels"].to(DEVICE)
        # Run through the model and get back the logits
        output = model.forward(
            input_ids,
            attention_mask
        )
        # Compute the loss and update the running sum for this epoch
        loss = loss_fn(output, label)
        train_loss_sum += loss.item()
        # Clear gradients from previous step
        optimizer.zero_grad()
        # Backpropagation -- compute new gradients
        loss.backward()
        # Update model parameters
        optimizer.step()
        # Output training info
        avg_train_loss = train_loss_sum / (batch_index + 1)
        print(
            "EPOCH: {} / {}, BATCH: {} / {}, TRAINING LOSS: {:.5f}".format(
                epoch + 1,
                EPOCHS,
                batch_index + 1,
                len(data_loader_train),
                avg_train_loss
            ),
            end="\r"
        )
    # At the end of each epoch, we evaluate the model on the validation data
    print()
    evaluate(model, data_loader_val)
    torch.save(model.module.state_dict(), MODEL_OUTPUT)
