from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")

text = "This film was ABSOLUTE GARBAGE! I hated every minute."

encoding = tokenizer(text)

encoding = tokenizer(text, return_tensors="pt")
model = AutoModel.from_pretrained("roberta-base")

output = model(encoding["input_ids"], encoding["attention_mask"])[0]
embeddings = output[0]

for i, token in enumerate(encoding.tokens()):
    print("{}: {}".format(encoding["input_ids"][0][i], token))
    print("Embedding: {}, ...\n".format(", ".join(map(lambda emb: "{:.5f}".format(emb), embeddings[i][:5].tolist()))))

