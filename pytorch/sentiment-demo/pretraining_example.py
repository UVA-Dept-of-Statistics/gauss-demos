import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE_MODEL_NAME = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_NAME)

text = "The capital of Spain is <mask> "
encoding = tokenizer(text, return_tensors="pt")

output = model(encoding["input_ids"], encoding["attention_mask"]).logits

mask_token_index = torch.where(encoding["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = output[0, mask_token_index, :]
predicted_token_id = torch.argmax(mask_token_logits).item()
predicted_token = tokenizer.decode(predicted_token_id)

print(predicted_token)