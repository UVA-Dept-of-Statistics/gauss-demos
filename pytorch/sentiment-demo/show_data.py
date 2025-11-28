import json

records = []
with open("/tmp/imdb-train.jsonl") as file:
    for line in list(file):
        record = json.loads(line)
        records.append(record)

print(json.dumps(records[0], ensure_ascii=False, indent=2))