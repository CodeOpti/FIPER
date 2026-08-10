import json

files = ["Mercuryv2-eval.jsonl", "Mercuryv2-train.jsonl", "PIEv2.jsonl"]


fields_to_keep = {"input", "target", "diff", "rate"}

merged_data = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())

                filtered_item = {key: item[key] for key in fields_to_keep if key in item}


                if all(k in filtered_item for k in ["input", "target", "diff", "rate"]):
                    merged_data.append(filtered_item)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {file}: {line[:50]}")

print(f"Merged samples: {len(merged_data)} records")


with open("OD-merged.jsonl", "w", encoding="utf-8") as out_file:
    for item in merged_data:
        out_file.write(json.dumps(item, ensure_ascii=True) + "\n")


