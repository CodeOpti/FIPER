import json
from tqdm import tqdm
from knowledge.code_embedder import CodeEmbedder

input_file = "OD-merged.jsonl"
output_file = "OD-merged-with-vec.jsonl"


ce = CodeEmbedder()


with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]


for row in tqdm(data, desc="Generating embeddings"):
    if "input" in row:
        try:
            vec = ce(row["input"], keep_tensor=False)
            row["vector"] = vec
        except Exception as e:
            row["vector"] = None
            print(f"Encoding failed: {e}")
    else:
        print(f"Missing input field: {row.idx}")


with open(output_file, "w", encoding="utf-8") as f:
    for row in data:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Processed {len(data)} records; results saved to {output_file}")
