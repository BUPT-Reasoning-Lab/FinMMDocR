import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Use GPU 4
from tqdm import tqdm
import json
from FlagEmbedding import BGEM3FlagModel


def main():
    model_name = "bge_m3"
    model = BGEM3FlagModel(model_name_or_path="./models/bge_m3")
    
    with open("./data/test.json", "r") as f:
        data = json.load(f)

    retrieved_results = []
    for example in tqdm(data):
        question_id = example["question_id"]
        question = example["question"]
        texts_path = example["texts"]
        with open(texts_path, "r") as f:
            texts = json.load(f)
        embeddings_1 = model.encode(
            question,
            batch_size=64,
            max_length=8192,  # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
        )["dense_vecs"]
        embeddings_2 = model.encode(texts)["dense_vecs"]
        similarity = embeddings_1 @ embeddings_2.T
        results = [{"page": i+1, "score": float(similarity[i])} for i in range(len(similarity))]
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        retrieved_results.append({
            "question_id": question_id,
            "question": question,
            "retrieved_results": results
        })

    with open(f"./retrieved_results/{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(retrieved_results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
