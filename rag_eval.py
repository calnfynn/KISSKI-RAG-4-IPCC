# evaluate_rag.py

import json
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import numpy as np
import argparse

def load_examples(jsonl_path):
    questions, generated, references = [], [], []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            questions.append(ex.get("question", ""))
            generated.append(ex.get("generated_answer", ""))
            references.append(ex.get("reference_answer", ""))  # empty string if missing
    return questions, generated, references

def evaluate_bertscore(candidates, references, lang="en"):
    P, R, F1 = bert_score(candidates, references, lang=lang)
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean())
    }

def evaluate_rouge(candidates, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
    avg_scores = {}
    for key in results[0]:
        avg_scores[key] = np.mean([r[key].fmeasure for r in results])
    return avg_scores

def evaluate_cosine(candidates, references, model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    emb_refs = model.encode(references, convert_to_tensor=True)
    emb_cands = model.encode(candidates, convert_to_tensor=True)
    scores = util.cos_sim(emb_cands, emb_refs)
    mean_sim = float(scores.diag().mean())
    return {"cosine_similarity": mean_sim}

def main(args):
    questions, generated, references = load_examples(args.input)

    # Optionally, filter empty references if your gold data is patchy
    filtered_gen, filtered_ref = [], []
    for g, r in zip(generated, references):
        if r.strip():  # has reference
            filtered_gen.append(g)
            filtered_ref.append(r)
    if not filtered_ref:
        print("No reference answers found in data! Populate 'reference_answer' for proper eval.")
        return

    print("Evaluating BERTScore...")
    bert = evaluate_bertscore(filtered_gen, filtered_ref)
    print("Evaluating ROUGE...")
    rouge = evaluate_rouge(filtered_gen, filtered_ref)
    print("Evaluating Cosine Similarity...")
    cosine = evaluate_cosine(filtered_gen, filtered_ref)

    print("\n=== RAG EVALUATION RESULTS ===")
    print("BERTScore:")
    for k, v in bert.items():
        print(f"  {k}: {v:.4f}")
    print("ROUGE:")
    for k, v in rouge.items():
        print(f"  {k}: {v:.4f}")
    print("Cosine similarity:")
    for k, v in cosine.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to .jsonl log from RAG run")
    args = parser.parse_args()
    main(args)
