#####
# Imports
#####

import json
from bert_score import score as bert_score      # For BERTScore (semantic similarity)
from rouge_score import rouge_scorer            # For ROUGE (n-gram overlap metrics)
from sentence_transformers import SentenceTransformer, util  # For cosine similarity
import numpy as np                              # For array operations

#####
# Data Loading
#####

def load_examples(jsonl_path):
    """
    Load generated and reference answers from a JSONL file.
    Each line should be a dict with 'generated_answer' and 'reference_answer' fields.
    Filters out examples with missing/empty answers.
    """
    generated, references = [], []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            g = ex.get("generated_answer", "")
            r = ex.get("reference_answer", "")
            if g.strip() and r.strip():
                generated.append(g)
                references.append(r)
    return generated, references

#####
# Evaluation Metrics
#####

def evaluate_bertscore(candidates, references, lang="en"):
    """
    Calculate BERTScore metrics (precision, recall, F1) between candidate and reference answers.
    Uses contextual embeddings for semantic similarity.
    """
    P, R, F1 = bert_score(candidates, references, lang=lang)
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean())
    }

def evaluate_rouge(candidates, references):
    """
    Compute ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) for candidate vs. reference answers.
    Scores are based on n-gram and sequence overlap.
    Returns average F1 scores for each ROUGE metric.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
    avg_scores = {}
    for key in results[0]:
        avg_scores[key] = np.mean([r[key].fmeasure for r in results])
    return avg_scores

def evaluate_cosine(candidates, references, model_name="all-mpnet-base-v2"):
    """
    Compute average cosine similarity between answer and reference using sentence embeddings.
    Captures semantic similarity regardless of exact wording.
    """
    model = SentenceTransformer(model_name)
    emb_refs = model.encode(references, convert_to_tensor=True)
    emb_cands = model.encode(candidates, convert_to_tensor=True)
    scores = util.cos_sim(emb_cands, emb_refs)
    mean_sim = float(scores.diag().mean())
    return {"cosine_similarity": mean_sim}

#####
# Main Evaluation Pipeline
#####

def main(jsonl_path):
    """
    Load examples, compute evaluation metrics, and print results.
    """
    generated, references = load_examples(jsonl_path)
    if not references:
        print("No reference answers found! Please provide 'reference_answer' in the log.")
        return

    print("BERTScore:")
    bert = evaluate_bertscore(generated, references)
    for k, v in bert.items():
        print(f"  {k}: {v:.4f}")

    print("ROUGE:")
    rouge = evaluate_rouge(generated, references)
    for k, v in rouge.items():
        print(f"  {k}: {v:.4f}")

    print("Cosine Similarity:")
    cosine = evaluate_cosine(generated, references)
    for k, v in cosine.items():
        print(f"  {k}: {v:.4f}")

#####
# Entrypoint
#####

if __name__ == "__main__":
    import sys
    # Use provided path or default to "txt/log.jsonl"
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "txt/log.jsonl"
    main(jsonl_path)
