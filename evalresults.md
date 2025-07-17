# Evaluation scores

```sh
(.venv) PS C:\Users\Fynn\Documents\KISSKI-RAG-4-IPCC> & C:/Users/Fynn/Documents/KISSKI-RAG-4-IPCC/.venv/Scripts/python.exe c:/Users/Fynn/Documents/KISSKI-RAG-4-IPCC/eval.py
BERTScore:
  precision: 0.8069
  recall: 0.8527
  f1: 0.8291
ROUGE:
  rouge1: 0.2757
  rouge2: 0.0495
  rougeL: 0.1518
Cosine Similarity:
  cosine_similarity: 0.7204
```

---

| Metric              | What it Scores                    | Good Score | Bad Score | *Our Score* | Notes          |
| -----------------   | --------------------------------- | ---------- | --------- | --------- | -------------- |
| BERTScore Precision | Semantic precision                | >0.8       | <0.6      | 0.8069    | good (just so) |
| BERTScore Recall    | Semantic recall                   | >0.8       | <0.6      | 0.8527    | good           |
| BERTScore (F1)      | Semantic overlap, paraphrasing OK | >0.8       | <0.6      | 0.8291    | good           |
| ROUGE-1             | Word overlap                      | >0.45      | <0.2      | 0.2757    | not good       |
| ROUGE-2             | Phrase overlap (2-word)           | >0.2       | <0.05     | 0.0495    | bad            |
| ROUGE-L             | Longest matching sequence         | >0.3       | <0.1      | 0.1518    | not good       |
| Cosine Similarity   | Embedding similarity (meaning)    | >0.7       | <0.5      | 0.7204    | good           |

---

All metrics can have values between 0 (worst) and 1 (best).

BERTScore & Cosine Similarity test for semantic matches, 
ROUGE tests for verbatim matches. 

BERT & Cosine are more helpful for RAG evaluation.