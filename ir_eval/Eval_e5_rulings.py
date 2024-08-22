from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
import pandas as pd
import torch

corpus, queries, qrels = GenericDataLoader(
    corpus_file='dataset/corpus_rulings.jsonl',
    query_file='dataset/queries.jsonl',
    qrels_file='dataset/qrels_rulings.tsv'
).load_custom()

torch.cuda.empty_cache()  # Clear any cached memory

model = DRES(models.SentenceBERT("paraphrase-multilingual-mpnet-base-v2"), batch_size=128)
retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=[1, 5, 10, 50, 100])

# Process queries in smaller chunks to reduce memory usage
chunk_size = 50000  # Adjust based on your GPU memory capacity
results = {}

for i in range(0, len(queries), chunk_size):
    chunk_queries = dict(list(queries.items())[i:i + chunk_size])
    chunk_results = retriever.retrieve(corpus, chunk_queries)
    results.update(chunk_results)
    torch.cuda.empty_cache()  # Clear cache after each chunk

ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

# Store results
metrics_dict = {
    "K": retriever.k_values,
    "NDCG": [ndcg[f"NDCG@{k}"] for k in retriever.k_values],
    "MAP": [_map[f"MAP@{k}"] for k in retriever.k_values],
    "Recall": [recall[f"Recall@{k}"] for k in retriever.k_values],
    "Precision": [precision[f"P@{k}"] for k in retriever.k_values],
}

metrics_df = pd.DataFrame(metrics_dict)
csv_filename = "e5_rulings_metrics.csv"
metrics_df.to_csv(csv_filename, index=False)

print(f"Metrics saved to {csv_filename}")
