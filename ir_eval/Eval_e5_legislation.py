from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
import pandas as pd

corpus, queries, qrels = GenericDataLoader(
    corpus_file='dataset/corpus_legislation.jsonl',
    query_file='dataset/queries.jsonl',
    qrels_file='dataset/qrels_legislation.tsv'
).load_custom()


model = DRES(models.SentenceBERT("paraphrase-multilingual-mpnet-base-v2"), batch_size=128)
retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values = [1,5,10,50,100])

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)


ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

# store results
metrics_dict = {
    "K": retriever.k_values,
    "NDCG": [ndcg[f"NDCG@{k}"] for k in retriever.k_values],
    "MAP": [_map[f"MAP@{k}"] for k in retriever.k_values],
    "Recall": [recall[f"Recall@{k}"] for k in retriever.k_values],
    "Precision": [precision[f"P@{k}"] for k in retriever.k_values],
}

metrics_df = pd.DataFrame(metrics_dict)

csv_filename = "e5_legislation_metrics.csv"
metrics_df.to_csv(csv_filename, index=False)

print(f"Metrics saved to {csv_filename}")
