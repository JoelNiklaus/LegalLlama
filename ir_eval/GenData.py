from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict

doc2doc = load_dataset("rcds/swiss_doc2doc_ir", trust_remote_code=True).rename_column('facts', 'text').rename_column('decision_id', '_id')
legislation = load_dataset("rcds/swiss_legislation", trust_remote_code=True).rename_column('uuid', '_id').rename_column('pdf_content', 'text')['train']
rulings = load_dataset("rcds/swiss_rulings", trust_remote_code=True).rename_column('full_text', 'text').rename_column('decision_id', '_id').rename_column('court', 'title')
legislation = legislation.filter(lambda entry: entry['text'] != "").select_columns(['_id', 'title', 'text'])
legislation.to_json('dataset/corpus_legislation.jsonl')
legislation.cleanup_cache_files()
print(legislation)


queries = doc2doc.filter(lambda entry: entry['text']).select_columns(['_id', 'text'])
queries = concatenate_datasets([queries['train'], queries['validation'], queries['test']])
queries.to_json('dataset/queries.jsonl')
reduced_query_ids = set(queries['_id'])
queries.cleanup_cache_files()
print(queries)

def gen_rulings():
  for split in doc2doc:
    for row in doc2doc[split]:
      _id = row.get('_id')
      if _id not in reduced_query_ids:
                continue
      cited_rulings_raw = row.get('cited_rulings')

      cited_rulings = [s.strip().strip("'") for s in cited_rulings_raw.strip("[]").split(',')]

      for cited_ruling in cited_rulings:
        yield {'_id': _id, 'cited_ruling': cited_ruling, 'score': 1}

swiss_ir_court_rulings = Dataset.from_generator(gen_rulings)
reduced_rulings = set(swiss_ir_court_rulings['cited_ruling'])
swiss_ir_court_rulings.to_csv('dataset/qrels_rulings.tsv', sep="\t")

swiss_rulings_corpus = rulings.filter(lambda entry: entry['text'] != "").filter(lambda entry: entry['_id'] in reduced_rulings).select_columns(['_id', 'title', 'text'])['train']
swiss_rulings_corpus.to_json('dataset/corpus_rulings.jsonl')
swiss_rulings_corpus.cleanup_cache_files()
print(swiss_rulings_corpus)

def gen_legislation():
  for split in doc2doc:
    for row in doc2doc[split]:
      _id = row.get('_id')
      if _id not in reduced_query_ids:
                continue
      laws_raw = row.get('laws')

      laws = [s.strip().strip("'") for s in laws_raw.strip("[]").split(',')]

      for law in laws:
        yield {'_id': _id, 'law': law, 'score': 1}

swiss_ir_laws = Dataset.from_generator(gen_legislation)
swiss_ir_laws.to_csv('dataset/qrels_legislation.tsv', sep="\t")