import json
import os
import pandas as pd
import glob
import re


def extract_scores(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    scores = {}
    for task, results in data['results'].items():
        acc_norm = results.get('acc_norm,none')
        acc = results.get('acc,none')
        # IMPORTANT: If acc_norm is not available, try to extract it from the acc string
        if acc_norm is not None:
            scores[task] = acc_norm
        elif acc is not None:
            scores[task] = acc

    return scores


def process_output_folder(output_dir):
    all_scores = {}

    for model in os.listdir(output_dir):
        model_path = os.path.join(output_dir, model)
        if os.path.isdir(model_path):
            model_scores = {}
            json_files = glob.glob(os.path.join(model_path, '**', 'results_*.json'), recursive=True)
            for json_file in json_files:
                scores = extract_scores(json_file)
                model_scores.update(scores)
            if model_scores:
                all_scores[model] = model_scores

    return all_scores


def rename_columns(df):
    rename_dict = {
        'arc_challenge': 'arc_en',
        'belebele_deu_Latn': 'belebele_de',
        'belebele_eng_Latn': 'belebele_en',
        'belebele_fra_Latn': 'belebele_fr',
        'belebele_ita_Latn': 'belebele_it',
        'belebele_spa_Latn': 'belebele_es',
        'truthfulqa_mc2': 'truthfulqa_en',
        'truthfulqa_de_mc2': 'truthfulqa_de',
        'truthfulqa_fr_mc2': 'truthfulqa_fr',
        'truthfulqa_es_mc2': 'truthfulqa_es',
        'truthfulqa_it_mc2': 'truthfulqa_it'
    }

    df.rename(columns=lambda x: rename_dict.get(x, x), inplace=True)
    return df


def aggregate_scores(df):
    # Aggregate dataset scores
    datasets = ['belebele', 'm_mmlu', 'truthfulqa', 'arc']
    for dataset in datasets:
        acc_columns = [col for col in df.columns if col.startswith(f"{dataset}_")]
        if acc_columns:
            agg_name = f"{dataset}_aggregate"
            df[agg_name] = df[acc_columns].mean(axis=1)

    # Aggregate language scores
    languages = ['en', 'de', 'fr', 'es', 'it']
    for lang in languages:
        lang_columns = [col for col in df.columns if f"_{lang}" in col]
        if lang_columns:
            df[f"{lang}_aggregate"] = df[lang_columns].mean(axis=1)

    is_x_aggregate = lambda col, x: bool(re.match(r'({})_aggregate'.format('|'.join(x)), col))

    # Overall language aggregate
    df["language_aggregate"] = df[[col for col in df.columns if is_x_aggregate(col, languages)]].mean(axis=1)

    # Overall dataset aggregate
    df["dataset_aggregate"] = df[[col for col in df.columns if is_x_aggregate(col, datasets)]].mean(axis=1)

    return df


# Set the path to your output directory
output_dir = './output'

# Process the output folder and get the scores
all_scores = process_output_folder(output_dir)

# Create a DataFrame
df = pd.DataFrame(all_scores).T

# Rename columns
df = rename_columns(df)

# Sort columns alphabetically for consistency
df = df.reindex(sorted(df.columns), axis=1)

# Add aggregate columns
df = aggregate_scores(df)

# Sort columns again to put aggregate columns at the end
df = df.reindex(sorted(df.columns), axis=1)

# Save the DataFrame to a CSV file in the output directory
csv_file = os.path.join(output_dir, 'model_scores.csv')
df.to_csv(csv_file)

print(f"Scores have been extracted and saved to {csv_file}")
