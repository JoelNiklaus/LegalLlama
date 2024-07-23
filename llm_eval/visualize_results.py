import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('output/model_scores.csv', index_col=0)

models_to_plot = [
    # "Qwen__Qwen2-0.5B-Instruct",

    # "Qwen__Qwen2-1.5B-Instruct",
    # "microsoft__Phi-3-mini-128k-instruct",

    # "CohereForAI__aya-23-8B",
    "mistralai__Mistral-7B-Instruct-v0.3",
    "meta-llama__Meta-Llama-3.1-8B-Instruct",
    "Qwen__Qwen2-7B-Instruct",
    "google__gemma-2-9b-it",
    "microsoft__Phi-3-small-128k-instruct",

    #"stabilityai__stablelm-2-12b-chat",
    #"Qwen__Qwen1.5-14B-Chat",
    #"mistralai__Mistral-Nemo-Instruct-2407",
    #"microsoft__Phi-3-medium-128k-instruct",
]

task_map = {
    'arc_aggregate': 'ARC',
    'belebele_aggregate': 'Belebele',
    'truthfulqa_aggregate': 'TruthfulQA',
    'm_mmlu_aggregate': 'M-MMLU'
}

language_map = {f"{lang}_aggregate": lang for lang in ['en', 'es', 'de', 'fr', 'it']}


def create_plot(data, plot_type, title_suffix):
    plt.figure(figsize=(15, 10))
    sns.set(style="whitegrid")
    data.plot(kind='bar', width=0.8)
    plt.title(f'Across {plot_type} {title_suffix}', fontsize=16)
    plt.xlabel(plot_type, fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(title='Models', loc='lower right', bbox_to_anchor=(1, 0))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'output/model_performance_{plot_type.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar chart has been saved as 'model_performance_{plot_type.lower()}.png'")


model_name_map = {model: model.split('__')[-1] for model in models_to_plot}

for plot_type, columns, name_map in [
    ('Tasks', task_map.keys(), task_map),
    ('Languages', language_map.keys(), language_map)
]:
    plot_data = df.loc[models_to_plot, columns].T
    plot_data.columns = [model_name_map[col] for col in plot_data.columns]
    plot_data.index = [name_map[idx] for idx in plot_data.index]

    title_suffix = f"({', '.join(name_map.values())})"
    create_plot(plot_data, plot_type, title_suffix)
