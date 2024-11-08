import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('output/model_scores.csv', index_col=0)

models_2b = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "utter-project/EuroLLM-1.7B-Instruct",
    "BSC-LT/salamandra-2b-instruct",
    "google/flan-t5-large",
]
models_3b = [
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/flan-t5-xl",
]
models_9b = [
    "microsoft/Phi-3-small-128k-instruct",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct",
    #"Qwen/Qwen2-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "CohereForAI/aya-expanse-8b",
    #"CohereForAI/aya-23-8B",
    #"mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Ministral-8B-Instruct-2410",
    "BSC-LT/salamandra-7b-instruct",
]
models_14b = [
    # "google/flan-t5-xxl", # TODO: to be run
    # "Qwen/Qwen2.5-14B-Instruct", # TODO: to be run
    "microsoft/Phi-3-medium-128k-instruct",
    "Qwen/Qwen1.5-14B-Chat",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "stabilityai/stablelm-2-12b-chat",
]
models_by_size = {
    '2b': models_2b,
    '3b': models_3b,
    '9b': models_9b,
    '14b': models_14b,
    'best': [models_2b[0], models_3b[0], models_9b[0], models_14b[0]],
}




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
    plt.savefig(f'output/model_performance_{plot_type.lower()}_{size}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar chart has been saved as 'model_performance_{plot_type.lower()}_{size}.png'")


for size in models_by_size:
    models_to_plot = models_by_size[size]
    models_to_plot = [model.replace("/", "__") for model in models_to_plot]

    task_map = {
        'arc_aggregate': 'ARC',
        'belebele_aggregate': 'Belebele',
        'truthfulqa_aggregate': 'TruthfulQA',
        'm_mmlu_aggregate': 'M-MMLU'
    }

    language_map = {f"{lang}_aggregate": lang for lang in ['en', 'es', 'de', 'fr', 'it']}

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
