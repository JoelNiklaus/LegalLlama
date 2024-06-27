import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('output/model_scores.csv', index_col=0)

# Select the columns to visualize
columns_to_plot = [
    'arc_aggregate', 'belebele_aggregate', 'm_mmlu_aggregate', 'truthfulqa_aggregate',
    'en_aggregate', 'de_aggregate', 'es_aggregate', 'fr_aggregate', 'it_aggregate'
]

# Select the first four models (rows)
models_to_plot = ["Qwen__Qwen2-0.5B-Instruct", "Qwen__Qwen2-1.5B-Instruct", "Qwen__Qwen2-7B-Instruct",
                  "microsoft__Phi-3-medium-128k-instruct"]

# Create a new DataFrame with selected data
plot_data = df.loc[models_to_plot, columns_to_plot]

# Transpose the DataFrame so that models become columns
plot_data = plot_data.T

# Set up the plot
plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")

# Create the bar plot
ax = plot_data.plot(kind='bar', width=0.8)

# Customize the plot
plt.title('Model Performance Across Tasks and Languages', fontsize=16)
plt.xlabel('Datasets', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('output/model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Bar chart has been saved as 'model_performance_comparison.png'")
