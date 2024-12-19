
import json
import os
from datasets import load_dataset
from ast import literal_eval



print("\nCreating OpenAI format dataset...")

openai_format = []

# Language pairs to translate between
language_pairs = [
    ("de", "fr"),
    ("fr", "de")
]

ds = load_dataset("joelniklaus/SwissLegalTranslations")

def formatting_prompts_func(examples):
    convos = []
    for example in examples["translation"]:
        if isinstance(example, str):
            try:
                example = literal_eval(example)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing string: {e}")
                print(f"Problematic string: {example}")
                continue

        # Get available languages from this specific example
        languages = list(example.keys())

        if len(languages) == 2:  # We need exactly two languages for translation
            lang1, lang2 = languages
            # Only create prompts for one direction because otherwise the training file is too large for OpenAI (limit 512MB)
            convos.append(create_prompt(example, lang1, lang2))
            # convos.append(create_prompt(example, lang2, lang1))

    return {"convos": convos}


def create_prompt(example, source_lang, target_lang):
    system = {"role": "system", "content": ""}
    prompt = {
        "role": "user",
        "content": f"{source_lang.upper()}: {example[source_lang]}\n{target_lang.upper()}:",
    }
    answer = {
        "role": "assistant",
        "content": f"{example[target_lang]}",
    }
    return {"messages": [system, prompt, answer]}

# Convert each example to OpenAI chat format
def process(dataset):
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count(),
        remove_columns=dataset.column_names,
    )
    return dataset

train = process(ds["train"])
validation = process(ds["validation"])
test = process(ds["test"])

# Save to JSONL file
print("Saving OpenAI format datasets to openai/...")
os.makedirs("openai", exist_ok=True)

# OpenAI allows 200MB max file size for finetuning per file
for split_name, split_data in [("train", train), ("validation", validation), ("test", test)]:
    output_path = f"openai/{split_name}.jsonl"
    print(f"Saving {split_name} split to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for example in split_data["convos"]:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

print("OpenAI format dataset created successfully")


