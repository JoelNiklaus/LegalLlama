from dataclasses import dataclass

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets


@dataclass
class Config:
    dataset: Dataset | DatasetDict
    text_col: str
    languages: list[str]


laws = Config(
    dataset=load_dataset("joelniklaus/SwissLawTranslations", "paragraph_level"),
    text_col="parText",
    languages=["de", "fr", "it", "rm", "en"],
)

decision_summaries = Config(
    dataset=load_dataset("joelniklaus/SwissDecisionSummaryTranslations", "text_level"),
    text_col="text",
    languages=["de", "fr", "it"],
)

# Exclude press releases because they are on the document level. The others are on the paragraph/sentence level.
# press_releases = Config(
#    dataset = load_dataset("joelniklaus/SwissSupremeCourtPressReleaseTranslations"),
#    text_col = "text",
#    languages = ["de", "fr", "it"],
# )


def create_translation_dataset(configs):
    def generate_translation_pairs(examples, text_col, languages):
        all_pairs = []
        # Get the number of examples in the batch
        batch_size = len(examples[f"{languages[0]}_{text_col}"])

        # Process each example in the batch
        for idx in range(batch_size):
            # Get translations for each language for this specific example
            translations = {
                lang: examples[f"{lang}_{text_col}"][idx] for lang in languages
            }

            # Filter out empty translations
            lang_keys = [key for key, value in translations.items() if value != ""]

            # Generate pairs for this example
            pairs = []
            for i in range(len(lang_keys)):
                for j in range(i + 1, len(lang_keys)):
                    lang1, lang2 = lang_keys[i], lang_keys[j]
                    json_obj = {lang1: translations[lang1], lang2: translations[lang2]}
                    pairs.append(str(json_obj))
            all_pairs.extend(pairs)

        return {"translation": all_pairs}

    combined_datasets = {}

    for split in ["train", "validation", "test"]:
        split_datasets = []
        for config in configs:
            dataset = config.dataset
            if split in dataset:
                split_dataset = dataset[split].map(
                    lambda example: generate_translation_pairs(
                        example, config.text_col, config.languages
                    ),
                    batched=True,
                    batch_size=1000,
                    remove_columns=dataset[split].column_names,
                )
                split_dataset = split_dataset.flatten()
                split_datasets.append(split_dataset)

        if split_datasets:
            combined_datasets[split] = concatenate_datasets(split_datasets)
            # Convert to pandas and drop duplicates
            df = combined_datasets[split].to_pandas()
            df = df.drop_duplicates(subset=["translation"]).reset_index(drop=True)
            combined_datasets[split] = Dataset.from_pandas(df)

    return DatasetDict(combined_datasets)


# Combine all configurations into a single dataset
combined_dataset = create_translation_dataset([laws, decision_summaries])

print(combined_dataset)
# Print one example from the training set
print("\nExample from training set:")
print(combined_dataset["train"][0]["translation"])

# Save dataset to csv files and push to hub
print("\nSaving dataset to csv files and pushing to hub...")
for split in combined_dataset.keys():
    combined_dataset[split].to_csv(f"{split}.csv", index=False)


# Calculate word count statistics
print("\nCalculating word count statistics...")
word_counts = []
for split in combined_dataset.keys():
    for example in combined_dataset[split]["translation"]:
        word_count = len(str(example).split())
        word_counts.append(word_count)

word_counts = sorted(word_counts)
total = len(word_counts)

statistics = {
    "median": word_counts[total//2],
    "mean": sum(word_counts)/total,
    "95th_percentile": word_counts[int(0.95 * total)],
    "99th_percentile": word_counts[int(0.99 * total)],
    "999th_percentile": word_counts[int(0.999 * total)],
    "9999th_percentile": word_counts[int(0.9999 * total)],
    "max": max(word_counts)
}

print("\nWord count statistics:")
print(f"Median: {statistics['median']:.0f} words")
print(f"Mean: {statistics['mean']:.0f} words") 
print(f"95th percentile: {statistics['95th_percentile']:.0f} words")
print(f"99th percentile: {statistics['99th_percentile']:.0f} words")
print(f"999th percentile: {statistics['999th_percentile']:.0f} words")
print(f"9999th percentile: {statistics['9999th_percentile']:.0f} words")
print(f"Max: {statistics['max']:.0f} words")


# Push to hub
combined_dataset.push_to_hub("joelniklaus/SwissLegalTranslations", private=True)

print("Dataset successfully pushed to hub at joelniklaus/SwissLegalTranslations")
