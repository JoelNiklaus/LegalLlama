"""
Adapted from here: https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing#scrollTo=Edrn7Rxmojtu

More documentation: https://github.com/unslothai/unsloth/wiki#train-on-completions--responses-only-do-not-train-on-inputs

"""

import os

from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

print("Loading unsloth...")

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",  # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",  # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
]  # More models at https://huggingface.co/unsloth

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3.5-mini-instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

print("Initializing PEFT model...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="phi-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
)


def formatting_prompts_func(examples):
    convos = []
    languages = ["de", "fr", "it", "rm", "en"]

    for example in zip(*examples.values()):
        example = {key: value for key, value in zip(examples.keys(), example)}
        # Create a prompt for each language pair
        for source_lang in languages:
            for target_lang in languages:
                if source_lang == target_lang:
                    continue
                if not example[f"{source_lang}_artText"] or not example[f"{target_lang}_artText"]:
                    continue  # skip when we don't have any translations for the language pair
                convos.append(create_prompt(example, source_lang, target_lang))
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}


def create_prompt(example, source_lang, target_lang):
    prompt = {
        "role": "user",
        "content": f"""
Translate the following Swiss law article from {source_lang} to {target_lang}:
{example[f'{source_lang}_artTitle']}
{example[f'{source_lang}_artText']}
            """,
    }
    answer = {
        "role": "assistant",
        "content": f"""
{example[f'{target_lang}_artTitle']}
{example[f'{target_lang}_artText']}
            """,
    }
    return [prompt, answer]


def preprocess(dataset):
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        batch_size=100,
        num_proc=NUM_CPUs,
        remove_columns=dataset.column_names
    )
    return dataset


NUM_CPUs = os.cpu_count()

print("Loading dataset...")
dataset = load_dataset("joelniklaus/SwissLawTranslations", "article_level")

print("Formatting prompts...")
train = preprocess(dataset["train"])
validation = preprocess(dataset["validation"])
test = preprocess(dataset["test"])

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train,
    eval_dataset=validation,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=NUM_CPUs,
    #callbacks=[EarlyStoppingCallback(3, 0.0)],
    packing=False,  # Can make training 5x faster for short sequences, but increases preprocessing time
    args=SFTConfig(
        # auto_find_batch_size=True,
        per_device_train_batch_size=8,
        # gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=10,  # This is just for debugging
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
        report_to="wandb",
    ),
)
# trainer = train_on_responses_only(trainer) # Needs separate instruction_part and response_part in the dataset

print("Training...")
trainer_stats = trainer.train()

### Inference

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="phi-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
)

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

messages = [
    {"from": "human", "value": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True)
tokenizer.batch_decode(outputs)

### Save the model

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving
