"""
Trains a model on the SwissLegalTranslations dataset. Only works on NVIDIA GPUs due to unsloth.

Adapted from here: https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing#scrollTo=Edrn7Rxmojtu
"""

from ast import literal_eval
import os

from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_dataset

print("Loading unsloth...")

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only


import argparse

parser = argparse.ArgumentParser(description="Train a model on the SwissLegalTranslations dataset")
parser.add_argument("--model_name", type=str)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--push_to_hub", type=bool, default=False)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=5)
# Can go down to 512 because when we look at the sentence level, they go rarely above 200 whitespace split words
parser.add_argument("--max_seq_length", type=int, default=512)
args = parser.parse_args()


model_name = args.model_name

dataset_name = "SwissLegalTranslations"
hf_model_name = f"unsloth/{model_name}-bnb-4bit"

if "llama" in model_name.lower():
    chat_template = "llama"  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth, phi-3
elif "qwen" in model_name.lower():
    chat_template = "chatml"
elif "gemma" in model_name.lower():
    chat_template = "chatml"
elif "phi" in model_name.lower():
    chat_template = "phi-3"
elif "mistral" in model_name.lower():
    chat_template = "mistral"
else:
    chat_template = "zephyr"

train_on_responses_only = False  # The loss starts lower, but training is not faster

total_batch_size = 128  # Keep this stable for reproducibility
gradient_accumulation_steps = int(total_batch_size / args.batch_size)
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
run_name = f"SLT-{model_name}"
device = "cuda"  # Unsloth only supports CUDA
seed = 42


print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=hf_model_name,
    max_seq_length=args.max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

print("Initializing PEFT model...")
model = FastLanguageModel.get_peft_model(
    model,
    r=args.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=seed,
    # rank stabilized LoRA: slightly higher training time, but better results at higher ranks (e.g., 256): https://huggingface.co/blog/damjan-k/rslora
    use_rslora=True,
    loftq_config=None,  # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template=chat_template,
)


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
            # Create prompts for both translation directions
            convos.append(create_prompt(example, lang1, lang2))
            convos.append(create_prompt(example, lang2, lang1))

    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}


def create_prompt(example, source_lang, target_lang):
    prompt = {
        "role": "user",
        "content": f"{source_lang.upper()}: {example[source_lang]}\n{target_lang.upper()}:",
    }
    answer = {
        "role": "assistant",
        "content": f"{example[target_lang]}",
    }
    return [prompt, answer]


def preprocess(dataset):
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        batch_size=1000,
        num_proc=NUM_CPUs,
        remove_columns=dataset.column_names,
    )
    # Add shuffling with a fixed seed for reproducibility
    dataset = dataset.shuffle(seed=seed)
    return dataset


NUM_CPUs = os.cpu_count()

print("Loading dataset...")
dataset = load_dataset("joelniklaus/SwissLegalTranslations")

print("Formatting prompts...")
train = preprocess(dataset["train"])
validation = preprocess(dataset["validation"])
test = preprocess(dataset["test"])

completion_only_collator = DataCollatorForCompletionOnlyLM(
    instruction_template="<|user|>",
    response_template="<|assistant|>",
    tokenizer=tokenizer,
    mlm=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train,
    eval_dataset=validation,
    dataset_text_field="text",
    max_seq_length=args.max_seq_length,
    data_collator=completion_only_collator if train_on_responses_only else None,
    dataset_num_proc=NUM_CPUs,
    callbacks=[EarlyStoppingCallback(3, 0.0)], # We evaluate every 100 steps, so we can have patience of 3
    packing=True,
    args=SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=1000,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        load_best_model_at_end=True,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=seed,
        output_dir=f"outputs/{run_name}",
        report_to="tensorboard",
    ),
)


print("Training...")
trainer_stats = trainer.train()
print(trainer_stats)


### Inference

print("Setting up inference...")

tokenizer = get_chat_template(
    tokenizer,
    chat_template=chat_template,
    mapping={
        "role": "from",
        "content": "value",
        "user": "human",
        "assistant": "gpt",
    },  # ShareGPT style
)

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

print("Inferring...")
article = "Die Todesstrafe ist abgeschafft. Niemand darf zu dieser Strafe verurteilt oder hingerichtet werden."
messages = [
    {
        "from": "human",
        "value": f"DE: {article}\nFR:",
    },
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to(device)

outputs = model.generate(input_ids=inputs, max_new_tokens=256, use_cache=True)
print(tokenizer.batch_decode(outputs))

### Save the model
print("Saving the model...")

# Save LoRA weights
model.save_pretrained(f"models/{run_name}")
tokenizer.save_pretrained(f"models/{run_name}")

if args.push_to_hub:
    model.push_to_hub(f"joelniklaus/{run_name}-LoRA", private=True)
    tokenizer.push_to_hub(f"joelniklaus/{run_name}-LoRA", private=True)

    # Save 16bit merged weights
    #model.save_pretrained_merged(
    #    f"models/{run_name}-16bit", tokenizer, save_method="merged_16bit"
    #)
    model.push_to_hub_merged(
        f"joelniklaus/{run_name}-16bit", tokenizer, save_method="merged_16bit", private=True
    )

    # Save 4bit merged weights
    #model.save_pretrained_merged(
    #    f"models/{run_name}-4bit", tokenizer, save_method="merged_4bit_forced"
    #)
    model.push_to_hub_merged(
        f"joelniklaus/{run_name}-4bit",
        tokenizer,
        save_method="merged_4bit_forced",
        private=True,
    )
