from unsloth import FastLanguageModel

max_seq_length = 512
dtype = None
load_in_4bit = True
run_names = ["SLT-Qwen2.5-72B-Instruct"]

for run_name in run_names:
    print(f"Uploading {run_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"models/{run_name}",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Save LoRA weights
    model.push_to_hub(f"joelniklaus/{run_name}-LoRA", private=True)
    tokenizer.push_to_hub(f"joelniklaus/{run_name}-LoRA", private=True)

    # Save 16bit merged weights
    #model.save_pretrained_merged(f"models/{run_name}-16bit", tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged(f"joelniklaus/{run_name}-16bit", tokenizer, save_method="merged_16bit", private=True)

    # Save 4bit merged weights
    #model.save_pretrained_merged(f"models/{run_name}-4bit", tokenizer, save_method="merged_4bit_forced")
    model.push_to_hub_merged(f"joelniklaus/{run_name}-4bit", tokenizer, save_method="merged_4bit_forced", private=True)


