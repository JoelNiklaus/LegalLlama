from unsloth import FastLanguageModel

temporary_location = "/ephemeral/_unsloth_temporary_saved_buffers"
max_seq_length = 512
dtype = None
load_in_4bit = True
hf_org = "SwiLTra-Bench"
run_names = ["SLT-gemma-2-9b-it"]

for run_name in run_names:
    print(f"Uploading {run_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"models/{run_name}",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Save LoRA weights
    model.push_to_hub(f"{hf_org}/{run_name}-LoRA", private=True)
    tokenizer.push_to_hub(f"{hf_org}/{run_name}-LoRA", private=True)

    # Save 16bit merged weights
    #model.save_pretrained_merged(f"models/{run_name}-16bit", tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged(f"{hf_org}/{run_name}-16bit", tokenizer, save_method="merged_16bit", private=True, temporary_location=temporary_location)

    # Save 4bit merged weights
    #model.save_pretrained_merged(f"models/{run_name}-4bit", tokenizer, save_method="merged_4bit_forced")
    model.push_to_hub_merged(f"{hf_org}/{run_name}-4bit", tokenizer, save_method="merged_4bit_forced", private=True, temporary_location=temporary_location)


