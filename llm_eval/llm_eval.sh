#!/bin/bash

activate () {
    echo "Activating virtual environment..."
    . .venv/bin/activate
}
cd ~/llm_eval
activate

# Inspired by the Euro-LLM-Leaderboard: https://huggingface.co/spaces/occiglot/euro-llm-leaderboard

MODELS=(
    #"Qwen/Qwen2-7B-Instruct"
    #"Qwen/Qwen2-1.5B-Instruct"
    #"Qwen/Qwen2-0.5B-Instruct"
    #"microsoft/Phi-3-medium-128k-instruct" # now evaluated on the leaderboard
    "google/gemma-2-9b-it"
    "microsoft/Phi-3-small-128k-instruct"
    "microsoft/Phi-3-mini-128k-instruct" # CUDA memory error for some reason
)
OUTPUT_DIR="./output"

# Create the main output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to run lm_eval with given parameters
run_lm_eval() {
    local model=$1
    local task_list=$2
    local num_fewshot=$3
    local task_name=$4

    echo "Running $task_name with $num_fewshot-shot for model $model"
    if ! lm_eval --model hf --model_args pretrained=$model --tasks $task_list --num_fewshot $num_fewshot --batch_size auto:4 --output_path "${OUTPUT_DIR}" --trust_remote_code; then
        echo "The evaluation for $task_name with model $model failed. Please check the logs in ${OUTPUT_DIR} for more information."
        return 1
    fi
    echo "Completed $task_name for $model"
    echo "--------------------"
}

# Main execution loop
for MODEL in "${MODELS[@]}"; do
    echo "Starting evaluations for model: $MODEL"

    # TruthfulQA: approx. 3m on A100 for Phi-3-medium-128k-instruct
    run_lm_eval "$MODEL" "truthfulqa_mc2,truthfulqa_de_mc2,truthfulqa_fr_mc2,truthfulqa_es_mc2,truthfulqa_it_mc2" 0 "TruthfulQA" || continue

    # Belebele: approx. 5m on A100 for Phi-3-medium-128k-instruct
    run_lm_eval "$MODEL" "belebele_eng_Latn,belebele_ita_Latn,belebele_deu_Latn,belebele_fra_Latn,belebele_spa_Latn" 5 "Belebele" || continue

    # ARC: approx. 20m on A100 for Phi-3-medium-128k-instruct
    run_lm_eval "$MODEL" "arc_challenge,arc_de,arc_es,arc_it,arc_fr" 25 "ARC" || continue

    # MMLU: approx. 1h on A100 for Phi-3-medium-128k-instruct
    run_lm_eval "$MODEL" "m_mmlu_en,m_mmlu_de,m_mmlu_fr,m_mmlu_es,m_mmlu_it" 5 "MMLU" || continue

    # HellaSwag: approx. 2.5h on A100 for Phi-3-medium-128k-instruct
    # (16h for all) ==> skip for cost reasons
    # run_lm_eval "$MODEL" "hellaswag,hellaswag_es,hellaswag_it,hellaswag_fr,hellaswag_de" 10 "HellaSwag" || continue

    # Possible additional supported benchmark: lambada_openai_mt_en,lambada_openai_mt_de,lambada_openai_mt_fr,lambada_openai_mt_es,lambada_openai_mt_it

    echo "All tasks completed for model: $MODEL"
done

echo "All evaluations completed."
