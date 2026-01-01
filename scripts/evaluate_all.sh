#!/bin/bash

# Set proxy
# export HTTP_PROXY='http://10.129.86.168:7890'
# export HTTPS_PROXY='http://10.129.86.168:7890'

prompt_types=(
    "cot"
    # "pot"
)
subsets=(
    "test"
)

model_outputs=(
    # "o4-mini-high.json"
    # "gpt4o.json"
    # "claude37_thinking.json"
    # "gemini.json"
    # "grok2.json"
    # "doubao-vision.json"
    # "doubao-thinking.json"
    # "llama4.json"
    # "gemma3.json"
    # "mistral.json"
    # "qwen2_5vl.json"
    # "o4-mini-high_text.json"
    # "gpt4o_text.json"
    # "claude37_thinking_text.json"
    # "gemini_text.json"
    # "grok3_text.json"
    # "doubao-vision_text.json"
    # "doubao-thinking_text.json"


    # "llama4_text.json"
    # "gemma3_text.json"
    # "mistral_text.json"
    # "qwen2_5vl_text.json"
    # "qwen3_text.json"
    # "deepseek-r1_text.json"
    # "deepseek-v3_text.json"
    # "llama3_3_text.json"

    # "doubao-vision_rag_colqwen.json"
    # "doubao-vision_rag_oracle.json"
    # "doubao-vision_text.json"
    # "doubao-vision.json"
    # "doubao-vision_rag_m3docrag.json"
    # "doubao-vision_rag_simpledoc_500.json"
    #"doubao-vision_rag_simpledoc.json"
    "doubao_vision_rag_vidorag_0720.json"
    # "doubao-vision_rag_mdocagent.json"
    # "doubao-vision_rag_vidorag_400.json"
    # "doubao-vision_rag_vidorag.json"
    # "qwen2.5_vl_7b_vrag_rl.json"
    # "colqwen2.5-v0.2_doubao_vision_rag_m3docrag.json"
    # "doubao_io_image_50.json"
    # "doubao-thinking_rag_colqwen.json"
    # "doubao-thinking_rag_oracle.json"
    # "doubao-thinking_text.json"
    # "doubao-thinking.json"


)

api_base="https://openrouter.ai/api/v1"
api_key="xxxx"

for prompt_type in "${prompt_types[@]}"; do
    for subset in "${subsets[@]}"; do
        echo "Evaluating $prompt_type on $subset set"
        raw_dir="outputs/$subset/raw_${prompt_type}_outputs"
        processed_dir="outputs/$subset/processed_${prompt_type}_outputs"
        result_file="outputs/results/${subset}_${prompt_type}_results.json"

        # remove result file if it exists
        if [ -f "$result_file" ]; then
            rm "$result_file"
        fi

        # Iterate over each file in the raw output directory
        # for raw_file in "$raw_dir"/*; do
        for raw_file in "${model_outputs[@]}"; do
            filename=$(basename "$raw_file")
            
            python evaluation.py \
                --prediction_path "outputs/$subset/raw_${prompt_type}_outputs/$raw_file" \
                --evaluation_output_dir "$processed_dir" \
                --prompt_type "$prompt_type" \
                --ground_truth_file "data/$subset.json" \
                --result_file "$result_file" \
                --api_base "$api_base" \
                --api_key "$api_key"
            echo "Finished evaluating $filename"
        done

        echo "Finished evaluating $prompt_type on $subset set"
    done
done