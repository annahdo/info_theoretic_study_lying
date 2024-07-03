# main experiments on 3 datasets
python info_theoretic_measures_script.py --model_name "HuggingFaceH4/zephyr-7b-beta" --dataset_name "Statements1000"
python info_theoretic_measures_script.py --model_name "HuggingFaceH4/zephyr-7b-beta" --dataset_name "cities"
python info_theoretic_measures_script.py --model_name "HuggingFaceH4/zephyr-7b-beta" --dataset_name "FreebaseStatements"


if [ -z "${HF_ACCESS_TOKEN}" ]; then
    echo "HF_ACCESS_TOKEN is not set, pls set this to your huggingface acces token (HF_ACCESS_TOKEN=<your_access_token>) to run the experiments with the llama models "
else
    echo "HF_ACCESS_TOKEN is set to '${HF_ACCESS_TOKEN}'"
    # experiments with llama models
    python info_theoretic_measures_script.py --model_name "meta-llama/Llama-2-7b-chat-hf" --dataset_name "Statements1000" --access_token $HF_ACCESS_TOKEN --assistant_token " [/INST] " --system_token "[INST]"
    python info_theoretic_measures_script.py --model_name "meta-llama/Llama-2-13b-chat-hf" --dataset_name "Statements1000" --access_token $HF_ACCESS_TOKEN --assistant_token " [/INST] " --system_token "[INST]"
    # experiments with llama models and tuned lens
    python info_theoretic_measures_script.py --model_name "meta-llama/Llama-2-7b-chat-hf" --dataset_name "Statements1000" --lens_type "tuned_lens" --access_token $HF_ACCESS_TOKEN --assistant_token " [/INST] " --system_token "[INST]"
    python info_theoretic_measures_script.py --model_name "meta-llama/Llama-2-13b-chat-hf" --dataset_name "Statements1000" --lens_type "tuned_lens" --access_token $HF_ACCESS_TOKEN --assistant_token " [/INST] " --system_token "[INST]"
fi

# experiments with alternative instruction
python info_theoretic_measures_script.py --model_name "HuggingFaceH4/zephyr-7b-beta" --dataset_name "Statements1000" --truth_format "<|system|>Continue the statement with accurate information.\n<|assistant|>{}" --lie_format "<|system|>Continue the statement with wrong information.\n<|assistant|>{}"
python info_theoretic_measures_script.py --model_name "HuggingFaceH4/zephyr-7b-beta" --dataset_name "Statements1000" --truth_format "<|system|>The following statement is true:\n<|assistant|>{}" --lie_format "<|system|>The following statement is false:\n<|assistant|>{}"

# experiments with xml format on 3 datasets
python info_theoretic_measures_script.py --model_name "HuggingFaceH4/zephyr-7b-beta" --dataset_name "Statements1000" --format_type "xml"
python info_theoretic_measures_script.py --model_name "HuggingFaceH4/zephyr-7b-beta" --dataset_name "cities" --format_type "xml"
python info_theoretic_measures_script.py --model_name "HuggingFaceH4/zephyr-7b-beta" --dataset_name "FreebaseStatements" --format_type "xml"
