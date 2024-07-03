#!/usr/bin/env python3

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import importlib
import hashlib

# Import custom modules
import utils, data, info_theory, plots
importlib.reload(utils)
importlib.reload(data)
importlib.reload(info_theory)
importlib.reload(plots)

from utils import *
from data import *
from info_theory import *
from plots import *


def load_model(model_name, lens_type="logit_lens", access_token=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    short_model_name = model_name.split("/")[-1]
    if model_name == "meta-llama/Llama-2-7b-chat-hf":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token=access_token).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    lenses = get_lens(lens_type, model.config.num_hidden_layers, model_name, hidden_size=model.config.hidden_size, device=device)

    return model, tokenizer, short_model_name, lenses

def run_all(model, tokenizer, lenses, short_model_name, dataset_name, format_type='freeform', system_token="<|system|>", assistant_token="\n<|assistant|>", truth_format=None, lie_format=None, max_new_tokens=10, batch_size=64, lens_type="logit_lens", k=10):
    dataset = load_data_set(dataset_name, format_type, assistant_token=assistant_token, system_token=system_token) 
    if lie_format is not None and truth_format is not None:
        change_format(dataset, lie_format, truth_format)
        # create hash from lie_format and truth_format
        hash = hashlib.md5(lie_format.encode()).hexdigest() + hashlib.md5(truth_format.encode()).hexdigest()
        format_type = f'custom_{hash}'
    print("#"*50)
    print("Generating truth and lie statements...")
    get_overlap_truth_lies(model, tokenizer, dataset, max_new_tokens=max_new_tokens, batch_size=batch_size)

    module_names = [f'model.layers.{i}' for i in range(model.config.num_hidden_layers)]
    token_positions = -max_new_tokens-1
    print("#"*50)
    print("Calculating hidden states...")
    dataset['hidden_states_lie'] = get_hidden_from_tokens(model, module_names, dataset['output_tokens_lie'], batch_size=batch_size, token_position=token_positions)
    dataset['hidden_states_truth'] = get_hidden_from_tokens(model, module_names, dataset['output_tokens_truth'], batch_size=batch_size, token_position=token_positions)

    print("#"*50)
    print("Calculating entropy...")
    entropy_truth = get_entropy(model, dataset['hidden_states_truth'], lenses=lenses)
    entropy_lie = get_entropy(model, dataset['hidden_states_lie'], lenses=lenses)
    save_path = f'plots/{short_model_name}_{dataset_name}_entropy_{lens_type}_{format_type}.pdf'
    plot_median_mean(entropy_truth, entropy_lie, y_label='Entropy', save_path=save_path, scale='log')

    print("#"*50)
    print("Calculating KL-divergence...")
    KL_truth = get_KL_divergence(model, dataset['hidden_states_truth'], lenses, mode='last')
    KL_lie = get_KL_divergence(model, dataset['hidden_states_lie'], lenses, mode='last')
    save_path = f'plots/{short_model_name}_{dataset_name}_KL_{lens_type}_{format_type}.pdf'
    plot_median_mean(KL_truth, KL_lie, y_label=f'KL divergence to last layer', save_path=save_path, scale='log')

    print("#"*50)
    print("Calculating probability of predicted token...")
    predicted_truth_tokens = np.array(dataset['answer_tokens_truth'])[:,0]
    prob_truth = get_probability(model, dataset['hidden_states_truth'], lenses, target_token=predicted_truth_tokens)
    predicted_lie_tokens = np.array(dataset['answer_tokens_lie'])[:,0]
    prob_lie = get_probability(model, dataset['hidden_states_lie'], lenses, target_token=predicted_lie_tokens)
    save_path = f'plots/{short_model_name}_{dataset_name}_prob_{lens_type}_{format_type}_predicted_token.pdf'
    plot_median_mean(prob_truth, prob_lie, save_path=save_path, y_label='Probability of predicted token', scale='linear')

    print("#"*50)
    print("Calculating probabilities of top k tokens...")
    top_k_prob_truth = torch.zeros((k,)+prob_truth.shape)
    top_k_prob_lie = torch.zeros((k,)+prob_lie.shape)
    top_k_truth_tokens = torch.topk(unembed(model, dataset['hidden_states_truth'][-1]), k, dim=-1)
    top_k_lie_tokens = torch.topk(unembed(model, dataset['hidden_states_lie'][-1]), k, dim=-1)
    for i in tqdm(range(k)):
        top_k_prob_truth[i] = get_probability(model, dataset['hidden_states_truth'], lenses, target_token=top_k_truth_tokens.indices[:,i], show_tqdm=False)
        top_k_prob_lie[i] = get_probability(model, dataset['hidden_states_lie'], lenses, target_token=top_k_lie_tokens.indices[:,i], show_tqdm=False)
    selected_layers = [20, 22, 24, 26, 28, 30, 31]
    save_path = f'plots/{short_model_name}_{dataset_name}_prob_{lens_type}_{format_type}_top_{k}_tokens.pdf'
    plot_h_bar(top_k_prob_truth, top_k_prob_lie, selected_layers, save_path=save_path)

    print("#"*50)
    print("Calculating matching tokens and summing over probabilities...")
    top_k_matching_truth = find_matching_tokens(tokenizer, k, top_k_truth_tokens.indices)
    top_k_matching_lie = find_matching_tokens(tokenizer, k, top_k_lie_tokens.indices)
    prob_truth_sums = (top_k_prob_truth*torch.transpose(top_k_matching_truth, 0,1).unsqueeze(1)).sum(dim=0)
    prob_lie_sums = (top_k_prob_lie*torch.transpose(top_k_matching_lie, 0,1).unsqueeze(1)).sum(dim=0)
    save_path = f'plots/{short_model_name}_{dataset_name}_prob_{lens_type}_{format_type}_predicted_token_summed.pdf'
    plot_median_mean(prob_truth_sums, prob_lie_sums, save_path=save_path, y_label='Probability of predicted token', scale='linear')

def main():
    parser = argparse.ArgumentParser(description="Run the pipeline for 'Information-theoretic detection of hidden cognition in LLMs'")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to use', choices=['Statements1000', 'cities', 'FreebaseStatements'])
    parser.add_argument('--format_type', type=str, default='freeform', help='Format type for truth/lie instruction (default: freeform)', choices=['freeform', 'xml'])
    parser.add_argument('--max_new_tokens', type=int, default=10, help='Max new tokens to generate to complete the statements (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lens_type', type=str, default='logit_lens', help='Lens type (default: logit_lens)', choices=['logit_lens', 'tuned_lens'])
    parser.add_argument('--top_k', type=int, default=10, help='Top K tokens (default: 10)')
    parser.add_argument('--access_token', type=str, default=None, help='Huggingface access token (default: None)')
    parser.add_argument('--system_token', type=str, default="<|system|>", help='Model specific system token (default: <|system|>)')
    parser.add_argument('--assistant_token', type=str, default="\n<|assistant|>", help='Model specific assistant token (default: \n<|assistant|>)')
    parser.add_argument('--truth_format', type=str, default=None, help='Optional alternative truth format (default: None)')
    parser.add_argument('--lie_format', type=str, default=None, help='Optional alternative lie format (default: None)')

    args = parser.parse_args()

    model, tokenizer, short_model_name, lenses = load_model(args.model_name, args.lens_type, args.access_token)
    run_all(model, tokenizer, lenses, short_model_name, args.dataset_name, args.format_type, args.system_token, args.assistant_token, args.truth_format, args.lie_format, args.max_new_tokens, args.batch_size, args.lens_type, args.top_k)

if __name__ == "__main__":
    main()
