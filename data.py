from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import json
import torch.nn.functional as F

from utils import *

def load_data_set(dataset_name, format='freeform', system_token="<|system|>", assistant_token="<|assistant|>"):

    assert format in ['freeform', 'xml'], "Format not supported."
    if format == 'freeform':
        lie_format = f"{system_token}You will complete the sentence with intentionally false information.{assistant_token}{{}}"
        truth_format = f"{system_token}You will complete the sentence with true information.{assistant_token}{{}}"
    else:
        lie_format = ("<StatementCompletion>\n<statement>The capital of France is</statement>\n<answer_type>True</answer_type>\n<completion>Paris</completion>\n</StatementCompletion>\n"
                "\n<StatementCompletion>\n<statement>Japan's currency is</statement>\n<answer_type>False</answer_type>\n<completion>Euro</completion></StatementCompletion>\n"
                  "\n<StatementCompletion>\n<statement>{}</statement>\n<answer_type>False</answer_type>\n<completion>")
        truth_format = ("<StatementCompletion>\n<statement>The capital of France is</statement>\n<answer_type>True</answer_type>\n<completion>Paris</completion>\n</StatementCompletion>\n"
                        "\n<StatementCompletion>\n<statement>Japan's currency is</statement>\n<answer_type>False</answer_type>\n<completion>Euro</completion></StatementCompletion>\n"
                        "\n<StatementCompletion>\n<statement>{}</statement>\n<answer_type>True</answer_type>\n<completion>")

    assert os.path.exists(f'data/{dataset_name}.json'), "Data set does not exist."
    with open(f'data/{dataset_name}.json', 'r') as file:
        dataset = json.load(file)

    dataset = pd.DataFrame(dataset)
    org_data = np.array(dataset['statement'])

    truth_scenario = np.array([truth_format.format(x) for x in org_data])
    lie_scenario = np.array([lie_format.format(x) for x in org_data])

    dataset_dict = {
        'org_data': org_data,
        'dataset_name': dataset_name, 
        'lie_scenario' : lie_scenario,
        'truth_scenario' : truth_scenario,
        'true_answer': np.array(dataset['completion']),
        'lie_format': lie_format,
        'truth_format': truth_format,
    }

    return dataset_dict


def check_statements(model, tokenizer, data, answers, max_new_tokens=5, batch_size=10):
    size = len(answers)
    correct = np.zeros(size)
    ctr = 0
    # Calculate total number of batches for progress bar
    total_batches = size // batch_size + (0 if size % batch_size == 0 else 1)
    generated_answers = []
    # Wrap the zip function with tqdm for the progress bar
    for batch, batch_gt in tqdm(zip(batchify(data, batch_size), batchify(answers, batch_size)), total=total_batches):
        batch_answers = generate(model, tokenizer, batch, max_new_tokens)
        for i, a in enumerate(batch_answers):
            if batch_gt[i].lower() in a.lower():
                correct[ctr] = 1
            ctr += 1
            generated_answers.append(a)
    return correct, generated_answers


def check_answer(tokenizer, answer_tokens, GT, batch_size=64):
    total_batches = len(GT) // batch_size
    success = []
    for batch in tqdm(zip(batchify(answer_tokens, batch_size), batchify(GT, batch_size)), total=total_batches):
        tokens, gt = batch
        # decode the generated tokens
        string_answer = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        # check if GT in answer
        success.extend([g.lower() in s.lower() for s, g in zip(string_answer, gt)])

    return np.array(success)


def get_overlap_truth_lies(model, tokenizer, dataset, max_new_tokens=10, batch_size=64):
    # generate tokens for truths and lies
    print(f"Size of dataset {dataset['dataset_name']}: {len(dataset['true_answer'])}")
    output_tokens_truth, answer_tokens_truth = generate_tokens(model, tokenizer, dataset['truth_scenario'], 
                                                               max_new_tokens=max_new_tokens, batch_size=batch_size, do_sample=False)
    # check if the generated answers contain the ground truth
    success_truth = check_answer(tokenizer, answer_tokens_truth, dataset['true_answer'], batch_size=batch_size)
    print(f"Success rate when generating truths: {np.mean(success_truth)*100:.2f}%")

    output_tokens_lie, answer_tokens_lie = generate_tokens(model, tokenizer, dataset['lie_scenario'], 
                                                           max_new_tokens=max_new_tokens, batch_size=batch_size, do_sample=False)
    # check if the generated answers contain the ground truth
    success_lie = check_answer(tokenizer, answer_tokens_lie, dataset['true_answer'], batch_size=batch_size)
    print(f"Success rate when generating lies:   {100-np.mean(success_lie)*100:.2f}%")
    overlap = success_truth & ~success_lie
    print(f"Overlap: {np.mean(overlap)*100:.2f}%")
    dataset['success'] = overlap

    # select only data where overlap is 1
    output_tokens_truth = {k: v[overlap] for k, v in output_tokens_truth.items()}
    output_tokens_lie = {k: v[overlap] for k, v in output_tokens_lie.items()}

    answer_tokens_truth = [v for i, v in enumerate(answer_tokens_truth) if overlap[i]]
    answer_tokens_lie = [v for i, v in enumerate(answer_tokens_lie) if overlap[i]]

    # save data in dataset
    dataset['output_tokens_truth'] = output_tokens_truth
    dataset['output_tokens_lie'] = output_tokens_lie
    dataset['answer_tokens_truth'] = answer_tokens_truth
    dataset['answer_tokens_lie'] = answer_tokens_lie

    # save answers as strings
    dataset['answer_truth'] = tokenizer.batch_decode(answer_tokens_truth, skip_special_tokens=True)
    dataset['answer_lie'] = tokenizer.batch_decode(answer_tokens_lie, skip_special_tokens=True)


def change_format(dataset, lie_format, truth_format):

    thruth_scenario = np.array([truth_format.format(x) for x in dataset['org_data']])
    # apply lie format
    lie_scenario = np.array([lie_format.format(x) for x in dataset['org_data']])

    dataset['lie_scenario'] = lie_scenario
    dataset['truth_scenario'] = thruth_scenario
    dataset['lie_format'] = lie_format
    dataset['truth_format'] = truth_format