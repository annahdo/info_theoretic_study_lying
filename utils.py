from tqdm import tqdm
import numpy as np
from baukit import TraceDict
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import re
from matplotlib import pyplot as plt
import torch.nn.functional as F
import os

def generate_tokens(model, tokenizer, data, max_new_tokens=10, batch_size=64, do_sample=False):
    assert tokenizer.padding_side == "left", "Not implemented for padding_side='right'"
    device = model.device
    total_batches = len(data) // batch_size
    output_tokens = {'input_ids': [], 'attention_mask': []}
    answer_tokens = []
    max_len = 0
    pad_token_id = tokenizer.eos_token_id
    for batch in tqdm(batchify(data, batch_size), total=total_batches):
        inputs = tokenizer(list(batch), return_tensors="pt", padding=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=pad_token_id).detach().cpu()
        n, il = inputs['input_ids'].shape
        _, ol = outputs.shape
        max_len = max(max_len, ol)
        output_tokens['input_ids'].extend(outputs)
        # define attention mask
        attention_mask = torch.where(outputs!=pad_token_id, 1, 0).long()
        output_tokens['attention_mask'].extend(attention_mask)
        answer_tokens.extend(outputs[:, il:])

    # convert to tensor
    output_token_tensor = torch.ones([len(data), max_len], dtype=torch.long) * tokenizer.pad_token_id
    attention_mask_tensor = torch.zeros([len(data), max_len], dtype=torch.long)

    for i, (input_ids, attention_mask) in enumerate(zip(output_tokens['input_ids'], output_tokens['attention_mask'])):
        output_token_tensor[i, -len(input_ids):] = input_ids
        attention_mask_tensor[i, -len(attention_mask):] = attention_mask

    output_tokens = {'input_ids': output_token_tensor, 'attention_mask': attention_mask_tensor}

    return output_tokens, answer_tokens


def generate(model, tokenizer, text, max_new_tokens=5, do_sample=False):
    text = list(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    _, input_length = inputs["input_ids"].shape
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
    answers = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    return answers


def batchify(lst, batch_size):
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def get_hidden_from_tokens(model, module_names, data, batch_size=10, token_position=-1, disable_tqdm=False):
    size = len(data['input_ids'])
    total_batches = size // batch_size + (0 if size % batch_size == 0 else 1)
    device = model.device
    # list of empty tensors for hidden states
    hidden_states = [None] * len(module_names)
    with torch.no_grad(), TraceDict(model, module_names) as return_dict:

        for input_ids, attention_mask in tqdm(zip(batchify(data['input_ids'], batch_size), batchify(data['attention_mask'], batch_size)), total=total_batches, disable=disable_tqdm):
            _ = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            #_ = model.generate(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), max_new_tokens=1, do_sample=False, pad_token_id=model.config.pad_token_id, temperature=0)
            for i, module_name in enumerate(module_names):
                # check for tuple output (in residual stream usually)
                if isinstance(return_dict[module_name].output, tuple):
                    output = return_dict[module_name].output[0][:, token_position, :].detach().cpu()
                else:
                    output = return_dict[module_name].output[:, token_position, :].detach().cpu()

                if hidden_states[i] is None:
                    hidden_states[i] = output
                else:
                    hidden_states[i] = torch.cat([hidden_states[i], output], dim=0)

        # convert list to tensor with new dimension at start
        hidden_states = torch.cat([t.unsqueeze(0) for t in hidden_states], dim=0)

    return hidden_states

def unembed(model, tensors, lens=None):
    device = model.device
    tensors = tensors.unsqueeze(0).to(device)
    if lens is not None:
        tensors = tensors + lens(tensors)
    tensors = model.model.norm(tensors)
    return model.lm_head(tensors).squeeze().detach().cpu().float()

def embed(model, tensors):
    device = model.device
    tensors = tensors.unsqueeze(0).to(device)
    tensors = model.model.embed_tokens(tensors)
    return tensors.squeeze().detach().cpu().float()


def get_lens(lens_type='logit_lens', num_hidden_layers=32, model_name=None, device='cuda', hidden_size=4096):
    
    # (tuned lens only works for models for which tuned lenses are available at https://huggingface.co/spaces/AlignmentResearch/tuned-lens/tree/main/lens)
    if lens_type == "tuned_lens":
        # get tuned lenses
        download_path = f"https://huggingface.co/spaces/AlignmentResearch/tuned-lens/resolve/main/lens/{model_name}/params.pt?download=true"
        target_path = f'data/lenses/{model_name}_params.pt'
        if not os.path.exists(target_path):
            os.makedirs(target_path.rsplit('/', 1)[0], exist_ok=True)
            try:
                os.system(f"wget {download_path} -O {target_path} -q")
            except:
                assert False, "Error in downloading lens parameters, tuned lens parameters might not exist for your model"

        raw_lenses = torch.load(target_path)

        lenses = []
        for i in range(num_hidden_layers):
            lens = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
            lens.weight = torch.nn.Parameter(raw_lenses[f'{i}.weight'].to(torch.float16))
            lens.bias = torch.nn.Parameter(raw_lenses[f'{i}.bias'].to(torch.float16))
            lens = lens.to(device)
            lenses.append(lens)

        # linear layer that has zero matrix as weight and zeros as bias
        lens = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        lens.weight = torch.nn.Parameter(torch.zeros([hidden_size, hidden_size], dtype=torch.float16).to(device))
        lens = lens.to(device)
        lenses.append(lens)

        # pop first layer
        _ = lenses.pop(0)

    elif lens_type == "logit_lens":
        lenses = [None]*num_hidden_layers
    else:
        raise NotImplementedError(f"lens_type: {lens_type} not implemented")

    return lenses


def print_examples(dataset, n=10):

    selected_GT = dataset['true_answer'][dataset['success']]
    selected_scenes = dataset['org_data'][dataset['success']]
    # inspect lies
    print(f"lie_format: {dataset['lie_format']}")
    print(f"truth_format: {dataset['truth_format']}\n")
    print("Examples with format: [statement/question] - [models completion]\n")
    # random indices
    np.random.seed(0)
    idx = np.random.choice(len(selected_scenes), n)
    for i in idx:
        print(f"{selected_scenes[i]}")
        print(f"\tGT: {selected_GT[i]}")
        print(f"\tgenerated lie:   {dataset['answer_lie'][i]}")
        print(f"\tgenerated truth: {dataset['answer_truth'][i]}")
        print("-"*20)

def pdist(x, y):
    diff = x.unsqueeze(2) - y.unsqueeze(1)
    distance_matrix = torch.norm(diff, dim=-1) 
    return distance_matrix.mean(dim=0)

def pcossim(x, y):
    x_expanded = x.unsqueeze(2)
    y_expanded = y.unsqueeze(1) 
    cosine_similarity_matrix = F.cosine_similarity(x_expanded, y_expanded, dim=-1)
    return cosine_similarity_matrix.mean(dim=0)


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


def find_matching_tokens(tokenizer, k, top_k_tokens):

    def check_arrays(array1, array2):
        result = [1 if a.lower().startswith(b.lower()) or b.lower().startswith(a.lower()) else 0 for a, b in zip(array1, array2)]
        return result

    top_k_strings = np.empty((top_k_tokens.shape), dtype='object')
    for i in range(k):
        top_k_strings[:,i]= tokenizer.batch_decode(top_k_tokens[:,i])

    top_k_matching = torch.zeros((top_k_tokens.shape))
    for i in range(k):
        top_k_matching[:,i] = torch.FloatTensor(check_arrays(top_k_strings[:,i], top_k_strings[:,0]))

    return top_k_matching