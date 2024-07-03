import torch
from utils import *

def get_entropy(model, hidden_states, lenses=None):

    num_modules, num_samples = hidden_states.shape[:2]

    if lenses is None:
        lenses = [None]*num_modules

    if len(hidden_states.shape)>3:
        entropy = torch.zeros(num_modules, num_samples, hidden_states.shape[2])
    else:
        entropy = torch.zeros(num_modules, num_samples)

    for i in tqdm(range(num_modules)):
        unembeded = unembed(model, hidden_states[i], lenses[i])
        entropy[i] = -(unembeded.softmax(dim=-1)*unembeded.log_softmax(dim=-1)).sum(-1)

    return entropy

def get_cross_entropy(model, hidden_states, lenses=None, mode='last', target_hidden_states=None, last_module=False):
    num_modules, num_samples = hidden_states.shape[:2]

    if lenses is None:
        lenses = [None]*num_modules

    considered_modules = num_modules if last_module else num_modules-1
    
    if len(hidden_states.shape)>3:
        cross_entropy = torch.zeros(considered_modules, num_samples, hidden_states.shape[2])
    else:
        cross_entropy = torch.zeros(considered_modules, num_samples)

    for i in tqdm(range(considered_modules)):

        unembedded = unembed(model, hidden_states[i], lenses[i]).softmax(dim=-1)
        if target_hidden_states is None:
            if mode == 'last':
                target = unembed(model, hidden_states[-1], lenses[i]).log_softmax(dim=-1)
            else:
                target = unembed(model, hidden_states[i+1], lenses[i]).log_softmax(dim=-1)
        else:
            target = unembed(model, target_hidden_states[i], lenses[i]).log_softmax(dim=-1)

        cross_entropy[i] = -(unembedded * target).sum(-1)

    return cross_entropy


def get_probability(model, hidden_states, lenses=None, target_token=None, show_tqdm=True):
    num_modules, num_samples = hidden_states.shape[:2]

    if lenses is None:
        lenses = [None]*num_modules

    # probability of predicted token over layers
    probs = torch.zeros([num_modules, num_samples])

    for i in tqdm(range(num_modules), disable=not show_tqdm):
        unembedded = unembed(model, hidden_states[i, torch.arange(num_samples), :], lenses[i])
        probs[i] = unembedded.softmax(dim=-1)[torch.arange(num_samples), target_token]

    return probs

def get_KL_divergence(model, hidden_states, lenses, mode='last', target_hidden_states=None, last_module=False):

    num_modules, num_samples = hidden_states.shape[:2]

    if lenses is None:
        lenses = [None]*num_modules

    considered_modules = num_modules if last_module else num_modules-1
    if len(hidden_states.shape)>3:
        KL = torch.zeros(considered_modules, num_samples, hidden_states.shape[2])
    else:
        KL = torch.zeros(considered_modules, num_samples)
    for i in tqdm(range(considered_modules)):

        unembeded = unembed(model, hidden_states[i], lenses[i])
        prob = unembeded.softmax(dim=-1)
        log_prob = unembeded.log_softmax(dim=-1)
        if target_hidden_states is None:
            if mode == 'last':
                log_prob_target = unembed(model, hidden_states[-1], lenses[i]).log_softmax(dim=-1)
            else:
                log_prob_target = unembed(model, hidden_states[i+1], lenses[i]).log_softmax(dim=-1)
        else:
            log_prob_target = unembed(model, target_hidden_states[i], lenses[i]).log_softmax(dim=-1)

        KL[i] = (prob * (log_prob-log_prob_target)).sum(-1)
        # KL[i] = torch.nn.functional.kl_div(unembedded, target, log_target=True, reduction='none').sum(dim=-1)

    return KL

def get_top_k_prob_token(model, tokens, batch_size, k=10):

    hidden_states = get_hidden_from_tokens(model, ['lm_head'], tokens, batch_size=batch_size, token_position=-1, disable_tqdm=True)[0]

    topk = torch.topk(hidden_states.softmax(dim=-1), k, dim=-1)

    return topk.indices, topk.values
