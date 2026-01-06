
import warnings
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")

import random
import os
import numpy as np
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
import datasets
import torch.nn.functional as F
import tqdm
import argparse
import json
from model import load_tokenizer, load_model
from data_builder import load_data
from metrics import get_roc_metrics, get_precision_recall_metrics
import scipy
import math
import jsonlines

from TPPL import BARTScorerSymmetricLogPPL


import matplotlib.pyplot as plt
import time
pct = 0.03
n = 10


import string
from transformers import BartTokenizer
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
assert bart_tokenizer.mask_token is not None, "Tokenizer has no mask token"
assert bart_tokenizer.mask_token == "<mask>", f"Unexpected mask token: {bart_tokenizer.mask_token}"

def is_punctuation_token(token: str) -> bool:
    return all(ch in string.punctuation for ch in token)

def fill_and_mask_token_level(token_ids, pct, tokenizer, skip_punct=True):
    maskable_indices = []
    for i, tid in enumerate(token_ids):
        tok = tokenizer.convert_ids_to_tokens(tid)
        if tok in tokenizer.all_special_tokens:
            continue
        if skip_punct and is_punctuation_token(tok):
            continue
        maskable_indices.append(i)

    if not maskable_indices:
        return []

    n_to_mask = max(1, int(pct * len(maskable_indices)))
    n_to_mask = min(n_to_mask, len(maskable_indices))
    mask_indices = np.random.choice(maskable_indices, size=n_to_mask, replace=False)
    return mask_indices.tolist()


def perturb_token_ids(input_ids, attention_mask, tokenizer, pct):
    token_ids = input_ids.tolist()
    mask_indices = fill_and_mask_token_level(token_ids, pct, tokenizer)

    masked_ids = token_ids.copy()
    for idx in mask_indices:
        masked_ids[idx] = tokenizer.mask_token_id

    return {
        "input_ids": torch.tensor(masked_ids, device=input_ids.device),
        "attention_mask": attention_mask.clone()
    }

def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples_2 = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples_2

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

def get_logrank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
    ranks, timesteps = matches[:, -1], matches[:, -2]
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  
    ranks = torch.log(ranks)
    return -ranks.mean().item()

def get_score(logits_ref, logits_score, labels, source_tokens, perturbed_tokens, basemodel):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples_2 = get_samples(logits_ref, labels)

    log_likelihood_x = get_likelihood(logits_score, labels)
    log_rank_x = get_logrank(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples_2)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)

    valuesPPL = PPL_scorer.score(texts_x=perturbed_tokens, tokenized_y=source_tokens, batch_size=10)
    mean_values_PPL = np.mean(valuesPPL)

    
    if basemodel == 'PPL_Fast':
        if 'gemini' in args.dataset_file and 'pubmed' in args.dataset_file:
            output_score = (((log_likelihood_x.squeeze(-1).item()  - miu_tilde.item())/ (sigma_tilde.item()))+2) * math.pow(math.e, -mean_values_PPL)
        else:
            output_score = ((log_likelihood_x.squeeze(-1).item() - miu_tilde.item()) / (sigma_tilde.item())) * math.pow(math.e, -mean_values_PPL)
    elif basemodel == 'PPL_lrr':
        if 'gpt-4' in args.dataset_file and 'pubmed' in args.dataset_file:
            output_score = (log_likelihood_x.squeeze(-1).item() / log_rank_x) * math.pow(math.e, mean_values_PPL)
        elif 'gpt-3.5-turbo' in args.dataset_file and 'pubmed' in args.dataset_file:
            output_score = (log_likelihood_x.squeeze(-1).item() / log_rank_x) * math.pow(math.e, mean_values_PPL)
        else:
            output_score = (log_likelihood_x.squeeze(-1).item() / log_rank_x) * math.pow(math.e, -mean_values_PPL)
    elif basemodel == 'PPL_likelihood':
        if 'gpt-4' in args.dataset_file and 'pubmed' in args.dataset_file:
            output_score = log_likelihood_x.squeeze(-1).item() * math.pow(math.e, -mean_values_PPL)
        elif 'gpt-3.5-turbo' in args.dataset_file and 'pubmed' in args.dataset_file:
            output_score = log_likelihood_x.squeeze(-1).item() * math.pow(math.e, -mean_values_PPL)
        else:
            output_score = log_likelihood_x.squeeze(-1).item() * math.pow(math.e, mean_values_PPL)
    elif basemodel == 'PPL_logrank':
        if 'gpt-4' in args.dataset_file and 'pubmed' in args.dataset_file:
            output_score = log_rank_x * math.pow(math.e, -mean_values_PPL)
        elif 'gpt-3.5-turbo' in args.dataset_file and 'pubmed' in args.dataset_file:
            output_score = log_rank_x * math.pow(math.e, -mean_values_PPL)
        else:
            output_score = log_rank_x * math.pow(math.e, mean_values_PPL)
    elif basemodel == 'PPL':
        if 'gpt-4' in args.dataset_file and 'pubmed' in args.dataset_file:
            output_score = mean_values_PPL
        elif 'gpt-3.5-turbo' in args.dataset_file and 'pubmed' in args.dataset_file:
            output_score = mean_values_PPL
        else:
            output_score = -mean_values_PPL

    else:
        output_score = log_likelihood_x.squeeze(-1).item()

    return output_score



def experiment():
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(
            args.reference_model_name, args.dataset, args.cache_dir
        )
        reference_model = load_model(
            args.reference_model_name, args.device, args.cache_dir
        )
        reference_model.eval()

    data = load_data(args.dataset_file)
    n_originals = len(data["original"])
    start_time = time.time()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = []

    perturbed_original_tokens = []
    perturbed_sampled_tokens = []
    original_tokens = []
    sampled_tokens = []

    for text in data["original"]:
        tok = bart_tokenizer(text, return_tensors="pt", add_special_tokens=True)
        tok = {k: v.squeeze(0) for k, v in tok.items()}
        original_tokens.append(tok)

        variants = [
            perturb_token_ids(tok["input_ids"], tok["attention_mask"], bart_tokenizer, pct)
            for _ in range(n)
        ]
        perturbed_original_tokens.append(variants)

    for text in data["sampled"]:
        tok = bart_tokenizer(text, return_tensors="pt", add_special_tokens=True)
        tok = {k: v.squeeze(0) for k, v in tok.items()}
        sampled_tokens.append(tok)

        variants = [
            perturb_token_ids(tok["input_ids"], tok["attention_mask"], bart_tokenizer, pct)
            for _ in range(n)
        ]
        perturbed_sampled_tokens.append(variants)

    for idx in tqdm.tqdm(range(n_originals), desc="computing"):

        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]

        tokenized = scoring_tokenizer(
            original_text,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False
        ).to(args.device)

        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]

            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                ref_tok = reference_tokenizer(
                    original_text,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False
                ).to(args.device)
                assert torch.all(ref_tok.input_ids[:, 1:] == labels)
                logits_ref = reference_model(**ref_tok).logits[:, :-1]

        original_crit = get_score(
            logits_ref,
            logits_score,
            labels,
            original_tokens[idx],                 
            perturbed_original_tokens[idx],       
            args.basemodel
        )

        tokenized = scoring_tokenizer(
            sampled_text,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False
        ).to(args.device)

        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]

            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                ref_tok = reference_tokenizer(
                    sampled_text,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False
                ).to(args.device)
                assert torch.all(ref_tok.input_ids[:, 1:] == labels)
                logits_ref = reference_model(**ref_tok).logits[:, :-1]

        sampled_crit = get_score(
            logits_ref,
            logits_score,
            labels,
            sampled_tokens[idx],
            perturbed_sampled_tokens[idx],
            args.basemodel
        )

        results.append({
            "original": original_text,
            "original_crit": original_crit,
            "sampled": sampled_text,
            "sampled_crit": sampled_crit
        })

    predictions = {
        "real": [x["original_crit"] for x in results],
        "samples": [x["sampled_crit"] for x in results],
    }

    fpr, tpr, roc_auc = get_roc_metrics(predictions["real"], predictions["samples"])
    p, r, pr_auc = get_precision_recall_metrics(predictions["real"], predictions["samples"])

    end_time = time.time()
    total_time = end_time - start_time

    name = args.basemodel
    n_samples = len(results)
    results_file = f'{args.output_file}.{name}.json'

    results_json = {
        'name': f'{name}_threshold',
        'info': {'n_samples': n_samples},
        'predictions': predictions,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        },
        'loss': 1 - pr_auc,
        'PR_AUC': pr_auc,
        'Time': total_time
    }

    with open(results_file, 'w') as fout:
        json.dump(results_json, fout)
        print(f'Results written into {results_file}')


    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"cost of time：{total_time:4f}")

    return roc_auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="data_test/pubmed")
    parser.add_argument('--dataset', type=str, default="pubmed")
    parser.add_argument('--dataset_file', type=str, default="data_test/pubmed_gpt-4")
    parser.add_argument('--reference_model_name', type=str, default="gpt2")
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--basemodel', type=str, default="standalone")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cpu") #cuda:0
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    base_tokenizer = scoring_tokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

    PPL_scorer =  BARTScorerSymmetricLogPPL(device=DEVICE, checkpoint='facebook/bart-base')

    experiment()

