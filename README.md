TMaT
This code is for the paper "MaT: Zero-Shot LLM-Generated Text Detection using Token Masking-Induced Perturbation".

Data
The following folders need to be created for experiments:
./exp 
./exp/data
./exp/results

Models loading
If you want to load models locally, place the files for the bart-base model in the 'facebook' directory.
For experiments with Open-Source LLMs, please download models and create directories in the following format:
gpt-neo-2.7B: 'EleutherAI/gpt-neo-2.7B'
I have used the GPT-neo-2.7B model and BART for my experiments.

Environment
Python Version: 3.11.13
PyTorch Version: 2.9.0+cu128
GPU: NVIDIA RTX-4090 GPU with 24GB memory

Install these libraries after Python and PyTorch installation:
random, os, numpy, datasets, tqdm, argparse, json, scipy, math, jsonlines, matplotlib, transformers

Demo
Please run the following commands from git-bash:
sh main.sh
