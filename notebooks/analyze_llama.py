#!/usr/bin/env python
# coding: utf-8


import os
import sys

# SPINQUANT_HOME = os.environ.get("SPINQUANT_HOME", "/opt/spinquant")
SPINQUANT_HOME = '/home/donghweeson/workspace/SpinQuant' # /home/donghweeson/workspace/SpinQuant
sys.path.append('/home/donghweeson/workspace/SpinQuant')

os.chdir(SPINQUANT_HOME)
print("Changed working directory to:", os.getcwd())

sys.argv = [
  "python",
  "--input_model", "models/llama2-7b",
  "--do_train", "False",
  "--do_eval", "True",
  "--per_device_eval_batch_size", "4",
  "--model_max_length", "2048",
  "--fp16", "True",
  "--bf16", "False",
  "--save_safetensors", "False",
  "--w_bits", "4",
  "--a_bits", "16",
  "--k_bits", "4",
  "--v_bits", "4",
  "--w_clip",
  "--a_asym",
  "--k_asym",
  "--v_asym",
  "--rotate",
  "--k_groupsize", "128",
  "--v_groupsize", "128",
  "--load_qmodel_path", "saved_models/llama2-7b/a16w4kv4-vasym.pt",
  "--optimized_rotation_path", "rotation_llama-2-7b/a16w4kv4-vsym/R.bin"
]


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import datetime
from logging import Logger

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq

log: Logger = utils.get_logger("spinquant")

import evaluate
from lm_eval import evaluator
from lm_eval.utils import make_table

from utils.quant_utils import find_qlayers, ActQuantWrapper
from functools import partial
import pickle

from utils.profile import (
  measure, profile, get_profiler, 
  get_profiled_df, plot_profiled_df,
  run_profile
)
import pstats
import importlib
import importlib.util
use_flash_attn2 = importlib.util.find_spec("flash_attn") is not None

attn_impl = "flash_attention_2" if use_flash_attn2 else "sdpa"

import pandas as pd
pd.set_option('display.max_colwidth', 100)

from matplotlib import pyplot as plt

task_names = ['hellaswag', 'arc_easy','arc_challenge', 'winogrande', 'openbookqa', "wikitext"]
# task_names = ['openbookqa']
# task_names = ['arc_easy']

CUDA_DEVICES = list(map(str.strip, os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")))
FIRST_GPU_ID = int(CUDA_DEVICES[0])
GPU_ID = 0


# dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
model_args, training_args, ptq_args = process_args_ptq()
print("------- ARGS ----------")
print("-----model args-----")
print(model_args)
print("------train args-------")
print(training_args)
print("-------- ptq args ---------")
print(ptq_args)
print("------- ARGS END ----------")

config = transformers.AutoConfig.from_pretrained(
    model_args.input_model, token=model_args.access_token, attn_implementation="sdpa"
)
# Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
process_word_embeddings = False
if config.tie_word_embeddings:
    config.tie_word_embeddings = False
    process_word_embeddings = True
dtype = torch.bfloat16 if training_args.bf16 else torch.float16
model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_args.input_model,
    config=config,
    torch_dtype=dtype,
    token=model_args.access_token,
)
if process_word_embeddings:
    model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
model.cuda()

model = ptq_model(ptq_args, model, model_args)

model.seqlen = training_args.model_max_length
log.info("Model PTQ completed {}".format(model))
log.info("Start to load tokenizer...")
tokenizer = LlamaTokenizerFast.from_pretrained(
    pretrained_model_name_or_path=model_args.input_model,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="right",
    use_fast=True,
    add_eos_token=False,
    add_bos_token=False,
    token=model_args.access_token,
)
log.info("Complete tokenizer loading...")

past_seq_len = 0
target_seq_len = 32
batch_size = 2

os.makedirs("./prof", exist_ok=True)

batch_size = [1, 2, 4]
past_seq_len = [8, 16, 32, 64, 128]
seq_len = [8, 16, 32, 64, 128]

for bs in batch_size:
  # prefill
  for sl in seq_len:
    print(f"PREFILL: bs {bs}, sl {sl}")
    run_profile(model, bs, 0, sl, "cuda", f"./prof/bs{bs}_sl{sl}_prf-cuda.png") 

  # generate
  for sl in past_seq_len:
    print(f"GENERATE: bs {bs}, sl {sl}")
    run_profile(model, bs, sl, 1, "cuda", f"./prof/bs{bs}_sl{sl}_gen-cuda.png")

