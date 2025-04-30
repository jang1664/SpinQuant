# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
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
import os

task_names = ['hellaswag', 'arc_easy','arc_challenge', 'winogrande', 'openbookqa', "wikitext"]
# task_names = ['openbookqa']
# task_names = ['arc_easy']

CUDA_DEVICES = list(map(str.strip, os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")))
FIRST_GPU_ID = int(CUDA_DEVICES[0])
print(FIRST_GPU_ID)
GPU_ID = 0

# os.environ["CUDA_VISIBLE_DEVICES"] = str(FIRST_GPU_ID)
# os.environ["LOCAL_RANK"] = str(FIRST_GPU_ID)

def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    print("------- ARGS ----------")
    print("-----model args-----")
    print(model_args)
    print("------train args-------")
    print(training_args)
    print("-------- ptq args ---------")
    print(ptq_args)
    print("------- ARGS END ----------")

    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
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

    # for l, layer in enumerate(model.model.layers):
    #     layer.self_attn.q_proj.quantizer.register_forward_hook(partial(forward_hook_act_quant, name=name))
    #     layer.self_attn.q_proj.register_forward_hook(forward_hook_weight_quant)
    
    model.seqlen = training_args.model_max_length
    if local_rank == 0:
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
    model.config.use_cache = False

    try:
      results = evaluator.simple_evaluate(
          model="hf",
          model_args={"pretrained" : model.to("cuda"),
                      "tokenizer" : tokenizer},
          tasks=task_names,
          num_fewshot=0,
          batch_size="auto",
          device="cuda",
          limit=128
      )
      print(make_table(results))
    except Exception as e:
      print("Error in evaluation")
      print(e)

    # testloader = data_utils.get_wikitext2(
    #     seed=ptq_args.seed,
    #     seqlen=2048,
    #     tokenizer=tokenizer,
    #     eval_mode=True,
    # )

    # dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    # log.info("wiki2 ppl is: {}".format(dataset_ppl))
    dist.barrier()

if __name__ == "__main__":
    train()
