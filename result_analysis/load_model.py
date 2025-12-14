import sys
import os
import torch
import transformers
from transformers import LlamaTokenizerFast
from eval_utils.modeling_llama import LlamaForCausalLM
from eval_utils.main import ptq_model
from utils.process_args import process_args_ptq
from utils import utils
from unittest.mock import patch
from logging import Logger

log: Logger = utils.get_logger("spinquant")

def load_model(
    input_model="models/llama2-7b",
    load_qmodel_path="saved_models/llama2-7b/a16w4kv4-vasym.pt",
    optimized_rotation_path="rotation_llama-2-7b/a16w4kv4-vsym/R.bin",
    w_bits=4,
    a_bits=16,
    k_bits=4,
    v_bits=4,
    k_groupsize=128,
    v_groupsize=128,
    w_clip=True,
    a_asym=True,
    k_asym=True,
    v_asym=True,
    rotate=True,
    per_device_eval_batch_size=4,
    model_max_length=2048,
    device="cuda",
    **kwargs
):
    """
    Load the model with the specified parameters.
    
    Args:
        input_model (str): Path to the input model.
        load_qmodel_path (str): Path to the quantized model checkpoint.
        optimized_rotation_path (str): Path to the optimized rotation checkpoint.
        w_bits (int): Number of bits for weights.
        a_bits (int): Number of bits for activations.
        k_bits (int): Number of bits for K-cache.
        v_bits (int): Number of bits for V-cache.
        k_groupsize (int): Group size for K-cache.
        v_groupsize (int): Group size for V-cache.
        w_clip (bool): Whether to clip weights.
        a_asym (bool): Whether to use asymmetric quantization for activations.
        k_asym (bool): Whether to use asymmetric quantization for K-cache.
        v_asym (bool): Whether to use asymmetric quantization for V-cache.
        rotate (bool): Whether to rotate the model.
        per_device_eval_batch_size (int): Batch size for evaluation.
        model_max_length (int): Maximum sequence length.
        device (str): Device to load the model on.
        **kwargs: Additional arguments to pass to the argument parser.
    
    Returns:
        tuple: (model, tokenizer)
    """
    
    # Construct sys.argv
    args = [
        "python",
        "--input_model", input_model,
        "--do_train", "False",
        "--do_eval", "True",
        "--per_device_eval_batch_size", str(per_device_eval_batch_size),
        "--model_max_length", str(model_max_length),
        "--fp16", "True",
        "--bf16", "False",
        "--save_safetensors", "False",
        "--w_bits", str(w_bits),
        "--a_bits", str(a_bits),
        "--k_bits", str(k_bits),
        "--v_bits", str(v_bits),
        "--k_groupsize", str(k_groupsize),
        "--v_groupsize", str(v_groupsize),
    ]

    if w_clip: args.append("--w_clip")
    if a_asym: args.append("--a_asym")
    if k_asym: args.append("--k_asym")
    if v_asym: args.append("--v_asym")
    if rotate: args.append("--rotate")
    
    if load_qmodel_path:
        args.extend(["--load_qmodel_path", load_qmodel_path])
    
    if optimized_rotation_path:
        args.extend(["--optimized_rotation_path", optimized_rotation_path])

    # Add any additional kwargs as arguments
    for key, value in kwargs.items():
        key = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value)])

    log.info(f"Loading model with args: {args}")

    # Patch sys.argv and call process_args_ptq
    with patch.object(sys, 'argv', args):
        model_args, training_args, ptq_args = process_args_ptq()

    log.info("------- ARGS ----------")
    log.info("-----model args-----")
    log.info(model_args)
    log.info("------train args-------")
    log.info(training_args)
    log.info("-------- ptq args ---------")
    log.info(ptq_args)
    log.info("------- ARGS END ----------")

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token, attn_implementation="eager"
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
        
    model.to(device)

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
    
    return model, tokenizer
