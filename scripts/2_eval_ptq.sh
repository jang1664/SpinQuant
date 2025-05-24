# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
export CUDA_VISIBLE_DEVICES=2
torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 True \
--bf16 False \
--save_safetensors False \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_clip \
--a_asym \
--k_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--rotate \
--save_qmodel_path "saved_models/qllama2-7b-16-4-4-128.pt" \
--optimized_rotation_path "rotation_llama-2-7b/a16w4kv4_fp16/R.bin" \

# --v_asym \
# --save_qmodel_path "saved_models/qllama2-7b-4-8-8-128-a16.pt" \
# --save_qmodel_path "saved_models/qllama2-7b-16-4-4-128-fp16.pt" \
# --optimized_rotation_path "rotation_llama-2-7b/a16w4kv4_fp16/R.bin" \

# --save_qmodel_path "saved_models/qllama3.1-8b-16-4-4-128.pt" \
# --optimized_rotation_path "rotation_llama-3.1-8b/a16w4kv4/R.bin" \
# --load_qmodel_path "saved_models/qllama3.1-8b-16-4-4-128.pt" \

# --save_qmodel_path "saved_models/qllama2-7b-16-4-4-128-fp16.pt" \
# --optimized_rotation_path "rotation_llama-2-7b/a16w4kv4_fp16/R.bin" \
# --load_qmodel_path "saved_models/qllama2-7b-16-4-4-128-fp16.pt" \
