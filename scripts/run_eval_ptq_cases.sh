# Define a function to run the torchrun command
run() {
    local input_model=$1
    local w_bits=$2
    local a_bits=$3
    local k_bits=$4
    local out_name=$5
    local zp_int8=$6
    local signed_kv=$7
    local zp_clamp=$8
    local scale_no_up=$9

    shift 9 # Shift the first 8 arguments
    local extra_args="$@" # Capture additional arguments

    # Process input_model to replace / with _
    local sanitized_model=$(echo "$input_model" | sed 's/.\//_/g')

    # Process extra_args to create a log-friendly string
    local extra_args_log=""
    if [[ -n "$extra_args" ]]; then
        extra_args_log=$(echo "$extra_args" | sed 's/--//g' | tr ' ' '_')
    fi

    config=${out_name##*/}

    echo "Running evaluation with the following parameters: ${input_model}, w_bits=${w_bits}, a_bits=${a_bits}, k_bits=${k_bits}, extra_args=${extra_args}"
    echo "Output will be saved to: saved_models/llama2-7b/${config}.pt"

    export CUDA_VISIBLE_DEVICES=3
    export ZP_INT8="$zp_int8"
    export SIGNED_KV="$signed_kv"
    export ZP_CLAMP="$zp_clamp"
    export SCALE_NO_UPCAST="$scale_no_up"
    torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
    --input_model "$input_model" \
    --do_train False \
    --do_eval True \
    --per_device_eval_batch_size 4 \
    --model_max_length 2048 \
    --fp16 True \
    --bf16 False \
    --save_safetensors False \
    --w_bits "$w_bits" \
    --a_bits "$a_bits" \
    --k_bits "$k_bits" \
    --v_bits "$k_bits" \
    --w_clip \
    --a_asym \
    --k_asym \
    --k_groupsize 128 \
    --v_groupsize 128 \
    --rotate \
    --optimized_rotation_path ${out_name}/R.bin \
    --save_qmodel_path "saved_models/llama3.2-3b/${config}.pt" \
    $extra_args 2>&1 | tee "logs/ptq_${sanitized_model}_w${w_bits}_a${a_bits}_k${k_bits}_${extra_args_log}_zpint8${zp_int8}_signedkv${signed_kv}_zpclamp${zp_clamp}_scalenoup${scale_no_up}.log"

}

# --load_qmodel_path "saved_models/llama3.2-3b/${config}.pt" \
# --load_qmodel_path "saved_models/qllama3.1-8b-16-4-4-128.pt" \

# Run cases
# run ./models/llama2-7b 4 16 4 rotation_llama-2-7b/a16w4kv4-vasym 0 0 1 0 --v_asym # default case
# run ./models/llama3.1-8b 4 16 4 rotation_llama-3.1-8b/a16w4kv4-vasym 0 0 1 0 --v_asym # default case
run ./models/llama3.2-3b 4 16 4 rotation_llama-3.2-3b/a16w4kv4-vasym 0 0 1 0 --v_asym # default case

# run ./models/llama2-7b 4 16 4 rotation_llama-2-7b/a16w4kv4-vsym 0 0 0 0
# run ./models/llama2-7b 4 16 4 rotation_llama-2-7b/a16w4kv4-vsym 0 0 1 0
# run ./models/llama2-7b 4 16 4 rotation_llama-2-7b/a16w4kv4-vsym 0 1 0 0
# run ./models/llama2-7b 4 16 4 rotation_llama-2-7b/a16w4kv4-vsym 0 1 1 0
# run ./models/llama2-7b 4 16 4 rotation_llama-2-7b/a16w4kv4-vsym 1 0 0 0 
# run ./models/llama2-7b 4 16 4 rotation_llama-2-7b/a16w4kv4-vsym 1 0 1 0
# run ./models/llama2-7b 4 16 4 rotation_llama-2-7b/a16w4kv4-vsym 1 1 0 0
# run ./models/llama2-7b 4 16 4 rotation_llama-2-7b/a16w4kv4-vsym 1 1 1 0

# run ./models/llama2-7b 4 16 4 rotation_llama-2-7b/a16w4kv4-vsym 1 1 0 1