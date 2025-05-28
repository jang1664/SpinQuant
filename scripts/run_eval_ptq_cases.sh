# Define a function to run the torchrun command
run_torchrun() {
    local input_model=$1
    local w_bits=$2
    local a_bits=$3
    local k_bits=$4
    shift 4 # Shift the first 4 arguments
    local extra_args="$@" # Capture additional arguments

    # Process input_model to replace / with _
    local sanitized_model=$(echo "$input_model" | sed 's/.\//_/g')

    # Process extra_args to create a log-friendly string
    local extra_args_log=""
    if [[ -n "$extra_args" ]]; then
        extra_args_log=$(echo "$extra_args" | sed 's/--//g' | tr ' ' '_')
    fi

    export CUDA_VISIBLE_DEVICES=0
    echo "Running evaluation with the following parameters: ${input_model}, w_bits=${w_bits}, a_bits=${a_bits}, k_bits=${k_bits}, extra_args=${extra_args}"

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
    $extra_args > "logs/${sanitized_model}_w${w_bits}_a${a_bits}_k${k_bits}_${extra_args_log}.log" 2>&1
}

# Run cases
run_torchrun ./models/llama2-7b 4 16 4 --v_asym 
run_torchrun ./models/llama2-7b 4 16 4 --v_asym --zp_int8
run_torchrun ./models/llama2-7b 4 16 4 --v_asym --signed_kv
run_torchrun ./models/llama2-7b 4 16 4 --v_asym --no_zp_clamp
run_torchrun ./models/llama2-7b 4 16 4 --v_asym --zp_int8 --signed_kv
run_torchrun ./models/llama2-7b 4 16 4 --v_asym --zp_int8 --no_zp_clamp
run_torchrun ./models/llama2-7b 4 16 4 --v_asym --signed_kv --no_zp_clamp
run_torchrun ./models/llama2-7b 4 16 4 --v_asym --zp_int8 --signed_kv --no_zp_clamp

run_torchrun ./models/llama2-7b 4 16 4
run_torchrun ./models/llama2-7b 4 16 4 --zp_int8
run_torchrun ./models/llama2-7b 4 16 4 --signed_kv
run_torchrun ./models/llama2-7b 4 16 4 --no_zp_clamp
run_torchrun ./models/llama2-7b 4 16 4 --zp_int8 --signed_kv
run_torchrun ./models/llama2-7b 4 16 4 --zp_int8 --no_zp_clamp
run_torchrun ./models/llama2-7b 4 16 4 --signed_kv --no_zp_clamp
run_torchrun ./models/llama2-7b 4 16 4 --zp_int8 --signed_kv --no_zp_clamp