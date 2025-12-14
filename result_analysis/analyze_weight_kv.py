import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from result_analysis.load_model import load_model
from result_analysis.catch_data import catch_tensors, TargetSpec
from eval_utils.rotation_utils import QKRotationWrapper
from utils.quant_utils import ActQuantWrapper
from utils import data_utils

def custom_process_fn(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    elif isinstance(data, tuple):
        return tuple(custom_process_fn(x) for x in data)
    elif isinstance(data, list):
        return [custom_process_fn(x) for x in data]
    return data

def analyze_distribution():
    # Load model
    print("Loading model...")
    # Arguments from analyze_llama.ipynb
    model, tokenizer = load_model(
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
        rotate=True
    )
    model.eval()

    # Prepare data
    print("Preparing data...")
    text = "Hello, this is a test sentence to analyze the distribution of weights and kv cache. " * 10
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    # Define specs
    specs = []
    
    # Spec for Weights (layer 0 only)
    # Note: We filter for layer 0 to save memory and time. Remove "layers.0." check to analyze all layers.
    def is_layer0_weight(name, module):
        return isinstance(module, ActQuantWrapper) and "layers.0." in name

    specs.append(TargetSpec(
        modules=is_layer0_weight,
        static_targets=["weight"],
        dynamic_targets=[]
    ))

    # Spec for Quantized K (layer 0 only)
    def is_layer0_k(name, module):
        return isinstance(module, QKRotationWrapper) and "layers.0." in name

    specs.append(TargetSpec(
        modules=is_layer0_k,
        dynamic_targets=["output"],
        static_targets=[]
    ))

    # Spec for Quantized V (v_proj output, layer 0 only)
    def is_v_proj_layer0(name, module):
        return isinstance(module, ActQuantWrapper) and "v_proj" in name and "layers.0." in name

    specs.append(TargetSpec(
        modules=is_v_proj_layer0,
        dynamic_targets=["output"],
        static_targets=[]
    ))

    # Catch tensors
    print("Catching tensors...")
    handles, data = catch_tensors(model, specs, process_fn=custom_process_fn)

    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        model(input_ids)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Analyze data
    print("Analyzing data...")
    
    weights = []
    k_cache = []
    v_cache = []

    for name, content in data.items():
        # Weights
        if "weight" in content:
            w = content["weight"]
            weights.append(w.flatten().float().numpy())
        
        # Quantized K (output of QKRotationWrapper)
        # QKRotationWrapper output is (q, k)
        if "output" in content and "qk_rotation_wrapper" in name:
             # content["output"] is a list of outputs (one per forward pass)
             for out in content["output"]:
                 # out is (q, k)
                 k = out[1]
                 k_cache.append(k.flatten().float().numpy())

        # Quantized V (output of v_proj)
        if "output" in content and "v_proj" in name:
            for out in content["output"]:
                v = out
                v_cache.append(v.flatten().float().numpy())

    # Plot distributions
    print("Plotting distributions...")
    os.makedirs("result_analysis/plots", exist_ok=True)

    if weights:
        print(f"Plotting weights from {len(weights)} layers")
        all_weights = np.concatenate(weights)
        plt.figure(figsize=(10, 6))
        plt.hist(all_weights, bins=100, log=True)
        plt.title("Weight Distribution")
        plt.xlabel("Value")
        plt.ylabel("Count (Log Scale)")
        plt.savefig("result_analysis/plots/weight_distribution.png")
        plt.close()
        print("Saved weight_distribution.png")
    else:
        print("No weights captured!")

    if k_cache:
        print(f"Plotting K cache from {len(k_cache)} layers")
        all_k = np.concatenate(k_cache)
        plt.figure(figsize=(10, 6))
        plt.hist(all_k, bins=100, log=True)
        plt.title("Quantized K Cache Distribution")
        plt.xlabel("Value")
        plt.ylabel("Count (Log Scale)")
        plt.savefig("result_analysis/plots/k_cache_distribution.png")
        plt.close()
        print("Saved k_cache_distribution.png")
    else:
        print("No K cache captured!")

    if v_cache:
        print(f"Plotting V cache from {len(v_cache)} layers")
        all_v = np.concatenate(v_cache)
        plt.figure(figsize=(10, 6))
        plt.hist(all_v, bins=100, log=True)
        plt.title("Quantized V Cache Distribution")
        plt.xlabel("Value")
        plt.ylabel("Count (Log Scale)")
        plt.savefig("result_analysis/plots/v_cache_distribution.png")
        plt.close()
        print("Saved v_cache_distribution.png")
    else:
        print("No V cache captured!")

if __name__ == "__main__":
    analyze_distribution()
