# Online Hadamard Power-of-Two Model Survey

Date: 2026-07-23

## Purpose and criterion

This survey reads the official Hugging Face `config.json` files for representative
open-weight model families. Base and Instruct checkpoints with identical model
shapes are grouped together.

The target criterion is deliberately limited to the Hadamard transforms that
remain online at inference:

1. **R3:** Q/K Hadamard rotation over each attention head.
2. **R4:** MLP Hadamard rotation over the input of `down_proj`.

R1 and R2 do not affect the final eligibility result because they are fused into
weights before inference.

For a dense model, R4 is direct power-of-two when `intermediate_size` is a power
of two. For an MoE model, every active expert dimension, including
`moe_intermediate_size` and `shared_expert_intermediate_size`, must be a power of
two. For nested multimodal configs, only the language model's `text_config` is
classified.

`head_dim` is taken directly from the config when present. Otherwise it is
calculated as `hidden_size / num_attention_heads`.

## What is fused and what remains online

| Rotation | Transform axis | Weight fused before inference? | Runtime online Hadamard? | Power-of-two relevant to this survey? |
| --- | --- | --- | --- | --- |
| R1 | `hidden_size` | **Yes.** Fused into embeddings, Q/K/V, O, MLP, and LM-head weights. | No | No |
| R2 | Per-head V/O `head_dim` | **Yes.** Fused into `v_proj` and `o_proj`. | No | No |
| R3 | Per-head Q/K `head_dim` after RoPE | **No.** The same orthogonal transform is applied to Q and K, preserving their dot product. | **Yes**, when K-cache quantization is enabled (`k_bits < 16`). | **Yes** |
| R4 | MLP `intermediate_size` | **Partially.** The inverse transform is fused into `down_proj.weight`. | **Yes.** The matching forward transform remains on the `down_proj` activation. | **Yes** |

The R4 `weight fused` entry does **not** mean that R4 is fully offline. It is a
paired transform: the inverse is stored in the weight, while the activation-side
Hadamard still runs online.

## Models whose R4 dimension is a power of two

The table includes all models found in the surveyed families whose R4 MLP
dimension is already a power of two. The final column is the target result:
both online Hadamard transforms can use a direct power-of-two kernel.

| Official model / architecture | R3 dimension | R3 weight fused? | R3 online? | R3 power-of-two? | R4 dimension | R4 inverse fused into weight? | R4 online? | R4 power-of-two? | Both online transforms power-of-two? |
| --- | ---: | --- | --- | --- | ---: | --- | --- | --- | --- |
| [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json) | 64 | No | Yes | Yes | 8192 | Yes | Yes | Yes | **Yes** |
| [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json) | 128 | No | Yes | Yes | 8192 | Yes | Yes | Yes | **Yes** |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base/blob/main/config.json) text backbone | 256; linear-attention subdimensions 128 | No | Architecture support required | Yes | expert 512; shared expert 512 | Yes, after architecture support | Yes, after architecture support | Yes | **Yes, geometrically** |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/main/config.json) text backbone | 256; linear-attention subdimensions 128 | No | Architecture support required | Yes | expert 1024; shared expert 1024 | Yes, after architecture support | Yes, after architecture support | Yes | **Yes, geometrically** |
| [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json) text backbone | 256; linear-attention subdimensions 128 | No | Architecture support required | Yes | expert 1024; shared expert 1024 | Yes, after architecture support | Yes, after architecture support | Yes | **Yes, geometrically** |
| [Mistral-Small-24B](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501/blob/main/config.json) | 128 | No | Architecture support required | Yes | 32768 | Yes, after architecture support | Yes, after architecture support | Yes | **Yes, geometrically** |
| [Mixtral-8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/blob/main/config.json) | 128 | No | Architecture support required | Yes | expert 16384 | Yes, after architecture support | Yes, after architecture support | Yes | **Yes, geometrically** |
| [Ministral-3-14B](https://huggingface.co/mistralai/Ministral-3-14B-Base-2512/blob/main/config.json) text backbone | 128 | No | Architecture support required | Yes | 16384 | Yes, after architecture support | Yes, after architecture support | Yes | **Yes, geometrically** |
| [Gemma-2B / Gemma-1.1-2B](https://huggingface.co/google/gemma-2b/blob/main/config.json) | 256 | No | Architecture support required | Yes | 16384 | Yes, after architecture support | Yes, after architecture support | Yes | **Yes, geometrically** |
| [Gemma-3-270M](https://huggingface.co/google/gemma-3-270m/blob/main/config.json) | 256 | No | Architecture support required | Yes | 2048 | Yes, after architecture support | Yes, after architecture support | Yes | **Yes, geometrically** |
| [Phi-3 Mini / Phi-3.5 Mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/config.json) | 96 | No | Architecture support required | **No** | 8192 | Yes, after architecture support | Yes, after architecture support | Yes | **No: R3 is 96** |
| [Phi-4 Mini](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json) | 128 | No | Architecture support required | Yes | 8192 | Yes, after architecture support | Yes, after architecture support | Yes | **Yes, geometrically** |

`Geometrically` means that the relevant config dimensions are powers of two. It
does not mean that the current SpinQuant Llama-specific monkeypatches and model
wrappers already support that architecture.

## Models that meet the online-only target

The following models have power-of-two dimensions for both R3 and R4:

- `meta-llama/Llama-3.2-1B`
- `meta-llama/Llama-3.2-3B`
- `Qwen/Qwen3.5-35B-A3B` text backbone
- `Qwen/Qwen3.5-122B-A10B` text backbone
- `Qwen/Qwen3.5-397B-A17B` text backbone
- `mistralai/Mistral-Small-24B`
- `mistralai/Mixtral-8x22B`
- `mistralai/Ministral-3-14B` text backbone
- `google/gemma-2b` and the same-shape Gemma-1.1-2B variants
- `google/gemma-3-270m`
- `microsoft/Phi-4-mini`

Among these, Llama-3.2-1B and Llama-3.2-3B are closest to the current SpinQuant
implementation. Other families require architecture-specific rotation and
wrapper support even though their dimensions satisfy the kernel constraint.

## Models that fail the R4 power-of-two condition

R3 is power-of-two for most standard Llama, Qwen, Mistral, and Gemma attention
heads. Their main blocker is R4.

| Family | Official model sizes and non-power-of-two R4 dimensions |
| --- | --- |
| Llama 2 | 7B: 11008; 13B: 13824; 70B: 28672 |
| Llama 3 | 8B: 14336; 70B: 28672 |
| Llama 3.1 | 8B: 14336; 70B: 28672; 405B: 53248 |
| Llama 3.3 | 70B: 28672 |
| Qwen2.5 | 0.5B: 4864; 1.5B: 8960; 3B: 11008; 7B: 18944; 14B: 13824; 32B: 27648; 72B: 29568 |
| Qwen3 dense | 0.6B: 3072; 1.7B: 6144; 4B: 9728; 8B: 12288; 14B: 17408; 32B: 25600 |
| Qwen3 MoE | 30B-A3B expert: 768; 235B-A22B expert: 1536 |
| Qwen3.5 dense | 0.8B: 3584; 2B: 6144; 4B: 9216; 9B: 12288; 27B: 17408 |
| Mistral | 7B: 14336; Nemo-12B: 14336; Large-123B: 28672 |
| Mixtral | 8x7B expert: 14336 |
| Ministral 3 | 3B: 9216; 8B: 14336 |
| Gemma | 7B: 24576 |
| Gemma 2 | 2B: 9216; 9B: 14336; 27B: 36864 |
| Gemma 3 | 1B: 6912; 4B: 10240; 12B: 15360; 27B: 21504 |
| Phi | Phi-2: 10240; Phi-3 Small: 14336; Phi-3 Medium: 17920; Phi-3.5 MoE: 6400; Phi-4: 17920 |
| DeepSeek LLM | 7B: 11008; 67B: 22016 |
| DeepSeek V2 | Lite dense/expert: 10944/1408; V2 dense/expert: 12288/1536 |

## R3-specific failures and special cases

| Model family | Q/K dimension | Result |
| --- | ---: | --- |
| Phi-2 | 80 | R3 is not direct power-of-two |
| Phi-3 Mini / Phi-3.5 Mini | 96 | R3 is not direct power-of-two even though R4 is 8192 |
| DeepSeek V2 | `qk_nope_head_dim 128 + qk_rope_head_dim 64 = 192` | Standard Q/K R3 is not direct power-of-two |
| DeepSeek V3 / R1 | `qk_nope_head_dim 128 + qk_rope_head_dim 64 = 192` | Standard Q/K R3 is not direct power-of-two |

DeepSeek V3 and R1 are also mixed on R4: expert MLP size 2048 is a power of
two, but the dense MLP size 18432 is not. They therefore fail the full online
criterion.

## SpinQuant implementation notes

- Current R3 is installed only when `k_bits < 16`.
- Current R3 runs after `apply_rotary_pos_emb`.
- Current R3 applies a Hadamard transform to the last Q and K dimensions at
  runtime; it is not fused into projection weights.
- Current R4 fuses its inverse into `down_proj.weight` and leaves the matching
  activation transform online.
- Current SpinQuant evaluation code derives R3 `head_dim` as
  `hidden_size / num_attention_heads`. Architectures that provide a distinct
  explicit `head_dim`, such as some recent Mistral and multimodal backbones,
  need an implementation update before the geometric result can be used.
- The `factorized` versus `zero_padding` option added in this repository
  currently selects the R4 implementation. R3 still uses the existing direct
  Fast Hadamard path and asserts that its dimension is a power of two.

## Official Hugging Face sources

- [Meta Llama collections](https://huggingface.co/meta-llama/collections)
- [Qwen collections](https://huggingface.co/Qwen/collections)
- [Mistral collections](https://huggingface.co/mistralai/collections)
- [Google Gemma collections](https://huggingface.co/google/collections)
- [Microsoft Phi-4 collection](https://huggingface.co/collections/microsoft/phi-4)
- [DeepSeek collections](https://huggingface.co/deepseek-ai/collections)

