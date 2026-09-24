# AQP8 vs AQP16 matrix-output accuracy

`error_A` is FP16 vs AQP8 W4/KV4; `error_B` is FP16 vs AQP16 W4/KV4. A/Q/P are 8 bit for AQP8 and 16 bit for AQP16.  Positive A−B or A/B above 1 means AQP8 has additional error.

Only Linear, raw QK, and PV matrix outputs are measured.  No logits, generation, or task-score metrics are included.

| Family | Workload | Model | Group | error_A MAE | error_B MAE | MAE A/B | error_A RMSE | error_B RMSE | RMSE A/B |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ppl | wikitext | llama2-7b | Linear | 0.0729243 | 0.0680035 | 1.0724 | 0.109903 | 0.102253 | 1.0748 |
| ppl | wikitext | llama2-7b | QK | 4.39728 | 3.19302 | 1.3772 | 5.85799 | 4.33982 | 1.3498 |
| ppl | wikitext | llama2-7b | PV | 0.021924 | 0.020853 | 1.0514 | 0.0351033 | 0.033475 | 1.0486 |
| zero_shot_mcq | arc_challenge | llama2-7b | Linear | 0.0768204 | 0.076633 | 1.0024 | 0.119888 | 0.121176 | 0.98937 |
| zero_shot_mcq | arc_challenge | llama2-7b | QK | 3.47582 | 3.3855 | 1.0267 | 4.70689 | 4.61752 | 1.0194 |
| zero_shot_mcq | arc_challenge | llama2-7b | PV | 0.0237414 | 0.0235883 | 1.0065 | 0.0382895 | 0.0383829 | 0.99757 |
| ppl | c4_validation | llama2-7b | Linear | 0.0746659 | 0.0726977 | 1.0271 | 0.111332 | 0.108053 | 1.0303 |
| ppl | c4_validation | llama2-7b | QK | 3.81577 | 3.24634 | 1.1754 | 5.04495 | 4.33743 | 1.1631 |
| ppl | c4_validation | llama2-7b | PV | 0.0222281 | 0.0218033 | 1.0195 | 0.0354008 | 0.0347513 | 1.0187 |
| zero_shot_mcq | hellaswag | llama2-7b | Linear | 0.0740941 | 0.0726295 | 1.0202 | 0.1341 | 0.131366 | 1.0208 |
| zero_shot_mcq | hellaswag | llama2-7b | QK | 3.1918 | 3.14577 | 1.0146 | 4.63363 | 4.60391 | 1.0065 |
| zero_shot_mcq | hellaswag | llama2-7b | PV | 0.0220487 | 0.021769 | 1.0129 | 0.0352718 | 0.034607 | 1.0192 |
| long_context_stress | ruler_niah_single_1_2k | llama2-7b | Linear | 0.0881016 | 0.0830334 | 1.061 | 0.139049 | 0.131579 | 1.0568 |
| long_context_stress | ruler_niah_single_1_2k | llama2-7b | QK | 5.73397 | 4.60577 | 1.245 | 8.02653 | 6.75837 | 1.1876 |
| long_context_stress | ruler_niah_single_1_2k | llama2-7b | PV | 0.0285968 | 0.0274781 | 1.0407 | 0.0456699 | 0.044324 | 1.0304 |
| zero_shot_mcq | winogrande | llama2-7b | Linear | 0.0698132 | 0.0685355 | 1.0186 | 0.107391 | 0.10528 | 1.02 |
| zero_shot_mcq | winogrande | llama2-7b | QK | 2.77621 | 2.74431 | 1.0116 | 3.83685 | 3.78876 | 1.0127 |
| zero_shot_mcq | winogrande | llama2-7b | PV | 0.0206949 | 0.020351 | 1.0169 | 0.0322833 | 0.0317954 | 1.0153 |
| ppl | wikitext | llama3.1-8b | Linear | 0.0700814 | 0.0676421 | 1.0361 | 0.112715 | 0.108454 | 1.0393 |
| ppl | wikitext | llama3.1-8b | QK | 7.24873 | 5.27256 | 1.3748 | 9.48518 | 7.2076 | 1.316 |
| ppl | wikitext | llama3.1-8b | PV | 0.0275581 | 0.027115 | 1.0163 | 0.0470304 | 0.0465493 | 1.0103 |
| zero_shot_mcq | arc_challenge | llama3.1-8b | Linear | 0.0869186 | 0.0867743 | 1.0017 | 0.13432 | 0.134341 | 0.99985 |
| zero_shot_mcq | arc_challenge | llama3.1-8b | QK | 5.74321 | 5.54404 | 1.0359 | 7.63966 | 7.45051 | 1.0254 |
| zero_shot_mcq | arc_challenge | llama3.1-8b | PV | 0.0409684 | 0.0412384 | 0.99345 | 0.0653806 | 0.0659417 | 0.99149 |
| ppl | c4_validation | llama3.1-8b | Linear | 0.0788975 | 0.0775666 | 1.0172 | 0.122181 | 0.119681 | 1.0209 |
| ppl | c4_validation | llama3.1-8b | QK | 7.37332 | 5.97854 | 1.2333 | 9.48644 | 7.84639 | 1.209 |
| ppl | c4_validation | llama3.1-8b | PV | 0.0391741 | 0.0391216 | 1.0013 | 0.062847 | 0.0628967 | 0.99921 |
| zero_shot_mcq | hellaswag | llama3.1-8b | Linear | 0.080368 | 0.080705 | 0.99582 | 0.124 | 0.124354 | 0.99716 |
| zero_shot_mcq | hellaswag | llama3.1-8b | QK | 4.60584 | 4.60226 | 1.0008 | 6.13281 | 6.12515 | 1.0013 |
| zero_shot_mcq | hellaswag | llama3.1-8b | PV | 0.0388105 | 0.0394611 | 0.98351 | 0.0617132 | 0.0628265 | 0.98228 |
| long_context_stress | ruler_niah_single_1_2k | llama3.1-8b | Linear | 0.0934009 | 0.0928074 | 1.0064 | 0.158121 | 0.159402 | 0.99196 |
| long_context_stress | ruler_niah_single_1_2k | llama3.1-8b | QK | 9.98984 | 8.41661 | 1.1869 | 13.4078 | 12.0317 | 1.1144 |
| long_context_stress | ruler_niah_single_1_2k | llama3.1-8b | PV | 0.0450628 | 0.0450048 | 1.0013 | 0.0700828 | 0.0704539 | 0.99473 |
| zero_shot_mcq | winogrande | llama3.1-8b | Linear | 0.0847882 | 0.0855125 | 0.99153 | 0.132878 | 0.13419 | 0.99022 |
| zero_shot_mcq | winogrande | llama3.1-8b | QK | 4.41516 | 4.42775 | 0.99716 | 6.0151 | 6.02646 | 0.99812 |
| zero_shot_mcq | winogrande | llama3.1-8b | PV | 0.0393998 | 0.0400892 | 0.9828 | 0.0626513 | 0.0638425 | 0.98134 |
| ppl | wikitext | llama3.2-3b | Linear | 0.0982051 | 0.0978266 | 1.0039 | 0.16534 | 0.16603 | 0.99585 |
| ppl | wikitext | llama3.2-3b | QK | 7.71708 | 6.65914 | 1.1589 | 10.5662 | 9.61217 | 1.0993 |
| ppl | wikitext | llama3.2-3b | PV | 0.0478206 | 0.0490149 | 0.97564 | 0.0857952 | 0.0887998 | 0.96616 |
| zero_shot_mcq | arc_challenge | llama3.2-3b | Linear | 0.16556 | 0.161802 | 1.0232 | 0.257691 | 0.251273 | 1.0255 |
| zero_shot_mcq | arc_challenge | llama3.2-3b | QK | 9.38122 | 8.90099 | 1.054 | 12.7088 | 12.02 | 1.0573 |
| zero_shot_mcq | arc_challenge | llama3.2-3b | PV | 0.100128 | 0.0972436 | 1.0297 | 0.149929 | 0.145489 | 1.0305 |
| ppl | c4_validation | llama3.2-3b | Linear | 0.134995 | 0.127106 | 1.0621 | 0.212919 | 0.199851 | 1.0654 |
| ppl | c4_validation | llama3.2-3b | QK | 8.07724 | 7.43597 | 1.0862 | 10.9716 | 10.2893 | 1.0663 |
| ppl | c4_validation | llama3.2-3b | PV | 0.0794768 | 0.0745398 | 1.0662 | 0.124181 | 0.116478 | 1.0661 |
| zero_shot_mcq | hellaswag | llama3.2-3b | Linear | 0.159833 | 0.149559 | 1.0687 | 0.250586 | 0.233041 | 1.0753 |
| zero_shot_mcq | hellaswag | llama3.2-3b | QK | 7.81215 | 7.10752 | 1.0991 | 10.7512 | 9.71977 | 1.1061 |
| zero_shot_mcq | hellaswag | llama3.2-3b | PV | 0.0964571 | 0.089426 | 1.0786 | 0.146456 | 0.135878 | 1.0779 |
| long_context_stress | ruler_niah_single_1_2k | llama3.2-3b | Linear | 0.131603 | 0.133745 | 0.98399 | 0.210037 | 0.213856 | 0.98214 |
| long_context_stress | ruler_niah_single_1_2k | llama3.2-3b | QK | 9.10125 | 8.9103 | 1.0214 | 12.2622 | 12.0908 | 1.0142 |
| long_context_stress | ruler_niah_single_1_2k | llama3.2-3b | PV | 0.0887612 | 0.092988 | 0.95454 | 0.130519 | 0.136662 | 0.95505 |
| zero_shot_mcq | winogrande | llama3.2-3b | Linear | 0.156549 | 0.150269 | 1.0418 | 0.248576 | 0.238004 | 1.0444 |
| zero_shot_mcq | winogrande | llama3.2-3b | QK | 7.60355 | 7.18211 | 1.0587 | 10.9524 | 10.4146 | 1.0516 |
| zero_shot_mcq | winogrande | llama3.2-3b | PV | 0.0936924 | 0.0877665 | 1.0675 | 0.143514 | 0.134262 | 1.0689 |

## Extended-workload provenance

- Zero-shot rows are fixed representative subsets; each answer choice is a separate prompt-plus-continuation forward request.
- The second PPL row is `allenai/c4` validation because the PG19 loader was unavailable in the experiment environment; its manifest pins the dataset/configuration and source-document count.
- RULER uses the official NVIDIA RULER `niah_single_1` configuration, seed 42, four generated 2K prompts, and the recorded upstream revision. It is an input stress workload only; retrieval correctness is not reported.
