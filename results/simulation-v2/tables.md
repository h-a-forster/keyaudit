# Simulation study: summary

Profile `default`, seed 20261024. Means over replicate panels with 95% intervals for the mean (normal approximation across replicates). Rates in percent.

| Scenario | Reps | Median / max accuracy | tau | True rate | E_model (all) | Region share | True in region | E_model in region | Calibrated (25% labels) |
|---|---|---|---|---|---|---|---|---|---|
| design | 10 | 0.53 / 0.83 | 1.02 | 2.87 [2.60, 3.14] | 2.45 [2.20, 2.69] | 86 [85, 87] | 2.44 [2.19, 2.68] | 2.38 [2.14, 2.61] | 2.86 [2.58, 3.14] |
| weak | 10 | 0.35 / 0.70 | 1.02 | 2.91 [2.77, 3.05] | 2.43 [2.30, 2.56] | 68 [66, 71] | 2.33 [2.18, 2.49] | 2.29 [2.15, 2.42] | 2.98 [2.75, 3.21] |
| families | 10 | 0.54 / 0.77 | 1.27 | 3.04 [2.89, 3.19] | 1.93 [1.79, 2.07] | 86 [84, 88] | 1.83 [1.66, 1.99] | 1.76 [1.61, 1.92] | 3.06 [2.89, 3.22] |
| tricks | 10 | 0.52 / 0.79 | 1.02 | 3.02 [2.80, 3.25] | 5.05 [4.70, 5.40] | 86 [84, 88] | 2.70 [2.43, 2.98] | 4.93 [4.53, 5.32] | 2.73 [2.26, 3.20] |

| Scenario | AP keymix | AP in region | AP top_disagreement | AP item_total | AP scalability | AP sign_lrt | AP one_coin_ds | Precision @ q<=0.10 | Recall @ q<=0.10 | Model FDR | Calibrated CI coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| design | 0.947 | 0.999 | 0.866 | 0.541 | 0.540 | 0.563 | 0.410 | 0.970 | 0.822 | 0.096 | 10/10 |
| weak | 0.866 | 0.995 | 0.596 | 0.372 | 0.371 | 0.350 | 0.254 | 0.949 | 0.662 | 0.096 | 10/10 |
| families | 0.914 | 0.998 | 0.831 | 0.473 | 0.471 | 0.479 | 0.523 | 1.000 | 0.569 | 0.095 | 10/10 |
| tricks | 0.550 | 0.577 | 0.517 | 0.366 | 0.365 | 0.376 | 0.320 | 0.508 | 0.855 | 0.098 | 9/10 |

Trick items (correct key, A1-A2 violated): flagged at q <= 0.10 in 86.16 [82.48, 89.83]% of cases, versus 85.54 [81.40, 89.67]% of true mis-keys and 0.21 [0.17, 0.24]% of other clean items.

Item bootstrap (panel fixed) coverage of nominal 95% intervals (B = 40): overall rate 13/30 (CP 95% 0.25-0.63); rate inside the competence region 30/30 (CP 95% 0.88-1.00, lower 99.9% bound 0.79). Median width of the overall interval 1.80 points; the point estimate lies inside its own interval in 30/30 panels.

Region-bounded interval for the overall rate (in-region interval plus the worst case outside the region): 30/30 (CP 95% 0.88-1.00), median width 16.58 points.
