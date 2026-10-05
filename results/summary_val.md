# Results on the val split (137 songs)

| Method | HR.5F | HR3F | HR.5F (trim) | HR3F (trim) | PWF | Sf | Label acc | Macro-F1 | Sections/song |
|---|---|---|---|---|---|---|---|---|---|
| CNN + BiGRU (ours) | 0.620 | 0.767 | 0.539 | 0.718 | 0.662 | 0.710 | 0.647 | 0.534 | 11.1 |
| MLP | 0.443 | 0.639 | 0.326 | 0.563 | 0.529 | 0.559 | 0.476 | 0.355 | 12.0 |
| Logistic regression | 0.332 | 0.481 | 0.168 | 0.351 | 0.496 | 0.455 | 0.417 | 0.227 | 9.0 |
| Foote novelty (unsupervised) | 0.423 | 0.662 | 0.304 | 0.592 | 0.464 | 0.000 | 0.357 | 0.052 | 12.3 |
| Majority class | 0.304 | 0.304 | 0.000 | 0.000 | 0.464 | 0.000 | 0.357 | 0.052 | 1.0 |
| MSAF, published with Harmonix* | 0.297 | 0.611 | 0.221 | 0.583 | 0.541 | 0.591 | - | - | - |

HR = boundary hit rate F-measure within ±0.5 s / ±3 s (trim: ignoring the first and last boundary); PWF = pairwise frame-clustering F; Sf = normalised conditional entropy F; Label acc = share of 0.1 s frames whose section type matches the expert; Macro-F1 = mean per-class F1 over the section types; Sections/song = predicted sections per song (the experts average 10.7).

*MSAF row: per-song results published with the Harmonix Set (Nieto et al., ISMIR 2019; unsupervised, annotated-beat features), averaged over the same songs; it predicts no section names.
