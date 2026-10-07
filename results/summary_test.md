# Results on the test split (136 songs)

| Method | HR.5F | HR3F | HR.5F (trim) | HR3F (trim) | PWF | Sf | Label acc | Macro-F1 | Sections/song |
|---|---|---|---|---|---|---|---|---|---|
| CNN + BiGRU (ours) | 0.600 | 0.760 | 0.513 | 0.709 | 0.666 | 0.706 | 0.636 | 0.508 | 10.8 |
| MLP | 0.449 | 0.622 | 0.339 | 0.546 | 0.539 | 0.567 | 0.499 | 0.365 | 12.6 |
| Logistic regression | 0.318 | 0.477 | 0.157 | 0.351 | 0.487 | 0.448 | 0.448 | 0.257 | 9.2 |
| Foote novelty (unsupervised) | 0.440 | 0.671 | 0.328 | 0.604 | 0.449 | 0.000 | 0.371 | 0.053 | 13.2 |
| Majority class | 0.297 | 0.297 | 0.000 | 0.000 | 0.449 | 0.000 | 0.371 | 0.053 | 1.0 |
| MSAF, published with Harmonix* | 0.262 | 0.592 | 0.184 | 0.561 | 0.542 | 0.599 | - | - | - |

HR = boundary hit rate F-measure within ±0.5 s / ±3 s (trim: ignoring the first and last boundary); PWF = pairwise frame-clustering F; Sf = normalised conditional entropy F; Label acc = share of 0.1 s frames whose section type matches the expert; Macro-F1 = mean per-class F1 over the section types; Sections/song = predicted sections per song (the experts average 10.9).

*MSAF row: per-song results published with the Harmonix Set (Nieto et al., ISMIR 2019; unsupervised, annotated-beat features), averaged over the same songs; it predicts no section names.
