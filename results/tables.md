### Per-category binary confirmation (mean +/- s.d., 5 seeds)

| Category | Model | Accuracy | F1 | Train (s) |
|---|---|---|---|---|
| granulated_sugar | Unigram BoW + LogReg | 0.965+/-0.010 | 0.958+/-0.012 | 0.01 |
| granulated_sugar | Word 1-2-gram + LogReg | 0.962+/-0.012 | 0.955+/-0.014 | 0.01 |
| granulated_sugar | Word 1-3-gram + LogReg | 0.959+/-0.012 | 0.951+/-0.014 | 0.01 |
| granulated_sugar | Char n-gram(3-5) + LogReg | 0.996+/-0.002 | 0.996+/-0.003 | 0.03 |
| granulated_sugar | BoW+MLP(256) | 0.982+/-0.009 | 0.979+/-0.010 | 0.39 |
| granulated_sugar | CNN | 0.943+/-0.010 | 0.932+/-0.013 | 0.46 |
| granulated_sugar | LSTM | 0.946+/-0.008 | 0.936+/-0.010 | 0.42 |
| milk | Unigram BoW + LogReg | 0.960+/-0.015 | 0.951+/-0.019 | 0.01 |
| milk | Word 1-2-gram + LogReg | 0.961+/-0.015 | 0.953+/-0.019 | 0.01 |
| milk | Word 1-3-gram + LogReg | 0.959+/-0.014 | 0.951+/-0.017 | 0.02 |
| milk | Char n-gram(3-5) + LogReg | 0.998+/-0.002 | 0.997+/-0.003 | 0.03 |
| milk | BoW+MLP(256) | 0.969+/-0.007 | 0.964+/-0.008 | 0.40 |
| milk | CNN | 0.959+/-0.009 | 0.950+/-0.012 | 0.25 |
| milk | LSTM | 0.950+/-0.020 | 0.939+/-0.025 | 0.40 |
| bread | Unigram BoW + LogReg | 0.965+/-0.014 | 0.958+/-0.016 | 0.01 |
| bread | Word 1-2-gram + LogReg | 0.968+/-0.009 | 0.962+/-0.011 | 0.01 |
| bread | Word 1-3-gram + LogReg | 0.966+/-0.009 | 0.960+/-0.011 | 0.01 |
| bread | Char n-gram(3-5) + LogReg | 0.995+/-0.001 | 0.994+/-0.001 | 0.04 |
| bread | BoW+MLP(256) | 0.977+/-0.008 | 0.973+/-0.009 | 0.38 |
| bread | CNN | 0.939+/-0.010 | 0.927+/-0.013 | 0.15 |
| bread | LSTM | 0.936+/-0.020 | 0.923+/-0.026 | 0.39 |
| beer | Unigram BoW + LogReg | 0.967+/-0.011 | 0.961+/-0.013 | 0.01 |
| beer | Word 1-2-gram + LogReg | 0.973+/-0.011 | 0.968+/-0.014 | 0.01 |
| beer | Word 1-3-gram + LogReg | 0.970+/-0.011 | 0.964+/-0.013 | 0.02 |
| beer | Char n-gram(3-5) + LogReg | 0.997+/-0.003 | 0.996+/-0.004 | 0.04 |
| beer | BoW+MLP(256) | 0.975+/-0.008 | 0.971+/-0.009 | 0.37 |
| beer | CNN | 0.951+/-0.010 | 0.941+/-0.012 | 0.15 |
| beer | LSTM | 0.952+/-0.009 | 0.944+/-0.010 | 0.39 |
| laundry_detergent | Unigram BoW + LogReg | 0.988+/-0.004 | 0.985+/-0.005 | 0.01 |
| laundry_detergent | Word 1-2-gram + LogReg | 0.985+/-0.007 | 0.982+/-0.009 | 0.01 |
| laundry_detergent | Word 1-3-gram + LogReg | 0.984+/-0.008 | 0.980+/-0.010 | 0.02 |
| laundry_detergent | Char n-gram(3-5) + LogReg | 1.000+/-0.000 | 1.000+/-0.000 | 0.04 |
| laundry_detergent | BoW+MLP(256) | 0.990+/-0.003 | 0.989+/-0.003 | 0.32 |
| laundry_detergent | CNN | 0.966+/-0.006 | 0.959+/-0.008 | 0.15 |
| laundry_detergent | LSTM | 0.964+/-0.008 | 0.957+/-0.010 | 0.40 |
| fresh_apples | Unigram BoW + LogReg | 0.985+/-0.008 | 0.982+/-0.010 | 0.01 |
| fresh_apples | Word 1-2-gram + LogReg | 0.985+/-0.008 | 0.982+/-0.010 | 0.01 |
| fresh_apples | Word 1-3-gram + LogReg | 0.985+/-0.010 | 0.982+/-0.012 | 0.02 |
| fresh_apples | Char n-gram(3-5) + LogReg | 1.000+/-0.000 | 1.000+/-0.000 | 0.04 |
| fresh_apples | BoW+MLP(256) | 0.988+/-0.004 | 0.986+/-0.004 | 0.36 |
| fresh_apples | CNN | 0.972+/-0.004 | 0.967+/-0.006 | 0.18 |
| fresh_apples | LSTM | 0.964+/-0.009 | 0.957+/-0.010 | 0.40 |

### Matched BoW vs CNN/LSTM (mean F1 over categories)

| Model | mean F1 (all categories) |
|---|---|
| Char n-gram(3-5) + LogReg | 0.997 |
| BoW+MLP(256) | 0.977 |
| Word 1-2-gram + LogReg | 0.967 |
| Unigram BoW + LogReg | 0.966 |
| Word 1-3-gram + LogReg | 0.965 |
| CNN | 0.946 |
| LSTM | 0.943 |

### Trie coverage

| Category | Items | Coverage | Positive recall (trie) | Unidentified |
|---|---|---|---|---|
| granulated_sugar | 1650 | 0.366 | 0.804 | 1046 |
| milk | 1650 | 0.440 | 0.840 | 924 |
| bread | 1650 | 0.439 | 0.817 | 926 |
| beer | 1650 | 0.499 | 0.859 | 826 |
| laundry_detergent | 1650 | 0.317 | 0.747 | 1127 |
| fresh_apples | 1650 | 0.346 | 0.816 | 1079 |

### Learning curve, unigram BoW (mean F1, 5 seeds)

| Category | 5% (~66) | 10% (~132) | 20% (~264) | 40% (~528) | 100% (~1320) |
|---|---|---|---|---|---|
| granulated_sugar | 0.872 | 0.906 | 0.923 | 0.939 | 0.958 |
| milk | 0.886 | 0.903 | 0.921 | 0.934 | 0.951 |
| bread | 0.871 | 0.885 | 0.905 | 0.925 | 0.958 |
| beer | 0.829 | 0.877 | 0.913 | 0.923 | 0.961 |
| laundry_detergent | 0.911 | 0.936 | 0.950 | 0.973 | 0.985 |
| fresh_apples | 0.951 | 0.954 | 0.958 | 0.969 | 0.982 |

### Consensus simulation: label-recovery accuracy (mean +/- s.d., 60 runs)

| k | Majority | Reliability-weighted | Dawid-Skene |
|---|---|---|---|
| 3 | 0.812+/-0.017 | 0.812+/-0.017 | 0.882+/-0.015 |
| 5 | 0.873+/-0.016 | 0.881+/-0.018 | 0.944+/-0.014 |
| 7 | 0.910+/-0.013 | 0.932+/-0.016 | 0.970+/-0.009 |
