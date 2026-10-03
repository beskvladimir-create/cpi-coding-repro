### Per-category binary confirmation (mean +/- s.d., 5 seeds)

| Category | Model | Accuracy | F1 | Train (s) |
|---|---|---|---|---|
| granulated_sugar | Unigram BoW + LogReg | 0.965+/-0.010 | 0.958+/-0.012 | 0.02 |
| granulated_sugar | Word 1-2-gram + LogReg | 0.962+/-0.012 | 0.955+/-0.014 | 0.03 |
| granulated_sugar | Word 1-3-gram + LogReg | 0.959+/-0.012 | 0.951+/-0.014 | 0.03 |
| granulated_sugar | Char n-gram(3-5) + LogReg | 0.996+/-0.002 | 0.996+/-0.003 | 0.05 |
| granulated_sugar | BoW+MLP(256) | 0.982+/-0.009 | 0.979+/-0.010 | 0.99 |
| granulated_sugar | CNN | 0.961+/-0.015 | 0.953+/-0.018 | 2.19 |
| granulated_sugar | LSTM | 0.959+/-0.015 | 0.950+/-0.018 | 2.41 |
| milk | Unigram BoW + LogReg | 0.960+/-0.015 | 0.951+/-0.019 | 0.03 |
| milk | Word 1-2-gram + LogReg | 0.961+/-0.015 | 0.953+/-0.019 | 0.03 |
| milk | Word 1-3-gram + LogReg | 0.959+/-0.014 | 0.951+/-0.017 | 0.03 |
| milk | Char n-gram(3-5) + LogReg | 0.998+/-0.002 | 0.997+/-0.003 | 0.06 |
| milk | BoW+MLP(256) | 0.969+/-0.007 | 0.964+/-0.008 | 1.09 |
| milk | CNN | 0.961+/-0.008 | 0.953+/-0.010 | 2.14 |
| milk | LSTM | 0.961+/-0.009 | 0.953+/-0.010 | 2.97 |
| bread | Unigram BoW + LogReg | 0.965+/-0.014 | 0.958+/-0.016 | 0.02 |
| bread | Word 1-2-gram + LogReg | 0.968+/-0.009 | 0.962+/-0.011 | 0.02 |
| bread | Word 1-3-gram + LogReg | 0.966+/-0.009 | 0.960+/-0.011 | 0.03 |
| bread | Char n-gram(3-5) + LogReg | 0.995+/-0.001 | 0.994+/-0.001 | 0.06 |
| bread | BoW+MLP(256) | 0.977+/-0.008 | 0.973+/-0.009 | 1.07 |
| bread | CNN | 0.964+/-0.006 | 0.957+/-0.008 | 2.14 |
| bread | LSTM | 0.958+/-0.008 | 0.950+/-0.011 | 2.65 |
| beer | Unigram BoW + LogReg | 0.967+/-0.011 | 0.961+/-0.013 | 0.02 |
| beer | Word 1-2-gram + LogReg | 0.973+/-0.011 | 0.968+/-0.014 | 0.02 |
| beer | Word 1-3-gram + LogReg | 0.970+/-0.011 | 0.964+/-0.013 | 0.03 |
| beer | Char n-gram(3-5) + LogReg | 0.997+/-0.003 | 0.996+/-0.004 | 0.07 |
| beer | BoW+MLP(256) | 0.975+/-0.008 | 0.971+/-0.009 | 1.06 |
| beer | CNN | 0.964+/-0.007 | 0.957+/-0.008 | 2.06 |
| beer | LSTM | 0.962+/-0.004 | 0.955+/-0.005 | 2.57 |
| laundry_detergent | Unigram BoW + LogReg | 0.988+/-0.004 | 0.985+/-0.005 | 0.02 |
| laundry_detergent | Word 1-2-gram + LogReg | 0.985+/-0.007 | 0.982+/-0.009 | 0.02 |
| laundry_detergent | Word 1-3-gram + LogReg | 0.984+/-0.008 | 0.980+/-0.010 | 0.03 |
| laundry_detergent | Char n-gram(3-5) + LogReg | 1.000+/-0.000 | 1.000+/-0.000 | 0.07 |
| laundry_detergent | BoW+MLP(256) | 0.990+/-0.003 | 0.989+/-0.003 | 0.93 |
| laundry_detergent | CNN | 0.983+/-0.006 | 0.980+/-0.007 | 1.84 |
| laundry_detergent | LSTM | 0.978+/-0.005 | 0.973+/-0.007 | 2.22 |
| fresh_apples | Unigram BoW + LogReg | 0.985+/-0.008 | 0.982+/-0.010 | 0.02 |
| fresh_apples | Word 1-2-gram + LogReg | 0.985+/-0.008 | 0.982+/-0.010 | 0.02 |
| fresh_apples | Word 1-3-gram + LogReg | 0.985+/-0.010 | 0.982+/-0.012 | 0.03 |
| fresh_apples | Char n-gram(3-5) + LogReg | 1.000+/-0.000 | 1.000+/-0.000 | 0.05 |
| fresh_apples | BoW+MLP(256) | 0.988+/-0.004 | 0.986+/-0.004 | 0.92 |
| fresh_apples | CNN | 0.978+/-0.006 | 0.973+/-0.006 | 1.85 |
| fresh_apples | LSTM | 0.968+/-0.011 | 0.963+/-0.012 | 2.34 |

### Matched BoW vs CNN/LSTM (mean F1 over categories)

| Model | mean F1 (all categories) |
|---|---|
| Char n-gram(3-5) + LogReg | 0.997 |
| BoW+MLP(256) | 0.977 |
| Word 1-2-gram + LogReg | 0.967 |
| Unigram BoW + LogReg | 0.966 |
| Word 1-3-gram + LogReg | 0.965 |
| CNN | 0.962 |
| LSTM | 0.957 |

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
