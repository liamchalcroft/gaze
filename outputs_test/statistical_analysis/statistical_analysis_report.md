# Statistical Significance Analysis Report

**Total tests performed:** 140
**Statistically significant results:** 14 (10.0%)

## Summary of Significant Findings

### Caption Task

- **web_search** vs **multiturn** on Caption Bert F1: p = 0.0011 (Effect Size: 274.249)
- **multiturn** vs **baseline** on Caption Bleu: p = 0.0000 (Effect Size: -0.279)
- **multiturn** vs **baseline** on Caption Bleu: p = 0.0000 (Effect Size: 3074.253)
- **multiturn** vs **baseline** on Caption Bert F1: p = 0.0001 (Effect Size: -0.221)
- **multiturn** vs **baseline** on Caption Bert F1: p = 0.0001 (Effect Size: 3105.867)
- **baseline** vs **visual** on Caption Bleu: p = 0.0000 (Effect Size: 0.316)
- **baseline** vs **visual** on Caption Bleu: p = 0.0000 (Effect Size: 2219.323)
- **baseline** vs **visual** on Caption Meteor: p = 0.0046 (Effect Size: 0.179)
- **baseline** vs **visual** on Caption Bert F1: p = 0.0000 (Effect Size: 0.260)
- **baseline** vs **visual** on Caption Bert F1: p = 0.0000 (Effect Size: 2287.143)

### Localization Task

- **comprehensive** vs **baseline** on Detection Map30: p = 0.0008 (Effect Size: 0.244)
- **comprehensive** vs **baseline** on Detection Map30: p = 0.0008 (Effect Size: 316.130)
- **comprehensive** vs **baseline** on Detection Map50: p = 0.0017 (Effect Size: 0.221)
- **comprehensive** vs **baseline** on Detection Map50: p = 0.0032 (Effect Size: 54.282)

## Statistical Test Interpretation Guide

### Effect Size Interpretation (Cohen's d)
- Small effect: d ≈ 0.2
- Medium effect: d ≈ 0.5
- Large effect: d ≈ 0.8

### P-value Interpretation
- p < 0.05: Statistically significant difference
- p < 0.01: Highly significant difference
- p < 0.001: Very highly significant difference

### Test Types Used
- **Paired t-test**: For continuous metrics (parametric)
- **Wilcoxon signed-rank**: For continuous metrics (non-parametric)
- **McNemar's test**: For binary classification metrics

---

*Statistical analysis performed with multiple testing correction*