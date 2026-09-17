# AQSE preregistered replicated study — recovered final report

- Experiment source commit: `091acf2e98b2b88720730eea276e253f0c96f5a6`
- Reporting source commit: `d42965ef6a38791fcdbdc1b7bbb3750a58d1f2f1`
- Completed replicas: `30 main + 12 negative + 12 positive`
- Conclusion: **evidence does not support the preregistered quantum-advantage criterion**
- Original execution wall time: unavailable; saved per-replica sums are in JSON

## Primary paired comparison

```json
{
  "delta_maximum": 0.08333333333333337,
  "delta_mean_interval": {
    "bootstrap_resamples": 2000,
    "confidence_level": 0.95,
    "lower": -0.022222222222222192,
    "point": -0.004166666666666652,
    "upper": 0.012500000000000015
  },
  "delta_median": 0.0,
  "delta_minimum": -0.125,
  "delta_standard_deviation": 0.050559607854097854,
  "proportion_delta_greater_than_zero": 0.3333333333333333,
  "wilcoxon": {
    "alternative": "two-sided",
    "p_value": 0.932805434372276,
    "statistic": 201.0,
    "zero_difference_policy": "Pratt includes zero differences in ranking"
  }
}
```

## Limitations

- exact-state simulator only; no QPU hardware evidence
- synthetic sensor domain only; no field-deployment validity
- positive control is mechanism-aligned and is not real-sensor evidence
- parameter-count and feature-space matching do not equalize hypothesis classes
- report recovery packages persisted scores and does not repeat predictions
