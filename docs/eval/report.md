**Eval set:** 19 questions (12 answerable, 7 unanswerable). Each scored over self-consistency samples; the threshold τ is swept analytically.

**Operating point (selective risk ≤ 2%): τ = 0.00**

| Metric | At τ = 0.00 (chosen) | At τ = 0.60 (default) |
|---|---|---|
| Coverage (answered) | 63.2% | 57.9% |
| **Confident-wrong rate** | **0.0%** | 0.0% |
| Selective risk (error among answered) | 0.0% | 0.0% |
| Execution accuracy (EX) | 100.0% | 100.0% |
| Abstention recall | 100.0% | 100.0% |
| Over-abstention | 0.0% | 8.3% |

![Risk-coverage curve](docs/eval/risk_coverage.png)