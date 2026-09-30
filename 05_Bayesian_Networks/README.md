# 05: Autoregressive language models as Bayesian networks
`ngram_lm.py`: one `NGramLM(order)` class for first-order P(Xt|Xt-1) and second-order P(Xt|Xt-2,Xt-1) CPTs from counts,
with argmax prediction, sampling/greedy generation, sentence probabilities and the sum-to-1 test.

Run: `python ngram_lm.py`

| | 1st order | 2nd order |
|---|---|---|
| Non-zero CPT entries | 17 | 19 |
| Unobserved contexts | 0 | 88% |
| Novel sentences in 5000 samples | 476 | 0 |

See `ANSWERS.md`, `report.pdf`.
