# Lab 05: Written answers

1. **Why the decomposition helps generation:** it reduces a sentence's joint probability to next-token conditionals, so generation is repeated sampling of X_t given the past, and the same factors score any sentence.
2. **Independence assumption:** P(X_t | X_1..X_{t-1}) = P(X_t | X_{t-1}), i.e. X_t is independent of X_1..X_{t-2} given X_{t-1}.
3. **First-order CPT:** <START>: the 1.0; the: cat .250, dog .250, mat .167, rug .167, park .167; cat: sat .667, ran .333; dog: same; sat: on 1.0; ran: to 1.0; on, to: the 1.0; mat, rug, park: <END> 1.0. Zero transitions, e.g. P(cat|cat), P(sat|the), P(ran|sat), P(<END>|the): 17 non-zero entries out of 121.
4. Counts: `self.counts`, a defaultdict(Counter) keyed by the context tuple.
5. P(Xt|Xt-1): the dict comprehension building `self.probs` in `__init__` (count / row total), read via `dist()`.
6. Next word: `sample()` draws with `random.choices` using the probabilities (option 2); `argmax()` (option 1) is used only in greedy mode. Greedy is deterministic and picks the single best word; sampling picks each word with its probability.
7. Unseen context: `dist()` returns {}, `sample()`/`argmax()` return None and `generate()` stops the sentence.
8. A total of 0.87 means missing probability mass: a bug such as a wrong denominator, dropped <END> transitions, or filtered tokens.
9. Predictions are not always what a human expects (after "the", cat vs dog is a tie): the model reflects data frequencies, not meaning or world knowledge.
10. Sampling varies more; greedy always takes the same word for a context (and loops in the first-order model).
11. Second-order vs first-order: (1) two parents X_{t-2}, X_{t-1} -> X_t; (2) CPT rows indexed by pairs (up to |V|^2 rows); (3) two words of context, separating "on the -> mat/rug" from "to the -> park"; (4) far more data needed.
12. More context sharpens predictions, but the CPT grows as |V|^n x |V| (121 to 1331 entries) while data stays fixed, so 88% of contexts are unseen: high variance and memorisation.
13. Approach B fixes the intended behaviour, needs an understood representation (CPT of counts), gives something to validate against, allows probabilistic invariant tests (sum = 1), and separates the model from its implementation.
14. The Bayesian-network view gives a factorisation of the joint, explicit independence assumptions, a view of how context enlarges the CPT, principled ancestral sampling, and a test of implementation against specification (normalisation).

## Comparison
| | 1st order | 2nd order |
|---|---|---|
| Non-zero CPT entries | 17 | 19 |
| Observed / possible contexts | 11 / 11 | 15 / 121 |
| Distinct sentences in 5000 samples | 482 | 6 |
| Novel sentences | 476 | 0 |
Example: P("the dog ran to the mat") is 0.0139 first-order and 0 second-order.
