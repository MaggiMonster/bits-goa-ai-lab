"""First- and second-order autoregressive language models (n-gram CPTs) built from counts.
Plain Python only: dicts, Counter, random.choices."""
import random
from collections import Counter, defaultdict

START, END = "<START>", "<END>"

DATA = """the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug""".splitlines()


def tokenise(lines):
    return [line.lower().split() for line in lines]


class NGramLM:
    """order=1: P(X_t | X_{t-1});  order=2: P(X_t | X_{t-2}, X_{t-1})."""

    def __init__(self, sentences, order=1):
        self.order = order
        self.counts = defaultdict(Counter)            # context tuple -> Counter(next token)
        for s in sentences:
            toks = [START] * order + s + [END]        # pad with `order` START tokens
            for i in range(order, len(toks)):
                self.counts[tuple(toks[i - order:i])][toks[i]] += 1
        # CPT: P(v | context) = C(context, v) / sum_k C(context, k)
        self.probs = {c: {v: n / sum(cnt.values()) for v, n in cnt.items()}
                      for c, cnt in self.counts.items()}
        self.words = sorted({t for s in sentences for t in s})
        self.vocab = self.words + [END]                # possible next tokens

    def dist(self, context):
        return self.probs.get(tuple(context), {})      # empty dict if context never observed

    def argmax(self, context):
        d = self.dist(context)
        return max(d, key=d.get) if d else None        # ties -> first inserted

    def sample(self, context, rng):
        d = self.dist(context)
        if not d:
            return None
        return rng.choices(list(d), weights=list(d.values()), k=1)[0]

    def generate(self, rng, greedy=False, max_len=30):
        ctx = [START] * self.order
        out = []
        for _ in range(max_len):
            nxt = self.argmax(ctx) if greedy else self.sample(ctx, rng)
            if nxt is None or nxt == END:               # unseen context or END stops generation
                break
            out.append(nxt)
            ctx = ctx[1:] + [nxt]
        return " ".join(out)

    def sentence_prob(self, sentence):
        toks = [START] * self.order + sentence.split() + [END]
        p = 1.0
        for i in range(self.order, len(toks)):
            p *= self.dist(toks[i - self.order:i]).get(toks[i], 0.0)
        return p

    def check_normalisation(self):
        return {c: sum(d.values()) for c, d in self.probs.items()}

    def n_parameters(self):
        return sum(len(d) for d in self.probs.values())    # non-zero CPT entries

    def possible_contexts(self):
        return (len(self.words) + 1) ** self.order         # words + <START>; <END> is never a context


def show_cpt(lm, words):
    for w in words:
        ctx = w if isinstance(w, tuple) else (w,)
        d = lm.dist(ctx)
        print(f"P(next | {' '.join(ctx)}): " + ", ".join(f"{v}={p:.3f}" for v, p in sorted(d.items(), key=lambda kv: -kv[1])))
        print(f"    zero-probability next words: {[v for v in lm.vocab if v not in d]}")


if __name__ == "__main__":
    sents = tokenise(DATA)
    m1, m2 = NGramLM(sents, 1), NGramLM(sents, 2)
    rng = random.Random(0)
    V = len(m1.vocab)
    print("Vocabulary (incl. <END>):", m1.vocab, f"|V|={V}")

    print("\n=== Q3: first-order CPT ===")
    show_cpt(m1, [START, "the", "cat", "dog", "sat", "ran", "on", "to", "mat", "rug", "park"])

    print("\n=== Normalisation test: sum_v P(v|w) for every context ===")
    for name, m in (("first-order", m1), ("second-order", m2)):
        tot = m.check_normalisation()
        ok = all(abs(t - 1) < 1e-9 for t in tot.values())
        print(f"{name}: {len(tot)} contexts, min={min(tot.values()):.12f} max={max(tot.values()):.12f} all close to 1: {ok}")
    for w, t in m1.check_normalisation().items():
        print("  ", " ".join(w), round(t, 6))

    print("\n=== Next-word prediction (argmax) for first-order model ===")
    for w in ("the", "cat", "dog", "sat", "ran", "on", "to"):
        d = m1.dist((w,))
        print(f"argmax P(w|{w}) = {m1.argmax((w,))}   dist={ {k: round(v, 3) for k, v in d.items()} }")

    print("\n=== Unseen context ===")
    print("dist('banana') =", m1.dist(("banana",)), "-> sample:", m1.sample(("banana",), rng))

    print("\n=== 20 sampled sentences (first-order, seed 0) ===")
    samples1 = [m1.generate(rng) for _ in range(20)]
    for s in samples1:
        print("  ", s)

    print("\n=== Mode A (greedy) vs Mode B (sampling), 5 sentences each, first-order ===")
    print("Greedy:")
    for _ in range(5):
        print("  ", m1.generate(rng, greedy=True))
    print("Sampling:")
    for _ in range(5):
        print("  ", m1.generate(rng))

    print("\n=== Second-order CPTs ===")
    show_cpt(m2, [(START, START), (START, "the"), ("the", "cat"), ("the", "dog"), ("cat", "sat"), ("dog", "ran"), ("on", "the"), ("to", "the")])

    print("\n=== 20 sampled sentences (second-order, seed 0) ===")
    rng = random.Random(0)
    for _ in range(20):
        print("  ", m2.generate(rng))
    print("Second-order greedy x5:")
    for _ in range(5):
        print("  ", m2.generate(rng, greedy=True))

    print("\n=== Comparison ===")
    train = {" ".join(s) for s in sents}
    N = 5000
    for name, m in (("first-order", m1), ("second-order", m2)):
        gen = [m.generate(random.Random(i)) for i in range(N)]
        distinct = set(gen)
        novel = distinct - train
        nctx, possible = len(m.probs), m.possible_contexts()
        print(f"{name}: parameters (non-zero CPT entries)={m.n_parameters()}, observed contexts={nctx}, "
              f"possible contexts={possible}, unobserved contexts={possible - nctx} ({100 * (possible - nctx) / possible:.0f}%), "
              f"full CPT size={possible * V}, distinct sentences in {N} samples={len(distinct)}, "
              f"novel (not in training)={len(novel)}")
        print("   novel examples:", sorted(novel)[:6])
    print("Training sentences:", len(train), "distinct")
    print("P(sentence) examples:")
    for s in ("the cat sat on the mat", "the cat ran to the park", "the dog ran to the mat", "the cat sat on the park"):
        print(f"   {s!r}: first-order={m1.sentence_prob(s):.5f}  second-order={m2.sentence_prob(s):.5f}")
