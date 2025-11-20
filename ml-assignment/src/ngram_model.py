# trigram-assignment/src/ngram_model.py

import re
import random
from collections import Counter, defaultdict

SENT_START = "<s>"
SENT_END = "</s>"

# tokens: words (including contractions/numbers) and sentence-ending punctuation
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.!?]")

class TrigramModel:
    def __init__(self, seed: int = 42):
        """
        Initializes the TrigramModel.
        Uses:
          - self.counts: dict mapping 2-word context (tuple) -> Counter(next_word -> count)
          - self.bigram_counts: Counter for bigrams (for simple backoff)
          - self.unigram_counts: Counter for unigram distribution (final fallback)
        """
        self.counts = defaultdict(Counter)   # (w1, w2) -> Counter(next_word)
        self.bigram_counts = Counter()       # (w2,) -> Counter(next_word) not stored as nested but total counts for backoff
        self.unigram_counts = Counter()
        self._random = random.Random(seed)

    # ----------------- tokenization & sentence split -----------------
    def _tokenize(self, text: str):
        # lower + regex tokenization; preserves .,!,? as tokens so we can split sentences
        text = text.lower()
        tokens = _TOKEN_RE.findall(text)
        return tokens

    def _split_sentences(self, tokens):
        sentences = []
        cur = []
        for t in tokens:
            cur.append(t)
            if t in ('.', '!', '?'):
                # sentence end - drop the punctuation token from final token list for modeling
                # but we treat it as end marker (we will append SENT_END)
                sentences.append([tok for tok in cur if tok not in ('.', '!', '?')])
                cur = []
        if cur:
            # leftover tokens (no terminating punctuation) still considered a sentence
            sentences.append([tok for tok in cur if tok not in ('.', '!', '?')])
        return sentences

    # ----------------- training -----------------
    def fit(self, text: str):
        """
        Trains the trigram model on the given text.

        Steps:
        - tokenize and split into sentences
        - for each sentence, pad with two SENT_START and append SENT_END
        - collect trigram counts and unigram counts for fallback
        """
        tokens = self._tokenize(text)
        sentences = self._split_sentences(tokens)

        # clear previous counts (allow re-fit)
        self.counts.clear()
        self.bigram_counts.clear()
        self.unigram_counts.clear()

        for sent in sentences:
            # skip empty sentences
            if not sent:
                continue
            # pad with two start tokens and an end token
            padded = [SENT_START, SENT_START] + sent + [SENT_END]
            # update unigram counts (including SENT_END but not start tokens)
            for w in padded:
                if w != SENT_START:  # don't count artificial start in unigram
                    self.unigram_counts[w] += 1
            # update trigram counts
            for i in range(len(padded) - 2):
                context = (padded[i], padded[i + 1])   # two-word context
                next_word = padded[i + 2]
                self.counts[context][next_word] += 1
                # also collect bigram-level info for backoff (context last word -> next_word)
                # here we increment a (single-word context) tuple to mirror fallback logic
                self.bigram_counts[(context[1], next_word)] += 1

    # ----------------- generation -----------------
    def _sample_from_counter(self, counter: Counter):
        """Sample a key from counter proportionally to counts."""
        words, weights = zip(*counter.items())
        return self._random.choices(words, weights=weights, k=1)[0]

    def _backoff_sample(self, context):
        """
        Backoff strategy:
        - try full trigram context (w1,w2)
        - if unseen, try bigram context (w2) by aggregating counts from self.counts
        - if still unseen, sample from unigram_counts
        """
        # trigram
        if context in self.counts and len(self.counts[context]) > 0:
            return self._sample_from_counter(self.counts[context])

        # bigram backoff: aggregate all counters where second item == context[1]
        w2 = context[1]
        agg = Counter()
        for (c1, c2), ctr in self.counts.items():
            if c2 == w2:
                agg.update(ctr)
        if agg:
            return self._sample_from_counter(agg)

        # unigram fallback
        if self.unigram_counts:
            return self._sample_from_counter(self.unigram_counts)

        # if nothing is available (model not trained), just return SENT_END
        return SENT_END

    def generate(self, max_length=50) -> str:
        """
        Generates new text using the trained trigram model.
        - starts with two SENT_START tokens
        - samples next words until SENT_END or max_length reached
        - returns a detokenized string
        """
        context = (SENT_START, SENT_START)
        output_tokens = []

        for _ in range(max_length):
            next_word = self._backoff_sample(context)
            if next_word == SENT_END:
                break
            output_tokens.append(next_word)
            context = (context[1], next_word)

        # simple detokenization: join with spaces
        if not output_tokens:
            return ""
        # attach punctuation if present (we didn't include punctuation tokens except sentence-enders that were removed)
        text = " ".join(output_tokens)
        # capitalize first character to make result look nicer
        if text:
            text = text[0].upper() + text[1:]
        return text
