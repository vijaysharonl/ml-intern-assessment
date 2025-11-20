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
        Trigram model:
          - self.counts: (w1, w2) -> Counter(next_word -> count)
          - self.bigram_counts: w2 -> Counter(next_word -> count)  (for backoff)
          - self.unigram_counts: Counter(word -> count)           (final fallback)
        """
        self.counts = defaultdict(Counter)       # (w1, w2) -> Counter(next_word)
        self.bigram_counts = defaultdict(Counter)  # w2 -> Counter(next_word)
        self.unigram_counts = Counter()
        self._random = random.Random(seed)

    # ----------------- tokenization & sentence split -----------------
    def _tokenize(self, text: str):
        # Lowercase + regex tokenization; keep .,!,? tokens so we can split sentences
        text = text.lower()
        tokens = _TOKEN_RE.findall(text)
        return tokens

    def _split_sentences(self, tokens):
        sentences = []
        cur = []
        for t in tokens:
            cur.append(t)
            if t in ('.', '!', '?'):
                # End of sentence: remove the punctuation tokens from sentence words
                sentences.append([tok for tok in cur if tok not in ('.', '!', '?')])
                cur = []
        if cur:
            # leftover tokens (no terminating punctuation) still considered a sentence
            sentences.append([tok for tok in cur if tok not in ('.', '!', '?')])
        return sentences

    # ----------------- training -----------------
    def fit(self, text: str):
        """
        Train the trigram model on the given raw text.
        - Tokenize, split into sentences
        - Pad each sentence with two <s> and one </s>
        - Build trigram counts, bigram counts (for backoff), and unigram counts
        """
        tokens = self._tokenize(text)
        sentences = self._split_sentences(tokens)

        # Clear previous counts (allow re-fit)
        self.counts.clear()
        self.bigram_counts.clear()
        self.unigram_counts.clear()

        for sent in sentences:
            if not sent:
                continue
            padded = [SENT_START, SENT_START] + sent + [SENT_END]

            # Update unigram counts (exclude artificial start tokens)
            for w in padded:
                if w != SENT_START:
                    self.unigram_counts[w] += 1

            # Update trigram & bigram counts
            for i in range(len(padded) - 2):
                context = (padded[i], padded[i + 1])  # (w1, w2)
                next_word = padded[i + 2]
                self.counts[context][next_word] += 1
                # Bigram context for backoff: store under w2 -> Counter(next_word)
                self.bigram_counts[context[1]][next_word] += 1

    # ----------------- generation -----------------
    def _sample_from_counter(self, counter: Counter):
        """Sample a key from a Counter proportionally to counts."""
        if not counter:
            return SENT_END
        words, weights = zip(*counter.items())
        return self._random.choices(words, weights=weights, k=1)[0]

    def _backoff_sample(self, context):
        """
        Backoff sampling:
          1) Try trigram context (w1, w2)
          2) If unseen, try bigram context w2
          3) If still unseen, use unigram distribution
          4) If model untrained, return SENT_END
        """
        # 1) trigram
        if context in self.counts and len(self.counts[context]) > 0:
            return self._sample_from_counter(self.counts[context])

        # 2) bigram backoff (w2)
        w2 = context[1]
        if w2 in self.bigram_counts and len(self.bigram_counts[w2]) > 0:
            return self._sample_from_counter(self.bigram_counts[w2])

        # 3) unigram fallback
        if self.unigram_counts:
            return self._sample_from_counter(self.unigram_counts)

        # Nothing available
        return SENT_END

    def generate(self, max_length=50) -> str:
        """
        Generate text using the trained trigram model.
        - Start from (<s>, <s>)
        - Sample next words until </s> or max_length
        - Return detokenized string
        """
        context = (SENT_START, SENT_START)
        output_tokens = []

        for _ in range(max_length):
            next_word = self._backoff_sample(context)
            if next_word == SENT_END:
                break
            output_tokens.append(next_word)
            context = (context[1], next_word)

        if not output_tokens:
            return ""

        text = " ".join(output_tokens)
        # Capitalize first character to look nicer
        if text:
            text = text[0].upper() + text[1:]
        return text
