# Trigram Language Model

This directory contains the core assignment files for the Trigram Language Model.

## How to Run 

    # AI/ML Intern Assignment – Trigram Language Model

    This project contains a full implementation of a **Trigram (3-gram) Language Model** built from scratch in Python.  
    The goal is to demonstrate understanding of text preprocessing, probabilistic language modeling, and clean software design.

---

## 📁 Project Structure

ml-assignment/
│
├── data/
│ └── example_corpus.txt
│
├── src/
│ ├── ngram_model.py ← your main implementation
│ ├── generate.py ← script to generate text
│ └── utils.py ← helper functions (optional)
│
├── tests/
│ └── test_ngram.py ← provided test cases
│
├── README.md ← (this file)
├── evaluation.md ← 1-page design summary
├── assignment.md
├── quick_start.md
└── requirements.txt


---

## 🛠 Installation

Ensure you are using **Python 3.8+**

```bash
pip install -r requirements.txt

## Running the Model
1. Train the model & generate text
python src/generate.py


This loads example_corpus.txt, trains the trigram model, and prints generated text.

##Running Tests

The assignment includes tests to verify your implementation:

pytest tests/test_ngram.py


Make sure tests pass before creating your Pull Request.

##What I Implemented

Inside src/ngram_model.py, the following were implemented:

✔ Text cleaning & tokenization
✔ Sentence splitting
✔ Padding with <s> and </s>
✔ Trigram count dictionary
✔ Probabilistic sampling (using random.choices)
✔ Backoff (trigram → bigram → unigram)
✔ Deterministic seeding for reproducibility

## Submitting Your Solution

Fork this repository

Implement the assignment

Commit & push your changes

Create a Pull Request

Submit the PR link + your resume in the form provided

If you want extended functionality (UNK handling, smoothing, saving model, etc.), feel free to add it, but it’s not required.


---

# **FINAL `evaluation.md` (place in root)**

```markdown
# Evaluation & Design Summary (Trigram Language Model)

This document explains the design choices behind the TrigramModel implementation and provides the steps required to evaluate the solution.

---

## 1. Tokenization & Text Cleaning

I used a simple and predictable regex-based tokenizer:



[A-Za-z0-9']+ | [.!?]


### Why this design?
- Keeps words clean (letters, numbers, contractions)
- Separates `.`, `!`, `?` as sentence-ending tokens
- Ensures controlled token behavior that matches test expectations
- Avoids dependence on external NLP libraries

All text is converted to **lowercase** to reduce vocabulary sparsity.

---

## 2. Sentence Splitting

Sentences are split whenever a `.`, `!`, or `?` token appears.

Why?
- Trigram models perform poorly if sentence boundaries leak across sentences
- This allows us to add proper padding tokens per sentence

---

## 3. Padding Strategy

Each sentence is padded as:

<s>, <s> , word1 , word2 , ... , <\s>


This allows:
- A consistent 2-word context for the first trigram
- Proper stopping during generation when `<\s>` appears

---

## 4. N-gram Count Storage

Trigrams are stored in:



self.counts[(w1, w2)] → Counter(next_word)


Reasons:
- Fast lookup
- Memory efficient
- Works cleanly with probabilistic sampling

Additionally:
- `unigram_counts` and a lightweight `bigram_counts` are maintained for backoff

---

## 5. Training Logic

For each padded sentence:
- Extract trigrams
- Update `(w1, w2) → w3` counts
- Update unigram and bigram counts

This ensures a fully functional backoff mechanism.

---

## 6. Generation (Probabilistic Sampling)

Generation starts with:

(<s>, <s>)


Next words are chosen using:

random.choices(words, weights)


### Why probabilistic sampling?

Because the assignment explicitly says:
> “You must probabilistically sample. Do NOT always choose the most likely word.”

This results in varied and natural-looking output.

---

## 7. Backoff Strategy

If the trigram context `(w1, w2)` was never seen in training:

1. Try bigram using all contexts `(*, w2)`
2. If still unseen, sample from unigram distribution
3. If all fails, return `<\s>`

This prevents generation from crashing or getting stuck.

---

## 8. Steps to Evaluate the Model

### **Run tests**


pytest tests/test_ngram.py


### **Generate sample text**


python src/generate.py


You should see coherent, random text that resembles the training corpus.

---

# 9. Summary

This implementation focuses on:
- Clean, readable Python
- Pure standard-library solution
- Deterministic optional random seed
- High test compatibility
- Correct probabilistic generation

The design follows all assignment requirements while staying simple and extensible.



## Design Choices

Please document your design choices in the `evaluation.md` file. This should be a 1-page summary of the decisions you made and why you made them.
