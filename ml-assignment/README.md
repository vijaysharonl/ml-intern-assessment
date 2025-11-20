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

