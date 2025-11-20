from src.ngram_model import TrigramModel

def main():
    # Create and train the TrigramModel
    model = TrigramModel()

    # Load the example training corpus
    with open("data/example_corpus.txt", "r") as f:
        text = f.read()

    model.fit(text)

    # Generate new text
    generated_text = model.generate()
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
