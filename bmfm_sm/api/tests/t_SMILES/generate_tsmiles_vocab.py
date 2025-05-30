import os
import pandas as pd
from collections import Counter

from DataSet.Tokenlizer import Tokenlizer


CSV_PATH = "bmfm_sm/api/tests/t_SMILES/output.csv"
OUT_PATH = "tsmiles_corpus.txt"

# 3) Read CSV and pull both columns
df = pd.read_csv(CSV_PATH, usecols=["tsmiles_item1", "tsmiles_item2"])

# 4) Concatenate
all_tsmiles = pd.concat([
    df["tsmiles_item1"],
    df["tsmiles_item2"]
]).dropna().unique()
all_tsmiles = sorted(all_tsmiles)


with open(OUT_PATH, "w") as fout:
    for smi in all_tsmiles:
        fout.write(smi + "\n")


CORPUS_PATH = OUT_PATH
VOCAB_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, 'resources', 'vocab.txt'
)

# Special tokens
SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']


def build_vocab(corpus_path, vocab_path, min_freq=1):
    # Initialize tokenizer from t-SMILES
    tokenizer = Tokenlizer(voc=VOCAB_PATH)
    freq = Counter()

    # Read corpus and count token frequencies
    with open(corpus_path, 'r') as f:
        for line in f:
            smiles = line.strip()
            if not smiles:
                continue
            tokens = tokenizer.tokenize(smiles)
            freq.update(tokens)

    # Filter tokens by frequency
    tokens = [tok for tok, ct in freq.items() if ct >= min_freq]
    tokens = sorted(tokens)

    # Combine special + sorted tokens
    vocab = SPECIAL_TOKENS + tokens

    # Write to file
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, 'w') as fout:
        for tok in vocab:
            fout.write(tok + '\n')

    print(f"Vocabulary of size {len(vocab)} saved to {vocab_path}")


if __name__ == '__main__':
    build_vocab(CORPUS_PATH, VOCAB_PATH)
