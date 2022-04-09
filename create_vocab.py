import utils.utils as utils
from collections import Counter
import itertools
import json
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")


def create_iterable(df_list):
    text = []
    total = 0
    for df in df_list:
        unique_contexts = list(df.context.unique())
        unique_questions = list(df.question.unique())
        total += df.context.nunique() + df.question.nunique()
        text.extend(unique_contexts + unique_questions)

    assert len(text) == total

    return text


def create_vocab(iterable, top_k=None):
    words = []
    for senteces in iterable:
        for word in nlp(senteces, disable=["parser", "tagger", "ner"]):
            words.append(word.text)
    word_counter = Counter(words)
    word_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
    print(f"raw-vocab: {len(word_vocab)}")
    word_vocab.insert(0, "<unk>")
    word_vocab.insert(1, "<pad>")
    print(f"vocab-length: {len(word_vocab)}")
    word2idx = {word: idx for idx, word in enumerate(word_vocab)}
    print(f"word2idx-length: {len(word2idx)}")
    idx2word = {v: k for k, v in word2idx.items()}

    return word2idx, idx2word, word_vocab


if __name__ == "__main__":
    with open("config.json", "r") as conf:
        config = json.loads(conf.read())
    train_path = config.get("qa_path") + "/train.csv"
    val_path = config.get("qa_path") + "/val.csv"
    print(train_path, val_path)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    vocab_list = create_iterable([train_df, val_df])
    vocab = create_vocab(vocab_list)
    with open(config.get("vocab_path"), "w") as f:
        json.dump(vocab, f)
    print("Vocab created")
