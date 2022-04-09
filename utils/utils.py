import nltk

nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
import re


def clean_text(text):
    punctuation_pattern = "[\.\?\!\,\:\;\-\_\[\]\(\)\{\}'\"\%\*\#\^]+"
    text_cleaned = re.sub(punctuation_pattern, "", text)
    text_cleaned = text_cleaned.lower()
    word_tokens = word_tokenize(text_cleaned)
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    # print(stop_words)
    # text_cleaned = " ".join([ps.stem(w) for w in word_tokens if not w in stop_words])
    # text_cleaned = " ".join([w for w in word_tokens if not w in stop_words])
    text_cleaned = " ".join([w for w in word_tokens])

    return text_cleaned


def white_space_tokenizer(text):
    return text.split(" ")
