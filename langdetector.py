import sys
import re
from os import listdir

ARG_ERROR = "Error wrong arguments. \nUsage: python " + sys.argv[0] + " train <N-gram size> <train_folder>\n" \
            + "or: python " + sys.argv[0] + " detect <N-gram size> <file>"

UNIGRAM_WEIGHTS_EXTENSION = "-unigram-weights.txt"
BIGRAM_WEIGHTS_EXTENSION = "-bigram-weights.txt"
WEIGHT_ENCODING = "utf-8"
SPACE = " "
NGRAM_WEIGHT_SEPARATOR = "\t"


def preprocess_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[0-9]", SPACE, text)
    text = text.replace(SPACE, "")
    return text


def save_weights(weights: dict, file_name: str):
    with open(file_name, "w", encoding=WEIGHT_ENCODING) as file:
        for ngram, count in weights.items():
            file.write(ngram + NGRAM_WEIGHT_SEPARATOR + str(count) + "\n")


def weight_unigrams(text: str) -> dict:
    unigrams = {}
    for char in text:
        unigrams[char] = unigrams.get(char, 0) + 1
    return unigrams


def weight_bigrams(text: str) -> dict:
    bigrams = {}
    for i in range(0, len(text)):
        if i + 1 < len(text):
            bigram = text[i] + text[i + 1]
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
    return bigrams


def train(ngram_size, train_data_path: str):
    for file_name in listdir(train_data_path):
        with open(train_data_path + "/" + file_name, "r", encoding="iso-8859-1", errors="ignore") as file:
            text = preprocess_text(file.read())
            if ngram_size == 1:
                weights = weight_unigrams(text)
                weights_file_name = file_name + UNIGRAM_WEIGHTS_EXTENSION
            else:
                weights = weight_bigrams(text)
                weights_file_name = file_name + BIGRAM_WEIGHTS_EXTENSION
            save_weights(weights, weights_file_name)


def load_weights_file(file_name: str) -> dict:
    weights = {}
    with open(file_name, "r", encoding=WEIGHT_ENCODING) as file:
        for line in file.readlines():
            aux_list = line.split(NGRAM_WEIGHT_SEPARATOR)
            if len(aux_list) == 2 and "" not in aux_list:
                weights[aux_list[0]] = int(aux_list[1])
    return weights


def load_trained_weights(extension: str) -> dict:
    weights = {}
    files = listdir()
    for file_name in files:
        extension_start = file_name.find(extension)
        if extension_start is not -1:
            language = file_name[:extension_start]
            weights[language] = load_weights_file(file_name)
    return weights


def detect_with_unigrams(text: str) -> str:
    text_unigrams = weight_unigrams(text)
    trained_unigrams = load_trained_weights(UNIGRAM_WEIGHTS_EXTENSION)
    # TODO pearson correlation comparison
    return "No yet implemented"


def detect_with_bigrams(text: str) -> str:
    text_bigrams = weight_bigrams(text)
    trained_bigrams = load_trained_weights(UNIGRAM_WEIGHTS_EXTENSION)
    # TODO pearson correlation comparison
    return "no yet implemented"


def detect_language(ngram_size, file: str) -> str:
    with open(file, "r") as file_d:
        text = preprocess_text(file_d.read())
        if ngram_size == 1:
            return detect_with_unigrams(text)
        else:
            return detect_with_bigrams(text)


def main(*args):
    if len(args) == 3:
        if args[0] == "train":
            train(int(args[1]), args[2])
        elif args[0] == "detect":
            print(detect_language(int(args[1]), args[2]))
        else:
            print(ARG_ERROR)
    else:
        print(ARG_ERROR)


if __name__ == '__main__':
    main(*sys.argv[1:])
