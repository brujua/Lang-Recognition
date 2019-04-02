import sys
import re
import numpy
from os import listdir
from typing import List

ARG_ERROR = "Error wrong arguments. \nUsage: python " + sys.argv[0] + " train <N-gram size> <train_folder>\n" \
            + "or: python " + sys.argv[0] + " detect <N-gram size> <test-file> <solutions-file>"

UNIGRAM_WEIGHTS_EXTENSION = "-unigram-weights.txt"
BIGRAM_WEIGHTS_EXTENSION = "-bigram-weights.txt"
WEIGHT_ENCODING = "utf-8"
SPACE = " "
NGRAM_WEIGHT_SEPARATOR = "\t"
SOLUTION_FILE = "solution"


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
    total_count = len(text)
    for char in text:
        unigrams[char] = unigrams.get(char, 0) + 1
    # adjust to relative frequency
    return {unigram: (count / total_count) for unigram, count in unigrams.items()}


def weight_bigrams(text: str) -> dict:
    bigrams = {}
    total_count = 0
    for i in range(0, len(text)):
        if i + 1 < len(text):
            bigram = text[i] + text[i + 1]
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
            total_count += 1
    # adjust to relative frequency
    return {bigram: (count / total_count) for bigram, count in bigrams.items()}


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
            line = line.replace("\n", "")
            aux_list = line.split(NGRAM_WEIGHT_SEPARATOR)
            if len(aux_list) == 2 and "" not in aux_list:
                weights[aux_list[0]] = float(aux_list[1])
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


def weight_with_defined_ngrams(text: str, ngrams: List[str]) -> dict:
    weights = {}
    total_count = 0
    for ngram in ngrams:
        weights[ngram] = text.count(ngram)
        total_count += 1
    return {ngram: (count / total_count) for ngram, count in weights.items()}


def calculate_correlation(dict1: dict, dict2: dict):
    # this function is responsible of converting the dictionary representation to the lists needed for correlation
    list1 = []
    list2 = []
    for key in dict1.keys():
        list1.append(dict1[key])
        list2.append(dict2.get(key, 0))
    return numpy.corrcoef(list1, list2)[0][1]


def detect_languages(ngram_size, file: str) -> List[str]:
    result = []
    with open(file, "r") as file_d:
        lines = file_d.readlines()
        for line in lines:
            text = preprocess_text(line)
            if ngram_size == 1:
                trained_ngrams = load_trained_weights(UNIGRAM_WEIGHTS_EXTENSION)
            else:
                trained_ngrams = load_trained_weights(BIGRAM_WEIGHTS_EXTENSION)
            text_unigrams = {}
            highest_corr = 0
            detected_language = None
            for language in trained_ngrams.keys():
                text_unigrams[language] = weight_with_defined_ngrams(text, trained_ngrams[language].keys())
                correlation = calculate_correlation(trained_ngrams[language], text_unigrams[language])
                if correlation > highest_corr:
                    highest_corr = correlation
                    detected_language = language
            result.append(detected_language)
    return result


def load_solution(file_name: str) -> List[str]:
    solutions = []
    with open(file_name, "r") as file:
        for line in file.readlines():
            data = line.split()
            if len(data) == 2:
                solutions.append(data[1])
    return solutions


def compare_solutions(expected: List[str], actuals: List[str]) -> str:
    total = len(expected)
    correct = 0
    for i in range(0, len(expected)):
        if expected[i] == actuals[i]:
            correct += 1
    return "Accuracy = " + '%.2f' % ((correct / total) * 100) + "%"


def main(*args):
    if len(args) >= 3:
        if args[0] == "train":
            train(int(args[1]), args[2])
        elif args[0] == "detect" and len(args) == 4:
            detected = detect_languages(int(args[1]), args[2])
            solution = load_solution(args[3])
            print(compare_solutions(solution, detected))
        else:
            print(ARG_ERROR)
    else:
        print(ARG_ERROR)


if __name__ == '__main__':
    main(*sys.argv[1:])
