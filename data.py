"""Reading messages. Do not modify."""
import csv
import re


def tokenize(text):
    """Lowercase words and numbers, in order: "Win $1000 NOW!" -> ["win", "1000", "now"]."""
    return re.findall(r"[a-z0-9]+", text.lower())


def read(path):
    """A list of (text, label) pairs from a CSV with label and text columns."""
    with open(path, newline="", encoding="utf-8") as f:
        return [(row["text"], row["label"]) for row in csv.DictReader(f)]


def load(path):
    """A list of (words, label) pairs."""
    return [(tokenize(text), label) for text, label in read(path)]
