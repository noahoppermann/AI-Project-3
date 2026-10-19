"""Class for holding model. Do not modify."""
from collections import Counter, defaultdict


class Model:
    def __init__(self):
        self.class_counts = Counter()            # label -> messages with that label
        self.word_counts = defaultdict(Counter)  # label -> word -> times seen in that label
        self.total_words = Counter()             # label -> words seen in that label
        self.vocabulary = set()                  # every word seen in training

    @property
    def labels(self):
        return sorted(self.class_counts)

    @property
    def num_examples(self):
        return sum(self.class_counts.values())
