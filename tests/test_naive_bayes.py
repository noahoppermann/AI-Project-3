import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if os.environ.get("SUBMISSION_DIR"):
    sys.path.insert(0, os.path.abspath(os.environ["SUBMISSION_DIR"]))

import naive_bayes as nb  # noqa: E402
import data               # noqa: E402
from model import Model   # noqa: E402

# These tests are a sample. Grading uses more, including the edge cases.
HINT = " Check the Requirements section of the README."

# The worksheet model, built by hand so the probability and classification
# tests do not depend on train.
WORKSHEET = {"spam": (2, {"win": 1, "cash": 2, "now": 1, "free": 1, "prize": 1}),
             "ham": (3, {"lunch": 2, "now": 2, "free": 1, "for": 1, "call": 1, "me": 1})}


def build(counts):
    m = Model()
    for c, (n, words) in counts.items():
        m.class_counts[c] = n
        m.word_counts[c].update(words)
        m.total_words[c] = sum(words.values())
        m.vocabulary.update(words)
    return m


def load(name):
    return data.load(os.path.join(ROOT, "data", name + ".csv"))


class NaiveBayesTest(unittest.TestCase):
    longMessage = False

    def close(self, got, want, what):
        self.assertIsInstance(got, (int, float), f"{what} returned {got!r}; expected a number.")
        self.assertAlmostEqual(got, want, places=9,
                               msg=f"{what} returned {got!r}; expected {want!r}." + HINT)

    def same(self, got, want, what):
        self.assertEqual(got, want, f"{what} gave {got!r}; expected {want!r}." + HINT)


class TestTraining(NaiveBayesTest):

    def setUp(self):
        self.model = nb.train(load("worksheet"))

    def test_class_counts(self):
        self.same(dict(self.model.class_counts), {"spam": 2, "ham": 3},
                  "class_counts on the worksheet")

    def test_word_counts(self):
        for c, (_, words) in WORKSHEET.items():
            self.same(dict(self.model.word_counts[c]), words,
                      f"word_counts['{c}'] on the worksheet")

    def test_total_words(self):
        self.same(dict(self.model.total_words), {"spam": 6, "ham": 8},
                  "total_words on the worksheet")

    def test_vocabulary(self):
        self.same(self.model.vocabulary,
                  {"call", "cash", "for", "free", "lunch", "me", "now", "prize", "win"},
                  "vocabulary on the worksheet")

class TestProbabilities(NaiveBayesTest):

    def setUp(self):
        self.model = build(WORKSHEET)

    def test_prior(self):
        self.close(nb.prior(self.model, "spam"), 2 / 5, "prior(model, 'spam')")
        self.close(nb.prior(self.model, "ham"), 3 / 5, "prior(model, 'ham')")

    def test_likelihood(self):
        self.close(nb.likelihood(self.model, "cash", "spam"), 3 / 15,
                   "likelihood(model, 'cash', 'spam')")
        self.close(nb.likelihood(self.model, "cash", "ham"), 1 / 17,
                   "likelihood(model, 'cash', 'ham')")

class TestClassification(NaiveBayesTest):

    def setUp(self):
        self.model = build(WORKSHEET)

    def test_log_score(self):
        words = ["free", "cash", "now"]
        self.close(nb.log_score(self.model, words, "spam"),
                   math.log(2 / 5) + math.log(2 / 15) + math.log(3 / 15) + math.log(2 / 15),
                   "log_score(model, ['free', 'cash', 'now'], 'spam')")
        self.close(nb.log_score(self.model, words, "ham"),
                   math.log(3 / 5) + math.log(2 / 17) + math.log(1 / 17) + math.log(3 / 17),
                   "log_score(model, ['free', 'cash', 'now'], 'ham')")

    def test_classify_worksheet(self):
        for words, label in load("worksheet_test"):
            self.same(nb.classify(self.model, words), label, f"classify(model, {words})")

class TestEvaluation(NaiveBayesTest):

    def test_worksheet_accuracy(self):
        self.close(nb.accuracy(nb.train(load("worksheet")), load("worksheet_test")), 1.0,
                   "accuracy on the worksheet")

    def test_sms_test_accuracy(self):
        self.close(nb.accuracy(nb.train(load("sms_train")), load("sms_test")), 0.95,
                   "accuracy on sms_test")

if __name__ == "__main__":
    unittest.main()
