import argparse
import math
import os

import data
import naive_bayes as nb

HERE = os.path.dirname(os.path.abspath(__file__))


def path(name):
    return os.path.join(HERE, "data", name + ".csv")


def worksheet(args):
    model = nb.train(data.load(path("worksheet")))
    print("label  messages  words  prior")
    for c in model.labels:
        print(f"{c:6} {model.class_counts[c]:8} {model.total_words[c]:6}  "
              f"{nb.prior(model, c):.4f}")
    print(f"\nvocabulary: {len(model.vocabulary)} words\n")
    print("word    " + "".join(f"{'P(w|' + c + ')':>12}" for c in model.labels))
    for w in sorted(model.vocabulary):
        print(f"{w:8}" + "".join(f"{nb.likelihood(model, w, c, args.k):12.4f}"
                                 for c in model.labels))
    print()
    for text, label in data.read(path("worksheet_test")):
        show(model, text, args.k, label)


def evaluate(args):
    model = nb.train(data.load(path("sms_train")))
    rows = data.read(path("sms_test"))
    print(f"accuracy on sms_test with k={args.k}: "
          f"{nb.accuracy(model, data.load(path('sms_test')), args.k):.3f}\n")
    print(f"{'actual':8}" + "".join(f"{'predicted ' + c:>16}" for c in model.labels))
    predicted = [(nb.classify(model, data.tokenize(text), args.k), label)
                 for text, label in rows]
    for actual in model.labels:
        print(f"{actual:8}" + "".join(f"{predicted.count((p, actual)):16}"
                                      for p in model.labels))
    wrong = [(text, label, p) for (text, label), (p, _) in zip(rows, predicted) if p != label]
    if wrong:
        print("\nMistakes:")
    for text, label, p in wrong:
        print(f"  {label} classified as {p}: {text}")


def classify(args):
    model = nb.train(data.load(path("sms_train")))
    show(model, args.text, args.k)


def words(args):
    model = nb.train(data.load(path("sms_train")))
    for c in model.labels:
        others = [o for o in model.labels if o != c]
        ratio = {w: min(nb.likelihood(model, w, c, args.k) / nb.likelihood(model, w, o, args.k)
                        for o in others)
                 for w in model.vocabulary}
        top = sorted(model.vocabulary, key=lambda w: (-ratio[w], w))[:args.n]
        print(f"Most {c}-like words: " + ", ".join(f"{w} ({ratio[w]:.1f}x)" for w in top))


def show(model, text, k, label=None):
    words = data.tokenize(text)
    scores = {c: nb.log_score(model, words, c, k) for c in model.labels}
    guess = nb.classify(model, words, k)
    unknown = [w for w in words if w not in model.vocabulary]
    print(f'"{text}"' + (f"  (actually {label})" if label else ""))
    for c, s in scores.items():
        print(f"  log-score {c:6} {s:9.3f}{'   <- classified' if c == guess else ''}")
    if unknown:
        print(f"  skipped unknown words: {', '.join(unknown)}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=float, default=1, help="smoothing strength (default 1)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("worksheet").set_defaults(fn=worksheet)
    sub.add_parser("evaluate").set_defaults(fn=evaluate)
    c = sub.add_parser("classify")
    c.add_argument("text")
    c.set_defaults(fn=classify)
    w = sub.add_parser("words")
    w.add_argument("--n", type=int, default=10)
    w.set_defaults(fn=words)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
