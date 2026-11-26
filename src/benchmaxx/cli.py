"""Command line: read decisions with the LLM, train a model, evaluate it, predict with it.

    benchmaxx read     --decisions train.jsonl --out features/train
    benchmaxx train    --features features/train --out models/mine [--out-of-domain other=features/other ...]
    benchmaxx evaluate --model models/mine --features features/test
    benchmaxx predict  --model models/mine --decisions new.jsonl --out predictions.jsonl
"""

import argparse
import json
from pathlib import Path

from benchmaxx.features import FeatureSet
from benchmaxx.metrics import evaluate
from benchmaxx.model import Benchmaxx
from benchmaxx.schema import read_decisions
from benchmaxx.training import TrainingConfig, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="benchmaxx", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    read = commands.add_parser("read", help="read decisions with the frozen LLM and store their features")
    read.add_argument("--decisions", required=True, type=Path, help="JSON Lines file of decisions")
    read.add_argument("--out", required=True, type=Path, help="directory for the features")
    read.add_argument("--llm", default="Qwen/Qwen3.5-4B")
    read.add_argument("--layers", default="12,20,28,32", help="hidden-state indices to keep")

    fit = commands.add_parser("train", help="train a correction head by cross-fitting, and optionally a domain gate")
    fit.add_argument("--features", required=True, type=Path, nargs="+", help="feature directories to train on")
    fit.add_argument("--out", required=True, type=Path, help="directory for the model")
    fit.add_argument("--config", type=Path, help="JSON file of TrainingConfig fields")
    fit.add_argument("--out-of-domain", nargs="*", default=[], metavar="GROUP=DIR",
                     help="features of domains not trained on; fits a domain gate (one group per name)")

    score = commands.add_parser("evaluate", help="accuracy, KL, ECE and AURC on labelled features")
    score.add_argument("--model", required=True, type=Path)
    score.add_argument("--features", required=True, type=Path)

    predict = commands.add_parser("predict", help="answer new decisions")
    predict.add_argument("--model", required=True, type=Path)
    predict.add_argument("--decisions", required=True, type=Path)
    predict.add_argument("--out", required=True, type=Path, help="JSON Lines file of predictions")

    args = parser.parse_args(argv)
    {"read": _read, "train": _train, "evaluate": _evaluate, "predict": _predict}[args.command](args)


def _read(args: argparse.Namespace) -> None:
    from benchmaxx.reader import LLMReader  # loads transformers only when needed
    reader = LLMReader(args.llm, layers=[int(x) for x in args.layers.split(",")])
    reader.read(read_decisions(args.decisions)).save(args.out)
    print(f"features written to {args.out}")


def _train(args: argparse.Namespace) -> None:
    config = TrainingConfig(**json.loads(args.config.read_text())) if args.config else TrainingConfig()
    features = FeatureSet.concat([FeatureSet.load(d) for d in args.features])
    groups: dict[str, list[FeatureSet]] = {}
    for item in args.out_of_domain:
        name, _, directory = item.partition("=")
        groups.setdefault(name, []).append(FeatureSet.load(directory))
    model = train(features, config, out_of_domain=groups or None)
    model.save(args.out)
    print(f"model written to {args.out}")


def _evaluate(args: argparse.Namespace) -> None:
    model = Benchmaxx.load(args.model)
    features = FeatureSet.load(args.features)
    predictions = model.predict_features(features)
    gold = [features.gold[int(features.offsets[i]) : int(features.offsets[i + 1])].numpy()
            for i in range(len(features))]
    print(json.dumps(evaluate([p.probs for p in predictions], gold), indent=2))


def _predict(args: argparse.Namespace) -> None:
    from benchmaxx.reader import LLMReader
    model = Benchmaxx.load(args.model)
    decisions = read_decisions(args.decisions)
    reader = LLMReader(model.llm["model"], layers=model.llm["layers"])
    with args.out.open("w") as f:
        for decision, prediction in zip(decisions, model.predict(decisions, reader), strict=True):
            f.write(json.dumps({"case_id": decision.case_id, "question": decision.question.text,
                                **prediction.__dict__}) + "\n")
    print(f"predictions written to {args.out}")


if __name__ == "__main__":
    main()
