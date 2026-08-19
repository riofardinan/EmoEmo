"""Entry point.

  python main.py --config exps/finetune_B0-I7.json
  python main.py --config exps/finetune_B0-I7.json --seed 1994 --device cuda:1

Any config field can be overridden from the command line.
"""

import argparse

from trainer import train
from utils.config import load_config
from utils.factory import available_methods


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-label class-incremental emotion decoding on GoEmotions."
    )
    parser.add_argument("--config", type=str, default="./exps/finetune_B0-I7.json",
                        help="JSON settings file.")
    parser.add_argument("--method", type=str, default=None,
                        help=f"Override the method. Available: {available_methods()}")
    parser.add_argument("--protocol", type=str, default=None,
                        help="Override the protocol, e.g. B0-I7, B16-I3.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--init_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--fp16", type=lambda v: v.lower() == "true", default=None)
    # Replay buffer budget (ER / RS / PRS / OCDM only).
    parser.add_argument("--fixed_memory", type=lambda v: v.lower() == "true",
                        default=None,
                        help="True routes ER through memory_per_class; False "
                             "gives every buffer method the same memory_size.")
    parser.add_argument("--memory_size", type=int, default=None)
    # Results directory suffix, so a variant does not overwrite the base run.
    parser.add_argument("--tag", type=str, default=None)
    # Emotional relation graph (AGCN / AESL only) — see utils/graph.py.
    parser.add_argument("--adj_estimator", type=str, default=None,
                        choices=["raw", "shrink", "shrink_pool", "empty"])
    parser.add_argument("--adj_alpha", type=float, default=None,
                        help="Shrinkage strength; negative selects method of moments.")
    parser.add_argument("--adj_pool_old", type=lambda v: v.lower() == "true",
                        default=None)
    parser.add_argument("--adj_subsample", type=float, default=None,
                        help="Fraction of rows the graph is estimated from; "
                             "training is unaffected.")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_config(args.config, overrides)
    train(cfg)


if __name__ == "__main__":
    main()
