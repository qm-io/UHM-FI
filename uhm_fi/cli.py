"""Safe command-line entry points for pre-training and downstream runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .annotation import COARSE_LABELS, FINE_LABELS
from .config import DownstreamConfig, PretrainingConfig
from .training import run_downstream, run_pretraining


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def pretrain_main() -> None:
    parser = argparse.ArgumentParser(
        description="UHM-FI image-text pre-training. Real training requires --execute."
    )
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Start the configured real pre-training job.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-validation-steps", type=int)
    parser.add_argument("--log-every-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging for this run.",
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-tag", action="append", dest="wandb_tags")
    parser.add_argument("--wandb-notes")
    args = parser.parse_args()

    config = PretrainingConfig.from_yaml(args.config)
    if args.device:
        config.trainer.device = args.device
    if args.resume:
        config.trainer.resume = str(Path(args.resume).expanduser().resolve())
    if args.epochs is not None:
        config.trainer.epochs = args.epochs
    if args.batch_size is not None:
        config.data.loader.batch_size = args.batch_size
    if args.gradient_accumulation is not None:
        config.trainer.gradient_accumulation = args.gradient_accumulation
    if args.max_train_steps is not None:
        config.trainer.max_train_steps = args.max_train_steps
    if args.max_validation_steps is not None:
        config.trainer.max_validation_steps = args.max_validation_steps
    if args.log_every_steps is not None:
        config.trainer.log_every_steps = args.log_every_steps
    if args.seed is not None:
        config.trainer.seed = args.seed
    if args.output_dir:
        config.trainer.output_dir = str(Path(args.output_dir).expanduser().resolve())
    if args.wandb:
        config.wandb.enabled = True
    if args.wandb_project:
        config.wandb.project = args.wandb_project
        config.wandb.enabled = True
    if args.wandb_entity:
        config.wandb.entity = args.wandb_entity
        config.wandb.enabled = True
    if args.wandb_run_name:
        config.wandb.run_name = args.wandb_run_name
        config.wandb.enabled = True
    if args.wandb_mode:
        config.wandb.mode = args.wandb_mode
        config.wandb.enabled = args.wandb_mode != "disabled"
    if args.wandb_tags:
        config.wandb.tags = tuple(args.wandb_tags)
        config.wandb.enabled = True
    if args.wandb_notes:
        config.wandb.notes = args.wandb_notes
        config.wandb.enabled = True
    config.validate()
    if not args.execute:
        _print(
            {
                "status": "configuration_valid",
                "training_started": False,
                "safety_gate": "Pass --execute to start real pre-training.",
                "config": config.to_dict(),
            }
        )
        return
    _print(run_pretraining(config))


def downstream_main() -> None:
    parser = argparse.ArgumentParser(
        description="UHM-FI downstream evaluation. Real training requires --execute."
    )
    parser.add_argument("--config", default="configs/downstream_rsna_linear.yaml")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Start the configured downstream training/evaluation job.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--resume",
        help="Resume downstream training from a new-format last.pt checkpoint.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-fraction", type=float)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging for this downstream run.",
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-tag", action="append", dest="wandb_tags")
    parser.add_argument("--wandb-notes")
    parser.add_argument(
        "--fixed-condition",
        choices=("bone", "breast", "chest"),
        help=(
            "Supply the task-level anatomy label required by CSAM ablations "
            "whose HMCS was removed."
        ),
    )
    args = parser.parse_args()

    config = DownstreamConfig.from_yaml(args.config)
    if args.device:
        config.trainer.device = args.device
    if args.checkpoint:
        config.pretrained_checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    if args.resume:
        config.trainer.resume = str(Path(args.resume).expanduser().resolve())
    if args.epochs is not None:
        config.trainer.epochs = args.epochs
    if args.train_fraction is not None:
        config.data.train_fraction = args.train_fraction
    if args.max_train_steps is not None:
        config.trainer.max_train_steps = args.max_train_steps
    if args.seed is not None:
        config.trainer.seed = args.seed
    if args.output_dir:
        config.trainer.output_dir = str(Path(args.output_dir).expanduser().resolve())
    if args.wandb:
        config.wandb.enabled = True
    if args.wandb_project:
        config.wandb.project = args.wandb_project
        config.wandb.enabled = True
    if args.wandb_entity:
        config.wandb.entity = args.wandb_entity
        config.wandb.enabled = True
    if args.wandb_run_name:
        config.wandb.run_name = args.wandb_run_name
        config.wandb.enabled = True
    if args.wandb_mode:
        config.wandb.mode = args.wandb_mode
        config.wandb.enabled = args.wandb_mode != "disabled"
    if args.wandb_tags:
        config.wandb.tags = tuple(args.wandb_tags)
        config.wandb.enabled = True
    if args.wandb_notes:
        config.wandb.notes = args.wandb_notes
        config.wandb.enabled = True
    if args.fixed_condition:
        fine_name = {
            "bone": "Musculoskeletal",
            "breast": "Breast",
            "chest": "Chest",
        }[args.fixed_condition]
        coarse_name = {
            "bone": "Bone",
            "breast": "Breast",
            "chest": "Chest",
        }[args.fixed_condition]
        config.task.fixed_fine_labels = tuple(
            float(label == fine_name) for label in FINE_LABELS
        )
        config.task.fixed_coarse_labels = tuple(
            float(label == coarse_name) for label in COARSE_LABELS
        )
    config.validate()
    if not args.execute:
        _print(
            {
                "status": "configuration_valid",
                "training_started": False,
                "safety_gate": "Pass --execute to start downstream training.",
                "config": config.to_dict(),
            }
        )
        return
    _print(run_downstream(config))
