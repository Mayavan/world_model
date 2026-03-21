from __future__ import annotations

"""CLI for dataset-seeded interactive action rollouts."""

import argparse
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np

from src.interactive_rollout import InteractiveRolloutSession
from src.utils.io import ensure_dir, timestamp_dir
from src.utils.video import side_by_side


def _parse_action_input(raw: str, *, n_future_frames: int) -> int | list[int] | None:
    text = raw.strip().lower()
    if not text:
        return None
    if "," in text or " " in text:
        normalized = text.replace(",", " ")
        parts = [p for p in normalized.split() if p]
        actions = [int(p) for p in parts]
        if len(actions) not in {1, n_future_frames}:
            raise ValueError(
                f"Expected 1 or {n_future_frames} action ids, got {len(actions)}"
            )
        return actions
    return int(text)


def _save_rgb_image(path: Path, image: np.ndarray) -> None:
    imageio.imwrite(path, (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))


def run_cli(args: argparse.Namespace) -> None:
    session = InteractiveRolloutSession(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device,
        sampling_steps=args.sampling_steps,
    )
    seed = session.load_seed(args.sample_idx)

    out_dir = ensure_dir(args.out_dir or timestamp_dir("runs", name="interactive_rollout"))
    print(f"Loaded sample {seed.sample_idx} from {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Actions: discrete ids in [0, {session.num_actions - 1}]")
    print(
        f"Single-action input repeats across the {session.n_future_frames}-step future plan. "
        f"You can also enter {session.n_future_frames} comma-separated action ids."
    )
    print(f"Artifacts will be written to: {out_dir}")

    context_path = Path(out_dir) / "context_seed.png"
    _save_rgb_image(context_path, session.render_context_strip())
    print(f"Saved seed context strip: {context_path}")

    while True:
        raw = input("Action? [int | comma-seq | reset | q] > ").strip()
        if not raw:
            continue
        if raw.lower() in {"q", "quit", "exit"}:
            print("Exiting interactive rollout.")
            break
        if raw.lower() in {"reset", "r"}:
            session.reset()
            reset_path = Path(out_dir) / "context_reset.png"
            _save_rgb_image(reset_path, session.render_context_strip())
            print(f"Reset to seed. Context strip saved to: {reset_path}")
            continue

        try:
            action = _parse_action_input(raw, n_future_frames=session.n_future_frames)
            if action is None:
                continue
            result = session.step(action)
        except Exception as exc:  # noqa: BLE001
            print(f"Input/rollout error: {exc}")
            continue

        step_idx = int(result["step_index"])
        pred_frame = np.asarray(result["predicted_frame"])
        pred_path = Path(out_dir) / f"step_{step_idx:04d}_pred.png"
        _save_rgb_image(pred_path, pred_frame)

        gt_frame = result.get("gt_frame")
        compare_path = None
        if gt_frame is not None:
            compare = side_by_side(np.asarray(gt_frame), pred_frame)
            compare_path = Path(out_dir) / f"step_{step_idx:04d}_gt_vs_pred.png"
            _save_rgb_image(compare_path, compare)

        context_strip = session.render_context_strip()
        context_path = Path(out_dir) / f"step_{step_idx:04d}_context.png"
        _save_rgb_image(context_path, context_strip)

        print(
            f"step={step_idx} applied_action={result['applied_action']} "
            f"action_plan={result['action_plan']} pred={pred_path.name}"
        )
        if compare_path is not None:
            print(
                f"  gt_action={result['gt_action']} matches_recorded_action="
                f"{result['matches_recorded_action']} compare={compare_path.name}"
            )
        print(f"  context={context_path.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive dataset-seeded world-model rollout")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", required=True, help="Path to offline dataset directory")
    parser.add_argument("--sample_idx", type=int, default=0, help="Dataset sample index to seed")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu or cuda")
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,
        help="Override sampling steps used by the latent flow model",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Directory for saved rollout artifacts (default: timestamped runs/ dir)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_cli(args)


if __name__ == "__main__":
    main()
