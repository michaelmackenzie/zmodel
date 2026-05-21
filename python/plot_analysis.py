#!/usr/bin/env python3
import argparse
import os
import sys
import dill

# Allow importing project modules when running from zmodel.
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from zmodel.analyze_plotting import plot_summary_artifacts


def _load_snapshot(snapshot_path):
    with open(snapshot_path, "rb") as handle:
        payload = dill.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Snapshot payload is not a dictionary")

    fit_model = payload.get("fit_model")
    summaries = payload.get("summaries")
    if fit_model is None:
        raise ValueError("Snapshot is missing 'fit_model'")
    if summaries is None:
        raise ValueError("Snapshot is missing 'summaries'")

    return payload, fit_model, summaries


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis plots from a saved analysis snapshot (.pkl)"
    )
    parser.add_argument("snapshot_file", help="Path to analysis snapshot pickle")
    parser.add_argument(
        "--plot-dir",
        default="plots_from_snapshot",
        help="Output directory for generated plots",
    )
    parser.add_argument(
        "--binned-bins",
        type=int,
        default=None,
        help="Override number of bins used for unbinned overlay histograms",
    )
    args = parser.parse_args()

    snapshot_path = os.path.abspath(args.snapshot_file)
    payload, fit_model, summaries = _load_snapshot(snapshot_path)

    if not summaries:
        print(f"Snapshot contains no summaries: {snapshot_path}")
        return

    config = payload.get("config", {})
    default_bins = int(config.get("binned_bins", 40))
    binned_bins = int(args.binned_bins) if args.binned_bins is not None else default_bins

    plot_dir = os.path.abspath(args.plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    plot_summary_artifacts(
        summaries=summaries,
        fit_model=fit_model,
        plot_dir=plot_dir,
        binned_bins=binned_bins,
    )

    print(f"Loaded snapshot: {snapshot_path}")
    print(f"Generated plots from {len(summaries)} fit summary entries")
    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()
