import dill
import os
import time

# Reduce TensorFlow C++ logging noise before zfit/tensorflow import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "3")
os.environ.setdefault("AUTOGRAPH_VERBOSITY", "0")

import numpy as np
import tensorflow as tf
import zfit

from build_model_from_text import build_model_from_card, parse_model_card
from model_io import load_fit_model
from analysis_core import configure_runtime, run_analysis
from analysis_overrides import apply_parameter_overrides
from analyze_plotting import plot_summary_artifacts


# Also silence python-side TensorFlow and absl warning emitters.
tf.get_logger().setLevel("ERROR")
try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity("error")
except Exception:
    pass

try:
    tf.config.optimizer.set_experimental_options({"loop_optimization": False})
except Exception:
    pass


def _load_analysis_model(model_file=None, input_card=None):
    if model_file is not None:
        return load_fit_model(os.path.abspath(model_file))

    card_path = os.path.abspath(input_card)
    card = parse_model_card(card_path)
    return build_model_from_card(card, os.path.dirname(card_path))


def _print_toy_summary(summary, is_observed_fit=False):
    poi_label = summary.get("poi_name", "poi")
    poi_fit = summary.get("poi_fit")
    poi_unc = summary.get("poi_unc_hesse")
    fit_text = f"{poi_fit:.3g}" if poi_fit is not None else "n/a"
    unc_text = f"{poi_unc:.3g}" if poi_unc is not None else "n/a"
    status_text = "valid" if summary['valid'] else "invalid"
    if summary.get("asimov_fit") or summary.get("toy_plot", {}).get("asimov"):
        label = "Asimov data"
    elif is_observed_fit or summary.get("observed_fit") or summary.get("toy_plot", {}).get("observed"):
        label = "Observed data"
    else:
        label = f"Toy {summary['toy']:3d}"
    print(
        f"{label}: {status_text:<7}, "
        f"{poi_label}={fit_text:<10} +- {unc_text:<10}, "
        f"time={summary.get('toy_time_s', float('nan')):.4f}s"
    )
    # if "count" in summary:
    #     print(f"  Toy count: {summary['count']}")
    if "poi_hat" in summary:
        print(f"  POI ({summary['poi_name']}) profiled best fit: {summary['poi_hat']:.6f}")
        print(
            f"  POI scan range: [{summary['poi_scan_low']:.6f}, {summary['poi_scan_high']:.6f}] "
            f"with {summary['poi_scan_points']} points"
        )
    if "cls_observed" in summary:
        print(f"  CLs observed upper limit: {summary['cls_observed']:.4f}")
    if "cls_scan_points" in summary:
        print(f"  CLs scan points: {summary['cls_scan_points']}")
    if "yield_upper_limit" in summary:
        print(f"  Yield upper limit: {summary['yield_upper_limit']:.4f}")
    if "cls_error" in summary:
        print(f"  CLs failed: {summary['cls_error']}")


def _save_analysis_snapshot(output_pkl, fit_model, summaries, args):
    payload = {
        "format": "analyze_model_snapshot_v1",
        "fit_model": fit_model,
        "input_data": fit_model.data,
        "summaries": summaries,
        "config": {
            "model_file": args.model_file,
            "input_card": args.input_card,
            "toys": args.toys,
            "fit_mode": args.fit_mode,
            "binned_bins": args.binned_bins,
            "graph_mode": args.graph_mode,
            "cls_alpha": args.cls,
            "signal_strength": args.signal_strength,
            "scan_max": args.scan_max,
            "cls_scan_points": args.cls_scan_points,
            "profile_scan": args.profile_scan,
            "poi_name": args.poi_name,
            "promote_poi": args.promote_poi,
            "poi_scan_points": args.poi_scan_points,
            "poi_scan_max": args.poi_scan_max,
            "set_parameters": args.set_parameters,
            "freeze_parameters": args.freeze_parameters,
            "set_parameter_ranges": args.set_parameter_ranges,
        },
    }

    output_path = os.path.abspath(output_pkl)
    with open(output_path, "wb") as handle:
        dill.dump(payload, handle)
    return output_path



def run_analysis_cli(args):
    fit_model = _load_analysis_model(model_file=args.model_file, input_card=args.input_card)
    apply_parameter_overrides(
        fit_model,
        set_values_spec=args.set_parameters,
        set_ranges_spec=args.set_parameter_ranges,
        freeze_spec=args.freeze_parameters,
    )

    zfit.settings.set_seed(args.seed)

    has_observed_data = hasattr(fit_model, "data") and fit_model.data is not None
    if args.toys is None:
        use_observed_data = has_observed_data
        use_asimov_data = False
        n_toys = 1
    elif args.toys == -1:
        use_observed_data = False
        use_asimov_data = True
        n_toys = 1
    elif args.toys < -1:
        raise ValueError("Only --toys -1 is supported as a special Asimov mode")
    else:
        use_observed_data = False
        use_asimov_data = False
        n_toys = int(args.toys)

    configure_runtime(args.graph_mode, fit_model, n_toys)
    total_start = time.perf_counter()
    summaries = run_analysis(
        fit_model,
        toys=n_toys,
        use_observed_data=use_observed_data,
        use_asimov_data=use_asimov_data,
        cls_alpha=args.cls,
        signal_strength=args.signal_strength,
        scan_max=args.scan_max,
        fit_mode=args.fit_mode,
        binned_bins=args.binned_bins,
        cls_scan_points=args.cls_scan_points,
        profile_scan=args.profile_scan,
        poi_name=args.poi_name,
        promote_poi=args.promote_poi,
        poi_scan_points=args.poi_scan_points,
        poi_scan_max=args.poi_scan_max,
        progress_callback=_print_toy_summary,
    )
    total_time_s = time.perf_counter() - total_start

    print(f"Analyzed model: {fit_model.model.name}")

    if args.plot:
        plot_summary_artifacts(
            summaries=summaries,
            fit_model=fit_model,
            plot_dir=os.path.abspath(args.plot_dir),
            binned_bins=args.binned_bins,
        )
        print(f"Saved plots to: {os.path.abspath(args.plot_dir)}")

    if summaries:
        print(f"Average time per toy: {total_time_s / len(summaries):.4f}s")
    print(f"Total execution time: {total_time_s:.4f}s")

    output_pkl=args.output_pkl
    if output_pkl is None: output_pkl = f'analysis_output_{args.seed}.pkl'
    snapshot_path = _save_analysis_snapshot(
        output_pkl=output_pkl,
        fit_model=fit_model,
        summaries=summaries,
        args=args,
    )
    print(f"Saved analysis snapshot to: {snapshot_path}")
