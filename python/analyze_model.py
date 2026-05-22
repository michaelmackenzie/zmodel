import dill
import json
import multiprocessing as mp
import os
import time

# Reduce TensorFlow C++ logging noise before zfit/tensorflow import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "3")
os.environ.setdefault("AUTOGRAPH_VERBOSITY", "0")

import numpy as np
import tensorflow as tf
import zfit

from zmodel.build_model_from_text import build_model_from_card, parse_model_card
from zmodel.model_io import load_fit_model
from zmodel.analysis_core import configure_runtime, run_analysis
from zmodel.analysis_overrides import apply_parameter_overrides
from zmodel.analyze_plotting import plot_summary_artifacts


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


def _print_dataset_summary(summary, is_observed_fit=False):
    poi_label = summary.get("poi_name", "poi")
    poi_fit = summary.get("poi_fit")
    poi_unc = summary.get("poi_unc_hesse")
    fit_text = f"{poi_fit:.3g}" if poi_fit is not None else "n/a"
    unc_text = f"{poi_unc:.3g}" if poi_unc is not None else "n/a"
    status_text = "valid" if summary['valid'] else "invalid"
    if summary.get("asimov_fit") or summary.get("dataset_plot", {}).get("asimov"):
        label = "Asimov data"
    elif is_observed_fit or summary.get("observed_fit") or summary.get("dataset_plot", {}).get("observed"):
        label = "Observed data"
    else:
        label = f"Toy {summary['dataset_id']:3d}"
    print(
        f"{label}: {status_text:<7}, "
        f"{poi_label}={fit_text:<10} +- {unc_text:<10}, "
        f"time={summary.get('dataset_time_s', float('nan')):.4f}s"
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
    if "cls_scan_max" in summary:
        print(f"  CLs scan max: {summary['cls_scan_max']:.4g}")
    if "cls_expected_quantiles" in summary:
        q = summary["cls_expected_quantiles"]
        print(
            "  CLs expected (asymptotic, b-only fit): "
            f"2.5%={q['2.5%']:.4f}, 16%={q['16%']:.4f}, 50%={q['50%']:.4f}, "
            f"84%={q['84%']:.4f}, 97.5%={q['97.5%']:.4f}"
        )
    if "cls_expected_error" in summary:
        print(f"  CLs expected failed: {summary['cls_expected_error']}")
    if "yield_upper_limit" in summary:
        print(f"  Yield upper limit: {summary['yield_upper_limit']:.4f}")
    if "cls_error" in summary:
        print(f"  CLs failed: {summary['cls_error']}")
    if "feldman_cousins" in summary:
        fc = summary["feldman_cousins"]
        if isinstance(fc, dict):
            if "fc_interval" in fc:
                print(f"  Feldman-Cousins interval: {fc['fc_interval']}")
            elif "fc_status" in fc:
                print(f"  Feldman-Cousins: {fc['fc_status']}")
            else:
                print(f"  Feldman-Cousins: {fc}")


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
            "cls_smart_scan": args.cls_smart_scan,
            "profile_scan": args.profile_scan,
            "poi_name": args.poi_name,
            "promote_poi": args.promote_poi,
            "poi_scan_points": args.poi_scan_points,
            "poi_scan_max": args.poi_scan_max,
            "feldman_cousins": args.feldman_cousins,
            "feldman_cousins_scan_points": args.fc_scan_points,
            "feldman_cousins_n_toys": args.fc_toys,
            "feldman_cousins_scan_max": args.fc_scan_max,
            "report_file": args.report_file,
            "nll_scan_points": args.nll_scan_points,
            "set_parameters": args.set_parameters,
            "freeze_parameters": args.freeze_parameters,
            "set_parameter_ranges": args.set_parameter_ranges,
        },
    }

    output_path = os.path.abspath(output_pkl)
    with open(output_path, "wb") as handle:
        dill.dump(payload, handle)
    return output_path


def _current_data_mode(use_observed_data, use_asimov_data):
    if use_observed_data:
        return "observed"
    if use_asimov_data:
        return "asimov"
    return "toy"


def _checkpoint_mismatches(checkpoint, expected):
    mismatches = []
    for key, expected_value in expected.items():
        if key not in checkpoint:
            mismatches.append((key, "<missing>", expected_value))
            continue
        if checkpoint.get(key) != expected_value:
            mismatches.append((key, checkpoint.get(key), expected_value))
    return mismatches


def _distribution_summary(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
    }


def _build_ensemble_evaluation_report(summaries, total_time_s):
    report = {
        "n_datasets": int(len(summaries)),
        "runtime": {
            "total_time_s": float(total_time_s),
            "average_time_s": float(total_time_s / len(summaries)) if summaries else None,
        },
    }

    if not summaries:
        return report

    valid_flags = [bool(summary.get("valid", False)) for summary in summaries]
    n_valid = int(sum(valid_flags))
    report["fit_quality"] = {
        "n_valid": n_valid,
        "n_invalid": int(len(summaries) - n_valid),
        "valid_fraction": float(n_valid / len(summaries)),
        "invalid_fraction": float((len(summaries) - n_valid) / len(summaries)),
    }

    report["poi_name"] = summaries[0].get("poi_name", "poi")

    poi_fits = [summary.get("poi_fit") for summary in summaries]
    poi_unc = [summary.get("poi_unc_hesse") for summary in summaries]
    poi_pulls = [summary.get("poi_pull") for summary in summaries]
    report["poi_fit"] = _distribution_summary(poi_fits)
    report["poi_unc_hesse"] = _distribution_summary(poi_unc)
    report["poi_pull"] = _distribution_summary(poi_pulls)

    coverage_values = []
    for summary in summaries:
        truth = summary.get("poi_true")
        fit = summary.get("poi_fit")
        unc = summary.get("poi_unc_hesse")
        if truth is None or fit is None or unc is None:
            continue
        truth = float(truth)
        fit = float(fit)
        unc = float(unc)
        if not (np.isfinite(truth) and np.isfinite(fit) and np.isfinite(unc) and unc > 0.0):
            continue
        coverage_values.append((truth, fit, unc))

    if coverage_values:
        within_1sigma = 0
        within_95pct = 0
        for truth, fit, unc in coverage_values:
            if abs(fit - truth) <= unc:
                within_1sigma += 1
            if abs(fit - truth) <= 1.96 * unc:
                within_95pct += 1
        n_cov = len(coverage_values)
        report["coverage"] = {
            "n": int(n_cov),
            "within_1sigma": float(within_1sigma / n_cov),
            "within_95pct": float(within_95pct / n_cov),
        }

    cls_obs = [summary.get("cls_observed") for summary in summaries if "cls_observed" in summary]
    cls_yield = [summary.get("yield_upper_limit") for summary in summaries if "yield_upper_limit" in summary]
    cls_failures = [summary for summary in summaries if "cls_error" in summary]
    if cls_obs or cls_yield or cls_failures:
        report["cls"] = {
            "observed_limit": _distribution_summary(cls_obs),
            "yield_upper_limit": _distribution_summary(cls_yield),
            "n_failures": int(len(cls_failures)),
            "failure_fraction": float(len(cls_failures) / len(summaries)),
        }

    fc_entries = [summary.get("feldman_cousins") for summary in summaries if "feldman_cousins" in summary]
    if fc_entries:
        fc_ok = 0
        fc_fail = 0
        fc_widths = []
        for entry in fc_entries:
            if not isinstance(entry, dict):
                fc_fail += 1
                continue
            status = str(entry.get("fc_status", "")).lower()
            if "ok" in status:
                fc_ok += 1
            else:
                fc_fail += 1
            interval = entry.get("fc_interval")
            if isinstance(interval, (list, tuple)) and len(interval) == 2:
                low, high = interval
                if low is not None and high is not None:
                    low = float(low)
                    high = float(high)
                    if np.isfinite(low) and np.isfinite(high):
                        fc_widths.append(high - low)

        report["feldman_cousins"] = {
            "n_evaluated": int(len(fc_entries)),
            "n_ok": int(fc_ok),
            "n_non_ok": int(fc_fail),
            "width": _distribution_summary(fc_widths),
        }

    return report


def _save_ensemble_report(report, output_pkl, report_file=None):
    if report_file:
        output_path = os.path.abspath(report_file)
    else:
        base, _ = os.path.splitext(os.path.abspath(output_pkl))
        output_path = f"{base}_ensemble_report.json"

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return output_path


def _split_dataset_ranges(n_datasets, n_jobs):
    n_jobs = max(1, min(int(n_jobs), int(n_datasets)))
    base = n_datasets // n_jobs
    rem = n_datasets % n_jobs
    ranges = []
    start = 0
    for worker_idx in range(n_jobs):
        size = base + (1 if worker_idx < rem else 0)
        end = start + size
        if start < end:
            ranges.append((worker_idx, start, end))
        start = end
    return ranges


def _run_parallel_worker(task):
    worker_index = int(task["worker_index"])
    start_index = int(task["start_index"])
    end_index = int(task["end_index"])

    fit_model = _load_analysis_model(model_file=task.get("model_file"), input_card=task.get("input_card"))
    apply_parameter_overrides(
        fit_model,
        set_values_spec=task.get("set_parameters"),
        set_ranges_spec=task.get("set_parameter_ranges"),
        freeze_spec=task.get("freeze_parameters"),
    )

    zfit.settings.set_seed(int(task["seed"]) + worker_index)
    configure_runtime(task["graph_mode"], fit_model, end_index - start_index)

    summaries = run_analysis(
        fit_model,
        toys=end_index,
        use_observed_data=False,
        use_asimov_data=False,
        cls_alpha=task.get("cls_alpha"),
        signal_strength=task.get("signal_strength"),
        scan_max=task.get("scan_max"),
        fit_mode=task["fit_mode"],
        binned_bins=int(task["binned_bins"]),
        cls_scan_points=task.get("cls_scan_points"),
        cls_smart_scan=bool(task.get("cls_smart_scan", False)),
        profile_scan=bool(task.get("profile_scan", False)),
        poi_name=task.get("poi_name"),
        promote_poi=bool(task.get("promote_poi", False)),
        poi_scan_points=int(task.get("poi_scan_points", 41)),
        poi_scan_max=task.get("poi_scan_max"),
        feldman_cousins_alpha=task.get("feldman_cousins_alpha"),
        feldman_cousins_scan_points=int(task.get("fc_scan_points", 21)),
        feldman_cousins_n_toys=int(task.get("fc_toys", 100)),
        feldman_cousins_scan_max=task.get("fc_scan_max"),
        progress_callback=None,
        checkpoint_freq=None,
        checkpoint_path=None,
        existing_results=None,
        resume_from_index=start_index,
        compute_nll_scan=bool(task.get("compute_nll_scan", False)),
        nll_scan_points=int(task.get("nll_scan_points", 121)),
    )
    return summaries



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

    n_jobs = int(getattr(args, "jobs", 1) or 1)
    if n_jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if int(args.nll_scan_points) < 3:
        raise ValueError("--nll-scan-points must be >= 3")

    configure_runtime(args.graph_mode, fit_model, n_toys)
    total_start = time.perf_counter()

    # Load checkpoint if resuming
    existing_results = []
    resume_from_index = 0
    if args.resume_from:
        try:
            with open(args.resume_from, "rb") as f:
                checkpoint = dill.load(f)
                expected_checkpoint_config = {
                    "data_mode": _current_data_mode(use_observed_data, use_asimov_data),
                    "fit_mode": args.fit_mode,
                    "cls_alpha": args.cls,
                    "signal_strength": args.signal_strength,
                    "scan_max": args.scan_max,
                    "cls_smart_scan": bool(args.cls_smart_scan),
                    "profile_scan": bool(args.profile_scan),
                    "poi_name": args.poi_name,
                    "poi_scan_points": int(args.poi_scan_points),
                    "poi_scan_max": args.poi_scan_max,
                    "feldman_cousins_alpha": args.feldman_cousins,
                    "feldman_cousins_scan_points": int(args.fc_scan_points),
                    "feldman_cousins_n_toys": int(args.fc_toys),
                    "feldman_cousins_scan_max": args.fc_scan_max,
                    "compute_nll_scan": bool(args.plot),
                    "nll_scan_points": int(args.nll_scan_points),
                }
                mismatches = _checkpoint_mismatches(checkpoint, expected_checkpoint_config)
                if mismatches:
                    mismatch_text = ", ".join(
                        [f"{k}: checkpoint={old!r}, current={new!r}" for k, old, new in mismatches]
                    )
                    raise ValueError(
                        "Checkpoint is incompatible with current analysis settings: "
                        f"{mismatch_text}"
                    )

                existing_results = checkpoint.get("summaries", [])
                resume_from_index = len(existing_results)
                if _current_data_mode(use_observed_data, use_asimov_data) == "toy":
                    print(f"Resumed from checkpoint: {len(existing_results)} toys already completed")
                else:
                    print(f"Resumed from checkpoint: {len(existing_results)} datasets already completed")
                if resume_from_index >= n_toys:
                    print(f"Already completed all {n_toys} datasets. Skipping analysis.")
                    summaries = existing_results
        except Exception as e:
            print(f"Warning: could not load checkpoint {args.resume_from}: {e}")

    if not hasattr(args, "resume_from") or not args.resume_from or resume_from_index < n_toys:
        can_parallelize = (
            n_jobs > 1
            and not use_observed_data
            and not use_asimov_data
            and n_toys > 1
            and not args.resume_from
            and args.checkpoint_freq is None
        )

        if can_parallelize:
            worker_ranges = _split_dataset_ranges(n_toys, n_jobs)
            tasks = []
            for worker_index, start_index, end_index in worker_ranges:
                tasks.append(
                    {
                        "worker_index": worker_index,
                        "start_index": start_index,
                        "end_index": end_index,
                        "model_file": args.model_file,
                        "input_card": args.input_card,
                        "set_parameters": args.set_parameters,
                        "freeze_parameters": args.freeze_parameters,
                        "set_parameter_ranges": args.set_parameter_ranges,
                        "seed": int(args.seed),
                        "graph_mode": args.graph_mode,
                        "cls_alpha": args.cls,
                        "signal_strength": args.signal_strength,
                        "scan_max": args.scan_max,
                        "fit_mode": args.fit_mode,
                        "binned_bins": int(args.binned_bins),
                        "cls_scan_points": args.cls_scan_points,
                        "cls_smart_scan": bool(args.cls_smart_scan),
                        "profile_scan": bool(args.profile_scan),
                        "poi_name": args.poi_name,
                        "promote_poi": bool(args.promote_poi),
                        "poi_scan_points": int(args.poi_scan_points),
                        "poi_scan_max": args.poi_scan_max,
                        "feldman_cousins_alpha": args.feldman_cousins,
                        "fc_scan_points": int(args.fc_scan_points),
                        "fc_toys": int(args.fc_toys),
                        "fc_scan_max": args.fc_scan_max,
                        "compute_nll_scan": bool(args.plot and start_index == 0),
                        "nll_scan_points": int(args.nll_scan_points),
                    }
                )

            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=len(tasks)) as pool:
                worker_results = pool.map(_run_parallel_worker, tasks)

            summaries = []
            for chunk in worker_results:
                summaries.extend(chunk)
            summaries.sort(key=lambda item: int(item.get("dataset_id", 0)))
            for summary in summaries:
                _print_dataset_summary(summary)
        else:
            if n_jobs > 1 and (use_observed_data or use_asimov_data):
                print("Parallel processing is only applied to generated toy datasets; running sequentially.")
            if n_jobs > 1 and args.resume_from:
                print("Parallel processing is disabled when --resume-from is used; running sequentially.")
            if n_jobs > 1 and args.checkpoint_freq is not None:
                print("Parallel processing is disabled when --checkpoint-freq is set; running sequentially.")

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
                cls_smart_scan=args.cls_smart_scan,
                profile_scan=args.profile_scan,
                poi_name=args.poi_name,
                promote_poi=args.promote_poi,
                poi_scan_points=args.poi_scan_points,
                poi_scan_max=args.poi_scan_max,
                feldman_cousins_alpha=args.feldman_cousins,
                feldman_cousins_scan_points=args.fc_scan_points,
                feldman_cousins_n_toys=args.fc_toys,
                feldman_cousins_scan_max=args.fc_scan_max,
                progress_callback=_print_dataset_summary,
                checkpoint_freq=args.checkpoint_freq,
                checkpoint_path=args.output_pkl + ".checkpoint" if args.checkpoint_freq else None,
                existing_results=existing_results,
                resume_from_index=resume_from_index,
                compute_nll_scan=args.plot,
                nll_scan_points=args.nll_scan_points,
            )
        total_time_s = time.perf_counter() - total_start
    else:
        total_time_s = 0

    print(f"Analyzed model: {fit_model.model.name}")
    if args.cls is not None and summaries:
        first = summaries[0]
        if "cls_observed" in first:
            print(f"CLs observed upper limit (alpha={args.cls:g}): {first['cls_observed']:.4f}")
            if "cls_expected_quantiles" in first:
                q = first["cls_expected_quantiles"]
                print(
                    "CLs expected (asymptotic, b-only fit): "
                    f"2.5%={q['2.5%']:.4f}, 16%={q['16%']:.4f}, 50%={q['50%']:.4f}, "
                    f"84%={q['84%']:.4f}, 97.5%={q['97.5%']:.4f}"
                )
        elif "cls_error" in first:
            print(f"CLs failed (alpha={args.cls:g}): {first['cls_error']}")

    if args.plot:
        plot_summary_artifacts(
            summaries=summaries,
            fit_model=fit_model,
            plot_dir=os.path.abspath(args.plot_dir),
            binned_bins=args.binned_bins,
        )
        print(f"Saved plots to: {os.path.abspath(args.plot_dir)}")

    if summaries:
        print(f"Average time per dataset: {total_time_s / len(summaries):.4f}s")
    print(f"Total execution time: {total_time_s:.4f}s")

    output_pkl = args.output_pkl
    if output_pkl is None:
        output_pkl = f"analysis_output_{args.seed}.pkl"

    ensemble_report = _build_ensemble_evaluation_report(summaries=summaries, total_time_s=total_time_s)
    report_path = _save_ensemble_report(
        report=ensemble_report,
        output_pkl=output_pkl,
        report_file=args.report_file,
    )
    print(f"Saved ensemble evaluation report to: {report_path}")

    snapshot_path = _save_analysis_snapshot(
        output_pkl=output_pkl,
        fit_model=fit_model,
        summaries=summaries,
        args=args,
    )
    print(f"Saved analysis snapshot to: {snapshot_path}")
