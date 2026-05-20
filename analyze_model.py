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
from scipy.optimize import minimize_scalar

from build_model_from_text import build_model_from_card, parse_model_card
from model_io import load_fit_model
from utilities import AsymptoticCalculator, POI, POIarray, UpperLimit


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


def _find_parameter_by_name(fit_model, parameter_name):
    for param in fit_model.model.get_params():
        if param.name == parameter_name:
            return param
    return None


def _is_likely_counting_model(fit_model):

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        mode = toy_plot.get("mode")
        signal_category = _get_signal_category(fit_model)

        if mode == "binned":
            edges = np.asarray(toy_plot["edges"], dtype=float)
            counts = np.asarray(toy_plot["counts"], dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            yerr = np.sqrt(np.maximum(counts, 1.0))

            ax.errorbar(centers, counts, yerr=yerr, fmt="o", color="black", markersize=4, capsize=2, label="Toy data")

            total_counts = np.zeros_like(counts, dtype=float)
            if getattr(fit_model, "shapes", None) and getattr(fit_model, "yields", None):
                for category, shape in fit_model.shapes.items():
                    if category not in fit_model.yields:
                        continue
                    total_counts = total_counts + _binned_component_counts(shape, fit_model.yields[category].value(), edges)
            if not np.any(total_counts):
                total_counts = _binned_model_counts_from_pdf(
                    fit_model.model,
                    float(fit_model.model.get_yield().value()) if hasattr(fit_model.model, "get_yield") else 1.0,
                    edges,
                )
            ax.step(edges[:-1], total_counts, where="post", color="black", linewidth=1.8, label="Total model")

            if getattr(fit_model, "shapes", None) and getattr(fit_model, "yields", None):
                background_counts = np.zeros_like(total_counts, dtype=float)
                signal_counts = None
                for category, shape in fit_model.shapes.items():
                    if category not in fit_model.yields:
                        continue
                    comp_counts = _binned_component_counts(shape, fit_model.yields[category].value(), edges)
                    if category == signal_category:
                        signal_counts = comp_counts
                    else:
                        background_counts = background_counts + comp_counts

                ax.step(edges[:-1], background_counts, where="post", color="tab:blue", linestyle="--", linewidth=1.6, label="Total background")
                if signal_counts is not None:
                    ax.step(edges[:-1], signal_counts, where="post", color="tab:red", linestyle="-.", linewidth=1.6, label="Signal")

            ax.set_xlabel(fit_model.obs.obs[0] if getattr(fit_model.obs, "obs", None) else "obs")
            ax.set_ylabel("Entries / bin width")

        else:
            values = np.asarray(toy_plot["values"], dtype=float)
            lower, upper = fit_model.obs_range
            edges = np.linspace(float(lower), float(upper), int(binned_bins) + 1)
            counts, edges = np.histogram(values, bins=edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            yerr = np.sqrt(np.maximum(counts, 1.0))

            ax.errorbar(centers, counts, yerr=yerr, fmt="o", color="black", markersize=4, capsize=2, label="Toy data")

            total_curve = _binned_model_counts_from_pdf(
                fit_model.model,
                float(fit_model.model.get_yield().value()) if hasattr(fit_model.model, "get_yield") else 1.0,
                edges,
            )
            ax.plot(centers, total_curve, color="black", linewidth=1.8, label="Total model")

            if getattr(fit_model, "shapes", None) and getattr(fit_model, "yields", None):
                background_curve = np.zeros_like(total_curve, dtype=float)
                signal_curve = None
                for category, shape in fit_model.shapes.items():
                    if category not in fit_model.yields:
                        continue
                    comp_curve = _binned_component_counts(shape, fit_model.yields[category].value(), edges)
                    if category == signal_category:
                        signal_curve = comp_curve
                    else:
                        background_curve = background_curve + comp_curve

                ax.plot(centers, background_curve, color="tab:blue", linestyle="--", linewidth=1.6, label="Total background")
                if signal_curve is not None:
                    ax.plot(centers, signal_curve, color="tab:red", linestyle="-.", linewidth=1.6, label="Signal")

            ax.set_xlabel(fit_model.obs.obs[0] if getattr(fit_model.obs, "obs", None) else "obs")
            ax.set_ylabel("Entries / bin width")

        ax.set_title(f"Toy {summary['toy']} Dataset and Fit Components")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"toy_{summary['toy']:04d}_dataset_fit.png"), dpi=140)
        plt.close(fig)
    finally:
        _restore_parameter_values(baseline_values)


def _plot_summary_artifacts(summaries, fit_model, plot_dir, binned_bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    # 1) Histogram plots for each fit parameter across toys.
    param_values = {}
    for summary in summaries:
        for name, value in summary.get("fit_params", {}).items():
            param_values.setdefault(name, []).append(value)

    for name, values in param_values.items():
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=min(40, max(10, int(np.sqrt(len(values)) * 2))), alpha=0.8)
        ax.set_title(f"Fit Parameter Distribution: {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Entries")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"fit_param_{name}.png"), dpi=140)
        plt.close(fig)

    # 2) POI pull distribution.
    pulls = [summary.get("poi_pull") for summary in summaries if summary.get("poi_pull") is not None]
    if pulls:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pulls, bins=min(40, max(10, int(np.sqrt(len(pulls)) * 2))), alpha=0.8)
        ax.set_title("POI Pull Distribution")
        ax.set_xlabel("(POI_fit - POI_true) / sigma_Hesse")
        ax.set_ylabel("Entries")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "poi_pull_distribution.png"), dpi=140)
        plt.close(fig)

    # 3) Dataset/component overlay for each toy.
    for summary in summaries:
        _plot_dataset_and_components(summary, fit_model, plot_dir, binned_bins)


def _is_likely_counting_model(fit_model):
    if fit_model.obs_range == (0.0, 1.0):
        return True
    obs_name = None
    if getattr(fit_model.obs, "obs", None):
        obs_name = fit_model.obs.obs[0]
    return obs_name == "count_obs"


def _configure_runtime(graph_mode, fit_model, toys):
    if graph_mode == "on":
        zfit.run.set_graph_mode(True)
        return
    if graph_mode == "off":
        zfit.run.set_graph_mode(False)
        return

    use_graph = not (_is_likely_counting_model(fit_model) or toys <= 5)
    zfit.run.set_graph_mode(use_graph)


def _resolve_fit_mode(fit_mode, fit_model):
    if fit_mode == "auto":
        return "binned" if _is_likely_counting_model(fit_model) else "unbinned"
    return fit_mode


def _build_counting_binned_space(fit_model):
    obs_name = "count_obs"
    if getattr(fit_model.obs, "obs", None):
        obs_name = fit_model.obs.obs[0]

    low, high = fit_model.obs_range
    edges = np.array([float(low), float(high)], dtype=float)
    binning = zfit.binned.VariableBinning(edges, name=obs_name)
    return zfit.Space(obs_name, binning=binning)


def _binning_edges_as_float_array(binning):
    edges = getattr(binning, "edges", binning)
    return np.asarray(list(edges), dtype=float)


def _build_binned_space(fit_model, bins):
    if _is_likely_counting_model(fit_model):
        return _build_counting_binned_space(fit_model)

    obs_names = getattr(fit_model.obs, "obs", None)
    if not obs_names or len(obs_names) != 1:
        raise ValueError("Binned fits currently support only 1D observables")

    low, high = fit_model.obs_range
    edges = np.linspace(float(low), float(high), int(bins) + 1)
    binning = zfit.binned.VariableBinning(edges, name=obs_names[0])
    return zfit.Space(obs_names[0], binning=binning)


def _make_binned_toy_data(model, binned_space):
    sample = model.sample(n="auto")
    values = np.asarray(sample.value(), dtype=float).reshape(-1)
    edges = _binning_edges_as_float_array(binned_space.binning)
    counts, _ = np.histogram(values, bins=edges)
    data = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts.astype(float))
    return data, values, edges, counts.astype(float)


def _make_counting_binned_toy_data(model, binned_space):
    expected = float(model.get_yield().value())
    count = int(np.random.poisson(expected))
    return zfit.data.BinnedData.from_tensor(space=binned_space, values=np.array([float(count)]))


def _capture_parameter_values(model):
    values = {}
    for param in model.get_params():
        if hasattr(param, "set_value"):
            values[param] = float(param.value())
    return values


def _restore_parameter_values(saved_values):
    for param, value in saved_values.items():
        param.set_value(value)


def _find_signal_parameter(fit_model):
    if fit_model.signal_process is not None:
        target = f"mu_{fit_model.signal_process}"
        for param in fit_model.model.get_params():
            if param.name == target:
                return param

    for param in fit_model.model.get_params():
        if param.name.startswith("mu_"):
            return param

    if fit_model.signal_process and fit_model.signal_process in fit_model.yields:
        return fit_model.yields[fit_model.signal_process]

    return None


def _plot_dataset_and_components(summary, fit_model, plot_dir, binned_bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    toy_plot = summary.get("toy_plot")
    if not toy_plot:
        return
    is_observed = bool(summary.get("observed_fit") or toy_plot.get("observed"))
    is_asimov = bool(summary.get("asimov_fit") or toy_plot.get("asimov"))

    os.makedirs(plot_dir, exist_ok=True)
    baseline_values = _capture_parameter_values(fit_model.model)
    fit_params = summary.get("fit_params", {})
    _restore_fit_params_by_name(fit_model, fit_params)

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        mode = toy_plot.get("mode")
        signal_category = _find_total_signal_category(fit_model)

        # Label for data points
        if is_asimov:
            data_label = "Asimov data"
        elif is_observed:
            data_label = "Observed data"
        else:
            data_label = "Toy data"

        if mode == "binned":
            edges = np.asarray(toy_plot["edges"], dtype=float)
            counts = np.asarray(toy_plot["counts"], dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            yerr = np.sqrt(np.maximum(counts, 1.0))

            ax.errorbar(centers, counts, yerr=yerr, fmt="o", color="black", markersize=4, capsize=2, label=data_label)

            total_counts = _binned_model_counts_from_pdf(
                fit_model.model,
                float(fit_model.model.get_yield().value()) if hasattr(fit_model.model, "get_yield") else 1.0,
                edges,
            )
            ax.step(edges[:-1], total_counts, where="post", color="black", linewidth=1.8, label="Total model")

            if getattr(fit_model, "shapes", None) and getattr(fit_model, "yields", None):
                background_counts = np.zeros_like(total_counts, dtype=float)
                signal_counts = None
                for category, shape in fit_model.shapes.items():
                    if category not in fit_model.yields:
                        continue
                    comp_counts = _binned_component_counts(shape, fit_model.yields[category].value(), edges)
                    if category == signal_category:
                        signal_counts = comp_counts
                    else:
                        background_counts = background_counts + comp_counts

                ax.step(edges[:-1], background_counts, where="post", color="tab:blue", linestyle="--", linewidth=1.6, label="Total background")
                if signal_counts is not None:
                    ax.step(edges[:-1], signal_counts, where="post", color="tab:red", linestyle="-.", linewidth=1.6, label="Signal")

            ax.set_xlabel(fit_model.obs.obs[0] if getattr(fit_model.obs, "obs", None) else "obs")
            ax.set_ylabel("Entries / bin")

        else:
            values = np.asarray(toy_plot.get("values", []), dtype=float)
            lower, upper = fit_model.obs_range
            bins = np.linspace(float(lower), float(upper), int(binned_bins) + 1)
            counts, edges = np.histogram(values, bins=bins) if values.size > 0 else (np.zeros(int(binned_bins)), np.linspace(float(lower), float(upper), int(binned_bins) + 1))
            centers = 0.5 * (edges[:-1] + edges[1:])
            yerr = np.sqrt(np.maximum(counts, 1.0))

            ax.errorbar(centers, counts, yerr=yerr, fmt="o", color="black", markersize=4, capsize=2, label=data_label)

            x_plot = np.linspace(float(lower), float(upper), 1000)
            total_curve = np.asarray(fit_model.model.pdf(x_plot), dtype=float).reshape(-1)
            total_yield = float(fit_model.model.get_yield().value()) if hasattr(fit_model.model, "get_yield") else 1.0
            total_curve = total_curve * total_yield * (x_plot[1] - x_plot[0])
            ax.plot(x_plot, total_curve, color="black", linewidth=1.8, label="Total model")

            if getattr(fit_model, "shapes", None) and getattr(fit_model, "yields", None):
                background_curve = np.zeros_like(x_plot, dtype=float)
                signal_curve = None
                for category, shape in fit_model.shapes.items():
                    if category not in fit_model.yields:
                        continue
                    comp_curve = _unbinned_component_curve(shape, fit_model.yields[category].value(), x_plot)
                    if category == signal_category:
                        signal_curve = comp_curve
                    else:
                        background_curve = background_curve + comp_curve

                ax.plot(x_plot, background_curve, color="tab:blue", linestyle="--", linewidth=1.6, label="Total background")
                if signal_curve is not None:
                    ax.plot(x_plot, signal_curve, color="tab:red", linestyle="-.", linewidth=1.6, label="Signal")

            ax.set_xlabel(fit_model.obs.obs[0] if getattr(fit_model.obs, "obs", None) else "obs")
            ax.set_ylabel("Entries / bin")

        if is_asimov:
            title = "Asimov Data and Fit Components"
        elif is_observed:
            title = "Observed Data and Fit Components"
        else:
            title = f"Toy {summary['toy']} Dataset and Fit Components"
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"toy_{summary['toy']:04d}_dataset_fit.png"), dpi=140)
        plt.close(fig)
    finally:
        _restore_parameter_values(baseline_values)


def _resolve_poi_parameter(fit_model, poi_name=None, promote_poi=False):
    if poi_name is not None:
        poi_param = _find_parameter_by_name(fit_model, poi_name)
        if poi_param is None:
            raise ValueError(f"Could not find parameter '{poi_name}' in model")
        if promote_poi and hasattr(poi_param, "floating"):
            poi_param.floating = True
        return poi_param

    poi_param = _find_signal_parameter(fit_model)
    if poi_param is not None and promote_poi and hasattr(poi_param, "floating"):
        poi_param.floating = True
    return poi_param


def _default_scan_max(signal_param, fit_model):
    if signal_param is None:
        return None
    if signal_param.name.startswith("mu_"):
        return 5.0
    if fit_model.signal_nominal_yield is not None:
        return max(50.0, 3.0 * fit_model.signal_nominal_yield)
    return 50.0


def _compute_cls(loss, minimizer, signal_param, alpha, scan_max, scan_points):
    scan_values = np.linspace(0.0, scan_max, int(scan_points))
    poinull = POIarray(signal_param, scan_values)
    poialt = POI(signal_param, 0.0)
    calculator = AsymptoticCalculator(input=loss, minimizer=minimizer)
    ul_engine = UpperLimit(calculator, poinull, poialt)
    return ul_engine.upperlimit(alpha=alpha, CLs=True)


def _default_cls_scan_points(fit_model, resolved_fit_mode, cls_scan_points):
    if cls_scan_points is not None:
        if cls_scan_points < 3:
            raise ValueError("--cls-scan-points must be >= 3")
        return int(cls_scan_points)

    if resolved_fit_mode == "binned" and _is_likely_counting_model(fit_model):
        return 9
    return 25


def _build_toy_data(fit_model, resolved_fit_mode, binned_space, is_counting):
    if resolved_fit_mode == "binned":
        if is_counting:
            expected = float(fit_model.model.get_yield().value())
            toy_count = int(np.random.poisson(expected))
            low, high = fit_model.obs_range
            edges = np.array([float(low), float(high)], dtype=float)
            counts = np.array([float(toy_count)], dtype=float)
            data = zfit.data.BinnedData.from_tensor(
                space=binned_space,
                values=counts,
            )
            toy_plot = {
                "mode": "binned",
                "edges": edges,
                "counts": counts,
            }
            return data, toy_count, toy_plot

        data, values, edges, counts = _make_binned_toy_data(fit_model.model, binned_space)
        toy_plot = {
            "mode": "binned",
            "edges": edges,
            "counts": counts,
            "values": values,
        }
        return data, None, toy_plot

    data = fit_model.model.sample(n="auto")
    values = np.asarray(data.value(), dtype=float).reshape(-1)
    toy_plot = {
        "mode": "unbinned",
        "values": values,
    }
    return data, None, toy_plot


def _build_asimov_binned_data(binned_model, binned_space):
    data = binned_model.to_binneddata()
    expected_counts = np.asarray(binned_model.values(), dtype=float).reshape(-1)
    edges = _binning_edges_as_float_array(binned_space.binning)
    toy_plot = {
        "mode": "binned",
        "edges": edges,
        "counts": expected_counts,
        "asimov": True,
    }
    return data, expected_counts, toy_plot


def _build_loss(fit_model, resolved_fit_mode, binned_model, data):
    if resolved_fit_mode == "binned":
        return zfit.loss.ExtendedBinnedNLL(
            model=binned_model,
            data=data,
            constraints=fit_model.constraints,
        )

    return zfit.loss.ExtendedUnbinnedNLL(
        model=fit_model.model,
        data=data,
        constraints=fit_model.constraints,
    )


def _default_poi_scan_upper(poi_param, fit_model, poi_scan_max):
    if poi_scan_max is not None:
        return float(poi_scan_max)

    upper = getattr(poi_param, "upper", None)
    if upper is not None and np.isfinite(float(upper)):
        return float(upper)

    if poi_param.name.startswith("mu_"):
        return _default_scan_max(poi_param, fit_model)

    center = float(poi_param.value())
    return center + 5.0


def _default_poi_scan_lower(poi_param):
    lower = getattr(poi_param, "lower", None)
    if lower is not None and np.isfinite(float(lower)):
        return float(lower)

    if poi_param.name.startswith("mu_"):
        return 0.0

    center = float(poi_param.value())
    return center - 5.0


def _run_profile_scan_for_loss(loss, poi_param, fit_model, poi_scan_points=41, poi_scan_max=None):
    if not getattr(poi_param, "floating", False):
        raise ValueError(f"POI '{poi_param.name}' must be floating to run a profile scan")

    scan_points = int(poi_scan_points)
    if scan_points < 3:
        raise ValueError("poi_scan_points must be >= 3")

    scan_low = _default_poi_scan_lower(poi_param)
    scan_high = _default_poi_scan_upper(poi_param, fit_model, poi_scan_max)
    if not np.isfinite(scan_low) or not np.isfinite(scan_high) or scan_high <= scan_low:
        raise ValueError("Invalid POI scan range")

    scan_values = np.linspace(scan_low, scan_high, scan_points)
    scan_nll = []
    minimizer = zfit.minimize.Minuit()
    original_value = float(poi_param.value())
    was_floating = getattr(poi_param, "floating", None)

    def _nll_at_poi(poi_value):
        poi_param.set_value(float(poi_value))
        result = minimizer.minimize(loss)
        return float(loss.value())

    try:
        poi_param.floating = False
        for value in scan_values:
            scan_nll.append(_nll_at_poi(value))
        scan_nll = np.asarray(scan_nll, dtype=float)
        best_idx = int(np.argmin(scan_nll))

        left = scan_values[max(0, best_idx - 1)]
        right = scan_values[min(scan_points - 1, best_idx + 1)]
        if right > left:
            local = minimize_scalar(_nll_at_poi, bounds=(left, right), method="bounded")
            poi_hat = float(local.x)
        else:
            poi_hat = float(scan_values[best_idx])

        # Evaluate validity at the best profiled point.
        poi_param.set_value(poi_hat)
        best_result = minimizer.minimize(loss)
        min_nll = float(loss.value())
        valid = bool(best_result.valid)
    finally:
        if was_floating is not None:
            poi_param.floating = was_floating

    poi_param.set_value(poi_hat if "poi_hat" in locals() else original_value)
    return {
        "valid": bool(valid) if "valid" in locals() else False,
        "min_nll": float(min_nll) if "min_nll" in locals() else float("nan"),
        "poi_name": poi_param.name,
        "poi_hat": float(poi_hat) if "poi_hat" in locals() else original_value,
        "poi_scan_points": scan_points,
        "poi_scan_low": float(scan_low),
        "poi_scan_high": float(scan_high),
    }


def _extract_hesse_error(result, poi_param):
    try:
        hesse = result.hesse(params=[poi_param])
    except Exception:
        return None

    entry = hesse.get(poi_param)
    if not isinstance(entry, dict):
        return None
    error = entry.get("error")
    if error is None:
        return None
    return float(error)


def _extract_fit_parameter_values(fit_result):
    values = {}
    for param, info in fit_result.params.items():
        value = info.get("value") if isinstance(info, dict) else None
        if value is not None:
            values[param.name] = float(value)
    return values


def _restore_fit_params_by_name(fit_model, fit_params):
    for name, value in fit_params.items():
        param = _find_parameter_by_name(fit_model, name)
        if param is not None and hasattr(param, "set_value"):
            param.set_value(value)


def _parse_parameter_value_map(spec):
    if spec is None:
        return {}

    assignments = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid parameter assignment '{item}'. Expected format name=value")
        name, value_text = item.split("=", 1)
        name = name.strip()
        value_text = value_text.strip()
        if not name:
            raise ValueError(f"Invalid parameter assignment '{item}'")
        assignments[name] = float(value_text)
    return assignments


def _parse_parameter_range_map(spec):
    if spec is None:
        return {}

    ranges = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item or ":" not in item:
            raise ValueError(f"Invalid range assignment '{item}'. Expected format name=low:high")
        name, bounds_text = item.split("=", 1)
        low_text, high_text = bounds_text.split(":", 1)
        name = name.strip()
        low = float(low_text.strip())
        high = float(high_text.strip())
        if not name:
            raise ValueError(f"Invalid range assignment '{item}'")
        if high <= low:
            raise ValueError(f"Invalid range for '{name}': high ({high}) must be > low ({low})")
        ranges[name] = (low, high)
    return ranges


def _parse_parameter_name_list(spec):
    if spec is None:
        return []
    return [item.strip() for item in spec.split(",") if item.strip()]


def _apply_parameter_overrides(fit_model, set_values_spec, set_ranges_spec, freeze_spec):
    value_updates = _parse_parameter_value_map(set_values_spec)
    range_updates = _parse_parameter_range_map(set_ranges_spec)
    freeze_names = _parse_parameter_name_list(freeze_spec)

    required_names = set(value_updates) | set(range_updates) | set(freeze_names)
    params_by_name = {}
    for name in required_names:
        param = _find_parameter_by_name(fit_model, name)
        if param is None:
            raise ValueError(f"Parameter '{name}' was not found in the model")
        params_by_name[name] = param

    for name, value in value_updates.items():
        params_by_name[name].set_value(value)

    for name, (low, high) in range_updates.items():
        param = params_by_name[name]
        if hasattr(param, "set_limits"):
            param.set_limits(low=low, high=high)
        else:
            param.lower = low
            param.upper = high

    for name in freeze_names:
        param = params_by_name[name]
        if not hasattr(param, "floating"):
            raise ValueError(f"Parameter '{name}' does not support floating/fixed state")
        param.floating = False


def _get_signal_category(fit_model):
    signal_category = getattr(fit_model, "signal_process", None)
    if signal_category is not None:
        return signal_category

    for name in _get_category_names(fit_model):
        if name.startswith("mu") or name.startswith("sig"):
            return name

    return None


def _binned_component_counts(pdf, yield_value, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    density = np.asarray(pdf.pdf(centers), dtype=float).reshape(-1)
    return density * float(yield_value) * widths


def _binned_model_counts_from_pdf(pdf, yield_value, edges):
    return _binned_component_counts(pdf, yield_value, edges)


def _unbinned_component_curve(pdf, yield_value, x_plot):
    density = np.asarray(pdf.pdf(x_plot), dtype=float).reshape(-1)
    step = float(x_plot[1] - x_plot[0]) if len(x_plot) > 1 else 1.0
    return density * float(yield_value) * step


def _get_category_names(fit_model):
    if getattr(fit_model, "process_names", None):
        return list(fit_model.process_names)

    if getattr(fit_model, "shapes", None):
        return list(fit_model.shapes.keys())

    return []


def _find_total_signal_category(fit_model):
    signal_process = getattr(fit_model, "signal_process", None)
    if signal_process is not None:
        return signal_process

    for name in _get_category_names(fit_model):
        if name.startswith("mu") or name.startswith("sig"):
            return name

    return None


def _sample_pdf_values_as_histogram(pdf, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    values = np.asarray(pdf.pdf(centers), dtype=float).reshape(-1)
    return values * widths


def _plot_dataset_and_components(summary, fit_model, plot_dir, binned_bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    toy_plot = summary.get("toy_plot")
    if not toy_plot:
        return

    fit_params = summary.get("fit_params", {})
    baseline_values = _capture_parameter_values(fit_model.model)
    _restore_fit_params_by_name(fit_model, fit_params)

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        is_counting = _is_likely_counting_model(fit_model)
        mode = toy_plot.get("mode")
        model = fit_model.model
        signal_category = _find_total_signal_category(fit_model)

        if mode == "binned":
            edges = np.asarray(toy_plot["edges"], dtype=float)
            counts = np.asarray(toy_plot["counts"], dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = np.diff(edges)
            yerr = np.sqrt(np.maximum(counts, 1.0))

            ax.errorbar(
                centers,
                counts,
                yerr=yerr,
                fmt="o",
                color="black",
                markersize=4,
                capsize=2,
                label="Toy data",
            )

            total_counts = np.asarray(summary.get("total_model_counts", []), dtype=float)
            if total_counts.size == 0:
                model_values = np.asarray(model.sample(n="auto").value(), dtype=float).reshape(-1)
                total_counts, _ = np.histogram(model_values, bins=edges)
            ax.step(edges[:-1], total_counts, where="post", color="black", linewidth=1.8, label="Total model")

            if summary.get("background_model_counts") is not None:
                bkg_counts = np.asarray(summary["background_model_counts"], dtype=float)
                ax.step(edges[:-1], bkg_counts, where="post", color="tab:blue", linestyle="--", linewidth=1.6, label="Total background")

            if summary.get("signal_model_counts") is not None:
                sig_counts = np.asarray(summary["signal_model_counts"], dtype=float)
                ax.step(edges[:-1], sig_counts, where="post", color="tab:red", linestyle="-.", linewidth=1.6, label="Signal")

            ax.set_xlabel(fit_model.obs.obs[0] if getattr(fit_model.obs, "obs", None) else "obs")
            ax.set_ylabel("Entries")

        else:
            values = np.asarray(toy_plot["values"], dtype=float)
            lower, upper = fit_model.obs_range
            bins = np.linspace(float(lower), float(upper), int(binned_bins) + 1)
            counts, edges = np.histogram(values, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            yerr = np.sqrt(np.maximum(counts, 1.0))

            ax.errorbar(
                centers,
                counts,
                yerr=yerr,
                fmt="o",
                color="black",
                markersize=4,
                capsize=2,
                label="Toy data",
            )

            total_y = np.zeros_like(counts, dtype=float)
            if getattr(fit_model, "shapes", None) and getattr(fit_model, "yields", None):
                for category, shape in fit_model.shapes.items():
                    if category not in fit_model.yields:
                        continue
                    total_y = total_y + _binned_component_counts(shape, fit_model.yields[category].value(), edges)
            if not np.any(total_y):
                x_plot = np.linspace(float(lower), float(upper), 1000)
                total_y = np.asarray(model.pdf(x_plot), dtype=float).reshape(-1)
                if hasattr(model, "get_yield"):
                    total_y = total_y * float(model.get_yield().value()) * (x_plot[1] - x_plot[0])
                ax.plot(x_plot, total_y, color="black", linewidth=1.8, label="Total model")
            else:
                ax.plot(centers, total_y, color="black", linewidth=1.8, label="Total model")

            signal_pdf = None
            for category in _get_category_names(fit_model):
                if category == signal_category:
                    signal_pdf = fit_model.shapes.get(category) if getattr(fit_model, "shapes", None) else None

            if signal_pdf is not None and getattr(fit_model, "yields", None) and signal_category in fit_model.yields:
                sig_y = _binned_component_counts(signal_pdf, fit_model.yields[signal_category].value(), edges)
                ax.plot(centers, sig_y, color="tab:red", linestyle="-.", linewidth=1.6, label="Signal")

            if getattr(fit_model, "shapes", None) and getattr(fit_model, "yields", None):
                bkg_y = np.zeros_like(total_y, dtype=float)
                for category, shape in fit_model.shapes.items():
                    if category == signal_category:
                        continue
                    if category not in fit_model.yields:
                        continue
                    comp_y = _binned_component_counts(shape, fit_model.yields[category].value(), edges)
                    bkg_y = bkg_y + comp_y
                ax.plot(centers, bkg_y, color="tab:blue", linestyle="--", linewidth=1.6, label="Total background")

            ax.set_xlabel(fit_model.obs.obs[0] if getattr(fit_model.obs, "obs", None) else "obs")
            ax.set_ylabel("Entries")

        ax.set_title(f"Toy {summary['toy']} Dataset and Fit Components")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"toy_{summary['toy']:04d}_dataset_fit.png"), dpi=140)
        plt.close(fig)
    finally:
        _restore_parameter_values(baseline_values)


def _plot_summary_artifacts(summaries, fit_model, plot_dir, binned_bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    # 1) Histogram plots for each fit parameter across toys.
    param_values = {}
    for summary in summaries:
        for name, value in summary.get("fit_params", {}).items():
            param_values.setdefault(name, []).append(value)

    for name, values in param_values.items():
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=min(40, max(10, int(np.sqrt(len(values)) * 2))), alpha=0.8)
        ax.set_title(f"Fit Parameter Distribution: {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Entries")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"fit_param_{name}.png"), dpi=140)
        plt.close(fig)

    # 2) POI pull distribution.
    pulls = [summary.get("poi_pull") for summary in summaries if summary.get("poi_pull") is not None]
    if pulls:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pulls, bins=min(40, max(10, int(np.sqrt(len(pulls)) * 2))), alpha=0.8)
        ax.set_title("POI Pull Distribution")
        ax.set_xlabel("(POI_fit - POI_true) / sigma_Hesse")
        ax.set_ylabel("Entries")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "poi_pull_distribution.png"), dpi=140)
        plt.close(fig)

    # 3) Dataset/component overlay for each toy.
    for summary in summaries:
        _plot_dataset_and_components(summary, fit_model, plot_dir, binned_bins)

def run_analysis(
    fit_model,
    toys,
    use_observed_data=False,
    use_asimov_data=False,
    cls_alpha=None,
    signal_strength=None,
    scan_max=None,
    fit_mode="auto",
    binned_bins=40,
    cls_scan_points=None,
    profile_scan=False,
    poi_name=None,
    promote_poi=False,
    poi_scan_points=41,
    poi_scan_max=None,
    progress_callback=None,
):
    resolved_fit_mode = _resolve_fit_mode(fit_mode, fit_model)
    is_counting = _is_likely_counting_model(fit_model)

    signal_param = _find_signal_parameter(fit_model)
    if cls_alpha is not None and signal_param is None:
        raise ValueError("Could not identify a signal parameter for CLs evaluation")
    cls_points = _default_cls_scan_points(fit_model, resolved_fit_mode, cls_scan_points)
    poi_param = _resolve_poi_parameter(
        fit_model,
        poi_name=poi_name,
        promote_poi=promote_poi,
    )
    if poi_param is None:
        raise ValueError("Could not identify a parameter of interest")

    starting_values = _capture_parameter_values(fit_model.model)
    summaries = []
    minimizer = zfit.minimize.Minuit()
    binned_space = None
    binned_model = None
    if resolved_fit_mode == "binned":
        binned_space = _build_binned_space(fit_model, binned_bins)
        binned_model = fit_model.model.to_binned(binned_space)

    if use_asimov_data and resolved_fit_mode != "binned":
        raise ValueError("--toys -1 is only supported for binned fits")

    for toy_index in range(toys):
        toy_start = time.perf_counter()
        _restore_parameter_values(starting_values)
        if signal_strength is not None and signal_param is not None and hasattr(signal_param, "set_value"):
            signal_param.set_value(signal_strength)

        if use_observed_data:
            if resolved_fit_mode == "binned":
                edges = _binning_edges_as_float_array(binned_space.binning)
                if hasattr(fit_model.data, "value"):
                    observed_values = np.asarray(fit_model.data.value(), dtype=float).reshape(-1)
                    counts, _ = np.histogram(observed_values, bins=edges)
                    if hasattr(fit_model.data, "to_binned"):
                        data = fit_model.data.to_binned(binned_space)
                    else:
                        data = zfit.data.BinnedData.from_tensor(
                            space=binned_space,
                            values=counts.astype(float),
                        )
                else:
                    observed_count = float(fit_model.data)
                    observed_values = np.array([observed_count], dtype=float)
                    counts = np.array([observed_count], dtype=float)
                    data = zfit.data.BinnedData.from_tensor(
                        space=binned_space,
                        values=counts,
                    )
                toy_plot = {
                    "mode": "binned",
                    "edges": edges,
                    "counts": counts.astype(float),
                    "values": observed_values,
                    "observed": True,
                }
            else:
                data = fit_model.data
                if hasattr(data, "value"):
                    observed_values = np.asarray(data.value(), dtype=float).reshape(-1)
                else:
                    observed_values = np.array([float(data)], dtype=float)
                toy_plot = {
                    "mode": "unbinned",
                    "values": observed_values,
                    "observed": True,
                }
            toy_count = None
        elif use_asimov_data:
            data, toy_count, toy_plot = _build_asimov_binned_data(binned_model, binned_space)
        else:
            data, toy_count, toy_plot = _build_toy_data(
                fit_model=fit_model,
                resolved_fit_mode=resolved_fit_mode,
                binned_space=binned_space,
                is_counting=is_counting,
            )
        loss = _build_loss(
            fit_model=fit_model,
            resolved_fit_mode=resolved_fit_mode,
            binned_model=binned_model,
            data=data,
        )

        if profile_scan:
            profile_summary = _run_profile_scan_for_loss(
                loss=loss,
                poi_param=poi_param,
                fit_model=fit_model,
                poi_scan_points=poi_scan_points,
                poi_scan_max=poi_scan_max,
            )
            fit_result = minimizer.minimize(loss)
            summary = {
                "toy": toy_index + 1,
                "valid": profile_summary["valid"] and bool(fit_result.valid),
                "edm": None,
                "fit_mode": resolved_fit_mode,
                "min_nll": profile_summary["min_nll"],
                "poi_name": profile_summary["poi_name"],
                "poi_hat": profile_summary["poi_hat"],
                "poi_scan_points": profile_summary["poi_scan_points"],
                "poi_scan_low": profile_summary["poi_scan_low"],
                "poi_scan_high": profile_summary["poi_scan_high"],
                "poi_fit": float(poi_param.value()),
                "poi_unc_hesse": _extract_hesse_error(fit_result, poi_param),
                "fit_params": _extract_fit_parameter_values(fit_result),
            }
        else:
            result = minimizer.minimize(loss)
            summary = {
                "toy": toy_index + 1,
                "valid": bool(result.valid),
                "edm": float(result.edm) if result.edm is not None else None,
                "fit_mode": resolved_fit_mode,
                "poi_name": poi_param.name,
                "poi_fit": float(poi_param.value()),
                "poi_unc_hesse": _extract_hesse_error(result, poi_param),
                "fit_params": _extract_fit_parameter_values(result),
            }
        if toy_count is not None:
            summary["count"] = toy_count
        summary["toy_time_s"] = time.perf_counter() - toy_start
        summary["toy_plot"] = toy_plot
        summary["observed_fit"] = bool(use_observed_data)
        summary["asimov_fit"] = bool(use_asimov_data)

        poi_true = signal_strength if signal_strength is not None else starting_values.get(poi_param)
        poi_unc = summary.get("poi_unc_hesse")
        if poi_true is not None and poi_unc is not None and np.isfinite(poi_unc) and poi_unc > 0.0:
            summary["poi_pull"] = (summary["poi_fit"] - float(poi_true)) / poi_unc
        else:
            summary["poi_pull"] = None

        if cls_alpha is not None:
            scan_upper = scan_max if scan_max is not None else _default_scan_max(signal_param, fit_model)
            try:
                cls_result = _compute_cls(
                    loss,
                    minimizer,
                    signal_param,
                    cls_alpha,
                    scan_upper,
                    cls_points,
                )
                observed = float(cls_result["observed"])
                summary["cls_observed"] = observed
                summary["cls_scan_points"] = cls_points
                if fit_model.signal_nominal_yield is not None and signal_param.name.startswith("mu_"):
                    summary["yield_upper_limit"] = observed * fit_model.signal_nominal_yield
            except Exception as exc:
                summary["cls_error"] = str(exc)

        summaries.append(summary)
        if progress_callback is not None:
            progress_callback(summary)

    return summaries


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
    _apply_parameter_overrides(
        fit_model,
        set_values_spec=args.set_parameters,
        set_ranges_spec=args.set_parameter_ranges,
        freeze_spec=args.freeze_parameters,
    )

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

    _configure_runtime(args.graph_mode, fit_model, n_toys)
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
        _plot_summary_artifacts(
            summaries=summaries,
            fit_model=fit_model,
            plot_dir=os.path.abspath(args.plot_dir),
            binned_bins=args.binned_bins,
        )
        print(f"Saved plots to: {os.path.abspath(args.plot_dir)}")

    if summaries:
        print(f"Average time per toy: {total_time_s / len(summaries):.4f}s")
    print(f"Total execution time: {total_time_s:.4f}s")

    snapshot_path = _save_analysis_snapshot(
        output_pkl=args.output_pkl,
        fit_model=fit_model,
        summaries=summaries,
        args=args,
    )
    print(f"Saved analysis snapshot to: {snapshot_path}")
