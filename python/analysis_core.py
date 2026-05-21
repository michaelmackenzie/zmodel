import numpy as np
import time
import zfit
from scipy.optimize import minimize_scalar

from zmodel.analysis_overrides import find_parameter_by_name
from zmodel.utilities import AsymptoticCalculator, POI, POIarray, UpperLimit


def is_likely_counting_model(fit_model):
    if fit_model.obs_range == (0.0, 1.0):
        return True
    obs_name = None
    if getattr(fit_model.obs, "obs", None):
        obs_name = fit_model.obs.obs[0]
    return obs_name == "count_obs"


def configure_runtime(graph_mode, fit_model, toys):
    if graph_mode == "on":
        zfit.run.set_graph_mode(True)
        return
    if graph_mode == "off":
        zfit.run.set_graph_mode(False)
        return

    use_graph = not (is_likely_counting_model(fit_model) or toys <= 5)
    zfit.run.set_graph_mode(use_graph)


def _resolve_fit_mode(fit_mode, fit_model):
    if fit_mode == "auto":
        return "binned" if is_likely_counting_model(fit_model) else "unbinned"
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
    if is_likely_counting_model(fit_model):
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


def _resolve_poi_parameter(fit_model, poi_name=None, promote_poi=False):
    if poi_name is not None:
        poi_param = find_parameter_by_name(fit_model, poi_name)
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

    if resolved_fit_mode == "binned" and is_likely_counting_model(fit_model):
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
            data = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts)
            toy_plot = {"mode": "binned", "edges": edges, "counts": counts}
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
    toy_plot = {"mode": "unbinned", "values": values}
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
        minimizer.minimize(loss)
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
    is_counting = is_likely_counting_model(fit_model)

    signal_param = _find_signal_parameter(fit_model)
    if cls_alpha is not None and signal_param is None:
        raise ValueError("Could not identify a signal parameter for CLs evaluation")
    cls_points = _default_cls_scan_points(fit_model, resolved_fit_mode, cls_scan_points)
    poi_param = _resolve_poi_parameter(fit_model, poi_name=poi_name, promote_poi=promote_poi)
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
                        data = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts.astype(float))
                else:
                    observed_count = float(fit_model.data)
                    observed_values = np.array([observed_count], dtype=float)
                    counts = np.array([observed_count], dtype=float)
                    data = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts)
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
                toy_plot = {"mode": "unbinned", "values": observed_values, "observed": True}
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
                cls_result = _compute_cls(loss, minimizer, signal_param, cls_alpha, scan_upper, cls_points)
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
