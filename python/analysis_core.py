import numpy as np
import time
import zfit
import dill
from scipy.optimize import minimize_scalar
from multiprocessing import Pool

from zmodel.analysis_overrides import find_parameter_by_name
from zmodel.utilities import AsymptoticCalculator, POI


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
    scan_values = np.linspace(0.0, float(scan_max), int(scan_points))
    calculator = AsymptoticCalculator(input=loss, minimizer=minimizer)

    # Warm start the scan from the current best-fit state once.
    minimizer.minimize(loss)

    def _to_float(value):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            return float("nan")
        return float(arr[0])

    def _limit_from_curve(mu_values, cls_values, threshold):
        mu_values = np.asarray(mu_values, dtype=float)
        cls_values = np.asarray(cls_values, dtype=float)
        valid = np.isfinite(cls_values)
        if not np.any(valid):
            return None

        muv = mu_values[valid]
        clsv = cls_values[valid]
        below = np.where(clsv <= threshold)[0]
        if below.size == 0:
            return float(muv[-1])

        idx = int(below[0])
        if idx == 0:
            return float(muv[0])

        x0, x1 = float(muv[idx - 1]), float(muv[idx])
        y0, y1 = float(clsv[idx - 1]), float(clsv[idx])
        if y1 == y0:
            return x1
        t = (threshold - y0) / (y1 - y0)
        return float(x0 + t * (x1 - x0))

    poialt = POI(signal_param, 0.0)
    observed_cls = []
    expected_cls_by_sigma = {s: [] for s in (-2, -1, 0, 1, 2)}

    # Intentionally do not reset nuisance parameters between scan points.
    # Each fit starts from the previous point's fitted state.
    for mu in scan_values:
        poinull = POI(signal_param, float(mu))
        pnull, palt = calculator.pvalue(poinull, poialt)
        pnull = _to_float(pnull)
        palt = _to_float(palt)
        if palt <= 0.0 or not np.isfinite(palt):
            observed_cls.append(np.nan)
        else:
            observed_cls.append(pnull / palt)

        expected_curves = calculator.expected_pvalue(
            poinull,
            poialt,
            nsigma=[-2, -1, 0, 1, 2],
            CLs=True,
        )
        for sigma, curve in zip((-2, -1, 0, 1, 2), expected_curves):
            expected_cls_by_sigma[sigma].append(_to_float(curve))

    observed_limit = _limit_from_curve(scan_values, observed_cls, alpha)
    expected_m2 = _limit_from_curve(scan_values, expected_cls_by_sigma[-2], alpha)
    expected_m1 = _limit_from_curve(scan_values, expected_cls_by_sigma[-1], alpha)
    expected_0 = _limit_from_curve(scan_values, expected_cls_by_sigma[0], alpha)
    expected_p1 = _limit_from_curve(scan_values, expected_cls_by_sigma[1], alpha)
    expected_p2 = _limit_from_curve(scan_values, expected_cls_by_sigma[2], alpha)

    return {
        "observed": observed_limit,
        "expected": expected_0,
        "expected_p1": expected_p1,
        "expected_m1": expected_m1,
        "expected_p2": expected_p2,
        "expected_m2": expected_m2,
    }


def _compute_cls_smart(loss, minimizer, signal_param, alpha, scan_max, scan_points):
    upper = max(float(scan_max), 1e-6)
    points = max(int(scan_points), 9)
    result = None

    # Expand scan range until the observed limit is comfortably inside range
    # and expected +2 sigma is available.
    for _ in range(6):
        result = _compute_cls(loss, minimizer, signal_param, alpha, upper, points)
        observed = result.get("observed") if isinstance(result, dict) else None
        expected_p2 = result.get("expected_p2") if isinstance(result, dict) else None

        if observed is None:
            return result, upper, points

        observed = float(observed)
        has_expected_p2 = expected_p2 is not None
        if observed <= 0.80 * upper and has_expected_p2:
            break

        upper *= 2.0
        points = max(points + 8, 25)

    if result is None:
        result = _compute_cls(loss, minimizer, signal_param, alpha, upper, points)

    observed = result.get("observed") if isinstance(result, dict) else None
    if observed is None:
        return result, upper, points

    observed = float(observed)
    if observed > 0.0:
        refined_upper = max(observed * 1.5, 1e-6)
        refined_points = max(points, 41)
        refined = _compute_cls(loss, minimizer, signal_param, alpha, refined_upper, refined_points)
        return refined, refined_upper, refined_points

    return result, upper, points


def _extract_expected_cls_quantiles(cls_result):
    if not isinstance(cls_result, dict):
        return None

    expected = cls_result.get("expected")
    if isinstance(expected, dict):
        q2p5 = expected.get("2.5") or expected.get("2p5") or expected.get("-2sigma")
        q16 = expected.get("16") or expected.get("16.0") or expected.get("-1sigma")
        q50 = expected.get("50") or expected.get("50.0") or expected.get("median")
        q84 = expected.get("84") or expected.get("84.0") or expected.get("+1sigma")
        q97p5 = expected.get("97.5") or expected.get("97p5") or expected.get("+2sigma")
    elif isinstance(expected, (list, tuple)) and len(expected) >= 5:
        q2p5, q16, q50, q84, q97p5 = expected[:5]
    else:
        q2p5 = cls_result.get("expected_m2")
        q16 = cls_result.get("expected_m1")
        q50 = cls_result.get("expected")
        q84 = cls_result.get("expected_p1")
        q97p5 = cls_result.get("expected_p2")

    # Fallback approximation when +/-2 sigma are unavailable from the backend.
    if q50 is not None and q84 is not None and q97p5 is None:
        q97p5 = float(q50) + 2.0 * (float(q84) - float(q50))
    if q50 is not None and q16 is not None and q2p5 is None:
        q2p5 = float(q50) - 2.0 * (float(q50) - float(q16))

    values = [q2p5, q16, q50, q84, q97p5]
    if any(value is None for value in values):
        return None

    return {
        "2.5%": float(q2p5),
        "16%": float(q16),
        "50%": float(q50),
        "84%": float(q84),
        "97.5%": float(q97p5),
    }


def _default_cls_scan_points(fit_model, resolved_fit_mode, cls_scan_points):
    if cls_scan_points is not None:
        if cls_scan_points < 3:
            raise ValueError("--cls-scan-points must be >= 3")
        return int(cls_scan_points)

    if resolved_fit_mode == "binned" and is_likely_counting_model(fit_model):
        return 9
    return 25


def _expected_counts_by_channel(fit_model):
    channel_expectations = {}
    term_channels = getattr(fit_model, "term_channels", {}) or {}
    for term_name, yield_param in getattr(fit_model, "yields", {}).items():
        channel = term_channels.get(term_name)
        if channel is None:
            continue
        channel_expectations[channel] = channel_expectations.get(channel, 0.0) + float(yield_param.value())
    return channel_expectations


def _build_toy_data(fit_model, resolved_fit_mode, binned_space, is_counting):
    if resolved_fit_mode == "binned":
        if is_counting:
            expected = float(fit_model.model.get_yield().value())
            channel_expectations = _expected_counts_by_channel(fit_model)
            channel_counts = {}
            if channel_expectations:
                for channel, value in channel_expectations.items():
                    channel_counts[channel] = float(np.random.poisson(max(0.0, value)))
                toy_count = int(sum(channel_counts.values()))
            else:
                toy_count = int(np.random.poisson(expected))
            low, high = fit_model.obs_range
            edges = np.array([float(low), float(high)], dtype=float)
            counts = np.array([float(toy_count)], dtype=float)
            data = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts)
            toy_plot = {"mode": "binned", "edges": edges, "counts": counts}
            if channel_counts:
                toy_plot["channel_counts"] = channel_counts
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


def _build_asimov_binned_data(binned_model, binned_space, fit_model):
    data = binned_model.to_binneddata()
    expected_counts = np.asarray(binned_model.values(), dtype=float).reshape(-1)
    edges = _binning_edges_as_float_array(binned_space.binning)
    toy_plot = {
        "mode": "binned",
        "edges": edges,
        "counts": expected_counts,
        "asimov": True,
    }
    channel_expectations = _expected_counts_by_channel(fit_model)
    if channel_expectations:
        toy_plot["channel_counts"] = channel_expectations
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
        # Set scan upper to 99% of the parameter upper bound to avoid boundary issues
        return 0.99 * float(upper)

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


def _compute_nll_scan_for_plot(loss, minimizer, poi_param, poi_unc, fit_model, poi_scan_max=None, n_points=51):
    """Profile-likelihood scan of the POI for plotting purposes.

    Scans the POI from (best_fit - 5*sigma) to (best_fit + 5*sigma),
    where sigma is the Hessian-estimated uncertainty.

    Returns a dict with keys 'poi_values', 'delta_nll_values', and 'poi_name',
    or None if the scan cannot be performed.
    """
    if not getattr(poi_param, "floating", False):
        return None

    # Get the current (best-fit) POI value
    poi_best_fit = float(poi_param.value())

    # Get parameter bounds
    param_lower = getattr(poi_param, "lower", None)
    param_upper = getattr(poi_param, "upper", None)

    # Set scan bounds based on Hessian uncertainty
    if poi_unc is not None and np.isfinite(poi_unc) and poi_unc > 0:
        poi_unc = float(poi_unc)
        scan_low = poi_best_fit - 5.0 * poi_unc
        scan_high = poi_best_fit + 5.0 * poi_unc
    else:
        # Fall back to default bounds based on parameter limits
        scan_low = _default_poi_scan_lower(poi_param)
        scan_high = _default_poi_scan_upper(poi_param, fit_model, poi_scan_max)

    # Clip to parameter bounds (with small margin from upper bound to avoid zfit boundary issues)
    if param_lower is not None:
        scan_low = max(scan_low, float(param_lower))
    if param_upper is not None:
        # Use 99% of upper bound to avoid boundary issues with zfit
        scan_high = min(scan_high, 0.99 * float(param_upper))

    if not np.isfinite(scan_low) or not np.isfinite(scan_high) or scan_high <= scan_low:
        return None
    scan_values = np.linspace(scan_low, scan_high, int(n_points))

    # Save best-fit values for all parameters (after global fit)
    bestfit_values = _capture_parameter_values(fit_model.model)
    was_floating = getattr(poi_param, "floating", True)

    nll_values = []
    try:
        poi_param.floating = False
        for v in scan_values:
            # Reset all floating nuisance parameters to their best-fit values before each scan point
            for param in fit_model.model.get_params():
                if param is not poi_param and getattr(param, "floating", False):
                    if hasattr(param, "set_value") and param in bestfit_values:
                        param.set_value(bestfit_values[param])
            poi_param.set_value(float(v))
            minimizer.minimize(loss)
            nll_values.append(float(loss.value()))
    except Exception:
        return None
    finally:
        poi_param.floating = was_floating
        _restore_parameter_values(bestfit_values)

    nll_arr = np.asarray(nll_values, dtype=float)
    delta_nll = nll_arr - float(np.nanmin(nll_arr))
    return {
        "poi_name": poi_param.name,
        "poi_values": scan_values.tolist(),
        "delta_nll_values": delta_nll.tolist(),
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


def _resolve_data_mode(use_observed_data, use_asimov_data):
    if use_observed_data:
        return "observed"
    if use_asimov_data:
        return "asimov"
    return "toy"


def _build_observed_input(fit_model, resolved_fit_mode, binned_space):
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
        if getattr(fit_model, "observed_counts_by_channel", None):
            toy_plot["channel_counts"] = {
                k: float(v)
                for k, v in fit_model.observed_counts_by_channel.items()
            }
        return data, None, toy_plot

    data = fit_model.data
    if hasattr(data, "value"):
        observed_values = np.asarray(data.value(), dtype=float).reshape(-1)
    else:
        observed_values = np.array([float(data)], dtype=float)
    toy_plot = {"mode": "unbinned", "values": observed_values, "observed": True}
    if getattr(fit_model, "observed_values_by_channel", None):
        toy_plot["channel_values"] = {
            k: np.asarray(v, dtype=float).reshape(-1)
            for k, v in fit_model.observed_values_by_channel.items()
        }
    return data, None, toy_plot


def _build_iteration_input(
    fit_model,
    resolved_fit_mode,
    binned_model,
    binned_space,
    is_counting,
    data_mode,
):
    if data_mode == "observed":
        return _build_observed_input(fit_model, resolved_fit_mode, binned_space)
    if data_mode == "asimov":
        return _build_asimov_binned_data(binned_model, binned_space, fit_model)
    return _build_toy_data(
        fit_model=fit_model,
        resolved_fit_mode=resolved_fit_mode,
        binned_space=binned_space,
        is_counting=is_counting,
    )


def _build_fit_summary(
    loss,
    minimizer,
    profile_scan,
    poi_param,
    fit_model,
    poi_scan_points,
    poi_scan_max,
    sample_index,
    resolved_fit_mode,
):
    if profile_scan:
        profile_summary = _run_profile_scan_for_loss(
            loss=loss,
            poi_param=poi_param,
            fit_model=fit_model,
            poi_scan_points=poi_scan_points,
            poi_scan_max=poi_scan_max,
        )
        fit_result = minimizer.minimize(loss)
        return {
            "toy": sample_index + 1,
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

    result = minimizer.minimize(loss)
    return {
        "toy": sample_index + 1,
        "valid": bool(result.valid),
        "edm": float(result.edm) if result.edm is not None else None,
        "fit_mode": resolved_fit_mode,
        "poi_name": poi_param.name,
        "poi_fit": float(poi_param.value()),
        "poi_unc_hesse": _extract_hesse_error(result, poi_param),
        "fit_params": _extract_fit_parameter_values(result),
    }


def _apply_cls_to_summary(
    summary,
    loss,
    minimizer,
    signal_param,
    fit_model,
    cls_alpha,
    scan_max,
    cls_points,
    cls_smart_scan,
):
    scan_upper = scan_max if scan_max is not None else _default_scan_max(signal_param, fit_model)
    try:
        if cls_smart_scan:
            cls_result, used_scan_max, used_scan_points = _compute_cls_smart(
                loss,
                minimizer,
                signal_param,
                cls_alpha,
                scan_upper,
                cls_points,
            )
        else:
            cls_result = _compute_cls(loss, minimizer, signal_param, cls_alpha, scan_upper, cls_points)
            used_scan_max = float(scan_upper)
            used_scan_points = int(cls_points)

        observed = float(cls_result["observed"])
        summary["cls_observed"] = observed
        summary["cls_scan_points"] = int(used_scan_points)
        summary["cls_scan_max"] = float(used_scan_max)
        if fit_model.signal_nominal_yield is not None and signal_param.name.startswith("mu_"):
            summary["yield_upper_limit"] = observed * fit_model.signal_nominal_yield
    except Exception as exc:
        summary["cls_error"] = str(exc)

    if "cls_error" in summary:
        return

    # Compute expected asymptotic CLs limits using nuisance parameters from a b-only fit.
    pre_bonly_values = _capture_parameter_values(fit_model.model)
    pre_bonly_float = bool(getattr(signal_param, "floating", True))
    try:
        signal_param.set_value(0.0)
        signal_param.floating = False
        minimizer.minimize(loss)
        bonly_values = _capture_parameter_values(fit_model.model)

        _restore_parameter_values(bonly_values)
        signal_param.floating = True
        signal_param.set_value(0.0)

        if cls_smart_scan:
            cls_expected_result, _, _ = _compute_cls_smart(
                loss,
                minimizer,
                signal_param,
                cls_alpha,
                scan_upper,
                cls_points,
            )
        else:
            cls_expected_result = _compute_cls(
                loss,
                minimizer,
                signal_param,
                cls_alpha,
                scan_upper,
                cls_points,
            )

        expected_quantiles = _extract_expected_cls_quantiles(cls_expected_result)
        if expected_quantiles is not None:
            summary["cls_expected_quantiles"] = expected_quantiles
    except Exception as exc:
        summary["cls_expected_error"] = str(exc)
    finally:
        signal_param.floating = pre_bonly_float
        _restore_parameter_values(pre_bonly_values)


def _compute_feldman_cousins_for_toy(
    fit_model,
    resolved_fit_mode,
    binned_space,
    binned_model,
    is_counting,
    poi_param,
    loss,
    feldman_cousins_alpha,
):
    fc_alpha = float(feldman_cousins_alpha)
    if not (0.0 < fc_alpha < 1.0):
        raise ValueError("Feldman-Cousins alpha must satisfy 0 < alpha < 1")
    fc_cl = 1.0 - fc_alpha

    fc_scan_points = 21
    fc_n_toys = 100
    poi_grid = np.linspace(0, _default_scan_max(poi_param, fit_model), fc_scan_points)
    fc_toy_fit_results = []

    minimizer = zfit.minimize.Minuit()
    starting_values = _capture_parameter_values(fit_model.model)
    original_floating = bool(getattr(poi_param, "floating", True))

    try:
        for poi_val in poi_grid:
            toy_fits = []
            for _ in range(fc_n_toys):
                _restore_parameter_values(starting_values)
                poi_param.set_value(poi_val)
                poi_param.floating = False

                if resolved_fit_mode == "binned":
                    toy_data, _, _ = _build_toy_data(fit_model, resolved_fit_mode, binned_space, is_counting)
                    toy_loss = _build_loss(fit_model, resolved_fit_mode, binned_model, toy_data)
                else:
                    toy_data, _, _ = _build_toy_data(fit_model, resolved_fit_mode, None, is_counting)
                    toy_loss = _build_loss(fit_model, resolved_fit_mode, None, toy_data)

                poi_param.floating = True
                try:
                    minimizer.minimize(toy_loss)
                    fit_val = float(poi_param.value())
                except Exception:
                    fit_val = np.nan
                toy_fits.append(fit_val)

            fc_toy_fit_results.append(toy_fits)

        _restore_parameter_values(starting_values)
        poi_param.floating = True
        try:
            minimizer.minimize(loss)
            observed_poi = float(poi_param.value())
        except Exception:
            observed_poi = np.nan

        fc_intervals = []
        for toy_fits in fc_toy_fit_results:
            toy_fits = np.array(toy_fits)
            if np.all(np.isnan(toy_fits)):
                fc_intervals.append((np.nan, np.nan))
                continue
            lower = np.nanpercentile(toy_fits, (fc_alpha / 2.0) * 100.0)
            upper = np.nanpercentile(toy_fits, (1.0 - fc_alpha / 2.0) * 100.0)
            fc_intervals.append((lower, upper))

        fc_in_interval = [poi for poi, (lo, hi) in zip(poi_grid, fc_intervals) if lo <= observed_poi <= hi]
        if fc_in_interval:
            fc_interval = (min(fc_in_interval), max(fc_in_interval))
        else:
            fc_interval = (np.nan, np.nan)

        return {
            "fc_interval": fc_interval,
            "fc_grid": poi_grid.tolist(),
            "fc_intervals": fc_intervals,
            "observed_poi": observed_poi,
            "fc_alpha": fc_alpha,
            "fc_confidence_level": fc_cl,
            "fc_status": "ok" if not np.isnan(fc_interval[0]) else "no interval found",
        }
    finally:
        poi_param.floating = original_floating
        _restore_parameter_values(starting_values)


def _run_single(
    fit_model,
    sample_index,
    data_mode,
    starting_values,
    signal_strength,
    signal_param,
    resolved_fit_mode,
    binned_model,
    binned_space,
    is_counting,
    minimizer,
    profile_scan,
    poi_param,
    poi_scan_points,
    poi_scan_max,
    cls_alpha,
    scan_max,
    cls_points,
    cls_smart_scan,
    feldman_cousins_alpha,
    compute_nll_scan,
):
    start_time = time.perf_counter()
    _restore_parameter_values(starting_values)
    if signal_strength is not None and signal_param is not None and hasattr(signal_param, "set_value"):
        signal_param.set_value(signal_strength)

    data, generated_count, data_plot = _build_iteration_input(
        fit_model=fit_model,
        resolved_fit_mode=resolved_fit_mode,
        binned_model=binned_model,
        binned_space=binned_space,
        is_counting=is_counting,
        data_mode=data_mode,
    )

    loss = _build_loss(
        fit_model=fit_model,
        resolved_fit_mode=resolved_fit_mode,
        binned_model=binned_model,
        data=data,
    )

    summary = _build_fit_summary(
        loss=loss,
        minimizer=minimizer,
        profile_scan=profile_scan,
        poi_param=poi_param,
        fit_model=fit_model,
        poi_scan_points=poi_scan_points,
        poi_scan_max=poi_scan_max,
        sample_index=sample_index,
        resolved_fit_mode=resolved_fit_mode,
    )

    if generated_count is not None:
        summary["count"] = generated_count
    summary["toy_time_s"] = time.perf_counter() - start_time
    summary["toy_plot"] = data_plot
    summary["observed_fit"] = (data_mode == "observed")
    summary["asimov_fit"] = (data_mode == "asimov")

    if compute_nll_scan:
        nll_scan = _compute_nll_scan_for_plot(
            loss=loss,
            minimizer=minimizer,
            poi_param=poi_param,
            poi_unc=summary.get("poi_unc_hesse"),
            fit_model=fit_model,
            poi_scan_max=poi_scan_max,
        )
        if nll_scan is not None:
            summary["nll_scan"] = nll_scan
            minimizer.minimize(loss)

    poi_true = signal_strength if signal_strength is not None else starting_values.get(poi_param)
    poi_unc = summary.get("poi_unc_hesse")
    if poi_true is not None and poi_unc is not None and np.isfinite(poi_unc) and poi_unc > 0.0:
        summary["poi_pull"] = (summary["poi_fit"] - float(poi_true)) / poi_unc
    else:
        summary["poi_pull"] = None

    if cls_alpha is not None:
        _apply_cls_to_summary(
            summary=summary,
            loss=loss,
            minimizer=minimizer,
            signal_param=signal_param,
            fit_model=fit_model,
            cls_alpha=cls_alpha,
            scan_max=scan_max,
            cls_points=cls_points,
            cls_smart_scan=cls_smart_scan,
        )

    if feldman_cousins_alpha is not None:
        try:
            summary["feldman_cousins"] = _compute_feldman_cousins_for_toy(
                fit_model=fit_model,
                resolved_fit_mode=resolved_fit_mode,
                binned_space=binned_space,
                binned_model=binned_model,
                is_counting=is_counting,
                poi_param=poi_param,
                loss=loss,
                feldman_cousins_alpha=feldman_cousins_alpha,
            )
        except Exception as exc:
            summary["feldman_cousins"] = {"fc_status": f"failed: {exc}"}

    return summary


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
    cls_smart_scan=False,
    profile_scan=False,
    poi_name=None,
    promote_poi=False,
    poi_scan_points=41,
    poi_scan_max=None,
    progress_callback=None,
    feldman_cousins_alpha=None,
    checkpoint_freq=None,  # Frequency of checkpointing
    checkpoint_path=None,   # Path to save checkpoints
    existing_results=None,
    resume_from_index=0,
    compute_nll_scan=False,
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
    summaries = list(existing_results) if existing_results else []
    minimizer = zfit.minimize.Minuit()
    binned_space = None
    binned_model = None
    if resolved_fit_mode == "binned":
        binned_space = _build_binned_space(fit_model, binned_bins)
        binned_model = fit_model.model.to_binned(binned_space)

    if use_asimov_data and resolved_fit_mode != "binned":
        raise ValueError("--toys -1 is only supported for binned fits")

    if checkpoint_freq is not None and checkpoint_freq < 1:
        raise ValueError("checkpoint_freq must be >= 1")

    data_mode = _resolve_data_mode(use_observed_data, use_asimov_data)
    resume_index = int(resume_from_index)
    if resume_index < 0:
        raise ValueError("resume_from_index must be >= 0")

    for sample_index in range(resume_index, toys):
        summary = _run_single(
            fit_model=fit_model,
            sample_index=sample_index,
            data_mode=data_mode,
            starting_values=starting_values,
            signal_strength=signal_strength,
            signal_param=signal_param,
            resolved_fit_mode=resolved_fit_mode,
            binned_model=binned_model,
            binned_space=binned_space,
            is_counting=is_counting,
            minimizer=minimizer,
            profile_scan=profile_scan,
            poi_param=poi_param,
            poi_scan_points=poi_scan_points,
            poi_scan_max=poi_scan_max,
            cls_alpha=cls_alpha,
            scan_max=scan_max,
            cls_points=cls_points,
            cls_smart_scan=cls_smart_scan,
            feldman_cousins_alpha=feldman_cousins_alpha,
            compute_nll_scan=(sample_index == 0 and compute_nll_scan),
        )

        summaries.append(summary)
        if progress_callback is not None:
            progress_callback(summary)

        # Save checkpoint if requested
        if checkpoint_freq is not None and (sample_index - resume_index + 1) % checkpoint_freq == 0 and checkpoint_path is not None:
            try:
                checkpoint_data = {
                    "summaries": summaries,
                    "completed_datasets": len(summaries),
                    "total_datasets": toys,
                    "data_mode": data_mode,
                    "fit_mode": resolved_fit_mode,
                    "cls_alpha": cls_alpha,
                    "signal_strength": signal_strength,
                    "scan_max": scan_max,
                    "cls_scan_points": cls_points,
                    "cls_smart_scan": bool(cls_smart_scan),
                    "profile_scan": bool(profile_scan),
                    "poi_name": poi_param.name,
                    "poi_scan_points": int(poi_scan_points),
                    "poi_scan_max": poi_scan_max,
                    "feldman_cousins_alpha": feldman_cousins_alpha,
                    "compute_nll_scan": bool(compute_nll_scan),
                }
                with open(checkpoint_path, "wb") as f:
                    dill.dump(checkpoint_data, f)
                label = "toys" if data_mode == "toy" else "datasets"
                print(f"Checkpoint saved: {len(summaries)}/{toys} {label} completed")
            except Exception as e:
                print(f"Warning: checkpoint save failed: {e}")

    return summaries
