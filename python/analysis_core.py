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


def _is_binned_dataset(dataset):
    if dataset is None:
        return False

    data_space = getattr(dataset, "space", None)
    if data_space is not None and getattr(data_space, "binned", False):
        return True

    # zfit BinnedData exposes values() and not value().
    has_values = callable(getattr(dataset, "values", None))
    has_value = callable(getattr(dataset, "value", None))
    return has_values and not has_value


def _has_histogram_input_data(fit_model):
    data = getattr(fit_model, "data", None)
    if isinstance(data, dict):
        return any(_is_binned_dataset(entry) for entry in data.values())
    return _is_binned_dataset(data)


def _native_binned_space_from_data(dataset):
    if not _is_binned_dataset(dataset):
        return None
    return getattr(dataset, "space", None)


def _resolve_fit_mode(fit_mode, fit_model):
    if fit_mode == "auto":
        if is_likely_counting_model(fit_model) or _has_histogram_input_data(fit_model):
            return "binned"
        return "unbinned"
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

    # zfit can expose edges as a per-axis container (e.g. tuple with one entry
    # for 1D), while np.histogram requires a flat 1D edge array.
    if isinstance(edges, (list, tuple)):
        if len(edges) == 1:
            edges = edges[0]
        else:
            raise ValueError("Binned fits currently support only 1D observables")

    if hasattr(edges, "numpy"):
        edges = edges.numpy()

    arr = np.asarray(edges, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim != 1:
        raise ValueError("Binned fits currently support only 1D observables")

    return arr


def _build_binned_space(fit_model, bins):
    native_space = _native_binned_space_from_data(getattr(fit_model, "data", None))
    if native_space is not None:
        return native_space

    if is_likely_counting_model(fit_model):
        return _build_counting_binned_space(fit_model)

    obs_names = getattr(fit_model.obs, "obs", None)
    if not obs_names or len(obs_names) != 1:
        raise ValueError("Binned fits currently support only 1D observables")

    low, high = fit_model.obs_range
    edges = np.linspace(float(low), float(high), int(bins) + 1)
    binning = zfit.binned.VariableBinning(edges, name=obs_names[0])
    return zfit.Space(obs_names[0], binning=binning)


def _build_channel_binned_spaces(fit_model, bins):
    channel_obs = getattr(fit_model, "channel_obs", {}) or {}
    channel_ranges = getattr(fit_model, "channel_obs_ranges", {}) or {}
    channel_data = getattr(fit_model, "data", {}) or {}
    spaces = {}
    for channel, obs_space in channel_obs.items():
        native_space = None
        if isinstance(channel_data, dict):
            native_space = _native_binned_space_from_data(channel_data.get(channel))
        if native_space is not None:
            spaces[channel] = native_space
            continue

        obs_names = getattr(obs_space, "obs", None)
        if not obs_names or len(obs_names) != 1:
            raise ValueError("Binned fits currently support only 1D observables per channel")

        low, high = channel_ranges.get(channel, tuple(float(x) for x in obs_space.limit1d))
        edges = np.linspace(float(low), float(high), int(bins) + 1)
        binning = zfit.binned.VariableBinning(edges, name=obs_names[0])
        spaces[channel] = zfit.Space(obs_names[0], binning=binning)
    return spaces


def _build_channel_binned_models(fit_model, channel_binned_spaces):
    channel_models = _channel_models(fit_model)
    return {
        channel: model.to_binned(channel_binned_spaces[channel])
        for channel, model in channel_models.items()
    }


def _make_binned_toy_data(model, binned_space):
    edges = _binning_edges_as_float_array(binned_space.binning)

    # Binned PDFs in zfit expect integer/None n in sample(); passing "auto"
    # triggers a TF cast error. Build toys from expected per-bin counts.
    is_binned_model = isinstance(model, zfit.core.binnedpdf.BaseBinnedPDF)
    if is_binned_model:
        try:
            expected_counts = np.asarray(model.values(), dtype=float).reshape(-1)
        except Exception:
            rel_counts = np.asarray(model.rel_counts(model.space), dtype=float).reshape(-1)
            total_yield = 1.0
            get_yield = getattr(model, "get_yield", None)
            if callable(get_yield):
                try:
                    total_yield = float(get_yield().value())
                except Exception:
                    total_yield = 1.0
            expected_counts = rel_counts * total_yield
        expected_counts = np.clip(
            np.nan_to_num(expected_counts, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            None,
        )
        counts = np.random.poisson(expected_counts).astype(float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        values = np.repeat(centers, counts.astype(int)).astype(float)
        data = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts)
        return data, values, edges, counts

    sample = model.sample(n="auto")
    values = np.asarray(sample.value(), dtype=float).reshape(-1)
    counts, _ = np.histogram(values, bins=edges)
    data = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts.astype(float))
    return data, values, edges, counts.astype(float)


def _guess_binned_space_for_model(fit_model, model, channel=None, bins=40):
    data = getattr(fit_model, "data", None)
    if channel is not None and isinstance(data, dict):
        native = _native_binned_space_from_data(data.get(channel))
        if native is not None:
            return native
    else:
        native = _native_binned_space_from_data(data)
        if native is not None:
            return native

    model_space = getattr(model, "space", None)
    if model_space is not None and getattr(model_space, "binned", False):
        return model_space

    if channel is not None:
        channel_obs = (getattr(fit_model, "channel_obs", {}) or {}).get(channel)
        if channel_obs is not None and getattr(channel_obs, "binned", False):
            return channel_obs

    low, high = None, None
    if channel is not None:
        channel_ranges = getattr(fit_model, "channel_obs_ranges", {}) or {}
        if channel in channel_ranges:
            low, high = channel_ranges[channel]
    if low is None or high is None:
        low, high = getattr(fit_model, "obs_range", (0.0, 1.0))

    obs_name = "obs"
    obs_names = getattr(model_space, "obs", None)
    if obs_names:
        obs_name = obs_names[0]
    edges = np.linspace(float(low), float(high), int(bins) + 1)
    return zfit.Space(obs_name, binning=zfit.binned.VariableBinning(edges, name=obs_name))


def _make_unbinned_toy_data_from_binned_model(fit_model, model, channel=None, bins=40):
    binned_space = _guess_binned_space_for_model(fit_model, model, channel=channel, bins=bins)
    if hasattr(model, "to_binned"):
        binned_model = model.to_binned(binned_space)
    else:
        binned_model = model

    expected_counts = np.asarray(binned_model.values(), dtype=float).reshape(-1)
    expected_counts = np.clip(np.nan_to_num(expected_counts, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    toy_counts = np.random.poisson(expected_counts).astype(float)

    edges = _binning_edges_as_float_array(binned_space.binning)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.repeat(centers, toy_counts.astype(int))
    unbinned_data = _values_to_unbinned_dataset(values, getattr(model, "space", None))
    return unbinned_data, values.astype(float), edges, toy_counts


def _channel_models(fit_model):
    return getattr(fit_model, "channel_models", {}) or {}


def _all_models(fit_model):
    channel_models = _channel_models(fit_model)
    if channel_models:
        return list(channel_models.values())
    return [fit_model.model]


def _resolve_process_key(process_map, process_name):
    if not process_map or process_name is None:
        return None

    if process_name in process_map:
        return process_name

    suffixed = [name for name in process_map if name.startswith(f"{process_name}__")]
    if len(suffixed) == 1:
        return suffixed[0]

    if "__" in process_name:
        base_name = process_name.split("__", 1)[0]
        if base_name in process_map:
            return base_name

    return None


def _all_params(fit_model):
    params = []
    seen = set()

    def _iter_child_params(param):
        children = getattr(param, "params", None)
        if children is None:
            return
        if isinstance(children, dict):
            iterable = children.values()
        else:
            iterable = children
        for child in iterable:
            candidate = child
            if isinstance(child, tuple) and len(child) >= 2:
                candidate = child[1]
            if hasattr(candidate, "value"):
                yield candidate

    def _collect(param):
        ident = id(param)
        if ident in seen:
            return
        seen.add(ident)
        params.append(param)
        for child in _iter_child_params(param):
            _collect(child)

    for model in _all_models(fit_model):
        for kwargs in ({}, {"floating": None}, {"floating": None, "is_yield": None}):
            try:
                model_params = list(model.get_params(**kwargs))
            except Exception:
                continue
            for param in model_params:
                _collect(param)

    for param in (getattr(fit_model, "yields", {}) or {}).values():
        if hasattr(param, "value"):
            _collect(param)

    return params


def _capture_fit_model_parameter_values(fit_model):
    values = {}
    for param in _all_params(fit_model):
        if not hasattr(param, "set_value"):
            continue
        try:
            values[param] = float(param.value())
        except Exception:
            # Some composed parameters are not directly settable/restorable.
            continue
    return values


def _capture_parameter_values(model):
    values = {}
    for kwargs in ({}, {"floating": None}, {"floating": None, "is_yield": None}):
        try:
            params = list(model.get_params(**kwargs))
        except Exception:
            continue
        for param in params:
            if not hasattr(param, "set_value"):
                continue
            try:
                values[param] = float(param.value())
            except Exception:
                continue
    return values


def _restore_parameter_values(saved_values):
    for param, value in saved_values.items():
        try:
            param.set_value(value)
        except Exception:
            # Skip parameters that cannot be set directly (e.g. composed params).
            continue


def _find_signal_parameter(fit_model):
    if fit_model.signal_process is not None:
        target = f"mu_{fit_model.signal_process}"
        for param in _all_params(fit_model):
            if param.name == target:
                return param

    for param in _all_params(fit_model):
        if param.name.startswith("mu_"):
            return param

    if fit_model.signal_process and getattr(fit_model, "yields", None):
        matched_key = _resolve_process_key(fit_model.yields, fit_model.signal_process)
        if matched_key is not None:
            return fit_model.yields[matched_key]

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
    channel_models = _channel_models(fit_model)

    if channel_models:
        if resolved_fit_mode == "binned":
            if not isinstance(binned_space, dict):
                raise ValueError("Expected per-channel binned spaces for channel-based binned fit")

            channel_data = {}
            channel_binned = {}
            for channel, model in channel_models.items():
                if channel not in binned_space:
                    raise ValueError(f"Missing binned space for channel '{channel}'")
                data, values, edges, counts = _make_binned_toy_data(model, binned_space[channel])
                channel_data[channel] = data
                channel_binned[channel] = {
                    "edges": edges.tolist(),
                    "counts": counts.tolist(),
                    "values": values.tolist(),
                }
            dataset_plot = {"mode": "binned", "channel_binned": channel_binned}
            return channel_data, None, dataset_plot

        channel_data = {}
        channel_values = {}
        for channel, model in channel_models.items():
            try:
                sample = model.sample(n="auto")
                values = np.asarray(sample.value(), dtype=float).reshape(-1)
                channel_data[channel] = sample
            except Exception:
                sample, values, _edges, _counts = _make_unbinned_toy_data_from_binned_model(
                    fit_model,
                    model,
                    channel=channel,
                )
                channel_data[channel] = sample
            channel_values[channel] = values
        dataset_plot = {"mode": "unbinned", "channel_values": channel_values}
        return channel_data, None, dataset_plot

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
            dataset_plot = {"mode": "binned", "edges": edges, "counts": counts}
            if channel_counts:
                dataset_plot["channel_counts"] = channel_counts
            return data, toy_count, dataset_plot

        data, values, edges, counts = _make_binned_toy_data(fit_model.model, binned_space)
        dataset_plot = {
            "mode": "binned",
            "edges": edges,
            "counts": counts,
            "values": values,
        }
        return data, None, dataset_plot

    try:
        data = fit_model.model.sample(n="auto")
        values = np.asarray(data.value(), dtype=float).reshape(-1)
    except Exception:
        data, values, _edges, _counts = _make_unbinned_toy_data_from_binned_model(
            fit_model,
            fit_model.model,
        )
    dataset_plot = {"mode": "unbinned", "values": values}
    return data, None, dataset_plot


def _build_asimov_binned_data(binned_model, binned_space, fit_model):
    if isinstance(binned_model, dict):
        data = {}
        channel_binned = {}
        for channel, model in binned_model.items():
            channel_data = model.to_binneddata()
            expected_counts = np.asarray(model.values(), dtype=float).reshape(-1)
            edges = _binning_edges_as_float_array(binned_space[channel].binning)
            data[channel] = channel_data
            channel_binned[channel] = {
                "edges": edges.tolist(),
                "counts": expected_counts.tolist(),
            }
        dataset_plot = {
            "mode": "binned",
            "channel_binned": channel_binned,
            "asimov": True,
        }
        return data, channel_binned, dataset_plot

    data = binned_model.to_binneddata()
    expected_counts = np.asarray(binned_model.values(), dtype=float).reshape(-1)
    edges = _binning_edges_as_float_array(binned_space.binning)
    dataset_plot = {
        "mode": "binned",
        "edges": edges,
        "counts": expected_counts,
        "asimov": True,
    }
    channel_expectations = _expected_counts_by_channel(fit_model)
    if channel_expectations:
        dataset_plot["channel_counts"] = channel_expectations
    return data, expected_counts, dataset_plot


def _build_loss(fit_model, resolved_fit_mode, binned_model, data):
    channel_models = _channel_models(fit_model)

    if channel_models:
        if not isinstance(data, dict):
            raise ValueError("Expected per-channel dataset dictionary for channel-based model")

        combined_loss = None
        for index, (channel, model) in enumerate(channel_models.items()):
            if channel not in data:
                raise ValueError(f"Missing dataset for channel '{channel}'")
            constraints = fit_model.constraints if index == 0 else []
            if resolved_fit_mode == "binned":
                if not isinstance(binned_model, dict) or channel not in binned_model:
                    raise ValueError(f"Missing binned model for channel '{channel}'")
                loss = zfit.loss.ExtendedBinnedNLL(
                    model=binned_model[channel],
                    data=data[channel],
                    constraints=constraints,
                )
            else:
                loss = zfit.loss.ExtendedUnbinnedNLL(
                    model=model,
                    data=data[channel],
                    constraints=constraints,
                )
            combined_loss = loss if combined_loss is None else combined_loss + loss
        return combined_loss

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


def _compute_nll_scan_for_plot(loss, minimizer, poi_param, poi_unc, fit_model, poi_scan_max=None, n_points=121):
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
    bestfit_values = _capture_fit_model_parameter_values(fit_model)
    was_floating = getattr(poi_param, "floating", True)

    nll_values = np.full(int(n_points), np.nan, dtype=float)
    try:
        poi_param.floating = False

        # Start from the point nearest the global best-fit value and profile
        # outward in both directions using warm starts to reduce scan artifacts.
        center_idx = int(np.argmin(np.abs(scan_values - poi_best_fit)))

        _restore_parameter_values(bestfit_values)
        poi_param.set_value(float(scan_values[center_idx]))
        minimizer.minimize(loss)
        nll_values[center_idx] = float(loss.value())

        for idx in range(center_idx + 1, len(scan_values)):
            poi_param.set_value(float(scan_values[idx]))
            minimizer.minimize(loss)
            nll_values[idx] = float(loss.value())

        _restore_parameter_values(bestfit_values)
        poi_param.set_value(float(scan_values[center_idx]))
        minimizer.minimize(loss)
        nll_values[center_idx] = float(loss.value())
        for idx in range(center_idx - 1, -1, -1):
            poi_param.set_value(float(scan_values[idx]))
            minimizer.minimize(loss)
            nll_values[idx] = float(loss.value())
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


def _estimate_poi_uncertainty_from_profile(loss, minimizer, poi_param, fit_model):
    scan = _compute_nll_scan_for_plot(
        loss=loss,
        minimizer=minimizer,
        poi_param=poi_param,
        poi_unc=None,
        fit_model=fit_model,
        n_points=41,
    )
    if not isinstance(scan, dict):
        return None

    x = np.asarray(scan.get("poi_values", []), dtype=float)
    y = np.asarray(scan.get("delta_nll_values", []), dtype=float)
    if x.size < 5 or y.size != x.size:
        return None

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 5:
        return None

    # A completely flat profile means the POI is unconstrained in this fit.
    if np.nanmax(y) < target:
        return float("inf")

    best_idx = int(np.argmin(y))
    target = 0.5

    def _crossing(xseg, yseg):
        if xseg.size < 2:
            return None
        for i in range(xseg.size - 1):
            y0, y1 = yseg[i], yseg[i + 1]
            if (y0 - target) == 0.0:
                return float(xseg[i])
            if (y0 - target) * (y1 - target) <= 0.0:
                if y1 == y0:
                    return float(0.5 * (xseg[i] + xseg[i + 1]))
                t = (target - y0) / (y1 - y0)
                return float(xseg[i] + t * (xseg[i + 1] - xseg[i]))
        return None

    x_left = x[: best_idx + 1][::-1]
    y_left = y[: best_idx + 1][::-1]
    x_right = x[best_idx:]
    y_right = y[best_idx:]

    left_cross = _crossing(x_left, y_left)
    right_cross = _crossing(x_right, y_right)
    center = float(x[best_idx])

    candidates = []
    if left_cross is not None:
        candidates.append(center - left_cross)
    if right_cross is not None:
        candidates.append(right_cross - center)
    if not candidates:
        return None

    unc = float(np.nanmean(np.asarray(candidates, dtype=float)))
    if np.isfinite(unc) and unc > 0.0:
        return unc
    return None


def _extract_fit_parameter_hesse_errors(fit_result):
    params = [param for param in fit_result.params.keys() if getattr(param, "floating", False)]
    if not params:
        return {}

    try:
        hesse = fit_result.hesse(params=params)
    except Exception:
        return {}

    errors = {}
    for param in params:
        entry = hesse.get(param)
        if not isinstance(entry, dict):
            continue
        error = entry.get("error")
        if error is None:
            continue
        error = float(error)
        if np.isfinite(error) and error > 0.0:
            errors[param.name] = error
    return errors


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


def _observed_dataset_to_values(dataset, fallback_space=None):
    value_method = getattr(dataset, "value", None)
    if callable(value_method):
        return np.asarray(value_method(), dtype=float).reshape(-1)

    values_method = getattr(dataset, "values", None)
    if callable(values_method):
        values = np.asarray(values_method(), dtype=float).reshape(-1)
        data_space = getattr(dataset, "space", fallback_space)
        obs_names = tuple(getattr(data_space, "obs", ()) or ()) if data_space is not None else ()
        has_binning = False
        if len(obs_names) == 1 and data_space is not None:
            try:
                _ = data_space.binning[obs_names[0]].edges
                has_binning = True
            except Exception:
                has_binning = False

        if has_binning:
            edges = np.asarray(data_space.binning[obs_names[0]].edges, dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            counts = np.maximum(np.rint(values).astype(int), 0)
            return np.repeat(centers, counts)

        return values

    return np.array([float(dataset)], dtype=float)


def _values_to_unbinned_dataset(values, obs_space):
    values = np.asarray(values, dtype=float).reshape(-1)
    unbinned_space = obs_space
    if obs_space is not None and hasattr(obs_space, "obs"):
        try:
            unbinned_space = zfit.Space(obs=obs_space.obs, limits=obs_space.limits)
        except Exception:
            unbinned_space = obs_space

    n_obs = len(getattr(unbinned_space, "obs", ()) or ()) if unbinned_space is not None else 1
    if n_obs <= 1:
        array = values.reshape(-1, 1)
    else:
        array = values.reshape(-1, n_obs)
    return zfit.Data.from_numpy(obs=unbinned_space, array=array)


def _build_observed_input(fit_model, resolved_fit_mode, binned_space):
    channel_models = _channel_models(fit_model)

    if channel_models and resolved_fit_mode == "binned":
        if not isinstance(fit_model.data, dict):
            raise ValueError("Observed data for channel-based binned fit must be a per-channel dictionary")
        if not isinstance(binned_space, dict):
            raise ValueError("Expected per-channel binned spaces for channel-based binned fit")

        channel_data = {}
        channel_binned = {}
        for channel, dataset in fit_model.data.items():
            if channel not in binned_space:
                raise ValueError(f"Missing binned space for channel '{channel}'")
            edges = _binning_edges_as_float_array(binned_space[channel].binning)
            values = _observed_dataset_to_values(dataset, binned_space[channel])
            counts, _ = np.histogram(values, bins=edges)
            channel_data[channel] = zfit.data.BinnedData.from_tensor(
                space=binned_space[channel],
                values=counts.astype(float),
            )
            channel_binned[channel] = {
                "edges": edges.tolist(),
                "counts": counts.astype(float).tolist(),
                "values": values.tolist(),
            }

        dataset_plot = {
            "mode": "binned",
            "channel_binned": channel_binned,
            "observed": True,
        }
        return channel_data, None, dataset_plot

    if resolved_fit_mode == "binned":
        edges = _binning_edges_as_float_array(binned_space.binning)
        if hasattr(fit_model.data, "value") or hasattr(fit_model.data, "values"):
            observed_values = _observed_dataset_to_values(fit_model.data, binned_space)
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

        dataset_plot = {
            "mode": "binned",
            "edges": edges,
            "counts": counts.astype(float),
            "values": observed_values,
            "observed": True,
        }
        if getattr(fit_model, "observed_counts_by_channel", None):
            dataset_plot["channel_counts"] = {
                k: float(v)
                for k, v in fit_model.observed_counts_by_channel.items()
            }
        return data, None, dataset_plot

    data = fit_model.data
    if channel_models:
        if not isinstance(data, dict):
            raise ValueError("Observed data for mixed-observable channels must be a per-channel dictionary")
        channel_data = {}
        channel_values = {}
        for channel, dataset in data.items():
            values = _observed_dataset_to_values(dataset)
            channel_values[channel] = values
            if hasattr(dataset, "value") and not hasattr(dataset, "values"):
                channel_data[channel] = dataset
            else:
                channel_data[channel] = _values_to_unbinned_dataset(values, channel_models[channel].space)
        dataset_plot = {"mode": "unbinned", "channel_values": channel_values, "observed": True}
        return channel_data, None, dataset_plot

    observed_values = _observed_dataset_to_values(data)
    unbinned_data = data
    if not hasattr(data, "value") or hasattr(data, "values"):
        unbinned_data = _values_to_unbinned_dataset(observed_values, fit_model.model.space)
    dataset_plot = {"mode": "unbinned", "values": observed_values, "observed": True}
    if getattr(fit_model, "observed_values_by_channel", None):
        dataset_plot["channel_values"] = {
            k: np.asarray(v, dtype=float).reshape(-1)
            for k, v in fit_model.observed_values_by_channel.items()
        }
    return unbinned_data, None, dataset_plot


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
            "dataset_id": sample_index + 1,
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
            "fit_param_hesse": _extract_fit_parameter_hesse_errors(fit_result),
        }

    result = minimizer.minimize(loss)
    poi_unc = _extract_hesse_error(result, poi_param)
    if poi_unc is None:
        poi_unc = _estimate_poi_uncertainty_from_profile(loss, minimizer, poi_param, fit_model)
    return {
        "dataset_id": sample_index + 1,
        "valid": bool(result.valid),
        "edm": float(result.edm) if result.edm is not None else None,
        "fit_mode": resolved_fit_mode,
        "poi_name": poi_param.name,
        "poi_fit": float(poi_param.value()),
        "poi_unc_hesse": poi_unc,
        "fit_params": _extract_fit_parameter_values(result),
        "fit_param_hesse": _extract_fit_parameter_hesse_errors(result),
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
    pre_bonly_values = _capture_fit_model_parameter_values(fit_model)
    pre_bonly_float = bool(getattr(signal_param, "floating", True))
    try:
        signal_param.set_value(0.0)
        signal_param.floating = False
        minimizer.minimize(loss)
        bonly_values = _capture_fit_model_parameter_values(fit_model)

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
    fc_scan_points=21,
    fc_n_toys=100,
    fc_scan_max=None,
):
    fc_alpha = float(feldman_cousins_alpha)
    if not (0.0 < fc_alpha < 1.0):
        raise ValueError("Feldman-Cousins alpha must satisfy 0 < alpha < 1")
    fc_cl = 1.0 - fc_alpha

    fc_scan_points = int(fc_scan_points)
    fc_n_toys = int(fc_n_toys)
    if fc_scan_points < 3:
        raise ValueError("Feldman-Cousins scan requires at least 3 POI grid points")
    if fc_n_toys < 1:
        raise ValueError("Feldman-Cousins requires at least 1 toy per POI point")

    grid_max = float(fc_scan_max) if fc_scan_max is not None else _default_scan_max(poi_param, fit_model)
    if not np.isfinite(grid_max) or grid_max <= 0.0:
        raise ValueError("Feldman-Cousins scan max must be finite and > 0")

    poi_grid = np.linspace(0.0, grid_max, fc_scan_points)
    fc_toy_fit_results = []

    minimizer = zfit.minimize.Minuit()
    starting_values = _capture_fit_model_parameter_values(fit_model)
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
            "fc_scan_points": int(fc_scan_points),
            "fc_n_toys": int(fc_n_toys),
            "fc_scan_max": float(grid_max),
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
    feldman_cousins_scan_points,
    feldman_cousins_n_toys,
    feldman_cousins_scan_max,
    compute_nll_scan,
    nll_scan_points,
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
    summary["dataset_time_s"] = time.perf_counter() - start_time
    summary["dataset_plot"] = data_plot
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
            n_points=nll_scan_points,
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
    if poi_true is not None and np.isfinite(float(poi_true)):
        summary["poi_true"] = float(poi_true)

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
                fc_scan_points=feldman_cousins_scan_points,
                fc_n_toys=feldman_cousins_n_toys,
                fc_scan_max=feldman_cousins_scan_max,
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
    feldman_cousins_scan_points=21,
    feldman_cousins_n_toys=100,
    feldman_cousins_scan_max=None,
    checkpoint_freq=None,  # Frequency of checkpointing
    checkpoint_path=None,   # Path to save checkpoints
    existing_results=None,
    resume_from_index=0,
    compute_nll_scan=False,
    nll_scan_points=121,
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

    starting_values = _capture_fit_model_parameter_values(fit_model)
    summaries = list(existing_results) if existing_results else []
    minimizer = zfit.minimize.Minuit()
    binned_space = None
    binned_model = None
    if resolved_fit_mode == "binned":
        if _channel_models(fit_model):
            binned_space = _build_channel_binned_spaces(fit_model, binned_bins)
            binned_model = _build_channel_binned_models(fit_model, binned_space)
        else:
            binned_space = _build_binned_space(fit_model, binned_bins)
            binned_model = fit_model.model.to_binned(binned_space)

    if use_asimov_data and resolved_fit_mode != "binned":
        raise ValueError("--toys -1 is only supported for binned fits")

    if checkpoint_freq is not None and checkpoint_freq < 1:
        raise ValueError("checkpoint_freq must be >= 1")
    if int(nll_scan_points) < 3:
        raise ValueError("nll_scan_points must be >= 3")

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
            feldman_cousins_scan_points=feldman_cousins_scan_points,
            feldman_cousins_n_toys=feldman_cousins_n_toys,
            feldman_cousins_scan_max=feldman_cousins_scan_max,
            compute_nll_scan=(sample_index == 0 and compute_nll_scan),
            nll_scan_points=int(nll_scan_points),
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
                    "feldman_cousins_scan_points": int(feldman_cousins_scan_points),
                    "feldman_cousins_n_toys": int(feldman_cousins_n_toys),
                    "feldman_cousins_scan_max": feldman_cousins_scan_max,
                    "compute_nll_scan": bool(compute_nll_scan),
                    "nll_scan_points": int(nll_scan_points),
                }
                with open(checkpoint_path, "wb") as f:
                    dill.dump(checkpoint_data, f)
                label = "toys" if data_mode == "toy" else "datasets"
                print(f"Checkpoint saved: {len(summaries)}/{toys} {label} completed")
            except Exception as e:
                print(f"Warning: checkpoint save failed: {e}")

    return summaries
