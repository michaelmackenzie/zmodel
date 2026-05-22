import os

import numpy as np


def _find_parameter_by_name(fit_model, parameter_name):
    seen = set()
    channel_models = getattr(fit_model, "channel_models", {}) or {}
    models = list(channel_models.values()) if channel_models else [fit_model.model]
    for model in models:
        for param in model.get_params():
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            if param.name == parameter_name:
                return param
    return None


def _capture_parameter_values(model):
    values = {}
    for param in model.get_params():
        if hasattr(param, "set_value"):
            values[param] = float(param.value())
    return values


def _capture_fit_model_parameter_values(fit_model):
    values = {}
    seen = set()
    channel_models = getattr(fit_model, "channel_models", {}) or {}
    models = list(channel_models.values()) if channel_models else [fit_model.model]
    for model in models:
        for param in model.get_params():
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            if hasattr(param, "set_value"):
                values[param] = float(param.value())
    return values


def _restore_parameter_values(saved_values):
    for param, value in saved_values.items():
        param.set_value(value)


def _restore_fit_params_by_name(fit_model, fit_params):
    for name, value in fit_params.items():
        param = _find_parameter_by_name(fit_model, name)
        if param is not None and hasattr(param, "set_value"):
            param.set_value(value)


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


def _safe_name(value):
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text)


def _summary_dataset_plot(summary):
    return summary.get("dataset_plot", {})


def _summary_dataset_id(summary):
    return int(summary.get("dataset_id", 0))


def _plot_categories(summary, fit_model):
    dataset_plot = _summary_dataset_plot(summary)
    channel_binned = dataset_plot.get("channel_binned")
    if isinstance(channel_binned, dict) and channel_binned:
        return list(channel_binned.keys())

    channel_values = dataset_plot.get("channel_values")
    if isinstance(channel_values, dict) and channel_values:
        return list(channel_values.keys())

    channel_counts = dataset_plot.get("channel_counts")
    if isinstance(channel_counts, dict) and channel_counts:
        return list(channel_counts.keys())

    channels = list(getattr(fit_model, "channels", []) or [])
    if len(channels) > 1:
        return channels

    return [None]


def _term_process_name(fit_model, term_name):
    term_processes = getattr(fit_model, "term_processes", {}) or {}
    if term_name in term_processes:
        return term_processes[term_name]
    if "__" in term_name:
        return term_name.split("__", 1)[0]
    return term_name


def _component_counts_by_channel(fit_model, edges, channel):
    total = np.zeros(len(edges) - 1, dtype=float)
    signal = np.zeros(len(edges) - 1, dtype=float)
    background = np.zeros(len(edges) - 1, dtype=float)
    term_channels = getattr(fit_model, "term_channels", {}) or {}

    for term_name, shape in getattr(fit_model, "shapes", {}).items():
        if term_name not in getattr(fit_model, "yields", {}):
            continue

        term_channel = term_channels.get(term_name)
        if channel is not None and term_channel is not None and term_channel != channel:
            continue

        comp = _binned_component_counts(shape, fit_model.yields[term_name].value(), edges)
        total = total + comp
        process = _term_process_name(fit_model, term_name)
        if getattr(fit_model, "signal_process", None) is not None and process == fit_model.signal_process:
            signal = signal + comp
        else:
            background = background + comp

    return total, signal, background


def _component_curve_by_channel(fit_model, x_plot, bin_width, channel):
    total = np.zeros(len(x_plot), dtype=float)
    signal = np.zeros(len(x_plot), dtype=float)
    background = np.zeros(len(x_plot), dtype=float)
    term_channels = getattr(fit_model, "term_channels", {}) or {}

    for term_name, shape in getattr(fit_model, "shapes", {}).items():
        if term_name not in getattr(fit_model, "yields", {}):
            continue

        term_channel = term_channels.get(term_name)
        if channel is not None and term_channel is not None and term_channel != channel:
            continue

        density = np.asarray(shape.pdf(x_plot), dtype=float).reshape(-1)
        comp = density * float(fit_model.yields[term_name].value()) * float(bin_width)
        total = total + comp

        process = _term_process_name(fit_model, term_name)
        if getattr(fit_model, "signal_process", None) is not None and process == fit_model.signal_process:
            signal = signal + comp
        else:
            background = background + comp

    return total, signal, background


def _parameter_shifted_value(param, delta):
    value = float(param.value()) + float(delta)
    lower = getattr(param, "lower", None)
    upper = getattr(param, "upper", None)
    if lower is not None and np.isfinite(float(lower)):
        value = max(value, float(lower))
    if upper is not None and np.isfinite(float(upper)):
        value = min(value, float(upper))
    return value


def _hessian_model_band(fit_model, fit_param_hesse, nominal_values, evaluate_total):
    if not fit_param_hesse:
        return None, None

    variance_up = np.zeros_like(nominal_values, dtype=float)
    variance_down = np.zeros_like(nominal_values, dtype=float)
    baseline_values = _capture_fit_model_parameter_values(fit_model)
    try:
        for param_name, sigma in fit_param_hesse.items():
            if not np.isfinite(float(sigma)) or float(sigma) <= 0.0:
                continue

            param = _find_parameter_by_name(fit_model, param_name)
            if param is None or not hasattr(param, "set_value"):
                continue

            base_value = float(param.value())
            up_value = _parameter_shifted_value(param, float(sigma))
            down_value = _parameter_shifted_value(param, -float(sigma))
            if up_value == base_value and down_value == base_value:
                continue

            param.set_value(up_value)
            up_values = np.asarray(evaluate_total(), dtype=float)

            param.set_value(down_value)
            down_values = np.asarray(evaluate_total(), dtype=float)

            up_delta = np.maximum(up_values - nominal_values, 0.0)
            up_delta = np.maximum(up_delta, nominal_values - down_values)

            down_delta = np.maximum(nominal_values - up_values, 0.0)
            down_delta = np.maximum(down_delta, down_values - nominal_values)

            variance_up = variance_up + np.square(up_delta)
            variance_down = variance_down + np.square(down_delta)

            param.set_value(base_value)
    finally:
        _restore_parameter_values(baseline_values)

    sigma_up = np.sqrt(np.maximum(variance_up, 0.0))
    sigma_down = np.sqrt(np.maximum(variance_down, 0.0))
    lower = np.maximum(nominal_values - sigma_down, 0.0)
    upper = nominal_values + sigma_up
    return lower, upper


def plot_dataset_and_components(summary, fit_model, plot_dir, binned_bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    dataset_plot = _summary_dataset_plot(summary)
    if not dataset_plot:
        return

    dataset_id = _summary_dataset_id(summary)

    fit_params = summary.get("fit_params", {})
    fit_param_hesse = summary.get("fit_param_hesse", {})
    baseline_values = _capture_fit_model_parameter_values(fit_model)
    _restore_fit_params_by_name(fit_model, fit_params)

    try:
        mode = dataset_plot.get("mode")
        categories = _plot_categories(summary, fit_model)

        if summary.get("asimov_fit") or dataset_plot.get("asimov"):
            data_label = "Asimov data"
            title_prefix = "Asimov Data and Fit Components"
        elif summary.get("observed_fit") or dataset_plot.get("observed"):
            data_label = "Observed data"
            title_prefix = "Observed Data and Fit Components"
        else:
            data_label = "Toy data"
            title_prefix = f"Toy {dataset_id} Dataset and Fit Components"

        for category in categories:
            fig, ax = plt.subplots(figsize=(8, 5))

            if mode == "binned":
                channel_binned = dataset_plot.get("channel_binned")
                if category is not None and isinstance(channel_binned, dict) and category in channel_binned:
                    edges = np.asarray(channel_binned[category]["edges"], dtype=float).reshape(-1)
                    counts = np.asarray(channel_binned[category]["counts"], dtype=float).reshape(-1)
                else:
                    edges = np.asarray(dataset_plot["edges"], dtype=float).reshape(-1)
                    counts = np.asarray(dataset_plot["counts"], dtype=float).reshape(-1)
                channel_counts = dataset_plot.get("channel_counts")
                if category is not None and isinstance(channel_counts, dict):
                    if len(counts) == 1:
                        counts = np.array([float(channel_counts.get(category, 0.0))], dtype=float)
                    else:
                        # Multi-bin per-channel datasets are only supported when explicit channel values are provided.
                        if "channel_values" not in dataset_plot:
                            plt.close(fig)
                            continue

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
                    label=data_label,
                )

                total_counts, sig_counts, bkg_counts = _component_counts_by_channel(fit_model, edges, category)
                if not np.any(total_counts):
                    total_counts = np.asarray(summary.get("total_model_counts", []), dtype=float)
                if total_counts.size:
                    band_low, band_high = _hessian_model_band(
                        fit_model,
                        fit_param_hesse,
                        total_counts,
                        lambda: _component_counts_by_channel(fit_model, edges, category)[0],
                    )
                    if band_low is not None and band_high is not None:
                        ax.fill_between(
                            edges[:-1],
                            band_low,
                            band_high,
                            step="post",
                            color="gray",
                            alpha=0.25,
                            linewidth=0.0,
                            label=r"Total model $\pm 1\sigma$",
                        )
                    ax.step(edges[:-1], total_counts, where="post", color="black", linewidth=1.8, label="Total model")
                if bkg_counts.size:
                    ax.step(edges[:-1], bkg_counts, where="post", color="tab:blue", linestyle="--", linewidth=1.6, label="Total background")
                if sig_counts.size:
                    ax.step(edges[:-1], sig_counts, where="post", color="tab:red", linestyle="-.", linewidth=1.6, label="Signal")

                ax.set_xlabel(fit_model.obs.obs[0] if getattr(fit_model.obs, "obs", None) else "obs")
                ax.set_ylabel("Entries")

            else:
                values_source = dataset_plot.get("values")
                obs_label = fit_model.obs.obs[0] if getattr(fit_model.obs, "obs", None) else "obs"
                lower, upper = fit_model.obs_range
                if category is not None:
                    channel_values = dataset_plot.get("channel_values")
                    if isinstance(channel_values, dict) and category in channel_values:
                        values_source = channel_values[category]
                    else:
                        plt.close(fig)
                        continue

                    channel_ranges = getattr(fit_model, "channel_obs_ranges", {}) or {}
                    if category in channel_ranges:
                        lower, upper = channel_ranges[category]

                    channel_obs = getattr(fit_model, "channel_obs", {}) or {}
                    channel_space = channel_obs.get(category)
                    if channel_space is not None and getattr(channel_space, "obs", None):
                        obs_label = channel_space.obs[0]

                values = np.asarray(values_source, dtype=float)
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
                    label=data_label,
                )

                bin_width = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
                n_curve = max(400, int(binned_bins) * 5)
                x_curve = np.linspace(float(lower), float(upper), n_curve)

                total_y, sig_y, bkg_y = _component_curve_by_channel(fit_model, x_curve, bin_width, category)
                if np.any(total_y):
                    band_low, band_high = _hessian_model_band(
                        fit_model,
                        fit_param_hesse,
                        total_y,
                        lambda: _component_curve_by_channel(fit_model, x_curve, bin_width, category)[0],
                    )
                    if band_low is not None and band_high is not None:
                        ax.fill_between(
                            x_curve,
                            band_low,
                            band_high,
                            color="gray",
                            alpha=0.25,
                            linewidth=0.0,
                            label=r"Total model $\pm 1\sigma$",
                        )
                    ax.plot(x_curve, total_y, color="black", linewidth=1.8, label="Total model")
                if np.any(sig_y):
                    ax.plot(x_curve, sig_y, color="tab:red", linestyle="-.", linewidth=1.6, label="Signal")
                if np.any(bkg_y):
                    ax.plot(x_curve, bkg_y, color="tab:blue", linestyle="--", linewidth=1.6, label="Total background")

                ax.set_xlabel(obs_label)
                ax.set_ylabel("Entries")

            title = title_prefix
            file_suffix = ""
            if category is not None:
                title = f"{title_prefix} [{category}]"
                file_suffix = f"_{_safe_name(category)}"

            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"data_{dataset_id:04d}_dataset_fit{file_suffix}.png"), dpi=140)
            plt.close(fig)
    finally:
        _restore_parameter_values(baseline_values)


def plot_summary_artifacts(summaries, fit_model, plot_dir, binned_bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

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

    for summary in summaries:
        plot_dataset_and_components(summary, fit_model, plot_dir, binned_bins)

    for summary in summaries:
        if "nll_scan" in summary:
            plot_nll_scan(summary, plot_dir)
            break  # only first dataset that has a scan


def plot_nll_scan(summary, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scan = summary["nll_scan"]
    poi_values = np.asarray(scan["poi_values"], dtype=float)
    delta_nll = np.asarray(scan["delta_nll_values"], dtype=float)
    poi_name = scan["poi_name"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(poi_values, delta_nll, color="tab:blue", linewidth=1.8)
    ax.axhline(0.5, color="tab:orange", linestyle="--", linewidth=1.2, label=r"1$\sigma$ ($\Delta$NLL = 0.5)")
    ax.axhline(2.0, color="tab:red", linestyle=":", linewidth=1.2, label=r"2$\sigma$ ($\Delta$NLL = 2.0)")

    # Mark best-fit value
    best_idx = int(np.argmin(delta_nll))
    ax.axvline(poi_values[best_idx], color="black", linestyle="-.", linewidth=1.0, label=f"Best fit: {poi_values[best_idx]:.3g}")

    dataset_plot = _summary_dataset_plot(summary)
    dataset_id = _summary_dataset_id(summary)

    if summary.get("asimov_fit") or dataset_plot.get("asimov"):
        title = f"NLL Profile – Asimov ({poi_name})"
    elif summary.get("observed_fit") or dataset_plot.get("observed"):
        title = f"NLL Profile – Observed data ({poi_name})"
    else:
        title = f"NLL Profile – Toy {dataset_id} ({poi_name})"

    ax.set_title(title)
    ax.set_xlabel(poi_name)
    ax.set_ylabel(r"$\Delta$ NLL")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"nll_profile_{dataset_id:04d}.png"), dpi=140)
    plt.close(fig)
