import os

import numpy as np


def _find_parameter_by_name(fit_model, parameter_name):
    for param in fit_model.model.get_params():
        if param.name == parameter_name:
            return param
    return None


def _capture_parameter_values(model):
    values = {}
    for param in model.get_params():
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


def _plot_categories(summary, fit_model):
    toy_plot = summary.get("toy_plot", {})
    channel_values = toy_plot.get("channel_values")
    if isinstance(channel_values, dict) and channel_values:
        return list(channel_values.keys())

    channel_counts = toy_plot.get("channel_counts")
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


def plot_dataset_and_components(summary, fit_model, plot_dir, binned_bins):
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
        mode = toy_plot.get("mode")
        categories = _plot_categories(summary, fit_model)

        if summary.get("asimov_fit") or toy_plot.get("asimov"):
            data_label = "Asimov data"
            title_prefix = "Asimov Data and Fit Components"
        elif summary.get("observed_fit") or toy_plot.get("observed"):
            data_label = "Observed data"
            title_prefix = "Observed Data and Fit Components"
        else:
            data_label = "Toy data"
            title_prefix = f"Toy {summary['toy']} Dataset and Fit Components"

        for category in categories:
            fig, ax = plt.subplots(figsize=(8, 5))

            if mode == "binned":
                edges = np.asarray(toy_plot["edges"], dtype=float).reshape(-1)
                counts = np.asarray(toy_plot["counts"], dtype=float).reshape(-1)
                channel_counts = toy_plot.get("channel_counts")
                if category is not None and isinstance(channel_counts, dict):
                    if len(counts) == 1:
                        counts = np.array([float(channel_counts.get(category, 0.0))], dtype=float)
                    else:
                        # Multi-bin per-channel datasets are only supported when explicit channel values are provided.
                        if "channel_values" not in toy_plot:
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
                    ax.step(edges[:-1], total_counts, where="post", color="black", linewidth=1.8, label="Total model")
                if bkg_counts.size:
                    ax.step(edges[:-1], bkg_counts, where="post", color="tab:blue", linestyle="--", linewidth=1.6, label="Total background")
                if sig_counts.size:
                    ax.step(edges[:-1], sig_counts, where="post", color="tab:red", linestyle="-.", linewidth=1.6, label="Signal")

                ax.set_xlabel(fit_model.obs.obs[0] if getattr(fit_model.obs, "obs", None) else "obs")
                ax.set_ylabel("Entries")

            else:
                values_source = toy_plot.get("values")
                if category is not None:
                    channel_values = toy_plot.get("channel_values")
                    if isinstance(channel_values, dict) and category in channel_values:
                        values_source = channel_values[category]
                    else:
                        plt.close(fig)
                        continue

                values = np.asarray(values_source, dtype=float)
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
                    label=data_label,
                )

                total_y, sig_y, bkg_y = _component_counts_by_channel(fit_model, edges, category)
                if np.any(total_y):
                    ax.plot(centers, total_y, color="black", linewidth=1.8, label="Total model")
                if np.any(sig_y):
                    ax.plot(centers, sig_y, color="tab:red", linestyle="-.", linewidth=1.6, label="Signal")
                if np.any(bkg_y):
                    ax.plot(centers, bkg_y, color="tab:blue", linestyle="--", linewidth=1.6, label="Total background")

                ax.set_xlabel(fit_model.obs.obs[0] if getattr(fit_model.obs, "obs", None) else "obs")
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
            fig.savefig(os.path.join(plot_dir, f"toy_{summary['toy']:04d}_dataset_fit{file_suffix}.png"), dpi=140)
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
            break  # only first toy that has a scan


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

    if summary.get("asimov_fit") or summary.get("toy_plot", {}).get("asimov"):
        title = f"NLL Profile – Asimov ({poi_name})"
    elif summary.get("observed_fit") or summary.get("toy_plot", {}).get("observed"):
        title = f"NLL Profile – Observed data ({poi_name})"
    else:
        title = f"NLL Profile – Toy {summary['toy']} ({poi_name})"

    ax.set_title(title)
    ax.set_xlabel(poi_name)
    ax.set_ylabel(r"$\Delta$ NLL")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"nll_profile_{summary['toy']:04d}.png"), dpi=140)
    plt.close(fig)
