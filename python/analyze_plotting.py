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
        fig, ax = plt.subplots(figsize=(8, 5))
        mode = toy_plot.get("mode")
        signal_category = _find_total_signal_category(fit_model)

        if summary.get("asimov_fit") or toy_plot.get("asimov"):
            data_label = "Asimov data"
        elif summary.get("observed_fit") or toy_plot.get("observed"):
            data_label = "Observed data"
        else:
            data_label = "Toy data"

        if mode == "binned":
            edges = np.asarray(toy_plot["edges"], dtype=float).reshape(-1)
            counts = np.asarray(toy_plot["counts"], dtype=float).reshape(-1)
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

            total_counts = np.asarray(summary.get("total_model_counts", []), dtype=float)
            if total_counts.size == 0:
                model_values = np.asarray(fit_model.model.sample(n="auto").value(), dtype=float).reshape(-1)
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
                label=data_label,
            )

            total_y = np.zeros_like(counts, dtype=float)
            if getattr(fit_model, "shapes", None) and getattr(fit_model, "yields", None):
                for category, shape in fit_model.shapes.items():
                    if category not in fit_model.yields:
                        continue
                    total_y = total_y + _binned_component_counts(shape, fit_model.yields[category].value(), edges)
            if not np.any(total_y):
                x_plot = np.linspace(float(lower), float(upper), 1000)
                total_y = np.asarray(fit_model.model.pdf(x_plot), dtype=float).reshape(-1)
                if hasattr(fit_model.model, "get_yield"):
                    total_y = total_y * float(fit_model.model.get_yield().value()) * (x_plot[1] - x_plot[0])
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

        if summary.get("asimov_fit") or toy_plot.get("asimov"):
            title = "Asimov Data and Fit Components"
        elif summary.get("observed_fit") or toy_plot.get("observed"):
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
