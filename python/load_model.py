import os

import numpy as np

from zmodel.model_io import load_fit_model


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _parameter_value_text(param):
    value = _safe_float(param.value()) if hasattr(param, "value") else None
    if value is None:
        return "n/a"
    lower = _safe_float(getattr(param, "lower", None))
    upper = _safe_float(getattr(param, "upper", None))
    range_text = ""
    if lower is not None and upper is not None:
        range_text = f" [{lower:g}, {upper:g}]"
    return f"{value:g}{range_text}"


def _dataset_observed_count(data):
    if data is None:
        return None
    if isinstance(data, (int, float)):
        return float(data)
    if isinstance(data, dict):
        total = 0.0
        for data_item in data.values():
            value = _dataset_observed_count(data_item)
            if value is None:
                return None
            total += value
        return total

    if hasattr(data, "values") and callable(data.values):
        values = np.asarray(data.values(), dtype=float)
        return float(np.sum(values))

    if hasattr(data, "value") and callable(data.value):
        values = np.asarray(data.value(), dtype=float)
        if values.ndim == 1:
            return float(values.shape[0])
        return float(np.sum(values))

    return None


def _iter_pdf_lines(pdf, indent=0, seen=None):
    if seen is None:
        seen = set()
    ident = id(pdf)
    prefix = "  " * indent
    if ident in seen:
        yield f"{prefix}- {type(pdf).__name__} {pdf.name} (already shown)"
        return
    seen.add(ident)

    params = []
    try:
        for param in pdf.get_params(floating=None):
            params.append(f"{param.name}={_parameter_value_text(param)}")
    except Exception:
        params = []

    param_text = f"; params: {', '.join(params)}" if params else ""
    yield f"{prefix}- {type(pdf).__name__} {pdf.name}{param_text}"

    for attr in ("pdfs", "models"):
        children = getattr(pdf, attr, None)
        if not children:
            continue
        for child in children:
            yield from _iter_pdf_lines(child, indent=indent + 1, seen=seen)


def _summarize_pdf_tree(model):
    lines = []
    seen = set()
    lines.append(f"Top-level PDF: {type(model).__name__} {model.name}")
    try:
        model_params = list(model.get_params(floating=None))
    except Exception:
        model_params = []
    if model_params:
        lines.append("Parameters:")
        for param in model_params:
            lines.append(f"  - {param.name} = {_parameter_value_text(param)}")
    for line in _iter_pdf_lines(model, indent=0, seen=seen):
        lines.append(line)
    return lines


def load_and_summarize_model(model_file: str, verbose: int = 0):
    model_path = os.path.abspath(model_file)
    fit_model = load_fit_model(model_path)
    observed_count = _dataset_observed_count(getattr(fit_model, "data", None))

    channel_models = getattr(fit_model, "channel_models", {}) or {}
    channel_names = list(channel_models.keys()) if channel_models else list(getattr(fit_model, "channels", []) or [])
    process_names = list(getattr(fit_model, "process_names", []) or [])

    poi_name = None
    for param in fit_model.model.get_params(floating=None):
        if getattr(param, "name", "").startswith("mu_"):
            poi_name = param.name
            break
    if poi_name is None and getattr(fit_model, "signal_process", None):
        poi_name = f"mu_{fit_model.signal_process}"

    if channel_models:
        seen = set()
        n_float = 0
        for model in channel_models.values():
            for param in model.get_params(floating=None):
                ident = id(param)
                if ident in seen:
                    continue
                seen.add(ident)
                n_float += 1
    else:
        n_float = len(fit_model.model.get_params(floating=None))

    pdf_lines = _summarize_pdf_tree(fit_model.model) if verbose else []
    if verbose > 1 and channel_models:
        pdf_lines.append("")
        pdf_lines.append("Channel PDFs:")
        for channel, pdf in channel_models.items():
            pdf_lines.append(f"Channel {channel}:")
            pdf_lines.extend(f"  {line}" for line in _iter_pdf_lines(pdf, indent=1, seen=set()))

    return {
        "model_path": model_path,
        "model_name": fit_model.model.name,
        "obs_range": fit_model.obs_range,
        "channels": channel_names,
        "processes": process_names or "unknown",
        "signal_process": fit_model.signal_process,
        "poi_name": poi_name,
        "constraints": len(fit_model.constraints),
        "floating_params": n_float,
        "observed_count": observed_count,
        "pdf_lines": pdf_lines,
    }