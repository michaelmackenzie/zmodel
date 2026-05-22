import os

from zmodel.model_io import load_fit_model


def load_and_summarize_model(model_file: str):
    model_path = os.path.abspath(model_file)
    fit_model = load_fit_model(model_path)
    observed_count = None
    # Try to extract observed count for both counting and binned/unbinned
    if hasattr(fit_model, "data") and fit_model.data is not None:
        try:
            # For counting models, data is a float/int
            if isinstance(fit_model.data, (int, float)):
                observed_count = float(fit_model.data)
            elif isinstance(fit_model.data, dict):
                import numpy as np
                total = 0.0
                for data_item in fit_model.data.values():
                    if hasattr(data_item, "values"):
                        vals = data_item.values()
                        total += float(np.sum(np.asarray(vals, dtype=float)))
                    elif hasattr(data_item, "value"):
                        vals = data_item.value()
                        arr = np.asarray(vals)
                        total += float(arr.shape[0]) if arr.ndim == 1 else float(np.sum(arr))
                    else:
                        total += float(data_item)
                observed_count = total
            # For binned/unbinned, try to sum entries
            elif hasattr(fit_model.data, "values"):
                vals = fit_model.data.values()
                import numpy as np
                observed_count = float(np.sum(np.asarray(vals, dtype=float)))
            elif hasattr(fit_model.data, "value"):
                vals = fit_model.data.value()
                import numpy as np
                arr = np.asarray(vals)
                observed_count = float(arr.shape[0]) if arr.ndim == 1 else float(np.sum(arr))
        except Exception:
            observed_count = None
    channel_models = getattr(fit_model, "channel_models", {}) or {}
    if channel_models:
        seen = set()
        n_float = 0
        for model in channel_models.values():
            for param in model.get_params():
                ident = id(param)
                if ident in seen:
                    continue
                seen.add(ident)
                n_float += 1
    else:
        n_float = len(fit_model.model.get_params())

    return {
        "model_path": model_path,
        "model_name": fit_model.model.name,
        "obs_range": fit_model.obs_range,
        "processes": fit_model.process_names or "unknown",
        "signal_process": fit_model.signal_process,
        "constraints": len(fit_model.constraints),
        "floating_params": n_float,
        "observed_count": observed_count,
    }