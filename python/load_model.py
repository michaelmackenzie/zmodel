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
    return {
        "model_path": model_path,
        "model_name": fit_model.model.name,
        "obs_range": fit_model.obs_range,
        "processes": fit_model.process_names or "unknown",
        "signal_process": fit_model.signal_process,
        "constraints": len(fit_model.constraints),
        "floating_params": len(fit_model.model.get_params()),
        "observed_count": observed_count,
    }