import os

from model_io import load_fit_model


def load_and_summarize_model(model_file: str):
    model_path = os.path.abspath(model_file)
    fit_model = load_fit_model(model_path)
    return {
        "model_path": model_path,
        "model_name": fit_model.model.name,
        "obs_range": fit_model.obs_range,
        "processes": fit_model.process_names or "unknown",
        "signal_process": fit_model.signal_process,
        "constraints": len(fit_model.constraints),
        "floating_params": len(fit_model.model.get_params()),
    }