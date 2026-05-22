def find_parameter_by_name(fit_model, parameter_name):
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


def _parse_parameter_value_map(spec):
    if spec is None:
        return {}

    assignments = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid parameter assignment '{item}'. Expected format name=value")
        name, value_text = item.split("=", 1)
        name = name.strip()
        value_text = value_text.strip()
        if not name:
            raise ValueError(f"Invalid parameter assignment '{item}'")
        assignments[name] = float(value_text)
    return assignments


def _parse_parameter_range_map(spec):
    if spec is None:
        return {}

    ranges = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item or ":" not in item:
            raise ValueError(f"Invalid range assignment '{item}'. Expected format name=low:high")
        name, bounds_text = item.split("=", 1)
        low_text, high_text = bounds_text.split(":", 1)
        name = name.strip()
        low = float(low_text.strip())
        high = float(high_text.strip())
        if not name:
            raise ValueError(f"Invalid range assignment '{item}'")
        if high <= low:
            raise ValueError(f"Invalid range for '{name}': high ({high}) must be > low ({low})")
        ranges[name] = (low, high)
    return ranges


def _parse_parameter_name_list(spec):
    if spec is None:
        return []
    return [item.strip() for item in spec.split(",") if item.strip()]


def apply_parameter_overrides(fit_model, set_values_spec, set_ranges_spec, freeze_spec):
    value_updates = _parse_parameter_value_map(set_values_spec)
    range_updates = _parse_parameter_range_map(set_ranges_spec)
    freeze_names = _parse_parameter_name_list(freeze_spec)

    required_names = set(value_updates) | set(range_updates) | set(freeze_names)
    params_by_name = {}
    for name in required_names:
        param = find_parameter_by_name(fit_model, name)
        if param is None:
            raise ValueError(f"Parameter '{name}' was not found in the model")
        params_by_name[name] = param

    for name, value in value_updates.items():
        params_by_name[name].set_value(value)

    for name, (low, high) in range_updates.items():
        param = params_by_name[name]
        if hasattr(param, "set_limits"):
            param.set_limits(low=low, high=high)
        else:
            param.lower = low
            param.upper = high

    for name in freeze_names:
        param = params_by_name[name]
        if not hasattr(param, "floating"):
            raise ValueError(f"Parameter '{name}' does not support floating/fixed state")
        param.floating = False
