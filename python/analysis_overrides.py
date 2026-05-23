def _all_unique_params(fit_model):
    seen = set()

    channel_models = getattr(fit_model, "channel_models", {}) or {}
    models = list(channel_models.values()) if channel_models else [fit_model.model]

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

    def _yield_param(param):
        ident = id(param)
        if ident in seen:
            return
        seen.add(ident)
        yield param
        for child in _iter_child_params(param):
            yield from _yield_param(child)

    def _iter_model_params(model):
        # zfit API support varies by model class/version; try the common variants.
        for kwargs in ({}, {"floating": None}, {"floating": None, "is_yield": None}):
            try:
                params = list(model.get_params(**kwargs))
            except Exception:
                continue
            for param in params:
                yield from _yield_param(param)

    for model in models:
        yield from _iter_model_params(model)

    # Some model compositions do not expose parameters through get_params,
    # but FitModel.yields contains the relevant optimization parameters.
    for param in (getattr(fit_model, "yields", {}) or {}).values():
        if hasattr(param, "value"):
            yield from _yield_param(param)


def _find_parameter_with_error(fit_model, parameter_name):
    params = list(_all_unique_params(fit_model))

    exact_matches = [param for param in params if param.name == parameter_name]
    if len(exact_matches) == 1:
        return exact_matches[0], None
    if len(exact_matches) > 1:
        names = sorted(param.name for param in exact_matches)
        return None, (
            f"Parameter name '{parameter_name}' is ambiguous; exact matches: {', '.join(names)}"
        )

    # Backward-compatible alias resolution:
    # if callers pass the base name, accept a unique channel-suffixed parameter.
    suffixed_matches = [
        param for param in params if param.name.startswith(f"{parameter_name}__")
    ]
    if len(suffixed_matches) == 1:
        return suffixed_matches[0], None
    if len(suffixed_matches) > 1:
        names = sorted(param.name for param in suffixed_matches)
        return None, (
            f"Parameter '{parameter_name}' matches multiple channel-specific parameters: "
            f"{', '.join(names)}. Use the full parameter name."
        )

    # Also try resolving a provided channel-specific name to its base equivalent.
    if "__" in parameter_name:
        base_name = parameter_name.split("__", 1)[0]
        base_matches = [param for param in params if param.name == base_name]
        if len(base_matches) == 1:
            return base_matches[0], None

    available_names = sorted(param.name for param in params)
    preview = ", ".join(available_names[:20])
    more = "" if len(available_names) <= 20 else f", ... (+{len(available_names) - 20} more)"
    return None, (
        f"Parameter '{parameter_name}' was not found in the model. "
        f"Available parameters include: {preview}{more}"
    )


def find_parameter_by_name(fit_model, parameter_name):
    param, _ = _find_parameter_with_error(fit_model, parameter_name)
    return param


def _resolve_required_parameters(fit_model, required_names):
    params_by_name = {}
    for name in required_names:
        param, error = _find_parameter_with_error(fit_model, name)
        if param is None:
            raise ValueError(error)
        params_by_name[name] = param
    return params_by_name


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
    params_by_name = _resolve_required_parameters(fit_model, required_names)

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
