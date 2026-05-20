import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import dill
import numpy as np
import zfit
import zfit.z.numpy as znp

from model_io import save_fit_model_bundle
from utilities import FitModel
from functions import *

@dataclass
class UncertaintySpec:
    name: str
    kind: str
    values: List[str]


@dataclass
class CardSpec:
    shape_files: Dict[str, str]
    is_counting: bool
    category: str
    process_names: List[str]
    process_ids: List[int]
    rates: Dict[str, float]
    uncertainties: List[UncertaintySpec]
    data_obs_file: Optional[str] = None
    observation_category: Optional[str] = None
    observation_count: Optional[float] = None


def _tokenize_card_line(line: str) -> List[str]:
    text = line.strip()
    if not text or text.startswith("#"):
        return []
    if "#" in text:
        text = text.split("#", 1)[0].strip()
    return text.split()


def parse_model_card(card_path: str) -> CardSpec:
    with open(card_path, "r", encoding="utf-8") as handle:
        lines = [_tokenize_card_line(line) for line in handle]

    tokens = [line for line in lines if line]

    shape_files: Dict[str, str] = {}
    category: Optional[str] = None
    process_names: Optional[List[str]] = None
    process_ids: Optional[List[int]] = None
    rates: Dict[str, float] = {}
    uncertainties: List[UncertaintySpec] = []
    process_line_count = 0
    data_obs_file: Optional[str] = None
    observation_category: Optional[str] = None
    observation_count: Optional[float] = None
    comment_markers = {"#", "//", "--"}

    for fields in tokens:
        key = fields[0].lower()
        for marker in comment_markers:
            key = key.split(marker, 1)[0].strip()
        if not key:
            continue

        if key in ("shape", "shapes") and len(fields) >= 2 and fields[1].lower() == "data_obs":
            if len(fields) != 3:
                raise ValueError(f"Invalid data_obs shape line: {' '.join(fields)}")
            if not fields[2].lower().endswith(".pkl"):
                raise ValueError(
                    f"Observed data file '{fields[2]}' must be a pickle file (.pkl)"
                )
            data_obs_file = fields[2]
            continue

        if key == "shapes":
            if len(fields) != 3:
                raise ValueError(f"Invalid shapes line: {' '.join(fields)}")
            if not fields[2].lower().endswith(".pkl"):
                raise ValueError(
                    f"Shape file '{fields[2]}' must be a pickle file (.pkl)"
                )
            shape_files[fields[1]] = fields[2]
            continue

        if key == "bin":
            if len(fields) != 2:
                raise ValueError(f"Invalid bin line: {' '.join(fields)}")
            category = fields[1]
            continue

        if key == "process":
            process_line_count += 1
            if process_line_count == 1:
                process_names = fields[1:]
            elif process_line_count == 2:
                process_ids = [int(item) for item in fields[1:]]
            else:
                raise ValueError("Model card has more than two process lines")
            continue

        if key == "rate":
            if process_names is None:
                raise ValueError("rate line appears before process names")
            values = fields[1:]
            if len(values) != len(process_names):
                raise ValueError("rate line length does not match process count")
            for process, value in zip(process_names, values):
                if value != "-":
                    rates[process] = float(value)
            continue

        if key == "observation":
            if len(fields) != 3:
                raise ValueError(
                    f"Invalid observation line: {' '.join(fields)}. Expected 'observation <category> <count>'"
                )
            observation_category = fields[1]
            observation_count = float(fields[2])
            continue

        if len(fields) < 3:
            raise ValueError(f"Invalid uncertainty line: {' '.join(fields)}")

        uncertainties.append(UncertaintySpec(name=fields[0], kind=fields[1], values=fields[2:]))

    if category is None:
        raise ValueError("Missing bin line")
    if process_names is None:
        raise ValueError("Missing process names line")
    if process_ids is None:
        raise ValueError("Missing process id line")
    if len(process_names) != len(process_ids):
        raise ValueError("process names and IDs length mismatch")

    is_counting = len(shape_files) == 0
    if not is_counting and "*" not in shape_files:
        raise ValueError("Missing default shape mapping: shapes * <file>")

    for unc in uncertainties:
        if len(unc.values) != len(process_names):
            raise ValueError(
                f"Uncertainty '{unc.name}' has {len(unc.values)} values, expected {len(process_names)}"
            )
        if is_counting and unc.kind.strip().lower() == "shape":
            raise ValueError(
                f"Shape uncertainty '{unc.name}' is not allowed for counting models (no shapes section provided)"
            )

    return CardSpec(
        shape_files=shape_files,
        is_counting=is_counting,
        category=category,
        process_names=process_names,
        process_ids=process_ids,
        rates=rates,
        uncertainties=uncertainties,
        data_obs_file=data_obs_file,
        observation_category=observation_category,
        observation_count=observation_count,
    )


def _load_shape_payload_from_file(file_path: str):
    with open(file_path, "rb") as handle:
        try:
            return pickle.load(handle)
        except Exception:
            handle.seek(0)
            return dill.load(handle)


def _resolve_shape_payloads(card: CardSpec, card_dir: str):
    payloads = {}
    process_to_payload = {}

    for target, rel_path in card.shape_files.items():
        full_path = rel_path if os.path.isabs(rel_path) else os.path.join(card_dir, rel_path)
        full_path = os.path.abspath(full_path)
        if full_path not in payloads:
            payloads[full_path] = _load_shape_payload_from_file(full_path)
        if target != "*":
            process_to_payload[target] = payloads[full_path]

    default_rel = card.shape_files["*"]
    default_path = default_rel if os.path.isabs(default_rel) else os.path.join(card_dir, default_rel)
    default_path = os.path.abspath(default_path)
    default_payload = payloads[default_path]

    for process in card.process_names:
        process_to_payload.setdefault(process, default_payload)

    return process_to_payload


def _observed_entries_from_dataset(data_obs) -> Optional[float]:
    try:
        values = data_obs.values()
        return float(np.sum(np.asarray(values, dtype=float)))
    except Exception:
        pass

    try:
        values = data_obs.value()
        arr = np.asarray(values)
        if arr.ndim == 0:
            return 1.0
        return float(arr.shape[0])
    except Exception:
        return None


def _extract_data_obs_payload(raw_payload):
    if raw_payload is None:
        return None

    if hasattr(raw_payload, "data_obs"):
        return getattr(raw_payload, "data_obs")

    if isinstance(raw_payload, dict):
        if "data_obs" in raw_payload:
            return raw_payload["data_obs"]
        data_block = raw_payload.get("data")
        if isinstance(data_block, dict) and "data_obs" in data_block:
            return data_block["data_obs"]

    return None


def _coerce_unbinned_data_obs(obs_space, payload):
    if payload is None:
        return None

    if hasattr(payload, "space") and hasattr(payload, "value"):
        return payload

    if isinstance(payload, dict) and "values" in payload:
        payload = payload["values"]

    values = np.asarray(payload, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values.reshape(-1, 1)

    return zfit.Data.from_numpy(obs=obs_space, array=values)


def _get_from_dict_candidates(source_dict, process, candidates):
    for name in candidates:
        if name in source_dict:
            return source_dict[name]

    process_dict = source_dict.get(process)
    if isinstance(process_dict, dict):
        for name in candidates:
            suffix = name.replace(f"{process}_", "", 1)
            if suffix in process_dict:
                return process_dict[suffix]
    return None


def get_nominal_pdf(payload, process: str):
    if hasattr(payload, "shapes") and process in payload.shapes:
        return payload.shapes[process]

    if hasattr(payload, "get_pdf"):
        return payload.get_pdf(process)

    if isinstance(payload, dict):
        for key in ("shapes", "pdfs", "process_pdfs"):
            source = payload.get(key)
            if isinstance(source, dict) and process in source:
                return source[process]
        if process in payload:
            return payload[process]

    if hasattr(payload, "PDFS"):
        pdfs = getattr(payload, "PDFS")
        if process in pdfs:
            return pdfs[process]

    for attr in (f"{process}_pdf", f"{process}_shape"):
        if hasattr(payload, attr):
            return getattr(payload, attr)

    raise ValueError(f"Could not find nominal PDF for process '{process}'")


def get_shape_variation_pdf(payload, process: str, uncertainty_name: str, direction: str):
    if hasattr(payload, "get_shape_variation"):
        return payload.get_shape_variation(process, uncertainty_name, direction)

    variation_name = f"{uncertainty_name}{direction}"
    candidate_names = [
        f"{process}_{variation_name}",
        f"{process}_{variation_name}_pdf",
        f"{process}_{variation_name}_shape",
    ]

    if isinstance(payload, dict):
        nested_variations = payload.get("shape_variations")
        if isinstance(nested_variations, dict):
            process_vars = nested_variations.get(process, {})
            if isinstance(process_vars, dict):
                unc_vars = process_vars.get(uncertainty_name, {})
                if isinstance(unc_vars, dict):
                    if direction in unc_vars:
                        return unc_vars[direction]
                    if variation_name in unc_vars:
                        return unc_vars[variation_name]

        for key in ("shapes", "pdfs", "process_pdfs", "shape_variations"):
            source = payload.get(key)
            if isinstance(source, dict):
                found = _get_from_dict_candidates(source, process, candidate_names)
                if found is not None:
                    return found

        found = _get_from_dict_candidates(payload, process, candidate_names)
        if found is not None:
            return found

    for name in candidate_names:
        if hasattr(payload, name):
            return getattr(payload, name)

    raise ValueError(
        f"Missing shape variation PDF '{process}_{uncertainty_name}{direction}' in shape module"
    )


def get_nominal_rate(payload, process: str, card_rate: Optional[float]):
    if card_rate is not None:
        return float(card_rate)

    if hasattr(payload, "yields") and process in payload.yields:
        value = payload.yields[process]
        if isinstance(value, zfit.Parameter):
            return float(value.value())
        if isinstance(value, (int, float)):
            return float(value)

    if isinstance(payload, dict):
        for key in ("rates", "yields", "nominal_rates"):
            source = payload.get(key)
            if isinstance(source, dict) and process in source:
                value = source[process]
                if isinstance(value, zfit.Parameter):
                    return float(value.value())
                return float(value)

    if hasattr(payload, "RATES"):
        rates = getattr(payload, "RATES")
        if process in rates:
            return float(rates[process])

    attr = f"{process}_yield"
    if hasattr(payload, attr):
        value = getattr(payload, attr)
        if isinstance(value, zfit.Parameter):
            return float(value.value())
        if isinstance(value, (int, float)):
            return float(value)

    return 1.0


def _clip(value):
    return znp.minimum(1.0, znp.maximum(-1.0, value))


def make_shape_morphed_pdf(nominal_pdf, up_pdf, down_pdf, theta, name: str):
    frac_up = zfit.ComposedParameter(
        f"frac_up_{name}",
        lambda t: znp.maximum(0.0, _clip(t)),
        params=[theta],
    )
    frac_down = zfit.ComposedParameter(
        f"frac_down_{name}",
        lambda t: znp.maximum(0.0, -_clip(t)),
        params=[theta],
    )
    return zfit.pdf.SumPDF(
        [up_pdf, down_pdf, nominal_pdf],
        fracs=[frac_up, frac_down],
        name=f"shape_morph_{name}",
    )


def multiply_factors(base: float, factors: List[zfit.Parameter], name: str):
    if not factors:
        param = zfit.Parameter(name, base, 0.0, max(base * 10.0, 1.0))
        param.floating = False
        return param

    return zfit.ComposedParameter(
        name,
        lambda *vals, b=base: b * znp.prod(znp.stack(vals)),
        params=list(factors),
    )


def _kind_token(kind: str) -> str:
    lowered = kind.strip().lower()
    if lowered == "lnn":
        return "lnN"
    if lowered == "gs":
        return "gs"
    if lowered == "shape":
        return "shape"
    raise ValueError(f"Unknown uncertainty type '{kind}'. Use lnN, gs, or shape.")


def build_model_from_card(card: CardSpec, card_dir: str):
    shapes = {}
    nominal_rates = {}
    process_payloads = {}

    observed_data = None

    if card.is_counting:
        obs_space = zfit.Space("count_obs", limits=(0.0, 1.0))
        obs_limits = (0.0, 1.0)
        for process in card.process_names:
            shapes[process] = zfit.pdf.Uniform(obs=obs_space, low=0.0, high=1.0, name=f"{process}_counting_pdf")
            nominal_rates[process] = float(card.rates.get(process, 1.0))

        if card.observation_count is not None:
            observed_data = float(card.observation_count)
    else:
        process_payloads = _resolve_shape_payloads(card, card_dir)
        for process in card.process_names:
            payload = process_payloads[process]
            shapes[process] = get_nominal_pdf(payload, process)
            nominal_rates[process] = get_nominal_rate(payload, process, card.rates.get(process))

        first_shape = shapes[card.process_names[0]]
        obs_space = first_shape.space
        obs_limits = tuple(float(x) for x in first_shape.space.limit1d)

        if card.data_obs_file is not None:
            obs_path = card.data_obs_file
            if not os.path.isabs(obs_path):
                obs_path = os.path.join(card_dir, obs_path)
            obs_payload = _load_shape_payload_from_file(os.path.abspath(obs_path))
            observed_data = _coerce_unbinned_data_obs(obs_space, _extract_data_obs_payload(obs_payload))

        if observed_data is not None and card.observation_count is not None:
            observed_entries = _observed_entries_from_dataset(observed_data)
            if observed_entries is not None and not np.isclose(observed_entries, card.observation_count, atol=0.5):
                raise ValueError(
                    f"Observation count ({card.observation_count}) does not match data_obs entries ({observed_entries})"
                )

    constraints = []
    rate_factors = {name: [] for name in card.process_names}

    process_id_map = dict(zip(card.process_names, card.process_ids))
    signal_processes = {name for name, proc_id in process_id_map.items() if proc_id < 0}

    for unc in card.uncertainties:
        kind = _kind_token(unc.kind)

        if kind in ("lnN", "gs"):
            theta = zfit.Parameter(f"nuis_{unc.name}", 0.0, -5.0, 5.0)
            constraints.append(
                zfit.constraint.GaussianConstraint(
                    params=theta,
                    observation=0.0,
                    uncertainty=1.0,
                )
            )
            for process, raw_value in zip(card.process_names, unc.values):
                if raw_value == "-":
                    continue
                value = float(raw_value)
                if kind == "lnN":
                    factor = zfit.ComposedParameter(
                        f"scale_{unc.name}_{process}",
                        lambda t, v=value: znp.power(v, t),
                        params=[theta],
                    )
                else:
                    sigma = value - 1.0 if value >= 1.0 else value
                    factor = zfit.ComposedParameter(
                        f"scale_{unc.name}_{process}",
                        lambda t, s=sigma: znp.maximum(0.0, 1.0 + s * t),
                        params=[theta],
                    )
                rate_factors[process].append(factor)
            continue

        theta = zfit.Parameter(f"nuis_shape_{unc.name}", 0.0, -1.0, 1.0)
        constraints.append(
            zfit.constraint.GaussianConstraint(
                params=theta,
                observation=0.0,
                uncertainty=1.0,
            )
        )
        if card.is_counting:
            raise ValueError(
                f"Shape uncertainty '{unc.name}' is not allowed for counting models"
            )
        for process, raw_value in zip(card.process_names, unc.values):
            if raw_value == "-":
                continue
            if raw_value != "1":
                raise ValueError(
                    f"Shape uncertainty value for {unc.name}/{process} must be '1' or '-', got {raw_value}"
                )
            payload = process_payloads[process]
            up_pdf = get_shape_variation_pdf(payload, process, unc.name, "Up")
            down_pdf = get_shape_variation_pdf(payload, process, unc.name, "Down")
            shapes[process] = make_shape_morphed_pdf(
                nominal_pdf=shapes[process],
                up_pdf=up_pdf,
                down_pdf=down_pdf,
                theta=theta,
                name=f"{unc.name}_{process}",
            )

    yields: Dict[str, zfit.Parameter] = {}
    for process in card.process_names:
        base = nominal_rates[process]
        all_factors = list(rate_factors[process])

        if process in signal_processes:
            mu = zfit.Parameter(f"mu_{process}", 1.0, 0.0, 10.0)
            all_factors.insert(0, mu)

        yields[process] = multiply_factors(
            base=base,
            factors=all_factors,
            name=f"yield_{process}",
        )

    extended_pdfs = {
        process: shapes[process].create_extended(yields[process])
        for process in card.process_names
    }
    model = zfit.pdf.SumPDF(list(extended_pdfs.values()), name=f"model_{card.category}")

    signal_name = next((name for name in card.process_names if process_id_map[name] < 0), None)
    signal_nominal_yield = nominal_rates.get(signal_name) if signal_name is not None else None

    return FitModel(
        obs=obs_space,
        obs_range=obs_limits,
        shapes=shapes,
        yields=yields,
        extended_pdfs=extended_pdfs,
        model=model,
        data=observed_data,
        process_names=list(card.process_names),
        signal_process=signal_name,
        constraints=constraints,
        loss=None,
        result=None,
        signal_nominal_yield=signal_nominal_yield,
    )


def build_and_save_model_from_card_file(input_card: str, output_file: str) -> str:
    card_path = os.path.abspath(input_card)
    card_dir = os.path.dirname(card_path)

    card = parse_model_card(card_path)
    fit_model = build_model_from_card(card, card_dir)

    # Always include observed data in the bundle if present
    # For counting models, this is just the observation_count
    if card.is_counting and card.observation_count is not None:
        fit_model.data = card.observation_count

    output_path = os.path.abspath(output_file)
    save_fit_model_bundle(fit_model, output_path, card=card, card_dir=card_dir)
    return output_path
