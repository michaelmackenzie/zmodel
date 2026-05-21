import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import dill
import numpy as np
import zfit
import zfit.z.numpy as znp

from zmodel.model_io import save_fit_model_bundle
from zmodel.utilities import FitModel
from zmodel.functions import *

@dataclass
class UncertaintySpec:
    name: str
    kind: str
    values: List[str]


@dataclass
class ConstraintSpec:
    name: str
    mean: float
    width: float


@dataclass
class ShapeSpec:
    process: str
    channel: str
    file: str


@dataclass
class CardSpec:
    shape_specs: List[ShapeSpec]
    is_counting: bool
    channels: List[str]
    bin_names: List[str]
    process_names: List[str]
    process_ids: List[int]
    rates: List[Optional[float]]
    uncertainties: List[UncertaintySpec]
    observations: Dict[str, float]
    data_obs_files: Dict[str, str]
    category: Optional[str] = None
    observation_count: Optional[float] = None
    param_constraints: List[ConstraintSpec] = None

    def __post_init__(self):
        if self.param_constraints is None:
            self.param_constraints = []
        if self.category is None and self.channels:
            self.category = self.channels[0]
        if self.observation_count is None and self.observations:
            self.observation_count = float(sum(self.observations.values()))


def _has_shape_mapping(shape_specs: List[ShapeSpec], process: str, channel: str) -> bool:
    for spec in shape_specs:
        if spec.process.lower() == "data_obs":
            continue
        process_match = spec.process == "*" or spec.process == process
        channel_match = spec.channel == "*" or spec.channel == channel
        if process_match and channel_match:
            return True
    return False


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

    shape_specs: List[ShapeSpec] = []
    bin_names: Optional[List[str]] = None
    process_names: Optional[List[str]] = None
    process_ids: Optional[List[int]] = None
    rates: Optional[List[Optional[float]]] = None
    uncertainties: List[UncertaintySpec] = []
    param_constraints: List[ConstraintSpec] = []
    process_line_count = 0
    observations: Dict[str, float] = {}
    data_obs_files: Dict[str, str] = {}
    legacy_observation_count: Optional[float] = None
    comment_markers = {"#", "//", "--"}

    for fields in tokens:
        key = fields[0].lower()
        for marker in comment_markers:
            key = key.split(marker, 1)[0].strip()
        if not key:
            continue

        if key == "shapes":
            if len(fields) not in (3, 4):
                raise ValueError(f"Invalid shapes line: {' '.join(fields)}")

            if len(fields) == 3:
                process_target = fields[1]
                channel_target = "*"
                file_name = fields[2]
            else:
                process_target = fields[1]
                channel_target = fields[2]
                file_name = fields[3]

            if not file_name.lower().endswith(".pkl"):
                raise ValueError(
                    f"Shape file '{file_name}' must be a pickle file (.pkl)"
                )

            if process_target.lower() == "data_obs":
                data_obs_files[channel_target] = file_name
            else:
                shape_specs.append(
                    ShapeSpec(process=process_target, channel=channel_target, file=file_name)
                )
            continue

        if key == "bin":
            if len(fields) < 2:
                raise ValueError(f"Invalid bin line: {' '.join(fields)}")
            bin_names = fields[1:]
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
            rates = [None if value == "-" else float(value) for value in values]
            continue

        if key == "observation":
            if len(fields) == 2:
                legacy_observation_count = float(fields[1])
                continue
            if len(fields) != 3:
                raise ValueError(
                    f"Invalid observation line: {' '.join(fields)}. Expected 'observation <category> <count>'"
                )
            observations[fields[1]] = float(fields[2])
            continue

        if len(fields) >= 4 and fields[1].lower() == "param":
            try:
                mean = float(fields[2])
                width = float(fields[3])
            except ValueError:
                raise ValueError(f"Invalid param constraint line: {' '.join(fields)}. Expected '<name> param <mean> <width>'")
            param_constraints.append(ConstraintSpec(name=fields[0], mean=mean, width=width))
            continue

        if len(fields) < 3:
            raise ValueError(f"Invalid uncertainty line: {' '.join(fields)}")

        uncertainties.append(UncertaintySpec(name=fields[0], kind=fields[1], values=fields[2:]))

    if bin_names is None:
        raise ValueError("Missing bin line")
    if process_names is None:
        raise ValueError("Missing process names line")
    if process_ids is None:
        raise ValueError("Missing process id line")
    if rates is None:
        raise ValueError("Missing rate line")
    if len(process_names) != len(process_ids):
        raise ValueError("process names and IDs length mismatch")
    if len(bin_names) == 1 and len(process_names) > 1:
        bin_names = [bin_names[0]] * len(process_names)
    if len(bin_names) != len(process_names):
        raise ValueError("bin line length does not match process count")

    channels = list(dict.fromkeys(bin_names))

    if legacy_observation_count is not None:
        if len(channels) != 1:
            raise ValueError("Single-value observation is only allowed for single-category cards")
        observations[channels[0]] = legacy_observation_count

    if observations:
        unknown_obs = [name for name in observations if name not in channels]
        if unknown_obs:
            raise ValueError(f"Observation category not present in bin line: {unknown_obs}")

    is_counting = len(shape_specs) == 0
    if not is_counting:
        for process, channel in zip(process_names, bin_names):
            if not _has_shape_mapping(shape_specs, process, channel):
                raise ValueError(
                    f"Missing shape mapping for process/channel '{process}/{channel}'. "
                    "Expected a matching line: shapes <process|*> <channel|*> <file>"
                )

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
        shape_specs=shape_specs,
        is_counting=is_counting,
        channels=channels,
        bin_names=bin_names,
        process_names=process_names,
        process_ids=process_ids,
        rates=rates,
        uncertainties=uncertainties,
        observations=observations,
        data_obs_files=data_obs_files,
        param_constraints=param_constraints,
    )


def _load_shape_payload_from_file(file_path: str):
    with open(file_path, "rb") as handle:
        try:
            return pickle.load(handle)
        except Exception:
            handle.seek(0)
            return dill.load(handle)


def _shape_mapping_rank(spec: ShapeSpec, process: str, channel: str) -> Optional[Tuple[int, int]]:
    process_match = spec.process == "*" or spec.process == process
    channel_match = spec.channel == "*" or spec.channel == channel
    if not (process_match and channel_match):
        return None
    specificity = int(spec.process != "*") + int(spec.channel != "*")
    return (specificity, 0)


def _resolve_shape_file_for_term(card: CardSpec, process: str, channel: str) -> str:
    best_spec = None
    best_rank = None
    for idx, spec in enumerate(card.shape_specs):
        rank = _shape_mapping_rank(spec, process, channel)
        if rank is None:
            continue
        ranked = (rank[0], idx)
        if best_rank is None or ranked > best_rank:
            best_rank = ranked
            best_spec = spec

    if best_spec is None:
        raise ValueError(
            f"No shape mapping found for process/channel '{process}/{channel}'"
        )
    return best_spec.file


def _resolve_shape_payloads(card: CardSpec, card_dir: str):
    payloads = {}

    term_payloads = []
    for process, channel in zip(card.process_names, card.bin_names):
        rel_path = _resolve_shape_file_for_term(card, process, channel)
        full_path = rel_path if os.path.isabs(rel_path) else os.path.join(card_dir, rel_path)
        full_path = os.path.abspath(full_path)
        if full_path not in payloads:
            payloads[full_path] = _load_shape_payload_from_file(full_path)
        term_payloads.append(payloads[full_path])

    return term_payloads


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
    term_names = []
    name_counts: Dict[str, int] = {}
    multiple_channels = len(set(card.bin_names)) > 1
    for process, channel in zip(card.process_names, card.bin_names):
        if multiple_channels:
            base = f"{process}__{channel}"
        else:
            base = process
        safe_base = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in base)
        index = name_counts.get(safe_base, 0)
        name_counts[safe_base] = index + 1
        if index:
            term_names.append(f"{safe_base}_{index}")
        else:
            term_names.append(safe_base)

    shapes = {}
    nominal_rates = {}
    process_payloads = []
    term_channels = {
        term_name: channel
        for term_name, channel in zip(term_names, card.bin_names)
    }
    term_processes = {
        term_name: process
        for term_name, process in zip(term_names, card.process_names)
    }
    observed_counts_by_channel: Dict[str, float] = {}
    observed_values_by_channel: Dict[str, np.ndarray] = {}

    observed_data = None

    if card.is_counting:
        obs_space = zfit.Space("count_obs", limits=(0.0, 1.0))
        obs_limits = (0.0, 1.0)
        for term_name, rate in zip(term_names, card.rates):
            shapes[term_name] = zfit.pdf.Uniform(obs=obs_space, low=0.0, high=1.0, name=f"{term_name}_counting_pdf")
            nominal_rates[term_name] = float(1.0 if rate is None else rate)

        if card.observations:
            observed_data = float(sum(card.observations.values()))
            observed_counts_by_channel = {k: float(v) for k, v in card.observations.items()}
        elif card.observation_count is not None:
            observed_data = float(card.observation_count)
    else:
        process_payloads = _resolve_shape_payloads(card, card_dir)
        for term_name, process, payload, rate in zip(term_names, card.process_names, process_payloads, card.rates):
            shapes[term_name] = get_nominal_pdf(payload, process)
            nominal_rates[term_name] = get_nominal_rate(payload, process, rate)

        first_shape = shapes[term_names[0]]
        obs_space = first_shape.space
        obs_limits = tuple(float(x) for x in first_shape.space.limit1d)

        for term_name in term_names[1:]:
            candidate_limits = tuple(float(x) for x in shapes[term_name].space.limit1d)
            if candidate_limits != obs_limits:
                raise ValueError(
                    "All channels must share the same observable limits to build a combined model"
                )

        if card.data_obs_files:
            # The current analysis pipeline expects one observed dataset.
            # If multiple channel files are provided, load each and merge rows.
            merged_rows = []
            for channel in card.channels:
                obs_rel = card.data_obs_files.get(channel, card.data_obs_files.get("*"))
                if obs_rel is None:
                    continue
                obs_path = obs_rel
                if not os.path.isabs(obs_path):
                    obs_path = os.path.join(card_dir, obs_path)
                obs_payload = _load_shape_payload_from_file(os.path.abspath(obs_path))
                channel_data = _coerce_unbinned_data_obs(obs_space, _extract_data_obs_payload(obs_payload))
                if channel_data is None:
                    continue
                channel_values = np.asarray(channel_data.value(), dtype=float)
                if channel_values.ndim == 1:
                    channel_values = channel_values.reshape(-1, 1)
                observed_values_by_channel[channel] = channel_values.reshape(-1)
                merged_rows.append(channel_values)

            if merged_rows:
                observed_data = zfit.Data.from_numpy(obs=obs_space, array=np.vstack(merged_rows))

        elif "*" in card.data_obs_files:
            obs_path = card.data_obs_files["*"]
            if not os.path.isabs(obs_path):
                obs_path = os.path.join(card_dir, obs_path)
            obs_payload = _load_shape_payload_from_file(os.path.abspath(obs_path))
            observed_data = _coerce_unbinned_data_obs(obs_space, _extract_data_obs_payload(obs_payload))
            if observed_data is not None and len(card.channels) == 1:
                observed_values_by_channel[card.channels[0]] = np.asarray(observed_data.value(), dtype=float).reshape(-1)

        expected_observation = float(sum(card.observations.values())) if card.observations else card.observation_count
        if observed_data is not None and expected_observation is not None:
            observed_entries = _observed_entries_from_dataset(observed_data)
            if observed_entries is not None and not np.isclose(observed_entries, expected_observation, atol=0.5):
                raise ValueError(
                    f"Observation count ({expected_observation}) does not match data_obs entries ({observed_entries})"
                )

    constraints = []
    rate_factors = {name: [] for name in term_names}

    signal_processes = {
        process
        for process, proc_id in zip(card.process_names, card.process_ids)
        if proc_id < 0
    }
    process_id_map = {
        process: proc_id
        for process, proc_id in zip(card.process_names, card.process_ids)
    }

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
            for term_name, process, raw_value in zip(term_names, card.process_names, unc.values):
                if raw_value == "-":
                    continue
                value = float(raw_value)
                if kind == "lnN":
                    factor = zfit.ComposedParameter(
                        f"scale_{unc.name}_{term_name}",
                        lambda t, v=value: znp.power(v, t),
                        params=[theta],
                    )
                else:
                    sigma = value - 1.0 if value >= 1.0 else value
                    factor = zfit.ComposedParameter(
                        f"scale_{unc.name}_{term_name}",
                        lambda t, s=sigma: znp.maximum(0.0, 1.0 + s * t),
                        params=[theta],
                    )
                rate_factors[term_name].append(factor)
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
        for term_name, process, payload, raw_value in zip(term_names, card.process_names, process_payloads, unc.values):
            if raw_value == "-":
                continue
            if raw_value != "1":
                raise ValueError(
                    f"Shape uncertainty value for {unc.name}/{process} must be '1' or '-', got {raw_value}"
                )
            up_pdf = get_shape_variation_pdf(payload, process, unc.name, "Up")
            down_pdf = get_shape_variation_pdf(payload, process, unc.name, "Down")
            shapes[term_name] = make_shape_morphed_pdf(
                nominal_pdf=shapes[term_name],
                up_pdf=up_pdf,
                down_pdf=down_pdf,
                theta=theta,
                name=f"{unc.name}_{term_name}",
            )

    yields: Dict[str, zfit.Parameter] = {}
    signal_strength_params: Dict[str, zfit.Parameter] = {}
    for term_name, process in zip(term_names, card.process_names):
        base = nominal_rates[term_name]
        all_factors = list(rate_factors[term_name])

        if process in signal_processes:
            mu = signal_strength_params.get(process)
            if mu is None:
                mu = zfit.Parameter(f"mu_{process}", 1.0, 0.0, 100.0)
                signal_strength_params[process] = mu
            all_factors.insert(0, mu)

        yields[term_name] = multiply_factors(
            base=base,
            factors=all_factors,
            name=f"yield_{term_name}",
        )

    extended_pdfs = {
        term_name: shapes[term_name].create_extended(yields[term_name])
        for term_name in term_names
    }
    model_name = f"model_{card.category}" if len(card.channels) == 1 else "model_combined"
    model = zfit.pdf.SumPDF(list(extended_pdfs.values()), name=model_name)

    # Apply explicit parameter Gaussian constraints from 'param' card lines.
    # Collect all named parameters from the model for lookup.
    all_params = {p.name: p for p in model.get_params(floating=None)}
    for cs in card.param_constraints:
        param = all_params.get(cs.name)
        if param is None:
            raise ValueError(
                f"param constraint references unknown parameter '{cs.name}'. "
                f"Available: {sorted(all_params.keys())}"
            )
        constraints.append(
            zfit.constraint.GaussianConstraint(
                params=param,
                observation=cs.mean,
                uncertainty=cs.width,
            )
        )

    signal_name = next((name for name, proc_id in process_id_map.items() if proc_id < 0), None)
    signal_nominal_yield = None
    if signal_name is not None:
        signal_nominal_yield = float(
            sum(
                nominal_rates[term_name]
                for term_name, process in zip(term_names, card.process_names)
                if process == signal_name
            )
        )

    return FitModel(
        obs=obs_space,
        obs_range=obs_limits,
        shapes=shapes,
        yields=yields,
        extended_pdfs=extended_pdfs,
        model=model,
        data=observed_data,
        process_names=list(term_names),
        signal_process=signal_name,
        constraints=constraints,
        loss=None,
        result=None,
        signal_nominal_yield=signal_nominal_yield,
        channels=list(card.channels),
        term_channels=term_channels,
        term_processes=term_processes,
        observed_counts_by_channel=observed_counts_by_channel,
        observed_values_by_channel=observed_values_by_channel,
    )


def build_and_save_model_from_card_file(input_card: str, output_file: str) -> str:
    card_path = os.path.abspath(input_card)
    card_dir = os.path.dirname(card_path)

    card = parse_model_card(card_path)
    fit_model = build_model_from_card(card, card_dir)

    # Always include observed data in the bundle if present
    # For counting models this is the summed observed count across categories.
    if card.is_counting and card.observations:
        fit_model.data = float(sum(card.observations.values()))
    elif card.is_counting and card.observation_count is not None:
        fit_model.data = card.observation_count

    output_path = os.path.abspath(output_file)
    save_fit_model_bundle(fit_model, output_path, card=card, card_dir=card_dir)
    return output_path
