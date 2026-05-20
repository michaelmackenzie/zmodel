import __main__
import json
import os
import pickle
from dataclasses import asdict
from typing import Any, Dict, Optional

import zfit
import zfit.z.numpy as znp

from utilities import FitModel


def _clip(value):
    return znp.minimum(1.0, znp.maximum(-1.0, value))


def _inject_hs3_helpers():
    __main__.znp = znp
    __main__._clip = _clip


def _serialize_card(card, card_dir: Optional[str]) -> Dict[str, Any]:
    payload = asdict(card)
    if card_dir is None:
        return payload

    resolved = {}
    for target, shape_file in payload["shape_files"].items():
        if os.path.isabs(shape_file):
            resolved[target] = shape_file
        else:
            resolved[target] = os.path.abspath(os.path.join(card_dir, shape_file))
    payload["shape_files"] = resolved
    return payload


def _deserialize_card(card_payload: Dict[str, Any]):
    from build_model_from_text import CardSpec, UncertaintySpec

    return CardSpec(
        shape_files=dict(card_payload["shape_files"]),
        is_counting=bool(card_payload.get("is_counting", False)),
        category=card_payload["category"],
        process_names=list(card_payload["process_names"]),
        process_ids=list(card_payload["process_ids"]),
        rates=dict(card_payload.get("rates", {})),
        uncertainties=[UncertaintySpec(**item) for item in card_payload.get("uncertainties", [])],
        data_obs_file=card_payload.get("data_obs_file"),
        observation_category=card_payload.get("observation_category"),
        observation_count=card_payload.get("observation_count"),
    )


def save_fit_model_bundle(fit_model: FitModel, output_file: str, card=None, card_dir: Optional[str] = None):
    hs3_payload = zfit.hs3.dumps(fit_model.model)
    try:
        json.dumps(hs3_payload)
        hs3_json_payload = hs3_payload
    except TypeError:
        hs3_json_payload = None

    bundle = {
        "format": "fit_model_bundle_v1",
        "fit_metadata": {
            "process_names": list(fit_model.process_names),
            "signal_process": fit_model.signal_process,
            "signal_nominal_yield": fit_model.signal_nominal_yield,
        },
        "hs3_model": hs3_json_payload,
    }

    if card is not None:
        bundle["card"] = _serialize_card(card, card_dir)

    ext = os.path.splitext(output_file)[1].lower()
    if ext == ".json":
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(bundle, handle, indent=2)
        return

    with open(output_file, "wb") as handle:
        pickle.dump(bundle, handle)


def _choose_top_model(distributions: Dict[str, Any]):
    if not distributions:
        raise ValueError("No distributions found in loaded HS3 payload")
    if len(distributions) == 1:
        return next(iter(distributions.values()))

    for name, model in distributions.items():
        if name.startswith("model_"):
            return model

    return next(iter(distributions.values()))


def _fit_model_from_hs3_payload(hs3_payload: Dict[str, Any], fit_metadata: Optional[Dict[str, Any]] = None):
    _inject_hs3_helpers()
    loaded = zfit.hs3.loads(hs3_payload)
    distributions = loaded.get("distributions", {})
    constraints = list(loaded.get("constraints", {}).values())
    model = _choose_top_model(distributions)

    signal_process = None
    if fit_metadata is not None:
        signal_process = fit_metadata.get("signal_process")

    if signal_process is None:
        for param in model.get_params():
            if param.name.startswith("mu_"):
                signal_process = param.name[3:]
                break

    return FitModel(
        obs=model.space,
        obs_range=tuple(float(x) for x in model.space.limit1d),
        shapes={},
        yields={},
        extended_pdfs={},
        model=model,
        data=None,
        process_names=list((fit_metadata or {}).get("process_names", [])),
        signal_process=signal_process,
        constraints=constraints,
        loss=None,
        result=None,
        signal_nominal_yield=(fit_metadata or {}).get("signal_nominal_yield"),
    )


def load_fit_model(model_file: str) -> FitModel:
    payload = None
    with open(model_file, "rb") as handle:
        try:
            payload = pickle.load(handle)
        except Exception:
            handle.seek(0)
            payload = json.load(handle)

    if payload.get("format") == "fit_model_bundle_v1":
        card_payload = payload.get("card")
        if card_payload is not None:
            from build_model_from_text import build_model_from_card

            card = _deserialize_card(card_payload)
            return build_model_from_card(card, os.path.dirname(os.path.abspath(model_file)))

        hs3_payload = payload.get("hs3_model")
        if hs3_payload is None:
            raise ValueError(
                "Saved bundle has no JSON-serializable HS3 payload and no card to rebuild from"
            )
        return _fit_model_from_hs3_payload(hs3_payload, payload.get("fit_metadata"))

    if "metadata" in payload and "distributions" in payload:
        return _fit_model_from_hs3_payload(payload)

    raise ValueError(f"Unsupported model file format in {model_file}")