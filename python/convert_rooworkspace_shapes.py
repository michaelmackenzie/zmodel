#!/usr/bin/env python3
import argparse
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import dill
import numpy as np
import zfit
import zfit.z.numpy as znp


class ConversionError(RuntimeError):
    pass


@dataclass
class WorkspaceConversionResult:
    workspace_name: str
    output_file: str
    n_variables: int
    n_datasets: int
    n_pdfs: int


_ROOT_LOGGING_CONFIGURED = False


def _configure_root_logging() -> None:
    global _ROOT_LOGGING_CONFIGURED
    if _ROOT_LOGGING_CONFIGURED:
        return

    import ROOT

    try:
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    except Exception:
        pass

    try:
        ROOT.gErrorIgnoreLevel = ROOT.kError
    except Exception:
        pass

    _ROOT_LOGGING_CONFIGURED = True


def _iter_roo_collection(collection) -> Iterable:
    if collection is None:
        return

    try:
        iterator = iter(collection)
    except TypeError:
        create_iterator = getattr(collection, "createIterator", None)
        if not callable(create_iterator):
            return

        iterator = create_iterator()
        while True:
            obj = iterator.Next()
            if not obj:
                break
            yield obj
        return

    for obj in iterator:
        yield obj

    if False:
        while True:
            obj = iterator.Next()
            if not obj:
                break
            yield obj


def _collect_workspaces(root_file) -> List:
    import ROOT

    workspaces = []

    def _scan_directory(directory):
        for key in directory.GetListOfKeys():
            name = key.GetName()
            obj = directory.Get(name)
            if obj is None:
                continue
            if obj.InheritsFrom("RooWorkspace"):
                workspaces.append(obj)
            elif obj.InheritsFrom("TDirectory"):
                _scan_directory(obj)

    _scan_directory(root_file)
    return workspaces


def _call_first(obj, method_names: List[str], context: str):
    for method_name in method_names:
        method = getattr(obj, method_name, None)
        if callable(method):
            return method()
    raise ConversionError(
        f"Could not get {context} from {obj.GetName()} ({obj.ClassName()}); "
        f"tried methods {method_names}"
    )


def _infer_limits(var) -> Tuple[float, float]:
    value = float(var.getVal())

    has_min = hasattr(var, "hasMin") and bool(var.hasMin())
    has_max = hasattr(var, "hasMax") and bool(var.hasMax())

    if has_min and has_max:
        low = float(var.getMin())
        high = float(var.getMax())
        if np.isfinite(low) and np.isfinite(high) and low < high:
            return low, high

    span = max(abs(value) * 2.0, 1.0)
    return value - span, value + span


def _roo_formula_to_python_expr(formula: str) -> str:
    expr = str(formula)
    replacements = {
        "TMath::Abs": "znp.abs",
        "TMath::Exp": "znp.exp",
        "TMath::Log": "znp.log",
        "TMath::Sqrt": "znp.sqrt",
        "TMath::Power": "znp.power",
        "pow": "znp.power",
        "abs": "znp.abs",
    }
    for old, new in replacements.items():
        expr = expr.replace(old, new)
    expr = re.sub(r"@(\d+)", lambda match: f"args[{match.group(1)}]", expr)
    return expr


def _make_composed_parameter(name: str, formula: str, dependencies: List[Any]):
    python_expr = _roo_formula_to_python_expr(formula)

    def _evaluate(*args, expr=python_expr):
        return eval(
            expr,
            {"__builtins__": {}},
            {"args": args, "np": np, "znp": znp},
        )

    param = zfit.ComposedParameter(name, _evaluate, params=list(dependencies))
    param._zmodel_rooformula = {
        "formula": str(formula),
        "dependency_names": [getattr(dep, "GetName", lambda: getattr(dep, "name", ""))() for dep in dependencies],
    }
    return param


class RooWorkspaceConverter:
    def __init__(self, workspace, name_prefix: str = ""):
        self.workspace = workspace
        self.name_prefix = name_prefix
        self._arg_cache: Dict[str, object] = {}
        self._pdf_cache: Dict[str, zfit.pdf.BasePDF] = {}
        self._space_cache: Dict[Tuple[str, float, float], zfit.Space] = {}

    def _zname(self, root_name: str) -> str:
        if self.name_prefix:
            return f"{self.name_prefix}{root_name}"
        return root_name

    def _space_for_observable(self, obs_var) -> zfit.Space:
        if not obs_var.InheritsFrom("RooRealVar"):
            raise ConversionError(
                f"Observable '{obs_var.GetName()}' in workspace '{self.workspace.GetName()}' "
                f"is not a RooRealVar ({obs_var.ClassName()})"
            )

        obs_name = obs_var.GetName()
        low, high = _infer_limits(obs_var)
        key = (obs_name, low, high)
        space = self._space_cache.get(key)
        if space is not None:
            return space

        space = zfit.Space(obs=obs_name, limits=(low, high))
        self._space_cache[key] = space
        return space

    def _convert_real_arg(self, roo_arg):
        name = roo_arg.GetName()
        cached = self._arg_cache.get(name)
        if cached is not None:
            return cached

        class_name = roo_arg.ClassName()

        if roo_arg.InheritsFrom("RooConstVar"):
            value = float(roo_arg.getVal())
            self._arg_cache[name] = value
            return value

        if roo_arg.InheritsFrom("RooRealVar"):
            if bool(roo_arg.isConstant()):
                value = float(roo_arg.getVal())
                self._arg_cache[name] = value
                return value

            low, high = _infer_limits(roo_arg)
            param = zfit.Parameter(
                name=self._zname(name),
                value=float(roo_arg.getVal()),
                lower=low,
                upper=high,
            )
            self._arg_cache[name] = param
            return param

        if roo_arg.InheritsFrom("RooFormulaVar"):
            formula = roo_arg.GetTitle()
            dependencies = []
            for index in range(64):
                try:
                    dependency = roo_arg.getParameter(index)
                except Exception:
                    break
                if dependency is None:
                    break
                dependencies.append(self._convert_real_arg(dependency))
            param = _make_composed_parameter(self._zname(name), formula, dependencies)
            self._arg_cache[name] = param
            return param

        raise ConversionError(
            f"Unsupported RooFit real argument '{name}' ({class_name}) in "
            f"workspace '{self.workspace.GetName()}'."
        )

    def _space_from_pdf_observables(self, roo_pdf) -> zfit.Space:
        observables = roo_pdf.getObservables(self.workspace.allVars())
        obs_vars = [obj for obj in _iter_roo_collection(observables) if obj.InheritsFrom("RooRealVar")]

        if len(obs_vars) != 1:
            obs_names = [obj.GetName() for obj in obs_vars]
            raise ConversionError(
                f"PDF '{roo_pdf.GetName()}' ({roo_pdf.ClassName()}) in workspace "
                f"'{self.workspace.GetName()}' has {len(obs_vars)} observables {obs_names}. "
                "Only 1D PDFs are supported in this initial converter."
            )

        return self._space_for_observable(obs_vars[0])

    def _convert_gaussian(self, roo_pdf):
        obs_var = _call_first(roo_pdf, ["x", "getX"], context="x observable")
        mean = _call_first(roo_pdf, ["mean", "getMean"], context="mean")
        sigma = _call_first(roo_pdf, ["sigma", "getSigma"], context="sigma")

        return zfit.pdf.Gauss(
            obs=self._space_for_observable(obs_var),
            mu=self._convert_real_arg(mean),
            sigma=self._convert_real_arg(sigma),
            name=self._zname(roo_pdf.GetName()),
        )

    def _convert_exponential(self, roo_pdf):
        obs_var = _call_first(roo_pdf, ["x", "getX"], context="x observable")
        coeff = _call_first(roo_pdf, ["c", "coef", "getCoef", "getCoefficient"], context="coefficient")

        return zfit.pdf.Exponential(
            obs=self._space_for_observable(obs_var),
            lam=self._convert_real_arg(coeff),
            name=self._zname(roo_pdf.GetName()),
        )

    def _convert_uniform(self, roo_pdf):
        obs_space = self._space_from_pdf_observables(roo_pdf)
        low, high = obs_space.limit1d
        return zfit.pdf.Uniform(
            obs=obs_space,
            low=float(low),
            high=float(high),
            name=self._zname(roo_pdf.GetName()),
        )

    def _convert_hist_pdf(self, roo_pdf):
        import ROOT

        obs_space = self._space_from_pdf_observables(roo_pdf)
        obs_name = tuple(obs_space.obs)[0]
        roo_datahist = roo_pdf.dataHist()

        if roo_datahist is None:
            raise ConversionError(
                f"RooHistPdf '{roo_pdf.GetName()}' in workspace '{self.workspace.GetName()}' has no data histogram."
            )

        try:
            hist_obj = roo_datahist.createHistogram(obs_name)
        except Exception:
            hist_obj = roo_datahist.createHistogram(roo_pdf.GetName(), self.workspace.var(obs_name))

        if hist_obj is None:
            raise ConversionError(
                f"Could not materialize histogram for RooHistPdf '{roo_pdf.GetName()}' in workspace '{self.workspace.GetName()}'."
            )

        edges = np.asarray([hist_obj.GetXaxis().GetBinLowEdge(1 + idx) for idx in range(hist_obj.GetNbinsX())], dtype=float)
        edges = np.concatenate([edges, [hist_obj.GetXaxis().GetBinUpEdge(hist_obj.GetNbinsX())]])
        counts = np.asarray([hist_obj.GetBinContent(1 + idx) for idx in range(hist_obj.GetNbinsX())], dtype=float)

        binned_space = zfit.Space(obs=obs_name, binning=zfit.binned.VariableBinning(edges, name=obs_name))
        binned_data = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts)
        if not hasattr(zfit.pdf, "HistogramPDF"):
            raise ConversionError("zfit.pdf.HistogramPDF is not available in this zfit version.")
        return zfit.pdf.HistogramPDF(data=binned_data, name=self._zname(roo_pdf.GetName()))

    def _convert_chebychev(self, roo_pdf):
        obs_space = self._space_from_pdf_observables(roo_pdf)
        coeffs = [self._convert_real_arg(obj) for obj in _iter_roo_collection(roo_pdf.coefList())]
        if not coeffs:
            raise ConversionError(
                f"RooChebychev '{roo_pdf.GetName()}' in workspace '{self.workspace.GetName()}' "
                "has no coefficients."
            )

        if not hasattr(zfit.pdf, "Chebyshev"):
            raise ConversionError(
                "zfit.pdf.Chebyshev is not available in this zfit version, so RooChebychev cannot be converted."
            )

        return zfit.pdf.Chebyshev(
            obs=obs_space,
            coeffs=coeffs,
            name=self._zname(roo_pdf.GetName()),
        )

    def _convert_crystal_ball(self, roo_pdf):
        obs_var = _call_first(roo_pdf, ["x", "getX"], context="x observable")
        obs_space = self._space_for_observable(obs_var)
        mu = _call_first(roo_pdf, ["x0", "mean", "m0", "getX0", "getMean"], context="mean")

        has_double_sided = all(
            callable(getattr(roo_pdf, name, None))
            for name in ("sigmaL", "sigmaR", "alphaL", "nL", "alphaR", "nR")
        )

        if has_double_sided:
            if not hasattr(zfit.pdf, "DoubleCB"):
                raise ConversionError(
                    "RooCrystalBall requires zfit.pdf.DoubleCB for asymmetric tails, "
                    "but DoubleCB is not available in this zfit version."
                )
            return zfit.pdf.DoubleCB(
                obs=obs_space,
                mu=self._convert_real_arg(mu),
                sigma=self._convert_real_arg(roo_pdf.sigmaL()),
                alphal=self._convert_real_arg(roo_pdf.alphaL()),
                nl=self._convert_real_arg(roo_pdf.nL()),
                alphar=self._convert_real_arg(roo_pdf.alphaR()),
                nr=self._convert_real_arg(roo_pdf.nR()),
                name=self._zname(roo_pdf.GetName()),
            )

        sigma = _call_first(roo_pdf, ["sigma", "getSigma"], context="sigma")
        alpha = _call_first(roo_pdf, ["alpha", "getAlpha"], context="alpha")
        n = _call_first(roo_pdf, ["n", "getN"], context="n")

        if not hasattr(zfit.pdf, "CrystalBall"):
            raise ConversionError(
                "zfit.pdf.CrystalBall is not available in this zfit version, so RooCrystalBall cannot be converted."
            )

        return zfit.pdf.CrystalBall(
            obs=obs_space,
            mu=self._convert_real_arg(mu),
            sigma=self._convert_real_arg(sigma),
            alpha=self._convert_real_arg(alpha),
            n=self._convert_real_arg(n),
            name=self._zname(roo_pdf.GetName()),
        )

    def _convert_add_pdf(self, roo_pdf):
        components = [self.convert_pdf(obj) for obj in _iter_roo_collection(roo_pdf.pdfList())]
        coeffs = [self._convert_real_arg(obj) for obj in _iter_roo_collection(roo_pdf.coefList())]

        if len(coeffs) != len(components) - 1:
            raise ConversionError(
                f"RooAddPdf '{roo_pdf.GetName()}' in workspace '{self.workspace.GetName()}' has "
                f"{len(components)} components and {len(coeffs)} coefficients. "
                "Only the fraction form with N-1 coefficients is supported."
            )

        return zfit.pdf.SumPDF(
            pdfs=components,
            fracs=coeffs,
            name=self._zname(roo_pdf.GetName()),
        )

    def convert_pdf(self, roo_pdf):
        name = roo_pdf.GetName()
        cached = self._pdf_cache.get(name)
        if cached is not None:
            return cached

        class_name = roo_pdf.ClassName()

        if class_name == "RooGaussian":
            converted = self._convert_gaussian(roo_pdf)
        elif class_name == "RooExponential":
            converted = self._convert_exponential(roo_pdf)
        elif class_name == "RooUniform":
            converted = self._convert_uniform(roo_pdf)
        elif class_name == "RooHistPdf":
            converted = self._convert_hist_pdf(roo_pdf)
        elif class_name == "RooChebychev":
            converted = self._convert_chebychev(roo_pdf)
        elif class_name == "RooCrystalBall":
            converted = self._convert_crystal_ball(roo_pdf)
        elif class_name == "RooAddPdf":
            converted = self._convert_add_pdf(roo_pdf)
        else:
            raise ConversionError(
                f"Unsupported PDF '{name}' in workspace '{self.workspace.GetName()}': "
                f"{class_name}. Supported: RooGaussian, RooExponential, RooUniform, RooHistPdf, "
                "RooChebychev, RooCrystalBall, RooAddPdf."
            )

        self._pdf_cache[name] = converted
        return converted

    def convert_variables(self) -> Dict[str, object]:
        converted = {}
        for var in _iter_roo_collection(self.workspace.allVars()):
            if not var.InheritsFrom("RooRealVar"):
                continue
            converted[var.GetName()] = self._convert_real_arg(var)
        return converted

    def convert_datasets(self) -> Dict[str, zfit.Data]:
        converted = {}

        for dataset in _iter_roo_collection(self.workspace.allData()):
            obs_set = dataset.get()
            obs_vars = [obj for obj in _iter_roo_collection(obs_set) if obj.InheritsFrom("RooRealVar")]

            if len(obs_vars) != 1:
                obs_names = [obj.GetName() for obj in obs_vars]
                raise ConversionError(
                    f"Dataset '{dataset.GetName()}' in workspace '{self.workspace.GetName()}' "
                    f"has {len(obs_vars)} observables {obs_names}. "
                    "Only 1D datasets are supported in this initial converter."
                )

            obs_var = obs_vars[0]
            obs_name = obs_var.GetName()
            obs_space = self._space_for_observable(obs_var)

            if dataset.InheritsFrom("RooDataHist"):
                hist_obj = None
                try:
                    hist_obj = dataset.createHistogram(obs_name)
                except Exception:
                    try:
                        hist_obj = dataset.createHistogram(dataset.GetName(), obs_var)
                    except Exception:
                        hist_obj = None

                if hist_obj is None:
                    raise ConversionError(
                        f"Could not materialize histogram for RooDataHist '{dataset.GetName()}' in workspace '{self.workspace.GetName()}'."
                    )

                edges = np.asarray(
                    [hist_obj.GetXaxis().GetBinLowEdge(1 + idx) for idx in range(hist_obj.GetNbinsX())],
                    dtype=float,
                )
                edges = np.concatenate([edges, [hist_obj.GetXaxis().GetBinUpEdge(hist_obj.GetNbinsX())]])
                counts = np.asarray(
                    [hist_obj.GetBinContent(1 + idx) for idx in range(hist_obj.GetNbinsX())],
                    dtype=float,
                )

                binned_space = zfit.Space(obs=obs_name, binning=zfit.binned.VariableBinning(edges, name=obs_name))
                converted[dataset.GetName()] = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts)
                continue

            n_entries = int(dataset.numEntries())
            values = np.empty((n_entries, 1), dtype=float)
            for idx in range(n_entries):
                row = dataset.get(idx)
                val_obj = row.find(obs_name)
                values[idx, 0] = float(val_obj.getVal())

            converted[dataset.GetName()] = zfit.Data.from_numpy(obs=obs_space, array=values)

        return converted

    def convert_pdfs(self) -> Dict[str, zfit.pdf.BasePDF]:
        converted = {}
        for roo_pdf in _iter_roo_collection(self.workspace.allPdfs()):
            converted[roo_pdf.GetName()] = self.convert_pdf(roo_pdf)
        return converted


def _build_payload(workspace, default_rate: float, name_prefix: str):
    converter = RooWorkspaceConverter(workspace=workspace, name_prefix=name_prefix)

    variables = converter.convert_variables()
    datasets = converter.convert_datasets()
    shapes = converter.convert_pdfs()

    payload = {
        "workspace": workspace.GetName(),
        "variables": variables,
        "datasets": datasets,
        "shapes": shapes,
        "rates": {name: float(default_rate) for name in shapes},
    }

    data_obs = datasets.get("data_obs")
    if data_obs is not None:
        # Preserve the full observed dataset object (including binned structure)
        # so downstream builders can recover counts or unbinned entries correctly.
        payload["data_obs"] = data_obs

    return payload


def _sanitize_name(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in name)


def _looks_like_root_file(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"root"
    except Exception:
        return False


def _as_float(value: Any) -> float:
    if hasattr(value, "value") and callable(value.value):
        return float(value.value())
    return float(value)


def _zfit_param_limits(param: Any) -> Tuple[float, float]:
    lower = getattr(param, "lower", None)
    upper = getattr(param, "upper", None)
    value = _as_float(param)

    lower = float(lower) if lower is not None and np.isfinite(float(lower)) else value - max(abs(value) * 2.0, 1.0)
    upper = float(upper) if upper is not None and np.isfinite(float(upper)) else value + max(abs(value) * 2.0, 1.0)
    if not lower < upper:
        lower, upper = value - 1.0, value + 1.0
    return lower, upper


def convert_zfit_parameter_to_roofit(param: Any, roo_name: Optional[str] = None, cache: Optional[Dict[str, Any]] = None):
    import ROOT

    name = roo_name or getattr(param, "name", None)
    if name is None:
        raise ConversionError("zfit parameter-like object has no name and no roo_name override provided")

    if cache is not None and name in cache:
        return cache[name]

    formula_meta = getattr(param, "_zmodel_rooformula", None)
    if formula_meta is not None:
        dependencies = list(getattr(param, "params", {}).values())
        roo_args = _make_roo_arg_list([
            convert_zfit_parameter_to_roofit(dep, cache=cache)
            for dep in dependencies
        ])
        roo_formula = ROOT.RooFormulaVar(name, name, formula_meta["formula"], roo_args, True)
        if cache is not None:
            cache[name] = roo_formula
        return roo_formula

    is_param = hasattr(param, "floating") and hasattr(param, "value")
    if not is_param:
        roo_obj = ROOT.RooConstVar(name, name, _as_float(param))
        if cache is not None:
            cache[name] = roo_obj
        return roo_obj

    value = _as_float(param)
    low, high = _zfit_param_limits(param)
    roo_var = ROOT.RooRealVar(name, name, value, low, high)
    roo_var.setConstant(not bool(getattr(param, "floating", True)))

    if cache is not None:
        cache[name] = roo_var
    return roo_var


def _get_zfit_obs_and_var(zfit_pdf, obs_cache: Dict[str, Any]):
    import ROOT

    space = zfit_pdf.space
    obs_names = tuple(space.obs or ())
    if len(obs_names) != 1:
        raise ConversionError(
            f"Only 1D zfit PDFs can be converted to RooFit, got observables={obs_names} for '{zfit_pdf.name}'"
        )

    obs_name = obs_names[0]
    low, high = tuple(float(x) for x in space.limit1d)
    obs_var = obs_cache.get(obs_name)
    if obs_var is None:
        obs_var = ROOT.RooRealVar(obs_name, obs_name, 0.5 * (low + high), low, high)
        obs_cache[obs_name] = obs_var
    return obs_name, obs_var


def _zfit_param_by_keys(zfit_pdf, keys: List[str]):
    params = getattr(zfit_pdf, "params", {})
    for key in keys:
        if key in params:
            return params[key]
    return None


def _make_roo_arg_list(items: List[Any]):
    import ROOT

    arg_list = ROOT.RooArgList()
    for item in items:
        arg_list.add(item)
    return arg_list


def _binned_data_to_th1(binned_data, hist_name: str):
    import ROOT

    space = binned_data.space
    obs_names = tuple(space.obs or ())
    if len(obs_names) != 1:
        raise ConversionError(
            f"Only 1D zfit binned data can be converted to a RooFit histogram, got observables={obs_names}."
        )

    obs_name = obs_names[0]
    edges = np.asarray(space.binning[obs_name].edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ConversionError(f"Invalid bin edges for zfit binned data '{hist_name}'")

    th1 = ROOT.TH1D(hist_name, hist_name, len(edges) - 1, edges)
    if hasattr(binned_data, "values") and callable(binned_data.values):
        values = np.asarray(binned_data.values(), dtype=float).reshape(-1)
    elif hasattr(binned_data, "value") and callable(binned_data.value):
        values = np.asarray(binned_data.value(), dtype=float).reshape(-1)
    else:
        raise ConversionError(f"Could not extract bin values from zfit binned data '{hist_name}'")
    if values.size != len(edges) - 1:
        raise ConversionError(
            f"Binned data '{hist_name}' has {values.size} bin values but {len(edges) - 1} bins"
        )

    for idx, value in enumerate(values, start=1):
        th1.SetBinContent(idx, float(value))

    return th1


def convert_zfit_pdf_to_roofit(
    zfit_pdf,
    cache: Optional[Dict[str, Any]] = None,
    param_cache: Optional[Dict[str, Any]] = None,
    obs_cache: Optional[Dict[str, Any]] = None,
):
    import ROOT

    if cache is None:
        cache = {}
    if param_cache is None:
        param_cache = {}
    if obs_cache is None:
        obs_cache = {}

    name = zfit_pdf.name
    if name in cache:
        return cache[name]

    class_name = type(zfit_pdf).__name__
    _, obs_var = _get_zfit_obs_and_var(zfit_pdf, obs_cache)

    if class_name in ("Gauss", "Gaussian"):
        mu = _zfit_param_by_keys(zfit_pdf, ["mu", "mean"])
        sigma = _zfit_param_by_keys(zfit_pdf, ["sigma"])
        if mu is None or sigma is None:
            raise ConversionError(f"Could not find mu/sigma parameters for zfit PDF '{name}'")
        roo_pdf = ROOT.RooGaussian(
            name,
            name,
            obs_var,
            convert_zfit_parameter_to_roofit(mu, cache=param_cache),
            convert_zfit_parameter_to_roofit(sigma, cache=param_cache),
        )
    elif class_name == "Exponential":
        lam = _zfit_param_by_keys(zfit_pdf, ["lam", "lambda"])
        if lam is None:
            raise ConversionError(f"Could not find lam parameter for zfit PDF '{name}'")
        roo_pdf = ROOT.RooExponential(
            name,
            name,
            obs_var,
            convert_zfit_parameter_to_roofit(lam, cache=param_cache),
        )
    elif class_name == "Uniform":
        roo_pdf = ROOT.RooUniform(name, name, ROOT.RooArgSet(obs_var))
    elif class_name == "HistogramPDF":
        data = None
        if hasattr(zfit_pdf, "to_binneddata"):
            data = zfit_pdf.to_binneddata()
        if data is None:
            data = getattr(zfit_pdf, "_data", None)
        if data is None:
            raise ConversionError(f"zfit HistogramPDF '{name}' has no attached binned data")
        th1 = _binned_data_to_th1(data, f"{name}_hist")
        roo_data_hist = ROOT.RooDataHist(name, name, ROOT.RooArgList(obs_var), th1)
        roo_pdf = ROOT.RooHistPdf(name, name, ROOT.RooArgSet(obs_var), roo_data_hist)
    elif class_name == "Chebyshev":
        params = getattr(zfit_pdf, "params", {})
        coeff_items = []
        for key in sorted(params.keys()):
            if str(key).startswith("coeff"):
                coeff_items.append(params[key])
        if not coeff_items:
            raise ConversionError(f"Could not extract Chebyshev coefficients for zfit PDF '{name}'")
        roo_coeffs = [
            convert_zfit_parameter_to_roofit(item, cache=param_cache)
            for item in coeff_items
        ]
        roo_pdf = ROOT.RooChebychev(name, name, obs_var, _make_roo_arg_list(roo_coeffs))
    elif class_name == "CrystalBall":
        mu = _zfit_param_by_keys(zfit_pdf, ["mu", "mean"])
        sigma = _zfit_param_by_keys(zfit_pdf, ["sigma"])
        alpha = _zfit_param_by_keys(zfit_pdf, ["alpha"])
        n = _zfit_param_by_keys(zfit_pdf, ["n"])
        if None in (mu, sigma, alpha, n):
            raise ConversionError(f"Could not extract CrystalBall parameters for zfit PDF '{name}'")

        roo_mu = convert_zfit_parameter_to_roofit(mu, cache=param_cache)
        roo_sigma = convert_zfit_parameter_to_roofit(sigma, cache=param_cache)
        roo_alpha = convert_zfit_parameter_to_roofit(alpha, cache=param_cache)
        roo_n = convert_zfit_parameter_to_roofit(n, cache=param_cache)
        roo_pdf = ROOT.RooCrystalBall(
            name,
            name,
            obs_var,
            roo_mu,
            roo_sigma,
            roo_sigma,
            roo_alpha,
            roo_n,
            roo_alpha,
            roo_n,
        )
    elif class_name == "DoubleCB":
        mu = _zfit_param_by_keys(zfit_pdf, ["mu", "mean"])
        sigma = _zfit_param_by_keys(zfit_pdf, ["sigma"])
        alphal = _zfit_param_by_keys(zfit_pdf, ["alphal", "alpha_l"])
        nl = _zfit_param_by_keys(zfit_pdf, ["nl", "n_l"])
        alphar = _zfit_param_by_keys(zfit_pdf, ["alphar", "alpha_r"])
        nr = _zfit_param_by_keys(zfit_pdf, ["nr", "n_r"])
        if None in (mu, sigma, alphal, nl, alphar, nr):
            raise ConversionError(f"Could not extract DoubleCB parameters for zfit PDF '{name}'")

        roo_pdf = ROOT.RooCrystalBall(
            name,
            name,
            obs_var,
            convert_zfit_parameter_to_roofit(mu, cache=param_cache),
            convert_zfit_parameter_to_roofit(sigma, cache=param_cache),
            convert_zfit_parameter_to_roofit(sigma, cache=param_cache),
            convert_zfit_parameter_to_roofit(alphal, cache=param_cache),
            convert_zfit_parameter_to_roofit(nl, cache=param_cache),
            convert_zfit_parameter_to_roofit(alphar, cache=param_cache),
            convert_zfit_parameter_to_roofit(nr, cache=param_cache),
        )
    elif class_name == "SumPDF":
        components = []
        if hasattr(zfit_pdf, "pdfs"):
            components = list(zfit_pdf.pdfs)
        if not components:
            raise ConversionError(
                f"Could not extract component PDFs from zfit SumPDF '{name}' for RooAddPdf conversion"
            )
        roo_components = [
            convert_zfit_pdf_to_roofit(item, cache=cache, param_cache=param_cache, obs_cache=obs_cache)
            for item in components
        ]

        params = getattr(zfit_pdf, "params", {})
        frac_items = []
        for key in sorted(params.keys()):
            if str(key).startswith("frac"):
                frac_items.append(params[key])
        if len(frac_items) != len(roo_components) - 1:
            raise ConversionError(
                f"zfit SumPDF '{name}' needs N-1 explicit fraction parameters to map to RooAddPdf. "
                f"Found {len(frac_items)} for {len(roo_components)} components."
            )

        roo_coeffs = [
            convert_zfit_parameter_to_roofit(item, cache=param_cache)
            for item in frac_items
        ]
        roo_pdf = ROOT.RooAddPdf(
            name,
            name,
            _make_roo_arg_list(roo_components),
            _make_roo_arg_list(roo_coeffs),
        )
    else:
        raise ConversionError(
            f"Unsupported zfit PDF '{name}' ({class_name}) for conversion to RooFit. "
            "Supported: Gauss, Exponential, Uniform, HistogramPDF, Chebyshev, CrystalBall, DoubleCB, SumPDF."
        )

    cache[name] = roo_pdf
    return roo_pdf


def convert_zfit_data_to_roofit(zfit_data, dataset_name: str = "data_obs", obs_cache: Optional[Dict[str, Any]] = None):
    import ROOT

    if obs_cache is None:
        obs_cache = {}

    space = zfit_data.space
    obs_names = tuple(space.obs or ())
    if len(obs_names) != 1:
        raise ConversionError(
            f"Only 1D zfit datasets can be converted to RooDataSet, got observables={obs_names}."
        )

    obs_name = obs_names[0]
    low, high = tuple(float(x) for x in space.limit1d)
    obs_var = obs_cache.get(obs_name)
    if obs_var is None:
        obs_var = ROOT.RooRealVar(obs_name, obs_name, 0.5 * (low + high), low, high)
        obs_cache[obs_name] = obs_var

    if hasattr(zfit_data, "space") and getattr(zfit_data.space, "binned", False):
        th1 = _binned_data_to_th1(zfit_data, f"{dataset_name}_hist")
        roo_data_hist = ROOT.RooDataHist(dataset_name, dataset_name, ROOT.RooArgList(obs_var), th1)
        return roo_data_hist

    arg_set = ROOT.RooArgSet(obs_var)
    dataset = ROOT.RooDataSet(dataset_name, dataset_name, arg_set)
    if hasattr(zfit_data, "value") and callable(zfit_data.value):
        values = np.asarray(zfit_data.value(), dtype=float).reshape(-1)
    elif hasattr(zfit_data, "values") and callable(zfit_data.values):
        values = np.asarray(zfit_data.values(), dtype=float).reshape(-1)
    else:
        raise ConversionError(f"Could not extract values from zfit dataset '{dataset_name}'")
    for value in values:
        obs_var.setVal(float(value))
        dataset.add(arg_set)
    return dataset


def convert_zfit_payload_to_rooworkspace(payload: Dict[str, Any], workspace_name: str = "converted_ws"):
    import ROOT

    _configure_root_logging()

    workspace = ROOT.RooWorkspace(workspace_name)
    ws_import = getattr(workspace, "import")
    recycle_opts = ROOT.RooFit.RecycleConflictNodes()

    pdf_cache: Dict[str, Any] = {}
    param_cache: Dict[str, Any] = {}
    obs_cache: Dict[str, Any] = {}

    variables = payload.get("variables", {})
    for var_name, var_obj in variables.items():
        roo_var = convert_zfit_parameter_to_roofit(var_obj, roo_name=var_name, cache=param_cache)
        ws_import(roo_var, recycle_opts)

    for pdf_name, zpdf in payload.get("shapes", {}).items():
        roo_pdf = convert_zfit_pdf_to_roofit(
            zpdf,
            cache=pdf_cache,
            param_cache=param_cache,
            obs_cache=obs_cache,
        )
        ws_import(roo_pdf, recycle_opts)

    for data_name, zdata in payload.get("datasets", {}).items():
        roo_data = convert_zfit_data_to_roofit(zdata, dataset_name=data_name, obs_cache=obs_cache)
        ws_import(roo_data, recycle_opts)

    return workspace


def _iter_fit_model_datasets(data_obj):
    if data_obj is None:
        return

    if isinstance(data_obj, dict):
        for name, dataset in data_obj.items():
            yield str(name), dataset
        return

    yield "data_obs", data_obj


def _collect_fit_model_payload(fit_model):
    payload = {
        "variables": {},
        "datasets": {},
        "shapes": {},
    }

    shapes = getattr(fit_model, "shapes", None)
    if isinstance(shapes, dict):
        payload["shapes"].update(shapes)

    model = getattr(fit_model, "model", None)
    if model is not None and getattr(model, "name", None) is not None:
        payload["shapes"][model.name] = model

    channel_models = getattr(fit_model, "channel_models", None)
    if isinstance(channel_models, dict):
        for name, pdf in channel_models.items():
            if getattr(pdf, "name", None) is not None:
                payload["shapes"][pdf.name] = pdf
            else:
                payload["shapes"][name] = pdf

    for term_name, pdf in list(payload["shapes"].items()):
        params = getattr(pdf, "get_params", None)
        if callable(params):
            for param in pdf.get_params(floating=None):
                payload["variables"][getattr(param, "name", term_name)] = param

    if hasattr(fit_model, "yields") and isinstance(fit_model.yields, dict):
        for name, param in fit_model.yields.items():
            payload["variables"].setdefault(name, param)

    for data_name, dataset in _iter_fit_model_datasets(getattr(fit_model, "data", None)):
        payload["datasets"][data_name] = dataset

    return payload


def export_zfit_analysis_to_root(input_file: str, output_root: str, workspace_name: Optional[str] = None):
    import ROOT
    from zmodel.model_io import load_fit_model

    input_path = os.path.abspath(input_file)
    output_path = os.path.abspath(output_root)

    if _looks_like_root_file(input_path):
        raise ConversionError(
            f"Input '{input_file}' looks like a ROOT file. '--output-root' mode expects a zfit analysis snapshot "
            "or fit-model bundle input (.pkl/.json), not a ROOT workspace. "
            "For ROOT->zfit conversion use: convert_rooworkspace <input.root> --output-dir <dir> [--output-prefix <name>]."
        )

    if not output_path.lower().endswith(".root"):
        raise ConversionError(
            f"Output path '{output_root}' must end with '.root' when using '--output-root'."
        )

    with open(input_path, "rb") as handle:
        try:
            payload = dill.load(handle)
        except Exception:
            handle.seek(0)
            payload = None

    fit_model = None
    if isinstance(payload, dict) and payload.get("format") == "analyze_model_snapshot_v1":
        fit_model = payload.get("fit_model")
    elif isinstance(payload, dict) and payload.get("format") == "fit_model_bundle_v1":
        try:
            fit_model = load_fit_model(input_path)
        except Exception as exc:
            raise ConversionError(
                f"Could not load fit-model bundle from '{input_file}' for ROOT export: {exc}"
            ) from exc
    elif hasattr(payload, "model"):
        fit_model = payload
    else:
        try:
            fit_model = load_fit_model(input_path)
        except Exception as exc:
            raise ConversionError(
                f"Input '{input_file}' is not a supported zfit analysis snapshot/fit-model bundle for '--output-root' export: {exc}"
            ) from exc

    if fit_model is None:
        raise ConversionError(f"Could not load a zfit analysis workspace from '{input_file}'")

    payload = _collect_fit_model_payload(fit_model)
    if workspace_name is None:
        workspace_name = getattr(getattr(fit_model, "model", None), "name", None) or "converted_ws"

    workspace = convert_zfit_payload_to_rooworkspace(payload, workspace_name=workspace_name)

    root_file = ROOT.TFile.Open(output_path, "RECREATE")
    if root_file is None or root_file.IsZombie():
        raise ConversionError(f"Could not create ROOT file '{output_path}'")
    root_file.cd()
    workspace.Write(workspace_name)
    root_file.Close()

    return output_path, workspace_name


def convert_root_file(
    root_path: str,
    output_dir: str,
    output_prefix: str,
    default_rate: float,
    include_prefix: bool = False,
) -> List[WorkspaceConversionResult]:
    import ROOT

    _configure_root_logging()

    root_file = ROOT.TFile.Open(root_path)
    if root_file is None or root_file.IsZombie():
        raise ConversionError(f"Could not open ROOT file '{root_path}'")

    workspaces = _collect_workspaces(root_file)
    if not workspaces:
        raise ConversionError(f"No RooWorkspace objects found in '{root_path}'")

    os.makedirs(output_dir, exist_ok=True)
    results = []
    merge_workspaces = len(workspaces) > 1

    merged_payload = {
        "workspace": workspaces[0].GetName() if len(workspaces) == 1 else "merged_workspaces",
        "workspaces": [ws.GetName() for ws in workspaces],
        "workspace_name_mapping": {},
        "variables": {},
        "datasets": {},
        "shapes": {},
        "rates": {},
    }

    merged_data_obs = None

    for workspace in workspaces:
        workspace_name = workspace.GetName()
        prefix = f"{_sanitize_name(workspace_name)}__" if (include_prefix or merge_workspaces) else ""
        payload = _build_payload(
            workspace=workspace,
            default_rate=default_rate,
            name_prefix=prefix,
        )

        merged_payload["workspace_name_mapping"][workspace_name] = prefix

        for key, value in payload["variables"].items():
            merged_key = f"{prefix}{key}" if prefix else key
            if merged_key in merged_payload["variables"]:
                raise ConversionError(
                    f"Variable name collision while merging workspaces: '{merged_key}'. "
                    "Use '--workspace-prefix' to force unique prefixed names."
                )
            merged_payload["variables"][merged_key] = value

        for key, value in payload["datasets"].items():
            merged_key = f"{prefix}{key}" if prefix else key
            if merged_key in merged_payload["datasets"]:
                raise ConversionError(
                    f"Dataset name collision while merging workspaces: '{merged_key}'. "
                    "Use '--workspace-prefix' to force unique prefixed names."
                )
            merged_payload["datasets"][merged_key] = value

        for key, value in payload["shapes"].items():
            merged_key = f"{prefix}{key}" if prefix else key
            if merged_key in merged_payload["shapes"]:
                raise ConversionError(
                    f"PDF name collision while merging workspaces: '{merged_key}'. "
                    "Use '--workspace-prefix' to force unique prefixed names."
                )
            merged_payload["shapes"][merged_key] = value

        for key, value in payload["rates"].items():
            merged_key = f"{prefix}{key}" if prefix else key
            merged_payload["rates"][merged_key] = value

        if "data_obs" in payload and merged_data_obs is None:
            merged_data_obs = payload["data_obs"]

        output_name = f"{output_prefix}.pkl"
        output_file = os.path.join(output_dir, output_name)

        results.append(
            WorkspaceConversionResult(
                workspace_name=workspace_name,
                output_file=output_file,
                n_variables=len(payload["variables"]),
                n_datasets=len(payload["datasets"]),
                n_pdfs=len(payload["shapes"]),
            )
        )

    if merged_data_obs is not None:
        merged_payload["data_obs"] = merged_data_obs

    with open(output_file, "wb") as handle:
        dill.dump(merged_payload, handle)

    root_file.Close()
    return results


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert RooWorkspace content in a ROOT file into zfit shape payload pickles "
            "usable by zmodel card 'shapes' lines, or export a zfit analysis snapshot back to ROOT."
        )
    )
    parser.add_argument("root_file", help="Input ROOT file path")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where one payload pickle per RooWorkspace is written",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for output files (default: input ROOT basename without extension)",
    )
    parser.add_argument(
        "--default-rate",
        type=float,
        default=1.0,
        help="Default nominal rate value assigned to each converted PDF",
    )
    parser.add_argument(
        "--workspace-prefix",
        action="store_true",
        help=(
            "Prefix converted zfit object names with '<workspace>__'. "
            "By default, original RooFit names are preserved wherever possible."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional ROOT file to write instead of the default shape payload pickle outputs",
    )
    parser.add_argument(
        "--workspace-name",
        type=str,
        default=None,
        help="Workspace name to use when exporting a zfit analysis snapshot to ROOT",
    )
    return parser


def main():
    args = _build_parser().parse_args()

    input_path = os.path.abspath(args.root_file)

    if args.output_root is not None:
        output_root = os.path.abspath(args.output_root)
        if _looks_like_root_file(input_path):
            raise ConversionError(
                "'--output-root' is the reverse export mode (zfit snapshot/bundle -> ROOT workspace). "
                "For converting ROOT workspaces to zfit shape pickles, omit '--output-root' and use '--output-dir'."
            )
        if not output_root.lower().endswith(".root"):
            raise ConversionError(
                f"Output path '{args.output_root}' must end with '.root' when using '--output-root'."
            )
        output_path, workspace_name = export_zfit_analysis_to_root(
            input_file=input_path,
            output_root=output_root,
            workspace_name=args.workspace_name,
        )
        print(f"Wrote RooWorkspace '{workspace_name}' to {output_path}")
        return

    output_dir = os.path.abspath(args.output_dir)
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(input_path))[0]

    results = convert_root_file(
        root_path=input_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        default_rate=args.default_rate,
        include_prefix=args.workspace_prefix,
    )

    print(f"Converted {len(results)} workspace(s) from {input_path}")
    for result in results:
        print(
            f"- {result.workspace_name}: {result.output_file} "
            f"(variables={result.n_variables}, datasets={result.n_datasets}, pdfs={result.n_pdfs})"
        )


if __name__ == "__main__":
    main()
