# zfit Modeling

This package is meant to assist physics analysis using the `zfit` framework.
Analysis models are considered to be the combination of signal/background processes with
associated PDFs, rates, uncertainties, and observed datasets.
This can be counting, binned, or unbinned models, with one or multiple separate search categories.
Analyses may be measuring one or more quantities, placing one- or two-sided limits on the parameters,
evaluating a measurement significance or goodness-of-fit, etc.

This package organizes analysis models into pkl files that contain the `zfit` model information,
with associated text ``data cards'' that contain information about the relevant processes, model files,
rate and shape uncertainties, and search categories to include.
These inputs are combined into a total model workspace that is used for analysis.

## Installation

Clone the repository and install dependencies in your own Python environment:

```bash
git clone <your-fork-or-repo-url>
cd zfit_modeling
python -m venv .venv
source .venv/bin/activate
pip install zfit uproot hist hepstats dill scipy tensorflow
```

## Usage

Build a model bundle from a text card:

```bash
./zmodel build examples/simple_model_card_example.txt model.json
```

Load a saved model bundle:

```bash
./zmodel load model.json
```

Run toy fits and optional CLs evaluation:

```bash
./zmodel analyze --model-file model.json --toys 10 --fit-mode auto
```

## Notes

- No user-specific file locations are required by default.
- Input cards should use relative paths (recommended) or absolute paths to your own data files.
- ROOT histogram loading utilities require explicit file and object paths; no hidden global path configuration is used.
