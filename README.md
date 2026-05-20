# zfit Modeling

This package is meant to assist physics analysis using the `zfit` framework.
Analysis models are considered to be the combination of signal/background processes with
associated PDFs, rates, uncertainties, and observed datasets.
This can be counting, binned, or unbinned models, with one or multiple separate search categories.
Analyses may be measuring one or more quantities, placing one- or two-sided limits on the parameters,
evaluating a measurement significance or goodness-of-fit, etc.

This package organizes analysis models into `.pkl` files that contain the `zfit` model information,
with associated text ``data cards'' that contain information about the relevant processes, model files,
rate and shape uncertainties, and search categories to include.
These inputs are combined into a total model workspace that is used for analysis.

## Installation

Clone the repository and install dependencies in your own Python environment:

```bash
git clone https://github.com/michaelmackenzie/zfit_modeling.git zfit_modeling
cd zfit_modeling

# For general users
python -m venv .venv
source .venv/bin/activate
pip install zfit uproot hist hepstats dill scipy tensorflow
```

For the Mu2e environment used by this repository, start from the shared setup and activate the matching Python stack:

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv rootana 2.5.0
```

## Usage

Build a model bundle from a text card. The default output is `model.pkl`, and observed data from the card is bundled when present:

```bash
./zmodel build examples/simple_model_card_example.txt
```

Load a saved model bundle. The summary includes the observed data count when it exists:

```bash
./zmodel load model.pkl
```

Run an analysis. If the saved model contains observed data, `analyze` fits that data by default. Use `--toys N` to generate toy datasets, or `--toys -1` to run the exact binned Asimov mode for validation:

```bash
./zmodel analyze --model-file model.pkl
./zmodel analyze --model-file model.pkl --toys 10 --fit-mode auto
./zmodel analyze --model-file model.pkl --toys -1
```

## Notes

- No user-specific file locations are required by default.
- Input cards should use relative paths (recommended) or absolute paths to your own data files.
- ROOT histogram loading utilities require explicit file and object paths; no hidden global path configuration is used.
