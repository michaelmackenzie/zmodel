# zmodel: zfit Modeling

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
git clone https://github.com/michaelmackenzie/zmodel.git zmodel
PATH="${PATH}:${PWD}/zmodel/bin"
PYTHONPATH="${PYTHONPATH}:${PWD}"

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

User-facing tool tools are organized under `bin/`:

- `bin/zmodel` for build/load/analyze
- `bin/plot_analysis.py` for plotting from snapshots

Build a model bundle from a text card. The default output is `model.pkl`, and observed data from the card is bundled when present:

```bash
zmodel build examples/simple_model_card_example.txt
```

Load a saved model bundle. The summary includes the observed data count when it exists:

```bash
zmodel load model.pkl
```

Run an analysis. If the saved model contains observed data, `analyze` fits that data by default. Use `--toys N` to generate toy datasets, or `--toys -1` to run the exact binned Asimov mode for validation:

```bash
zmodel analyze --model-file model.pkl
zmodel analyze --model-file model.pkl --toys 10 --fit-mode auto --plot
zmodel analyze --model-file model.pkl --toys -1
zmodel analyze --model-file model.pkl --toys -1 --feldman-cousins 0.1 --fc-scan-points 10 --fc-toys 50 --fc-scan-max 5.0
```

Each analysis run now also writes an ensemble evaluation report in JSON format. By default the file is derived from `--output-pkl`, for example `analysis_output_ensemble_report.json`. Use `--report-file` to choose a custom path.

Generate plots from a saved analysis snapshot:

```bash
plot_analysis.py analysis_output.pkl --plot-dir plots_from_snapshot
```

Convert RooFit workspaces in a ROOT file into zfit shape payloads that can be used as `shapes` inputs:

```bash
python python/convert_rooworkspace_shapes.py input.root --output-dir shapes
```

The converter writes one pickle per `RooWorkspace` and currently supports a focused subset of PDFs with clear zfit equivalents, including Gaussian, exponential, uniform, histogram, Chebyshev, and Crystal Ball forms. Unsupported PDFs raise an error.

The same module also exposes helper functions for the reverse direction, converting zfit parameters, datasets, and PDFs back into RooFit objects when needed. To export a saved zfit analysis snapshot back into a ROOT file containing a `RooWorkspace`, use:

```bash
python python/convert_rooworkspace_shapes.py analysis_output.pkl --output-root workspace.root
```

Convert text cards between Combine and zmodel formats:

```bash
# Combine -> zmodel (auto-detects Combine input)
python python/convert_datacard_format.py \
	/exp/mu2e/app/users/mmackenz/conv/ConvAna/analysis/datacards/combine_mumem_20_r0102.txt \
	examples/mumem_20_zmodel_card.txt \
	--shapes-file shapes/workspace_mumem_20_r0102.pkl

# zmodel -> Combine
python python/convert_datacard_format.py \
	examples/mumem_20_zmodel_card.txt \
	examples/mumem_20_combine_card.txt \
	--direction zmodel-to-combine \
	--root-file workspaces/workspace_mumem_20_r0102.root \
	--workspace-name workspace \
	--pdf-template 'mumem_20_$PROCESS_pdf' \
	--syst-template 'mumem_20_$PROCESS_pdf_$SYSTEMATIC'
```

## Notes

- No user-specific file locations are required by default.
- Input cards should use relative paths (recommended) or absolute paths to your own data files.
