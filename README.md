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

```bash
git clone https://github.com/michaelmackenzie/zfit_modeling.git
```

## Usage

In the Mu2e environment, the underlying python environment must first be enabled:
```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv rootana 2.5.0
```

Converting input model information into a complete model workspace:
```bash


```
