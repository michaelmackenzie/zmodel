import numpy as np
import uproot
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import zfit
import zfit.z.numpy as znp
import hist
from hepstats.hypotests import UpperLimit
from hepstats.hypotests.calculators import AsymptoticCalculator
from hepstats.hypotests.parameters import POI
from hepstats.hypotests.parameters import POIarray

# Histogram and selection information
selection_set    = 20
hist_path        = '/exp/mu2e/data/users/mmackenz/conv_ana/histograms/'
hist_mode        = 2
path_in_file     = f'Ana/ConvAna_ConvAna/Hist/trk_{selection_set}'
tree_in_file     = f'Ana/ConvAna_ConvAna/data/Norm'
hist_name        = 'obs'
target_bin_width = 0.5


@dataclass
class FitModel:
    obs: zfit.Space
    obs_range: tuple
    shapes: Dict[str, zfit.pdf.BasePDF]
    yields: Dict[str, zfit.Parameter]
    extended_pdfs: Dict[str, zfit.pdf.BasePDF]
    model: zfit.pdf.BasePDF
    data: Any
    process_names: List[str] = field(default_factory=list)
    signal_process: Optional[str] = None
    constraints: List[Any] = field(default_factory=list)
    loss: Optional[Any] = None
    result: Optional[Any] = None
    signal_nominal_yield: Optional[float] = None

# Define a custom PowerLaw PDF class
class PowerLaw(zfit.pdf.ZPDF):
    """
    Custom 1D Power-Law PDF: f(x) = x^(gamma)
    For a falling spectrum like DIO, gamma will optimize to a negative number.
    """
    # Define the parameter names that this PDF depends on
    _PARAMS = ("gamma",)

    @zfit.supports(norm=False)
    def _pdf(self, x, norm, params):
        # Extract the coordinate tensor (axis 0)
        data = x[0]
        gamma = params["gamma"]

        # Return the unnormalized mathematical definition
        return znp.power(data, gamma)

# ==============================================================================
# Helper Function: Convert a ROOT Histogram to zfit-compatible BinnedData
# ==============================================================================
def load_binned_data_from_root(name, obs_space, scale = 1., nexpected = -1):
    file_path = f'{hist_path}ConvAna.cnv_ana.{name}.m{hist_mode}.hist'
    with uproot.open(file_path) as f:
        # Access the histogram inside the target directory
        raw_hist = f[f"{path_in_file}/{hist_name}"]
        if not raw_hist:
            raise Exception(f"No histogram for {name}")
        h = raw_hist.to_hist()
        if nexpected > 0:
            tree = f[tree_in_file]
            nseen = np.sum(tree["nseen"].array())
            if nseen != nexpected:
                print(f'{name} has {nseen} sampled count, but expected {nexpected} --> Scaling to compensate!')
                scale *= nexpected / nseen

    lower_lim, upper_lim = obs_space.limit1d

    # Slice using coordinate locators (the 'j' suffix means 'value in axis units')
    h = h[hist.loc(lower_lim) : hist.loc(upper_lim)]

    # Apply rebinning on the truncated histogram if needed
    bin_width = h.axes[0].widths[0]
    rebin_factor = int(target_bin_width / bin_width)
    if rebin_factor > 1:
        h = h[::hist.rebin(rebin_factor)]

    # Extract bin edges and counts (excluding underflow/overflow)
    counts = h.values()
    edges = h.axes[0].edges

    # Apply the scale factor
    counts = [ count * scale for count in counts ]

    # Create a zfit-compatible binning structure
    binning = zfit.binned.VariableBinning(edges, name=obs_space.obs[0])
    binned_space = zfit.Space(obs_space.obs[0], binning=binning)

    # Pack into zfit BinnedData structure
    return zfit.data.BinnedData.from_tensor(space=binned_space, values=counts)
