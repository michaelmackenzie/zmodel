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
    channels: List[str] = field(default_factory=list)
    term_channels: Dict[str, str] = field(default_factory=dict)
    term_processes: Dict[str, str] = field(default_factory=dict)
    observed_counts_by_channel: Dict[str, float] = field(default_factory=dict)
    observed_values_by_channel: Dict[str, Any] = field(default_factory=dict)
    channel_models: Dict[str, zfit.pdf.BasePDF] = field(default_factory=dict)
    channel_obs: Dict[str, zfit.Space] = field(default_factory=dict)
    channel_obs_ranges: Dict[str, tuple] = field(default_factory=dict)

# ==============================================================================
# Helper Function: Convert a ROOT Histogram to zfit-compatible BinnedData
# ==============================================================================
def load_binned_data_from_root(
    name,
    obs_space,
    file_path,
    *,
    histogram_path,
    target_bin_width,
    scale=1.0,
    nexpected=-1,
    tree_path=None,
):
    """Load binned data from a ROOT histogram using explicit, portable inputs.

    Parameters
    ----------
    name : str
        Dataset/process name used only for logging and errors.
    obs_space : zfit.Space
        Observable space defining the target range.
    file_path : str
        ROOT input file path.
    histogram_path : str
        Path to histogram within the ROOT file, e.g. "dir/hist".
    target_bin_width : float
        Desired output bin width in axis units.
    scale : float, optional
        Global scale factor applied to histogram counts.
    nexpected : int or float, optional
        If > 0, normalize sampled histogram counts to this expected count.
    tree_path : str, optional
        ROOT tree path used when `nexpected > 0` to read branch "nseen".
    """
    with uproot.open(file_path) as f:
        # Access the histogram inside the target directory
        raw_hist = f[histogram_path]
        if not raw_hist:
            raise Exception(f"No histogram for {name}")
        h = raw_hist.to_hist()
        if nexpected > 0:
            if tree_path is None:
                raise ValueError("tree_path is required when nexpected > 0")
            tree = f[tree_path]
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
