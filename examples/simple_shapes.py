import pickle
import dill
import numpy as np
import zfit

obs = zfit.Space("mass", limits=(100.0, 110.0))

sig_mu = zfit.Parameter("sig_mu", 105.0, 103.0, 107.0)
sig_mu_offset = zfit.Parameter("sig_mu_offset", 0.0, -1.0, 1.0)
sig_mu_eff = zfit.ComposedParameter(
    "sig_mu_eff",
    lambda mu, d: mu + d,
    params=[sig_mu, sig_mu_offset],
)
sig_sigma = zfit.Parameter("sig_sigma", 0.35, 0.02, 2.0)
sig_pdf = zfit.pdf.Gauss(obs=obs, mu=sig_mu_eff, sigma=sig_sigma, name="sig_pdf")
sig_resUp = zfit.pdf.Gauss(obs=obs, mu=sig_mu_eff, sigma=0.50, name="sig_resUp")
sig_resDown = zfit.pdf.Gauss(obs=obs, mu=sig_mu_eff, sigma=0.20, name="sig_resDown")


bkg_lambda = zfit.Parameter("bkg_lambda", -0.25, -3.0, -0.001)
bkg_pdf = zfit.pdf.Exponential(obs=obs, lam=bkg_lambda, name="bkg_pdf")
bkg_slopeUp = zfit.pdf.Exponential(obs=obs, lam=-0.15, name="bkg_slopeUp")
bkg_slopeDown = zfit.pdf.Exponential(obs=obs, lam=-0.45, name="bkg_slopeDown")


RATES = {
    "sig": 12.0,
    "bkg": 80.0,
}
RNG_SEED = 12345


def _sample_component(pdf, yield_mean, rng):
    n_events = int(rng.poisson(float(yield_mean)))
    if n_events <= 0:
        return np.empty(0, dtype=float)
    return np.asarray(pdf.sample(n=n_events).value(), dtype=float).reshape(-1)


def make_toy_data_obs(rng_seed=RNG_SEED):
    rng = np.random.default_rng(int(rng_seed))
    sig_values = _sample_component(sig_pdf, RATES["sig"], rng)
    bkg_values = _sample_component(bkg_pdf, RATES["bkg"], rng)
    toy_values = np.concatenate([sig_values, bkg_values]) if (sig_values.size or bkg_values.size) else np.empty(0, dtype=float)
    rng.shuffle(toy_values)
    return toy_values


def make_shape_payload():
    return {
        "shapes": {
            "sig": sig_pdf,
            "bkg": bkg_pdf,
            "sig_resUp": sig_resUp,
            "sig_resDown": sig_resDown,
            "bkg_slopeUp": bkg_slopeUp,
            "bkg_slopeDown": bkg_slopeDown,
        },
        "rates": dict(RATES),
        "data_obs": {
            "values": make_toy_data_obs(),
        },
    }


def write_shape_payload(output_file="simple_shapes.pkl"):
    payload = make_shape_payload()
    with open(output_file, "wb") as handle:
        dill.dump(payload, handle)
    print(f"Wrote shape payload: {output_file}")


if __name__ == "__main__":
    write_shape_payload()
