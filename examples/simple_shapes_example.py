import pickle

import dill
import zfit


obs = zfit.Space("mass", limits=(100.0, 110.0))


sig_mu = zfit.Parameter("example_sig_mu", 105.0, 103.0, 107.0)
sig_sigma = zfit.Parameter("example_sig_sigma", 0.35, 0.02, 2.0)
sig_pdf = zfit.pdf.Gauss(obs=obs, mu=sig_mu, sigma=sig_sigma, name="sig_pdf")
sig_resUp = zfit.pdf.Gauss(obs=obs, mu=sig_mu, sigma=0.50, name="sig_resUp")
sig_resDown = zfit.pdf.Gauss(obs=obs, mu=sig_mu, sigma=0.20, name="sig_resDown")


bkg_lambda = zfit.Parameter("example_bkg_lambda", -0.25, -3.0, -0.001)
bkg_pdf = zfit.pdf.Exponential(obs=obs, lam=bkg_lambda, name="bkg_pdf")
bkg_slopeUp = zfit.pdf.Exponential(obs=obs, lam=-0.15, name="bkg_slopeUp")
bkg_slopeDown = zfit.pdf.Exponential(obs=obs, lam=-0.45, name="bkg_slopeDown")


RATES = {
    "sig": 12.0,
    "bkg": 80.0,
}


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
    }


def write_shape_payload(output_file="simple_shapes_example.pkl"):
    payload = make_shape_payload()
    with open(output_file, "wb") as handle:
        dill.dump(payload, handle)
    print(f"Wrote shape payload: {output_file}")


if __name__ == "__main__":
    write_shape_payload()