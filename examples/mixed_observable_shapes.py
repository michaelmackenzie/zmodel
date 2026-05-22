import dill
import numpy as np
import zfit


RNG_SEED = 13579


def _make_catA_payload():
    obs = zfit.Space("massA", limits=(100.0, 110.0))

    sig_pdf = zfit.pdf.Gauss(obs=obs, mu=105.0, sigma=0.35, name="sig_pdf_catA")
    sig_resUp = zfit.pdf.Gauss(obs=obs, mu=105.0, sigma=0.50, name="sig_resUp_catA")
    sig_resDown = zfit.pdf.Gauss(obs=obs, mu=105.0, sigma=0.22, name="sig_resDown_catA")

    bkg_pdf = zfit.pdf.Exponential(obs=obs, lam=-0.22, name="bkg_pdf_catA")
    bkg_slopeUp = zfit.pdf.Exponential(obs=obs, lam=-0.16, name="bkg_slopeUp_catA")
    bkg_slopeDown = zfit.pdf.Exponential(obs=obs, lam=-0.30, name="bkg_slopeDown_catA")

    return {
        "shapes": {
            "sig": sig_pdf,
            "bkg": bkg_pdf,
            "sig_resUp": sig_resUp,
            "sig_resDown": sig_resDown,
            "bkg_slopeUp": bkg_slopeUp,
            "bkg_slopeDown": bkg_slopeDown,
        },
        "rates": {
            "sig": 9.0,
            "bkg": 62.0,
        },
        "_nominal_pdfs": {
            "sig": sig_pdf,
            "bkg": bkg_pdf,
        },
    }


def _make_catB_payload():
    obs = zfit.Space("energyB", limits=(0.0, 200.0))

    sig_pdf = zfit.pdf.Gauss(obs=obs, mu=92.0, sigma=8.0, name="sig_pdf_catB")
    sig_resUp = zfit.pdf.Gauss(obs=obs, mu=92.0, sigma=11.0, name="sig_resUp_catB")
    sig_resDown = zfit.pdf.Gauss(obs=obs, mu=92.0, sigma=5.5, name="sig_resDown_catB")

    bkg_pdf = zfit.pdf.Exponential(obs=obs, lam=-0.018, name="bkg_pdf_catB")
    bkg_slopeUp = zfit.pdf.Exponential(obs=obs, lam=-0.012, name="bkg_slopeUp_catB")
    bkg_slopeDown = zfit.pdf.Exponential(obs=obs, lam=-0.027, name="bkg_slopeDown_catB")

    return {
        "shapes": {
            "sig": sig_pdf,
            "bkg": bkg_pdf,
            "sig_resUp": sig_resUp,
            "sig_resDown": sig_resDown,
            "bkg_slopeUp": bkg_slopeUp,
            "bkg_slopeDown": bkg_slopeDown,
        },
        "rates": {
            "sig": 5.0,
            "bkg": 38.0,
        },
        "_nominal_pdfs": {
            "sig": sig_pdf,
            "bkg": bkg_pdf,
        },
    }


def _sample_component(pdf, yield_mean, rng):
    n_events = int(rng.poisson(float(yield_mean)))
    if n_events <= 0:
        return np.empty(0, dtype=float)
    return np.asarray(pdf.sample(n=n_events).value(), dtype=float).reshape(-1)


def _attach_toy_data_obs(payload, rng):
    rates = payload["rates"]
    nominal_pdfs = payload.pop("_nominal_pdfs")
    sig_values = _sample_component(nominal_pdfs["sig"], rates["sig"], rng)
    bkg_values = _sample_component(nominal_pdfs["bkg"], rates["bkg"], rng)
    toy_values = np.concatenate([sig_values, bkg_values]) if (sig_values.size or bkg_values.size) else np.empty(0, dtype=float)
    rng.shuffle(toy_values)
    payload["data_obs"] = {"values": toy_values}
    return payload


def write_mixed_observable_payloads(prefix="mixed_observable_shapes", rng_seed=RNG_SEED):
    rng = np.random.default_rng(int(rng_seed))
    file_a = f"{prefix}_catA.pkl"
    file_b = f"{prefix}_catB.pkl"

    with open(file_a, "wb") as handle:
        dill.dump(_attach_toy_data_obs(_make_catA_payload(), rng), handle)
    with open(file_b, "wb") as handle:
        dill.dump(_attach_toy_data_obs(_make_catB_payload(), rng), handle)

    print(f"Wrote shape payload: {file_a}")
    print(f"Wrote shape payload: {file_b}")


if __name__ == "__main__":
    write_mixed_observable_payloads()
