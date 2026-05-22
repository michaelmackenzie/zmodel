import dill
import numpy as np
import zfit


obs = zfit.Space("mass", limits=(100.0, 110.0))
RNG_SEED = 24680


def _make_category_payload(category):
    if category == "catA":
        sig_mu = 105.0
        sig_sigma = 0.35
        bkg_lam = -0.25
        rates = {"sig": 10.0, "bkg": 70.0}
    elif category == "catB":
        sig_mu = 105.4
        sig_sigma = 0.45
        bkg_lam = -0.18
        rates = {"sig": 6.0, "bkg": 45.0}
    else:
        raise ValueError(f"Unknown category '{category}'")

    sig_pdf = zfit.pdf.Gauss(obs=obs, mu=sig_mu, sigma=sig_sigma, name=f"sig_pdf_{category}")
    sig_resUp = zfit.pdf.Gauss(obs=obs, mu=sig_mu, sigma=sig_sigma * 1.35, name=f"sig_resUp_{category}")
    sig_resDown = zfit.pdf.Gauss(obs=obs, mu=sig_mu, sigma=sig_sigma * 0.65, name=f"sig_resDown_{category}")

    bkg_pdf = zfit.pdf.Exponential(obs=obs, lam=bkg_lam, name=f"bkg_pdf_{category}")
    bkg_slopeUp = zfit.pdf.Exponential(obs=obs, lam=bkg_lam * 0.7, name=f"bkg_slopeUp_{category}")
    bkg_slopeDown = zfit.pdf.Exponential(obs=obs, lam=bkg_lam * 1.3, name=f"bkg_slopeDown_{category}")

    return {
        "shapes": {
            "sig": sig_pdf,
            "bkg": bkg_pdf,
            "sig_resUp": sig_resUp,
            "sig_resDown": sig_resDown,
            "bkg_slopeUp": bkg_slopeUp,
            "bkg_slopeDown": bkg_slopeDown,
        },
        "rates": rates,
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


def write_two_category_payloads(prefix="simple_shapes", rng_seed=RNG_SEED):
    rng = np.random.default_rng(int(rng_seed))
    payload_a = _attach_toy_data_obs(_make_category_payload("catA"), rng)
    payload_b = _attach_toy_data_obs(_make_category_payload("catB"), rng)

    file_a = f"{prefix}_catA.pkl"
    file_b = f"{prefix}_catB.pkl"

    with open(file_a, "wb") as handle:
        dill.dump(payload_a, handle)
    with open(file_b, "wb") as handle:
        dill.dump(payload_b, handle)

    print(f"Wrote shape payload: {file_a}")
    print(f"Wrote shape payload: {file_b}")


if __name__ == "__main__":
    write_two_category_payloads()
