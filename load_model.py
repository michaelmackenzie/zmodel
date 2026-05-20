import argparse
import os

from model_io import load_fit_model


def main():
    parser = argparse.ArgumentParser(description="Load a saved zfit model bundle or HS3 file")
    parser.add_argument("model_file", help="Path to the saved model JSON file")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_file)
    fit_model = load_fit_model(model_path)

    print(f"Loaded model from {model_path}")
    print(f"Model name: {fit_model.model.name}")
    print(f"Observable range: {fit_model.obs_range}")
    print(f"Processes: {fit_model.process_names or 'unknown'}")
    print(f"Signal process: {fit_model.signal_process}")
    print(f"Constraints: {len(fit_model.constraints)}")
    print(f"Floating params: {len(fit_model.model.get_params())}")


if __name__ == "__main__":
    main()