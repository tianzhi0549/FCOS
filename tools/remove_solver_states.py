# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip
import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser(description="Remove the solver states stored in a trained model")
    parser.add_argument(
        "model",
        default="models/FCOS_R_50_FPN_1x.pth",
        help="path to the input model file",
    )

    args = parser.parse_args()

    model = torch.load(args.model)
    del model["optimizer"]
    del model["scheduler"]

    filename_wo_ext, ext = os.path.splitext(args.model)
    output_file = filename_wo_ext + "_wo_solver_states" + ext
    torch.save(model, output_file)
    print("Done. The model without solver states is saved to {}".format(output_file))

if __name__ == "__main__":
    main()

