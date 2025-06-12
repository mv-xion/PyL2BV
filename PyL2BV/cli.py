import argparse
import matplotlib.pyplot as plt
from PyL2BV.pyl2bv_code.model_runner import run_retrieval


def main():
    parser = argparse.ArgumentParser(description="Run the model.")
    parser.add_argument(
        "input_folder_path",
        type=str,
        help="Path to the input folder",
    )
    parser.add_argument(
        "input_type",
        type=str,
        help="Type of input file",
    )
    parser.add_argument(
        "model_folder_path",
        type=str,
        help="Path to the model folder",
    )
    parser.add_argument(
        "conversion_factor",
        type=float,
        help="Image conversion factor",
    )
    parser.add_argument(
        "chunk_size",
        type=int,
        help="Processing chunk size",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Flag to enable plotting",
    )
    parser.add_argument(
        "--debug_log",
        action="store_true",
        help="Flag to enable debug logging",
    )

    args = parser.parse_args()

    result = run_retrieval(
        input_folder_path=args.input_folder_path,
        input_type=args.input_type,
        model_folder_path=args.model_folder_path,
        conversion_factor=args.conversion_factor,
        chunk_size=args.chunk_size,
        show_message_callback=None,
        plotting=args.plot,
        debug_log=args.debug_log
    )

    if result.success and result.plots and args.plot:
        for img, title, cmap in result.plots:
            plt.imshow(img, cmap=cmap)
            plt.title(title)
            plt.colorbar()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
