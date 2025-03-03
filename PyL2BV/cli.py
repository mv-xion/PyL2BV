import argparse
from .pyl2bv_code.model_runner import run_retrieval


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

    completion_message = run_retrieval(
        input_folder_path=args.input_folder_path,
        input_type=args.input_type,
        model_folder_path=args.model_folder_path,
        conversion_factor=args.conversion_factor,
        show_message_callback=None,
        plot=args.plot,
        debug_log=args.debug_log
    )


if __name__ == "__main__":
    main()
