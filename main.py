from src.main import main
from src.gradio import gradio_main
from src.logger import setup_default_logger, setup_logger
import json
import logging

import argparse


logger = logging.getLogger()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config"
                        , default=DEFAULT_PATH_TO_CONFIG
                        , help="Path to config")
    parser.add_argument("--mode", choices=["stdio", "gradio"] 
                        , default="stdio"
                        , help="Path to config")
    parser.add_argument("--searcher", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    setup_default_logger()
    args = parse_arguments()
    
    configs = None 

    try:
        with open(args.config, 'r', encoding='utf8') as config_file:
            configs = json.load(config_file)
    except Exception as ex:
        logger.error(f"Failed to read configs: {ex}")
        exit(1)
    setup_logger(configs.get("logging"))
    
    if args.mode == "gradio":
        gradio_main(share=True)
    else:
        main(configs, args.searcher)