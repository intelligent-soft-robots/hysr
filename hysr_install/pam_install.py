import logging
import subprocess
from pathlib import Path

SCRIPT_FILE: str = "pam.sh"


def run() -> None:

    logging.basicConfig(level=logging.INFO)

    script_folder = Path(__file__).absolute().parents[0]
    script_file = script_folder / SCRIPT_FILE

    if not script_file.is_file():
        logging.error(
            "Failed to find the pam installation script file {} "
            "(searched in {})".format(SCRIPT_FILE, script_folder)
        )
        exit(1)

    logging.info("calling the script file {}".format(script_file))

    ret = subprocess.call(str(script_file))

    if ret == 0:
        logging.info("successfully installed pam")
        exit(0)

    else:
        logging.error("failed to install pam with error code: {}".format(ret))
        exit(1)
