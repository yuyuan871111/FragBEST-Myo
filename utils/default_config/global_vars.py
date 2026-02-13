import configparser
import sys

# Read config file.
config = configparser.ConfigParser()
path_args = __file__.split("/")[0:-1]
root_path = "/".join(path_args)
config.read(f"{root_path}/../config.cfg")
config.sections()

propka3_bin = ""
if "PROPKA3_BIN" in config["ThirdParty"]:
    propka3_bin = config["ThirdParty"]["PROPKA3_BIN"]
else:
    print("ERROR: PROPKA3_BIN not set. Variable should point to PROPKA3_BIN program.")
    sys.exit(1)
