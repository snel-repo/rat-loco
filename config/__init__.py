# __init__.py

import pathlib

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

path = pathlib.Path(__file__).parent / "config.toml"
with path.open(mode="rb") as fp:
    config = tomllib.load(fp)
