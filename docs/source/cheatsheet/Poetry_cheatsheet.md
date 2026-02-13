# Poetry cheatsheet
| Poetry                                                           | Conda/PIP                           | Description                                                                                                                                                             |
| ------------------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `poetry init`: does not install python                           | `conda create -n [env_name] python` | Initialise the package environment                                                                                                                                      |
| `poetry env use [python_path]`                                   | -                                   | Select which version of the python to use.<br /> <br />Activate or create a new virtualenv for the current project.                                                     |
| `poetry shell`                                                   | `conda activate [env_name]`         | Activate the virtual environment                                                                                                                                        |
| `exit`                                                           | `conda deactivate`                  | Deactivate the virtual environment                                                                                                                                      |
| `poetry add [package_name] [-vvv]`                               | `pip install [package_name]`        | Install the package, where`-vvv` flag is to show the full logs of the installation.                                                                                     |
| `poetry add -G dev [package_name]`                               | -                                   | Install the package only for developing                                                                                                                                 |
| `poetry remove [pakage_name]`                                    | `pip uninstall [package_name]`      | Uninstall the package                                                                                                                                                   |
| `poetry remove -G dev [pakage_name]`                             | `pip uninstall [package_name]`      | Uninstall the package only for developing                                                                                                                               |
| (file)`poetry.lock`                                              | (file)`requirement.txt`             | The requirement of the package list                                                                                                                                     |
| (file)`pyproject.toml`                                           | (file)`requirement.txt`             | Python project configuration                                                                                                                                            |
| `poetry lock`                                                    | -                                   | Update the configuration file`poetry.lock`, based on `pyproject.toml`                                                                                                   |
| `poetry install`                                                 | -                                   | Install the project dependencies                                                                                                                                        |
| `poetry update`                                                  | -                                   | List the packages available to update                                                                                                                                   |
| `poetry show [--tree]`                                           | `conda list`                        | Show the python packages (and dependencies).                                                                                                                            |
| `poetry run jupyter notebook`                                    | -                                   | Run jupyter notebook server at background. Using server-kernel in VScode's jupyter notebook to get the python kernel. Use`Control+C` to leave and shut down the kernel. |
| `poetry export -f requirement.txt -o requirement.txt --with dev` | `pip freeze > requirement.txt`      | Export the lock file to alternative formats.                                                                                                                            |
| `poetry build`                                                   | -                                   | Build the packages needed.                                                                                                                                              |
| `poetry publish`                                                 | -                                   | Publish the packages to PyPI so that people can simply install pacakge by`pip install [your_package_name]`.                                                             |
|                                                                  |                                     |                                                                                                                                                                         |



## Trouble shooting

1. If the `poetry add [package-name]` stucks in `'SecretService Keyring'`, set variable `$PYTHON_KEYRING_BACKEND` in your linux environment.
    ```bash
    export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
    ```
    Solved from [Regression: Poetry 1.7 hangs instead of asking to unlock keyring · Issue #8623 · python-poetry/poetry (github.com)](https://github.com/python-poetry/poetry/issues/8623)
