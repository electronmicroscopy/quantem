# Developer Instructions

We use [uv](https://docs.astral.sh/uv/) to manage the package.

Getting started:

- [install uv](https://docs.astral.sh/uv/getting-started/installation/)
- `git clone` the repo and `cd` into the directory
- run `uv sync` to install all the dependencies in an editable environment
- run `uv sync --all-packages` to also install `quantem.widget` (optional)

The following will set up the pre-commit and [ruff](https://github.com/astral-sh/ruff) for linting and formatting. These commands only need to be run once when first setting up your dev environment: 

- `uv tool install pre-commit` 
- `uv tool install ruff`
- `pre-commit install`

Once these have been installed, the `.pre-commit-config.yaml` file will be run when trying to `git commit`. Errors that cannot be auto-fixed  will be listed and you will have to resolve them before committing. In many cases you will get a warning that the formatting and auto-fixes have been applied; you can stage the changes for commit with `git add -u` and the pre-commit should allow you to commit your changes.

Dependency management:

- use `uv add package_name` to add dependencies
- use `uv remove package_name` to remove dependencies
- use `uv add dev_package_name --dev` to add a dev dependency, i.e. that devs need (e.g. pytest) but you don't want shipped to users
- use `uv pip install testing_package_name` to install a package you think you might need, but don't want to add to dependencies just yet

Running python/scripts in environment:

- use `uv run python`, `uv run jupyterlab` etc. to automatically activate the environment and run your command
- alternatively use `source .venv/bin/activate` to explicitly activate environment and use `python`, `jupyterlab` etc. as usual
  - note that if you're using an IDE like VS Code, it probably activates the environment automatically
  
