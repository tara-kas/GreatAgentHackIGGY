# GreatAgentHackIGGY
GitHub Repo for Great Agent Hack

## Setup with UV

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

### Installation

If you don't have `uv` installed, install it first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Project Initialization

1. **Create virtual environment and sync dependencies:**
   ```bash
   uv sync
   ```
   This will:
   - Create a `.venv` virtual environment
   - Install all dependencies from `pyproject.toml`
   - Generate a `uv.lock` file

2. **Activate the virtual environment:**
   ```bash
   # macOS/Linux
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

3. **Run scripts:**
   ```bash
   # Using uv run (recommended - automatically uses the venv)
   uv run TrackC/quicktest.py
   
   # Or activate venv first, then run normally
   python TrackC/quicktest.py
   ```

### Managing Dependencies

- **Add a new dependency:**
  ```bash
  uv add package-name
  ```
  This automatically updates `pyproject.toml` and installs the package.

- **Add a development dependency:**
  ```bash
  uv add --dev package-name
  ```

- **Remove a dependency:**
  ```bash
  uv remove package-name
  ```

- **Update lock file:**
  ```bash
  uv lock
  ```

- **Sync dependencies (after pulling changes):**
  ```bash
  uv sync
  ```

- **List installed packages:**
  ```bash
  uv pip list
  ```

### Quick Start

```bash
# 1. Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync dependencies
uv sync

# 3. Run the test script
uv run TrackC/quicktest.py
```
