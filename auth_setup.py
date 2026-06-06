"""
Authenticate and initialize the Earth Engine connection.

Usage
-----
1. Put your project ID in config.py (GEE_PROJECT).
2. Run once to authenticate (opens a browser / pastes a token):
       python auth_setup.py
3. Thereafter, every script just calls `init_ee()` from this module.
"""

import ee

from config import GEE_PROJECT


def init_ee(project: str | None = None):
    """Initialize EE, authenticating the first time if needed."""
    project = project or GEE_PROJECT
    if project == "REPLACE_WITH_YOUR_PROJECT_ID":
        raise SystemExit(
            "Set GEE_PROJECT in config.py to your Earth Engine Cloud project ID "
            "before running."
        )
    try:
        ee.Initialize(project=project)
    except Exception:
        # Not authenticated yet -> trigger the interactive auth flow.
        ee.Authenticate()
        ee.Initialize(project=project)
    return project


if __name__ == "__main__":
    proj = init_ee()
    # Smoke test: round-trip a trivial computation through the server.
    info = ee.Number(1).add(1).getInfo()
    print(f"Earth Engine initialized on project '{proj}'. Sanity check 1+1 = {info}")
