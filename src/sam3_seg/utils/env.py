"""Load secrets from the project-root .env into the process environment.

Call load_env() at the top of any script that needs credentials.
Uses exactly the variable names the libraries themselves look for
(KAGGLE_USERNAME, KAGGLE_KEY, HF_TOKEN), so no further plumbing is needed.
"""

import os

from dotenv import load_dotenv

from sam3_seg.utils.config import project_root


def load_env(required: list[str] | None = None) -> None:
    load_dotenv(project_root() / ".env")
    missing = [k for k in (required or []) if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required env vars: {missing}. "
            f"Add them to {project_root() / '.env'}"
        )