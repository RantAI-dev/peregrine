"""ik_llama.cpp engine benchmark adapter."""

from pathlib import Path
from typing import Optional

from .llama_cpp import LlamaCppAdapter


class IkLlamaCppAdapter(LlamaCppAdapter):
    """Adapter for pinned ik_llama.cpp fork (named in docs/targets.md Section 2 Claim 2)."""

    DEFAULT_PINNED_TAG = "master"
    DEFAULT_PINNED_COMMIT = "pinned"
    GIT_REPO_URL = "https://github.com/ikawrakow/ik_llama.cpp.git"

    def __init__(
        self,
        pinned_tag: str = DEFAULT_PINNED_TAG,
        pinned_commit: str = DEFAULT_PINNED_COMMIT,
        custom_bin_dir: Optional[Path] = None,
    ):
        super().__init__(
            pinned_tag=pinned_tag,
            pinned_commit=pinned_commit,
            custom_bin_dir=custom_bin_dir,
        )
        self.name = "ik_llama.cpp"
        self.pinned_version = f"{pinned_tag} ({pinned_commit})"
