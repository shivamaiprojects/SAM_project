"""Verify Hugging Face authentication and access to the gated SAM 3 checkpoints."""

from huggingface_hub import HfApi, whoami
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from sam3_seg.utils.env import load_env

REPOS = ("facebook/sam3", "facebook/sam3.1")
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".safetensors")


def main() -> None:
    load_env(required=["HF_TOKEN"])

    user = whoami()
    print(f"Authenticated as: {user['name']}")

    api = HfApi()
    for repo in REPOS:
        try:
            files = api.list_repo_files(repo)
        except GatedRepoError:
            print(f"\n{repo}: GATED — access not granted for this account")
            continue
        except RepositoryNotFoundError:
            print(f"\n{repo}: NOT FOUND or not visible to this token")
            continue

        checkpoints = [f for f in files if f.endswith(CHECKPOINT_SUFFIXES)]
        print(f"\n{repo}: ACCESS OK ({len(files)} files)")
        for name in checkpoints:
            print(f"    {name}")


if __name__ == "__main__":
    main()