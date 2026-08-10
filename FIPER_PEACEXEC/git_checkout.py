"""Create isolated detached Git worktrees for repository evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterator


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    command = [
        "git",
        "-c",
        f"safe.directory={repository}",
        "-C",
        str(repository),
        *arguments,
    ]
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


@contextmanager
def detached_worktree(
    repository_path: str | Path,
    commit: str,
    temporary_root: str | Path | None = None,
) -> Iterator[Path]:
    """Yield an isolated worktree at ``commit`` without modifying the source clone."""

    repository = Path(repository_path).expanduser().resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"Repository does not exist: {repository}")
    top_level = Path(_git(repository, "rev-parse", "--show-toplevel").stdout.strip())
    if top_level.resolve() != repository:
        raise ValueError(f"repo_path must be the Git repository root: {repository}")
    _git(repository, "rev-parse", "--verify", f"{commit}^{{commit}}")

    parent = Path(temporary_root).resolve() if temporary_root else None
    if parent:
        parent.mkdir(parents=True, exist_ok=True)
    session_root = Path(
        tempfile.mkdtemp(prefix="fiper-peacexec-", dir=str(parent) if parent else None)
    ).resolve()
    worktree = session_root / "worktree"
    try:
        _git(repository, "worktree", "add", "--detach", str(worktree), commit)
        yield worktree
    finally:
        if worktree.exists():
            try:
                _git(repository, "worktree", "remove", "--force", str(worktree))
            except subprocess.CalledProcessError:
                pass
        if session_root.exists():
            shutil.rmtree(session_root)
