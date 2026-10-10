#!/usr/bin/env python3
"""Mechanically enforces test-driven bug fixing on a commit range.

"Write a failing test first" is unverifiable as a prompt instruction — an
agent can claim compliance and nothing checks. This gate makes it structural
by re-deriving the fact that matters: the new test actually failed before the
fix existed. Four checks, all required:

  SHAPE   a `test:`/`test(scope):` commit is followed by a `fix:`/`feat:`
          commit (optionally scoped).
  PURITY  no `test:` commit touches instrumentation source
          (packages/*/opentelemetry/**) — a fix smuggled into the test
          commit would defeat the RED check below.
  RED     each `test:` commit, checked out in isolation via `git worktree`,
          must FAIL when its new/changed test files run there.
  GREEN   at head, the same test files must pass.

Pure logic (subject classification, test-file extraction, purity checking)
is unit-tested directly in test_tdd_gate.py. Everything past the plumbing
divider does real git/subprocess work and is instead exercised by running
this script against real branches (see the task's VERIFY section).
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

SUBJECT_RE = re.compile(r"^(test|fix|feat)(\([^)]*\))?:\s*\S")
IMPURE_GLOB = "packages/*/opentelemetry/*"
PYTEST_STARTED_MARKER = "test session starts"


# --------------------------------------------------------------------------
# Pure logic — no subprocess, no filesystem. Unit-tested directly.
# --------------------------------------------------------------------------


def classify_subject(subject: str) -> str | None:
    """Return 'test', 'fix', or 'feat' for a commit subject, else None.

    Accepts an optional `(scope)` between the type and the colon, e.g.
    `test(ollama): reproduce ...`.
    """
    match = SUBJECT_RE.match(subject.strip())
    return match.group(1) if match else None


def validate_shape(kinds: list[str | None]) -> str | None:
    """Require a 'test' classification followed later by a 'fix'/'feat' one.

    Returns an error message, or None if the shape is valid. Unrelated
    commits may be interleaved; only relative order of test vs. fix/feat
    matters.
    """
    test_idxs = [i for i, k in enumerate(kinds) if k == "test"]
    if not test_idxs:
        return (
            "no commit subject starts with `test:` or `test(scope):` — the "
            "required shape is a test commit followed by a fix/feat commit"
        )
    fix_idxs = [i for i, k in enumerate(kinds) if k in ("fix", "feat")]
    if not fix_idxs:
        return "a test: commit exists, but no fix:/feat: commit follows it"
    if not any(f > t for t in test_idxs for f in fix_idxs):
        return (
            "a test: commit and a fix:/feat: commit both exist, but no "
            "fix:/feat: commit comes after any test: commit"
        )
    return None


def find_impure_paths(paths: list[str]) -> list[str]:
    """Paths that touch instrumentation source (packages/*/opentelemetry/**);
    a test: commit must have none."""
    return [p for p in paths if fnmatch.fnmatch(p, IMPURE_GLOB)]


def extract_test_files(paths: list[str]) -> list[str]:
    """Paths that are `.py` files under a `tests/` directory."""
    return [p for p in paths if p.endswith(".py") and "tests" in Path(p).parts[:-1]]


def package_for_path(path: str) -> str | None:
    """Extract `<pkg>` from `packages/<pkg>/...`, or None."""
    parts = Path(path).parts
    return parts[1] if len(parts) >= 2 and parts[0] == "packages" else None


def group_by_package(paths: list[str]) -> dict[str, list[str]]:
    """Group test-file paths by package, each made relative to its package
    directory (e.g. packages/foo/tests/x.py -> {"foo": ["tests/x.py"]})."""
    groups: dict[str, list[str]] = {}
    for p in paths:
        pkg = package_for_path(p)
        if pkg is None:
            continue
        rel = str(Path(*Path(p).parts[2:]))
        groups.setdefault(pkg, []).append(rel)
    return groups


def pytest_actually_ran(output: str) -> bool:
    """False if `uv run pytest` errored before pytest itself started (e.g. an
    unresolved environment) — such a run must NOT be treated as a pass."""
    return PYTEST_STARTED_MARKER in output


# --------------------------------------------------------------------------
# Plumbing: git, subprocess, worktrees.
# --------------------------------------------------------------------------


@dataclass
class Commit:
    sha: str
    subject: str
    kind: str | None = None
    changed_paths: list[str] = field(default_factory=list)


@dataclass
class PytestResult:
    ran: bool
    returncode: int
    output: str


def run_git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{proc.stderr}")
    return proc.stdout


def get_commits(base: str, head: str, cwd: Path) -> list[Commit]:
    out = run_git(["log", "--reverse", "--format=%H%x1f%s", f"{base}..{head}"], cwd=cwd)
    commits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        sha, subject = line.split("\x1f", 1)
        commits.append(Commit(sha=sha, subject=subject, kind=classify_subject(subject)))
    return commits


def get_changed_paths(sha: str, cwd: Path) -> list[str]:
    out = run_git(["show", "--name-only", "--format=", sha], cwd=cwd)
    return [line for line in out.splitlines() if line.strip()]


def run_pytest(paths: list[str], cwd: Path) -> PytestResult:
    # --all-groups mirrors every package's `install` nx target (`uv sync
    # --all-groups`) — the "test" dependency group (VCR, provider SDKs, etc.)
    # is not part of uv's default group and would otherwise be missing.
    proc = subprocess.run(
        ["uv", "run", "--all-groups", "pytest", "-v", *paths],
        cwd=cwd, capture_output=True, text=True,
    )
    output = proc.stdout + proc.stderr
    return PytestResult(ran=pytest_actually_ran(output), returncode=proc.returncode, output=output)


class Worktree:
    """A temporary `git worktree` at a given sha, always cleaned up (even on
    failure) — this is what makes the RED check isolated."""

    def __init__(self, repo_root: Path, sha: str):
        self.repo_root = repo_root
        self.sha = sha
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="tdd-gate-wt-"))
        shutil.rmtree(self.path)  # git worktree add needs to create it itself
        run_git(["worktree", "add", "--detach", str(self.path), self.sha], cwd=self.repo_root)
        return self.path

    def __exit__(self, *exc_info) -> None:
        if self.path is None:
            return
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(self.path)],
            cwd=self.repo_root, capture_output=True,
        )
        shutil.rmtree(self.path, ignore_errors=True)
        subprocess.run(["git", "worktree", "prune"], cwd=self.repo_root, capture_output=True)


# --------------------------------------------------------------------------
# The gate itself.
# --------------------------------------------------------------------------


def indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def run_pytest_per_package(test_paths: list[str], cwd: Path) -> list[tuple[str, PytestResult]]:
    return [
        (pkg, run_pytest(rel_paths, cwd / "packages" / pkg))
        for pkg, rel_paths in sorted(group_by_package(test_paths).items())
    ]


def check_results(results: list[tuple[str, PytestResult]], phase: str, expect_fail: bool) -> str | None:
    """Print each package's captured pytest output and return an error
    message (or None) per the RED/GREEN contract for that phase."""
    for pkg, result in results:
        print(f"  --- uv run pytest output ({pkg}) ---")
        print(indent(result.output))
        if not result.ran:
            return (
                f"{phase}: FAIL — pytest never started for {pkg} (uv could not "
                "resolve the environment); an unrunnable check does not count as a pass"
            )
        if expect_fail and result.returncode == 0:
            return (
                f"{phase}: FAIL — the test PASSED before the fix exists ({pkg}); "
                "it never failed, so it proves nothing"
            )
        if not expect_fail and result.returncode != 0:
            return f"{phase}: FAIL — the test still fails at head ({pkg})"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()

    repo_root = Path(run_git(["rev-parse", "--show-toplevel"], cwd=Path.cwd()).strip())
    print(f"TDD gate: {args.base}..{args.head}\n")

    commits = get_commits(args.base, args.head, repo_root)
    if not commits:
        print(f"No commits found in range {args.base}..{args.head}.")
        return 1

    print("Commits in range:")
    for c in commits:
        print(f"  {c.sha[:10]}  [{c.kind or '?'}]  {c.subject}")
    print()

    shape_error = validate_shape([c.kind for c in commits])
    if shape_error:
        print(f"SHAPE: FAIL — {shape_error}")
        return 1
    print("SHAPE: ok — a test: commit is followed by a fix:/feat: commit")

    test_commits = [c for c in commits if c.kind == "test"]
    for c in test_commits:
        c.changed_paths = get_changed_paths(c.sha, repo_root)

    impure = [(c, find_impure_paths(c.changed_paths)) for c in test_commits]
    impure = [(c, offenders) for c, offenders in impure if offenders]
    if impure:
        print("PURITY: FAIL")
        for c, offenders in impure:
            print(f"  {c.sha[:10]} ({c.subject}) touches instrumentation source:")
            for path in offenders:
                print(f"    {path}")
        return 1
    print("PURITY: ok — no test: commit touches packages/*/opentelemetry/**")

    all_test_files: list[str] = []
    for c in test_commits:
        files = extract_test_files(c.changed_paths)
        if not files:
            print(
                f"RED: FAIL — {c.sha[:10]} ({c.subject}) adds/modifies no *.py "
                "file under a tests/ directory, so there is nothing to prove failed"
            )
            return 1
        all_test_files.extend(files)

        print(f"\nRED check for {c.sha[:10]} ({c.subject}):")
        print(f"  test files: {files}")
        with Worktree(repo_root, c.sha) as wt_path:
            error = check_results(run_pytest_per_package(files, wt_path), "RED", expect_fail=True)
            if error:
                print(error)
                return 1
    print("RED: ok — every new test failed in isolation before its fix")

    print(f"\nGREEN check at {args.head}:")
    with Worktree(repo_root, args.head) as wt_path:
        error = check_results(run_pytest_per_package(all_test_files, wt_path), "GREEN", expect_fail=False)
        if error:
            print(error)
            return 1
    print("GREEN: ok — the tests pass at head")

    print("\nTDD gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
