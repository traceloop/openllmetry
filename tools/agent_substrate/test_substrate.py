"""Guards the agent substrate's portability invariants.

The substrate is only vendor-neutral if nothing drifts back into a vendor-specific
file. These assertions are the enforcement; the prose in AGENTS.md is not.
"""

import glob
import os
import subprocess
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
)

PROCEDURES = REPO_ROOT / "docs" / "ai" / "procedures"
KNOWLEDGE = REPO_ROOT / "docs" / "ai" / "knowledge"
CLAUDE_DIR = REPO_ROOT / ".claude"
SKILLS = CLAUDE_DIR / "skills"

MAX_WRAPPER_LINES = 15


class TestCanonicalInstructions:
    def test_agents_md_is_the_canonical_regular_file(self):
        path = REPO_ROOT / "AGENTS.md"
        assert path.is_file(), "AGENTS.md must exist at the repo root"
        assert not path.is_symlink(), "AGENTS.md is canonical; it must not be a symlink"

    def test_claude_md_is_a_symlink_to_agents_md(self):
        """CLAUDE.md must be a pointer, never a second copy that can drift."""
        path = REPO_ROOT / "CLAUDE.md"
        assert path.is_symlink(), (
            "CLAUDE.md must be a symlink to AGENTS.md, not a regular file — "
            "two copies drift apart and the knowledge becomes vendor-specific"
        )
        assert os.readlink(path) == "AGENTS.md"

    def test_git_records_claude_md_as_a_symlink(self):
        """A symlink on disk is not enough; git must have stored mode 120000."""
        out = subprocess.check_output(
            ["git", "ls-files", "-s", "CLAUDE.md"], cwd=REPO_ROOT, text=True
        )
        assert out.split()[0] == "120000", f"expected git mode 120000, got: {out!r}"


class TestNoKnowledgeInVendorDirs:
    def test_claude_dir_contains_only_pointers(self):
        """Every file under .claude/ must be an allowlisted pointer, never knowledge.

        This walks the whole `.claude/` tree, not just `.claude/skills/` — a future
        `.claude/commands/*.md` or `.claude/agents/*.md` with real procedural content
        must be caught here too. Rather than only rejecting oversized `.md` files
        (which lets a short note, or any non-markdown file, carry real knowledge and
        pass), this is an allowlist: the only files permitted anywhere under
        `.claude/` are `SKILL.md` wrappers directly under `.claude/skills/<name>/`,
        and `.claude/settings.local.json`. Wrappers still must not exceed
        MAX_WRAPPER_LINES.
        """
        offenders = []
        for path in CLAUDE_DIR.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(REPO_ROOT)
            is_settings_local = path == CLAUDE_DIR / "settings.local.json"
            is_skill_wrapper = path.name == "SKILL.md" and path.parent.parent == SKILLS
            if is_settings_local:
                continue
            if is_skill_wrapper:
                lines = path.read_text().splitlines()
                if len(lines) > MAX_WRAPPER_LINES:
                    offenders.append(f"{rel} ({len(lines)} lines) exceeds a pointer's size")
                continue
            offenders.append(f"{rel} is not an allowlisted pointer file")
        assert not offenders, (
            "these paths under .claude/ hold content that belongs in docs/ai/ instead: "
            f"{offenders}"
        )

    def test_every_skill_wrapper_points_at_a_real_procedure(self):
        wrappers = sorted(SKILLS.glob("*/SKILL.md"))
        assert wrappers, "expected skill wrappers under .claude/skills/*/SKILL.md"
        for wrapper in wrappers:
            text = wrapper.read_text()
            assert "docs/ai/procedures/" in text, (
                f"{wrapper.relative_to(REPO_ROOT)} does not reference a procedure file"
            )
            referenced = [
                tok.strip("`.,)")
                for tok in text.split()
                if tok.strip("`.,)").startswith("docs/ai/procedures/")
            ]
            for ref in referenced:
                assert (REPO_ROOT / ref).is_file(), (
                    f"{wrapper.relative_to(REPO_ROOT)} points at {ref}, which does not exist"
                )


class TestProcedures:
    @pytest.mark.parametrize(
        "name",
        [
            "fix-instrumentation-bug",
            "add-instrumentation",
            "record-cassette",
            "semconv-conformance",
        ],
    )
    def test_procedure_exists(self, name):
        assert (PROCEDURES / f"{name}.md").is_file()

    def test_agents_md_links_every_procedure(self):
        """A procedure nobody can find from the entry point is not discoverable."""
        agents = (REPO_ROOT / "AGENTS.md").read_text()
        for path in PROCEDURES.glob("*.md"):
            if path.name == "README.md":
                continue
            assert path.name in agents, (
                f"{path.name} is not linked from AGENTS.md, so an agent starting at the "
                "canonical entry point will never find it"
            )


class TestOkfBundle:
    def _concepts(self):
        return [
            Path(p)
            for p in sorted(glob.glob(str(KNOWLEDGE / "*.md")))
            if Path(p).name not in ("index.md", "log.md")
        ]

    def test_bundle_root_declares_okf_version(self):
        text = (KNOWLEDGE / "index.md").read_text()
        assert 'okf_version: "0.2"' in text

    def test_only_the_root_declares_okf_version(self):
        """OKF permits okf_version in the bundle root's frontmatter only."""
        for path in KNOWLEDGE.glob("*.md"):
            if path.name == "index.md":
                continue
            assert "okf_version:" not in path.read_text(), (
                f"{path.name} declares okf_version; only index.md may"
            )

    def test_log_exists(self):
        assert (KNOWLEDGE / "log.md").is_file()

    def test_bundle_has_concepts(self):
        assert self._concepts(), (
            "the bundle has no concepts — an empty knowledge base teaches nothing"
        )

    def test_every_file_has_parseable_frontmatter_with_a_type(self):
        for path in KNOWLEDGE.glob("*.md"):
            text = path.read_text()
            assert text.startswith("---\n"), f"{path.name} has no frontmatter block"
            frontmatter = text.split("---", 2)[1]
            assert "type:" in frontmatter, f"{path.name} frontmatter has no type field"
            type_line = next(
                line for line in frontmatter.splitlines() if line.startswith("type:")
            )
            assert type_line.split(":", 1)[1].strip(), f"{path.name} has an empty type"

    def test_every_concept_is_listed_in_the_index(self):
        index = (KNOWLEDGE / "index.md").read_text()
        for path in self._concepts():
            assert path.name in index, (
                f"{path.name} is not listed in index.md, so retrieval by an agent "
                "reading the index will miss it"
            )

    def test_concepts_carry_provenance(self):
        """OKF trust tiers depend on `generated`; a concept without it is anonymous."""
        for path in self._concepts():
            assert "generated:" in path.read_text(), (
                f"{path.name} has no `generated:` block, so its trust tier is unknowable"
            )

    def test_concepts_carry_an_expiry(self):
        """Empirical facts rot. A concept with no `stale_after` is trusted forever,
        which is exactly the failure mode this bundle exists to avoid."""
        for path in self._concepts():
            text = path.read_text()
            frontmatter = text.split("---", 2)[1]
            stale_line = next(
                (line for line in frontmatter.splitlines() if line.startswith("stale_after:")),
                None,
            )
            assert stale_line is not None, (
                f"{path.name} has no `stale_after:` date, so it would be trusted forever "
                "even as the fact it records rots"
            )
            value = stale_line.split(":", 1)[1].strip()
            assert value, (
                f"{path.name} declares `stale_after:` with an empty value, so it would be "
                "trusted forever even as the fact it records rots"
            )
            try:
                date.fromisoformat(value)
            except ValueError:
                pytest.fail(
                    f"{path.name} declares `stale_after: {value}`, which is not a valid "
                    "ISO-8601 date, so the expiry cannot be relied on"
                )
