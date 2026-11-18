"""Unit tests for the TDD gate's pure logic.

These test plain data in, plain data out — no real git repo, no subprocess,
no filesystem. The git/subprocess plumbing in tdd_gate.py is exercised
instead by running the script against real branches (see the task's VERIFY
section), because faking git history well enough to unit-test it would be
less trustworthy than the real thing.
"""
from tdd_gate import (
    classify_subject,
    extract_test_files,
    find_impure_paths,
    validate_shape,
)


class TestClassifySubject:
    def test_recognises_plain_test(self):
        assert classify_subject("test: reproduce missing finish reasons") == "test"

    def test_recognises_scoped_test(self):
        assert classify_subject("test(ollama): reproduce missing finish reasons") == "test"

    def test_recognises_plain_fix(self):
        assert classify_subject("fix: emit gen_ai.response.finish_reasons") == "fix"

    def test_recognises_scoped_fix(self):
        assert classify_subject("fix(ollama): emit gen_ai.response.finish_reasons") == "fix"

    def test_recognises_plain_feat(self):
        assert classify_subject("feat: add litellm instrumentation") == "feat"

    def test_recognises_scoped_feat(self):
        assert classify_subject("feat(litellm): add package instrumentation") == "feat"

    def test_rejects_non_conforming_subject(self):
        assert classify_subject("docs: update the README") is None

    def test_rejects_bare_type_with_no_colon(self):
        assert classify_subject("test something without a colon") is None

    def test_rejects_empty_subject(self):
        assert classify_subject("") is None


class TestValidateShape:
    def test_valid_test_then_fix(self):
        assert validate_shape(["test", "fix"]) is None

    def test_valid_test_then_feat(self):
        assert validate_shape(["test", "feat"]) is None

    def test_valid_with_unrelated_commits_interleaved(self):
        assert validate_shape([None, "test", None, "fix", None]) is None

    def test_rejects_no_test_commit(self):
        error = validate_shape(["fix", "feat"])
        assert error is not None
        assert "test:" in error

    def test_rejects_test_with_no_following_fix(self):
        error = validate_shape(["test", None])
        assert error is not None
        assert "fix" in error

    def test_rejects_fix_before_test(self):
        """A fix: commit that precedes the test: commit does not satisfy the
        shape — the fix must come after, proving the test drove it."""
        error = validate_shape(["fix", "test"])
        assert error is not None

    def test_rejects_empty_range(self):
        assert validate_shape([]) is not None


class TestFindImpurePaths:
    def test_flags_instrumentation_source(self):
        paths = ["packages/opentelemetry-instrumentation-ollama/opentelemetry/instrumentation/ollama/span_utils.py"]
        assert find_impure_paths(paths) == paths

    def test_does_not_flag_tests_paths(self):
        paths = ["packages/opentelemetry-instrumentation-ollama/tests/test_chat.py"]
        assert find_impure_paths(paths) == []

    def test_mixed_list_flags_only_the_instrumentation_path(self):
        impure = "packages/foo/opentelemetry/instrumentation/foo/bar.py"
        pure = "packages/foo/tests/test_bar.py"
        assert find_impure_paths([impure, pure]) == [impure]

    def test_empty_list_is_pure(self):
        assert find_impure_paths([]) == []


class TestExtractTestFiles:
    def test_extracts_py_file_under_tests_dir(self):
        paths = ["packages/opentelemetry-instrumentation-ollama/tests/test_chat.py"]
        assert extract_test_files(paths) == paths

    def test_ignores_non_test_directory_py_files(self):
        paths = ["packages/opentelemetry-instrumentation-ollama/opentelemetry/instrumentation/ollama/span_utils.py"]
        assert extract_test_files(paths) == []

    def test_ignores_non_py_files_under_tests(self):
        paths = ["packages/opentelemetry-instrumentation-ollama/tests/cassettes/test_chat.yaml"]
        assert extract_test_files(paths) == []

    def test_mixed_list_extracts_only_the_test_file(self):
        test_file = "packages/foo/tests/test_bar.py"
        source_file = "packages/foo/opentelemetry/instrumentation/foo/bar.py"
        assert extract_test_files([test_file, source_file]) == [test_file]

    def test_empty_list_yields_no_test_files(self):
        assert extract_test_files([]) == []
