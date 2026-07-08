"""
Unit tests for module-level condition helpers.
"""

from traceloop.sdk.guardrail.conditions import (
    gt,
    lt,
    gte,
    lte,
    between,
    eq,
    is_true,
    is_false,
    is_truthy,
    is_falsy,
)


class TestIsTrue:
    def test_with_true(self):
        assert is_true()(True) is True

    def test_with_false(self):
        assert is_true()(False) is False

    def test_with_none(self):
        assert is_true()(None) is False

    def test_rejects_truthy_non_bool(self):
        condition = is_true()
        assert condition(1) is False
        assert condition("yes") is False
        assert condition([1, 2]) is False


class TestIsFalse:
    def test_with_false(self):
        assert is_false()(False) is True

    def test_with_true(self):
        assert is_false()(True) is False

    def test_with_none(self):
        assert is_false()(None) is False

    def test_rejects_falsy_non_bool(self):
        condition = is_false()
        assert condition(0) is False
        assert condition("") is False
        assert condition([]) is False


class TestIsTruthy:
    def test_with_truthy_values(self):
        condition = is_truthy()
        assert condition(True) is True
        assert condition(1) is True
        assert condition("yes") is True
        assert condition([1, 2]) is True

    def test_with_falsy_values(self):
        condition = is_truthy()
        assert condition(False) is False
        assert condition(0) is False
        assert condition("") is False
        assert condition(None) is False
        assert condition([]) is False


class TestIsFalsy:
    def test_with_falsy_values(self):
        condition = is_falsy()
        assert condition(False) is True
        assert condition(0) is True
        assert condition("") is True
        assert condition(None) is True
        assert condition([]) is True

    def test_with_truthy_values(self):
        condition = is_falsy()
        assert condition(True) is False
        assert condition(1) is False
        assert condition("yes") is False
        assert condition([1, 2]) is False


class TestGt:
    def test_greater(self):
        assert gt(0.5)(0.8) is True

    def test_equal(self):
        assert gt(0.5)(0.5) is False

    def test_less(self):
        assert gt(0.5)(0.3) is False

    def test_none(self):
        assert gt(0.5)(None) is False

    def test_integers(self):
        condition = gt(10)
        assert condition(15) is True
        assert condition(10) is False
        assert condition(5) is False


class TestLt:
    def test_less(self):
        assert lt(1000)(500) is True

    def test_equal(self):
        assert lt(1000)(1000) is False

    def test_greater(self):
        assert lt(1000)(1500) is False

    def test_none(self):
        assert lt(1000)(None) is False

    def test_floats(self):
        condition = lt(0.5)
        assert condition(0.3) is True
        assert condition(0.5) is False
        assert condition(0.7) is False


class TestGte:
    def test_greater(self):
        assert gte(0.8)(0.9) is True

    def test_equal(self):
        assert gte(0.8)(0.8) is True

    def test_less(self):
        assert gte(0.8)(0.7) is False

    def test_none(self):
        assert gte(0.8)(None) is False

    def test_integers(self):
        condition = gte(50)
        assert condition(60) is True
        assert condition(50) is True
        assert condition(40) is False


class TestLte:
    def test_less(self):
        assert lte(0.5)(0.3) is True

    def test_equal(self):
        assert lte(0.5)(0.5) is True

    def test_greater(self):
        assert lte(0.5)(0.7) is False

    def test_none(self):
        assert lte(0.5)(None) is False

    def test_integers(self):
        condition = lte(100)
        assert condition(80) is True
        assert condition(100) is True
        assert condition(120) is False


class TestBetween:
    def test_in_range(self):
        assert between(50, 200)(100) is True

    def test_at_min(self):
        assert between(50, 200)(50) is True

    def test_at_max(self):
        assert between(50, 200)(200) is True

    def test_below(self):
        assert between(50, 200)(30) is False

    def test_above(self):
        assert between(50, 200)(250) is False

    def test_none(self):
        assert between(50, 200)(None) is False

    def test_floats(self):
        condition = between(0.3, 0.8)
        assert condition(0.5) is True
        assert condition(0.3) is True
        assert condition(0.8) is True
        assert condition(0.2) is False
        assert condition(0.9) is False


class TestEq:
    def test_matching_string(self):
        assert eq("approved")("approved") is True

    def test_different_string(self):
        assert eq("approved")("rejected") is False

    def test_none(self):
        assert eq("approved")(None) is False

    def test_various_types(self):
        assert eq(42)(42) is True
        assert eq(42)(43) is False
        assert eq(0.8)(0.8) is True
        assert eq(0.8)(0.7) is False
        assert eq(True)(True) is True
        assert eq(True)(False) is False
