"""
Unit tests for guardrail conditions.

Tests all built-in condition methods from the Condition class.
"""
from dataclasses import dataclass
from traceloop.sdk.guardrail.condition import Condition


@dataclass
class MockResult:
    """Mock object with attributes for condition testing."""
    success: bool = False
    score: float = 0.0
    matched: bool = False
    contains_pii: bool = False
    status: str = ""
    count: int = 0
    confidence: float = 0.0
    toxicity: float = 0.0
    latency_ms: float = 0.0


class TestConditionSuccess:
    """Tests for Condition.success()."""

    def test_success_with_dict_true(self):
        """Test success condition with dict where success is True."""
        condition = Condition.success()
        result = {"success": True}
        assert condition(result) is True

    def test_success_with_dict_false(self):
        """Test success condition with dict where success is False."""
        condition = Condition.success()
        result = {"success": False}
        assert condition(result) is False

    def test_success_with_object_true(self):
        """Test success condition with object where success is True."""
        condition = Condition.success()
        result = MockResult(success=True)
        assert condition(result) is True

    def test_success_with_object_false(self):
        """Test success condition with object where success is False."""
        condition = Condition.success()
        result = MockResult(success=False)
        assert condition(result) is False

    def test_success_with_missing_field(self):
        """Test success condition when success field is missing."""
        condition = Condition.success()
        result = {"score": 0.8}
        assert condition(result) is False


class TestConditionIsTrue:
    """Tests for Condition.is_true()."""

    def test_is_true_with_dict(self):
        """Test is_true condition with dict."""
        condition = Condition.is_true("matched")
        result = {"matched": True}
        assert condition(result) is True

    def test_is_true_with_object(self):
        """Test is_true condition with object."""
        condition = Condition.is_true("matched")
        result = MockResult(matched=True)
        assert condition(result) is True

    def test_is_true_with_false_value(self):
        """Test is_true condition when field is False."""
        condition = Condition.is_true("matched")
        result = {"matched": False}
        assert condition(result) is False

    def test_is_true_with_none_value(self):
        """Test is_true condition when field is None."""
        condition = Condition.is_true("matched")
        result = {"matched": None}
        assert condition(result) is False

    def test_is_true_with_missing_field(self):
        """Test is_true condition when field is missing."""
        condition = Condition.is_true("matched")
        result = {"score": 0.8}
        assert condition(result) is False


class TestConditionIsFalse:
    """Tests for Condition.is_false()."""

    def test_is_false_with_dict(self):
        """Test is_false condition with dict."""
        condition = Condition.is_false("contains_pii")
        result = {"contains_pii": False}
        assert condition(result) is True

    def test_is_false_with_object(self):
        """Test is_false condition with object."""
        condition = Condition.is_false("contains_pii")
        result = MockResult(contains_pii=False)
        assert condition(result) is True

    def test_is_false_with_true_value(self):
        """Test is_false condition when field is True."""
        condition = Condition.is_false("contains_pii")
        result = {"contains_pii": True}
        assert condition(result) is False

    def test_is_false_with_none_value(self):
        """Test is_false condition when field is None."""
        condition = Condition.is_false("contains_pii")
        result = {"contains_pii": None}
        assert condition(result) is False

    def test_is_false_with_missing_field(self):
        """Test is_false condition when field is missing."""
        condition = Condition.is_false("contains_pii")
        result = {"score": 0.8}
        assert condition(result) is False


class TestConditionEquals:
    """Tests for Condition.equals()."""

    def test_equals_with_matching_value(self):
        """Test equals condition with matching value."""
        condition = Condition.equals("approved", field="status")
        result = {"status": "approved"}
        assert condition(result) is True

    def test_equals_with_different_value(self):
        """Test equals condition with different value."""
        condition = Condition.equals("approved", field="status")
        result = {"status": "rejected"}
        assert condition(result) is False

    def test_equals_with_missing_field(self):
        """Test equals condition when field is missing."""
        condition = Condition.equals("approved", field="status")
        result = {"score": 0.8}
        assert condition(result) is False

    def test_equals_with_various_types(self):
        """Test equals condition with various data types."""
        # Integer
        condition = Condition.equals(42, field="count")
        assert condition({"count": 42}) is True
        assert condition({"count": 43}) is False

        # Float
        condition = Condition.equals(0.8, field="score")
        assert condition({"score": 0.8}) is True
        assert condition({"score": 0.7}) is False

        # Boolean
        condition = Condition.equals(True, field="active")
        assert condition({"active": True}) is True
        assert condition({"active": False}) is False


class TestConditionGreaterThan:
    """Tests for Condition.greater_than()."""

    def test_greater_than_with_larger_value(self):
        """Test greater_than when value is larger."""
        condition = Condition.greater_than(0.5, field="score")
        result = {"score": 0.8}
        assert condition(result) is True

    def test_greater_than_with_equal_value(self):
        """Test greater_than when value is equal."""
        condition = Condition.greater_than(0.5, field="score")
        result = {"score": 0.5}
        assert condition(result) is False

    def test_greater_than_with_smaller_value(self):
        """Test greater_than when value is smaller."""
        condition = Condition.greater_than(0.5, field="score")
        result = {"score": 0.3}
        assert condition(result) is False

    def test_greater_than_with_default_score_field(self):
        """Test greater_than with default 'score' field."""
        condition = Condition.greater_than(10)
        result = {"score": 15}
        assert condition(result) is True
        result = {"score": 5}
        assert condition(result) is False

    def test_greater_than_with_custom_field(self):
        """Test greater_than with custom field name."""
        condition = Condition.greater_than(10, field="count")
        result = {"count": 20}
        assert condition(result) is True

    def test_greater_than_with_missing_field(self):
        """Test greater_than when field is missing (defaults to 0)."""
        condition = Condition.greater_than(5, field="missing")
        result = {"score": 0.8}
        # Missing field defaults to 0, which is not > 5
        assert condition(result) is False


class TestConditionLessThan:
    """Tests for Condition.less_than()."""

    def test_less_than_with_smaller_value(self):
        """Test less_than when value is smaller."""
        condition = Condition.less_than(1000, field="latency_ms")
        result = {"latency_ms": 500}
        assert condition(result) is True

    def test_less_than_with_equal_value(self):
        """Test less_than when value is equal."""
        condition = Condition.less_than(1000, field="latency_ms")
        result = {"latency_ms": 1000}
        assert condition(result) is False

    def test_less_than_with_larger_value(self):
        """Test less_than when value is larger."""
        condition = Condition.less_than(1000, field="latency_ms")
        result = {"latency_ms": 1500}
        assert condition(result) is False

    def test_less_than_with_default_score_field(self):
        """Test less_than with default 'score' field."""
        condition = Condition.less_than(10)
        result = {"score": 5}
        assert condition(result) is True
        result = {"score": 15}
        assert condition(result) is False

    def test_less_than_with_custom_field(self):
        """Test less_than with custom field name."""
        condition = Condition.less_than(100, field="count")
        result = {"count": 50}
        assert condition(result) is True

    def test_less_than_with_missing_field(self):
        """Test less_than when field is missing (defaults to inf)."""
        condition = Condition.less_than(1000, field="missing")
        result = {"score": 0.8}
        # Missing field defaults to inf, which is not < 1000
        assert condition(result) is False


class TestConditionGreaterThanOrEqual:
    """Tests for Condition.greater_than_or_equal()."""

    def test_gte_with_larger_value(self):
        """Test gte when value is larger."""
        condition = Condition.greater_than_or_equal(0.8, field="confidence")
        result = {"confidence": 0.9}
        assert condition(result) is True

    def test_gte_with_equal_value(self):
        """Test gte when value is equal."""
        condition = Condition.greater_than_or_equal(0.8, field="confidence")
        result = {"confidence": 0.8}
        assert condition(result) is True

    def test_gte_with_smaller_value(self):
        """Test gte when value is smaller."""
        condition = Condition.greater_than_or_equal(0.8, field="confidence")
        result = {"confidence": 0.7}
        assert condition(result) is False

    def test_gte_with_custom_field(self):
        """Test gte with custom field name."""
        condition = Condition.greater_than_or_equal(50, field="count")
        assert condition({"count": 60}) is True
        assert condition({"count": 50}) is True
        assert condition({"count": 40}) is False

    def test_gte_with_missing_field(self):
        """Test gte when field is missing (defaults to 0)."""
        condition = Condition.greater_than_or_equal(5, field="missing")
        result = {"score": 0.8}
        # Missing field defaults to 0, which is not >= 5
        assert condition(result) is False


class TestConditionLessThanOrEqual:
    """Tests for Condition.less_than_or_equal()."""

    def test_lte_with_smaller_value(self):
        """Test lte when value is smaller."""
        condition = Condition.less_than_or_equal(0.5, field="toxicity")
        result = {"toxicity": 0.3}
        assert condition(result) is True

    def test_lte_with_equal_value(self):
        """Test lte when value is equal."""
        condition = Condition.less_than_or_equal(0.5, field="toxicity")
        result = {"toxicity": 0.5}
        assert condition(result) is True

    def test_lte_with_larger_value(self):
        """Test lte when value is larger."""
        condition = Condition.less_than_or_equal(0.5, field="toxicity")
        result = {"toxicity": 0.7}
        assert condition(result) is False

    def test_lte_with_custom_field(self):
        """Test lte with custom field name."""
        condition = Condition.less_than_or_equal(100, field="count")
        assert condition({"count": 80}) is True
        assert condition({"count": 100}) is True
        assert condition({"count": 120}) is False

    def test_lte_with_missing_field(self):
        """Test lte when field is missing (defaults to inf)."""
        condition = Condition.less_than_or_equal(1000, field="missing")
        result = {"score": 0.8}
        # Missing field defaults to inf, which is not <= 1000
        assert condition(result) is False


class TestConditionBetween:
    """Tests for Condition.between()."""

    def test_between_with_value_in_range(self):
        """Test between when value is within range."""
        condition = Condition.between(50, 200, field="count")
        result = {"count": 100}
        assert condition(result) is True

    def test_between_with_value_at_min_boundary(self):
        """Test between when value is at minimum boundary."""
        condition = Condition.between(50, 200, field="count")
        result = {"count": 50}
        assert condition(result) is True

    def test_between_with_value_at_max_boundary(self):
        """Test between when value is at maximum boundary."""
        condition = Condition.between(50, 200, field="count")
        result = {"count": 200}
        assert condition(result) is True

    def test_between_with_value_below_range(self):
        """Test between when value is below range."""
        condition = Condition.between(50, 200, field="count")
        result = {"count": 30}
        assert condition(result) is False

    def test_between_with_value_above_range(self):
        """Test between when value is above range."""
        condition = Condition.between(50, 200, field="count")
        result = {"count": 250}
        assert condition(result) is False

    def test_between_with_custom_field(self):
        """Test between with custom field name."""
        condition = Condition.between(0.3, 0.8, field="confidence")
        assert condition({"confidence": 0.5}) is True
        assert condition({"confidence": 0.3}) is True
        assert condition({"confidence": 0.8}) is True
        assert condition({"confidence": 0.2}) is False
        assert condition({"confidence": 0.9}) is False

    def test_between_with_default_score_field(self):
        """Test between with default 'score' field."""
        condition = Condition.between(0.5, 1.0)
        assert condition({"score": 0.7}) is True
        assert condition({"score": 0.5}) is True
        assert condition({"score": 1.0}) is True
        assert condition({"score": 0.3}) is False

    def test_between_with_missing_field(self):
        """Test between when field is missing (returns None)."""
        condition = Condition.between(50, 200, field="missing")
        result = {"score": 0.8}
        assert condition(result) is False

    def test_between_with_none_value(self):
        """Test between when field value is None."""
        condition = Condition.between(50, 200, field="count")
        result = {"count": None}
        assert condition(result) is False
