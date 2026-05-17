"""
Unit tests for guardrail conditions.

Tests all built-in condition methods from the Condition class.

Note: In the new design, conditions receive the extracted field value directly
(not the full result object). The field extraction is done by the evaluator's
as_guard method using the condition_field setting.
"""
import warnings

from traceloop.sdk.guardrail.condition import Condition


class TestConditionIsTrue:
    """Tests for Condition.is_true()."""

    def test_is_true_with_true_value(self):
        """Test is_true condition when value is True."""
        condition = Condition.is_true()
        assert condition(True) is True

    def test_is_true_with_false_value(self):
        """Test is_true condition when value is False."""
        condition = Condition.is_true()
        assert condition(False) is False

    def test_is_true_with_none_value(self):
        """Test is_true condition when value is None."""
        condition = Condition.is_true()
        assert condition(None) is False

    def test_is_true_with_truthy_values(self):
        """Test is_true condition with truthy but non-boolean values."""
        condition = Condition.is_true()
        # Only exact True should pass, not truthy values
        assert condition(1) is False
        assert condition("yes") is False
        assert condition([1, 2]) is False


class TestConditionIsFalse:
    """Tests for Condition.is_false()."""

    def test_is_false_with_false_value(self):
        """Test is_false condition when value is False."""
        condition = Condition.is_false()
        assert condition(False) is True

    def test_is_false_with_true_value(self):
        """Test is_false condition when value is True."""
        condition = Condition.is_false()
        assert condition(True) is False

    def test_is_false_with_none_value(self):
        """Test is_false condition when value is None."""
        condition = Condition.is_false()
        assert condition(None) is False

    def test_is_false_with_falsy_values(self):
        """Test is_false condition with falsy but non-boolean values."""
        condition = Condition.is_false()
        # Only exact False should pass, not falsy values
        assert condition(0) is False
        assert condition("") is False
        assert condition([]) is False


class TestConditionIsTruthy:
    """Tests for Condition.is_truthy()."""

    def test_is_truthy_with_true(self):
        condition = Condition.is_truthy()
        assert condition(True) is True

    def test_is_truthy_with_false(self):
        condition = Condition.is_truthy()
        assert condition(False) is False

    def test_is_truthy_with_none(self):
        condition = Condition.is_truthy()
        assert condition(None) is False

    def test_is_truthy_with_truthy_values(self):
        condition = Condition.is_truthy()
        assert condition(1) is True
        assert condition("yes") is True
        assert condition([1, 2]) is True

    def test_is_truthy_with_falsy_values(self):
        condition = Condition.is_truthy()
        assert condition(0) is False
        assert condition("") is False
        assert condition([]) is False


class TestConditionIsFalsy:
    """Tests for Condition.is_falsy()."""

    def test_is_falsy_with_false(self):
        condition = Condition.is_falsy()
        assert condition(False) is True

    def test_is_falsy_with_true(self):
        condition = Condition.is_falsy()
        assert condition(True) is False

    def test_is_falsy_with_none(self):
        condition = Condition.is_falsy()
        assert condition(None) is True

    def test_is_falsy_with_falsy_values(self):
        condition = Condition.is_falsy()
        assert condition(0) is True
        assert condition("") is True
        assert condition([]) is True

    def test_is_falsy_with_truthy_values(self):
        condition = Condition.is_falsy()
        assert condition(1) is False
        assert condition("yes") is False
        assert condition([1, 2]) is False


class TestConditionEquals:
    """Tests for Condition.equals()."""

    def test_equals_with_matching_string(self):
        """Test equals condition with matching string value."""
        condition = Condition.equals("approved")
        assert condition("approved") is True

    def test_equals_with_different_string(self):
        """Test equals condition with different string value."""
        condition = Condition.equals("approved")
        assert condition("rejected") is False

    def test_equals_with_none_value(self):
        """Test equals condition when value is None."""
        condition = Condition.equals("approved")
        assert condition(None) is False

    def test_equals_with_various_types(self):
        """Test equals condition with various data types."""
        # Integer
        condition = Condition.equals(42)
        assert condition(42) is True
        assert condition(43) is False

        # Float
        condition = Condition.equals(0.8)
        assert condition(0.8) is True
        assert condition(0.7) is False

        # Boolean
        condition = Condition.equals(True)
        assert condition(True) is True
        assert condition(False) is False


class TestConditionGreaterThan:
    """Tests for Condition.greater_than()."""

    def test_greater_than_with_larger_value(self):
        """Test greater_than when value is larger."""
        condition = Condition.greater_than(0.5)
        assert condition(0.8) is True

    def test_greater_than_with_equal_value(self):
        """Test greater_than when value is equal."""
        condition = Condition.greater_than(0.5)
        assert condition(0.5) is False

    def test_greater_than_with_smaller_value(self):
        """Test greater_than when value is smaller."""
        condition = Condition.greater_than(0.5)
        assert condition(0.3) is False

    def test_greater_than_with_none_value(self):
        """Test greater_than when value is None."""
        condition = Condition.greater_than(0.5)
        assert condition(None) is False

    def test_greater_than_with_integer(self):
        """Test greater_than with integer values."""
        condition = Condition.greater_than(10)
        assert condition(15) is True
        assert condition(10) is False
        assert condition(5) is False


class TestConditionLessThan:
    """Tests for Condition.less_than()."""

    def test_less_than_with_smaller_value(self):
        """Test less_than when value is smaller."""
        condition = Condition.less_than(1000)
        assert condition(500) is True

    def test_less_than_with_equal_value(self):
        """Test less_than when value is equal."""
        condition = Condition.less_than(1000)
        assert condition(1000) is False

    def test_less_than_with_larger_value(self):
        """Test less_than when value is larger."""
        condition = Condition.less_than(1000)
        assert condition(1500) is False

    def test_less_than_with_none_value(self):
        """Test less_than when value is None."""
        condition = Condition.less_than(1000)
        assert condition(None) is False

    def test_less_than_with_float(self):
        """Test less_than with float values."""
        condition = Condition.less_than(0.5)
        assert condition(0.3) is True
        assert condition(0.5) is False
        assert condition(0.7) is False


class TestConditionGreaterThanOrEqual:
    """Tests for Condition.greater_than_or_equal()."""

    def test_gte_with_larger_value(self):
        """Test gte when value is larger."""
        condition = Condition.greater_than_or_equal(0.8)
        assert condition(0.9) is True

    def test_gte_with_equal_value(self):
        """Test gte when value is equal."""
        condition = Condition.greater_than_or_equal(0.8)
        assert condition(0.8) is True

    def test_gte_with_smaller_value(self):
        """Test gte when value is smaller."""
        condition = Condition.greater_than_or_equal(0.8)
        assert condition(0.7) is False

    def test_gte_with_none_value(self):
        """Test gte when value is None."""
        condition = Condition.greater_than_or_equal(0.8)
        assert condition(None) is False

    def test_gte_with_integer(self):
        """Test gte with integer values."""
        condition = Condition.greater_than_or_equal(50)
        assert condition(60) is True
        assert condition(50) is True
        assert condition(40) is False


class TestConditionLessThanOrEqual:
    """Tests for Condition.less_than_or_equal()."""

    def test_lte_with_smaller_value(self):
        """Test lte when value is smaller."""
        condition = Condition.less_than_or_equal(0.5)
        assert condition(0.3) is True

    def test_lte_with_equal_value(self):
        """Test lte when value is equal."""
        condition = Condition.less_than_or_equal(0.5)
        assert condition(0.5) is True

    def test_lte_with_larger_value(self):
        """Test lte when value is larger."""
        condition = Condition.less_than_or_equal(0.5)
        assert condition(0.7) is False

    def test_lte_with_none_value(self):
        """Test lte when value is None."""
        condition = Condition.less_than_or_equal(0.5)
        assert condition(None) is False

    def test_lte_with_integer(self):
        """Test lte with integer values."""
        condition = Condition.less_than_or_equal(100)
        assert condition(80) is True
        assert condition(100) is True
        assert condition(120) is False


class TestConditionBetween:
    """Tests for Condition.between()."""

    def test_between_with_value_in_range(self):
        """Test between when value is within range."""
        condition = Condition.between(50, 200)
        assert condition(100) is True

    def test_between_with_value_at_min_boundary(self):
        """Test between when value is at minimum boundary."""
        condition = Condition.between(50, 200)
        assert condition(50) is True

    def test_between_with_value_at_max_boundary(self):
        """Test between when value is at maximum boundary."""
        condition = Condition.between(50, 200)
        assert condition(200) is True

    def test_between_with_value_below_range(self):
        """Test between when value is below range."""
        condition = Condition.between(50, 200)
        assert condition(30) is False

    def test_between_with_value_above_range(self):
        """Test between when value is above range."""
        condition = Condition.between(50, 200)
        assert condition(250) is False

    def test_between_with_float_range(self):
        """Test between with float values."""
        condition = Condition.between(0.3, 0.8)
        assert condition(0.5) is True
        assert condition(0.3) is True
        assert condition(0.8) is True
        assert condition(0.2) is False
        assert condition(0.9) is False

    def test_between_with_none_value(self):
        """Test between when value is None."""
        condition = Condition.between(50, 200)
        assert condition(None) is False


class TestConditionDeprecation:
    """Tests that all Condition methods emit DeprecationWarning."""

    def _assert_deprecation(self, func, *args):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(*args)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
        return result

    def test_is_true_deprecation(self):
        condition = self._assert_deprecation(Condition.is_true)
        assert condition(True) is True

    def test_is_false_deprecation(self):
        condition = self._assert_deprecation(Condition.is_false)
        assert condition(False) is True

    def test_is_truthy_deprecation(self):
        condition = self._assert_deprecation(Condition.is_truthy)
        assert condition(1) is True

    def test_is_falsy_deprecation(self):
        condition = self._assert_deprecation(Condition.is_falsy)
        assert condition(0) is True

    def test_between_deprecation(self):
        condition = self._assert_deprecation(Condition.between, 0.3, 0.7)
        assert condition(0.5) is True

    def test_equals_deprecation(self):
        condition = self._assert_deprecation(Condition.equals, "approved")
        assert condition("approved") is True

    def test_greater_than_deprecation(self):
        condition = self._assert_deprecation(Condition.greater_than, 0.5)
        assert condition(0.8) is True

    def test_less_than_deprecation(self):
        condition = self._assert_deprecation(Condition.less_than, 0.5)
        assert condition(0.3) is True

    def test_greater_than_or_equal_deprecation(self):
        condition = self._assert_deprecation(Condition.greater_than_or_equal, 0.8)
        assert condition(0.8) is True

    def test_less_than_or_equal_deprecation(self):
        condition = self._assert_deprecation(Condition.less_than_or_equal, 0.5)
        assert condition(0.5) is True
