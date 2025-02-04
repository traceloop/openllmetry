from importlib.metadata import version
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.instrumentation.groq.utils import model_as_dict

# Describe block for all tests related to model_as_dict


@pytest.mark.describe("model_as_dict function")
class TestModelAsDict:

    @pytest.mark.happy_path
    def test_model_as_dict_with_pydantic_v1(self):
        """
        Test that model_as_dict correctly calls the dict() method on a Pydantic v1 model.
        """
        mock_model = MagicMock()
        mock_model.dict.return_value = {"model": "test_model"}

        with patch("opentelemetry.instrumentation.groq.utils.version", return_value="1.9.0"):
            result = model_as_dict(mock_model)

        assert result == {"model": "test_model"}
        mock_model.dict.assert_called_once()

    @pytest.mark.happy_path
    def test_model_as_dict_with_pydantic_v2(self):
        """
        Test that model_as_dict correctly calls the model_dump() method on a Pydantic v2 model.
        """
        mock_model = MagicMock()
        mock_model.model_dump.return_value = {"model": "test_model"}

        with patch("opentelemetry.instrumentation.groq.utils.version", return_value="2.0.0"):
            result = model_as_dict(mock_model)

        assert result == {"model": "test_model"}
        mock_model.model_dump.assert_called_once()

    @pytest.mark.edge_case
    def test_model_as_dict_with_non_pydantic_object(self):
        """
        Test that model_as_dict returns the object itself if it is not a Pydantic model.
        """
        non_pydantic_object = {"model": "non_pydantic_model"}

        result = model_as_dict(non_pydantic_object)

        assert result == non_pydantic_object

    @pytest.mark.edge_case
    def test_model_as_dict_with_empty_model(self):
        """
        Test that model_as_dict handles an empty model gracefully.
        """
        empty_model = MagicMock()
        empty_model.dict.return_value = {}

        with patch("opentelemetry.instrumentation.groq.utils.version", return_value="1.9.0"):
            result = model_as_dict(empty_model)

        assert result == {}
        empty_model.dict.assert_called_once()

    @pytest.mark.edge_case
    def test_model_as_dict_with_none(self):
        """
        Test that model_as_dict returns None when given None as input.
        """
        result = model_as_dict(None)

        assert result is None
