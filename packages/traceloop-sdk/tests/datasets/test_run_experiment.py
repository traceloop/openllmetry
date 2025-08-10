import json
import pytest
from unittest.mock import patch, MagicMock
from traceloop.sdk.datasets.dataset import Dataset
from traceloop.sdk.experiments.model import ExperimentRunResult, ExperimentResult
from traceloop.sdk.experiments.experiment import Experiment
from .mock_response import get_dataset_by_slug_json


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
@pytest.mark.asyncio
async def test_run_experiment_with_input_mapping():
    """Test running experiments on all dataset rows with input mapping"""
    
    # Mock the dataset response
    with patch.object(Dataset, '_get_http_client_static') as mock_get_client:
        mock_client = MagicMock()
        
        # Mock get_by_slug to return dataset data
        mock_client.get.return_value = json.loads(get_dataset_by_slug_json)
        mock_get_client.return_value = mock_client
        
        # Mock Experiment.evaluate to return experiment results
        mock_experiment_result = ExperimentResult(
            experiment_id="test-experiment-id",
            evaluator_slug="test-evaluator",
            total_runs=4,
            successful_runs=4,
            failed_runs=0,
            results=[
                ExperimentRunResult(
                    input_index=0,
                    input_data={"question": "Laptop", "answer": "999.99"},
                    result="This is a laptop result",
                    error=None,
                    execution_time=1.5
                ),
                ExperimentRunResult(
                    input_index=1,
                    input_data={"question": "Phone", "answer": "599.99"},
                    result="This is a phone result", 
                    error=None,
                    execution_time=1.2
                ),
                ExperimentRunResult(
                    input_index=2,
                    input_data={"question": "Mouse", "answer": "29.99"},
                    result="This is a mouse result", 
                    error=None,
                    execution_time=0.8
                ),
                ExperimentRunResult(
                    input_index=3,
                    input_data={"question": "Keyboard", "answer": "79.99"},
                    result="This is a keyboard result", 
                    error=None,
                    execution_time=1.0
                )
            ],
            total_execution_time=4.5
        )
        
        with patch.object(Experiment, 'evaluate', return_value=mock_experiment_result) as mock_evaluate:
            # Define input mapping
            input_mapping = {
                "question": "product",
                "answer": "price"
            }
            
            # Run experiment
            result = await Dataset.run_experiment(
                slug="product-inventory-2", 
                evaluator_slug="test-evaluator",
                input_mapping=input_mapping
            )
            
            # Verify results
            assert isinstance(result, ExperimentResult)
            assert result.evaluator_slug == "test-evaluator"
            assert result.total_runs == 4
            assert result.successful_runs == 4
            assert result.failed_runs == 0
            assert len(result.results) == 4
            
            # Verify that Experiment.evaluate was called with correct arguments
            mock_evaluate.assert_called_once()
            call_args = mock_evaluate.call_args
            assert call_args[1]['evaluator_slug'] == "test-evaluator"
            assert call_args[1]['dataset_slug'] == "product-inventory-2"
            assert len(call_args[1]['inputs']) == 4
            
            # Verify input mapping worked correctly
            inputs = call_args[1]['inputs']
            for input_data in inputs:
                assert "question" in input_data
                assert "answer" in input_data


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
@pytest.mark.asyncio
async def test_run_experiment_invalid_column_mapping():
    """Test that invalid column mapping raises appropriate error"""
    
    with patch.object(Dataset, '_get_http_client_static') as mock_get_client:
        mock_client = MagicMock()
        mock_client.get.return_value = json.loads(get_dataset_by_slug_json)
        mock_get_client.return_value = mock_client
        
        # Define invalid input mapping (nonexistent column)
        input_mapping = {
            "question": "nonexistent_column"
        }
        
        # Should raise ValueError for invalid column
        try:
            await Dataset.run_experiment(
                slug="product-inventory-2",
                evaluator_slug="test-evaluator",
                input_mapping=input_mapping
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "nonexistent_column" in str(e)
            assert "not found in dataset" in str(e)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
@pytest.mark.asyncio  
async def test_run_experiment_with_errors():
    """Test handling of experiment execution with some errors"""
    
    with patch.object(Dataset, '_get_http_client_static') as mock_get_client:
        mock_client = MagicMock()
        
        # Mock get_by_slug to return dataset data
        mock_client.get.return_value = json.loads(get_dataset_by_slug_json)
        mock_get_client.return_value = mock_client
        
        # Mock Experiment.evaluate to return results with some errors
        mock_experiment_result = ExperimentResult(
            experiment_id="test-experiment-id",
            evaluator_slug="test-evaluator",
            total_runs=4,
            successful_runs=2,
            failed_runs=2,
            results=[
                ExperimentRunResult(
                    input_index=0,
                    input_data={"question": "Laptop", "answer": "999.99"},
                    result="This is a laptop result",
                    error=None,
                    execution_time=1.5
                ),
                ExperimentRunResult(
                    input_index=1,
                    input_data={"question": "Phone", "answer": "599.99"},
                    result=None,
                    error="Evaluation failed",
                    execution_time=0.5
                ),
                ExperimentRunResult(
                    input_index=2,
                    input_data={"question": "Mouse", "answer": "29.99"},
                    result="This is a mouse result",
                    error=None,
                    execution_time=0.8
                ),
                ExperimentRunResult(
                    input_index=3,
                    input_data={"question": "Keyboard", "answer": "79.99"},
                    result=None,
                    error="Timeout error",
                    execution_time=2.0
                )
            ],
            total_execution_time=4.8
        )
        
        with patch.object(Experiment, 'evaluate', return_value=mock_experiment_result):
            input_mapping = {
                "question": "product", 
                "answer": "price"
            }
            
            result = await Dataset.run_experiment(
                slug="product-inventory-2",
                evaluator_slug="test-evaluator",
                input_mapping=input_mapping
            )
            
            # Should return experiment result with mixed success/failure
            assert isinstance(result, ExperimentResult)
            assert result.total_runs == 4
            assert result.successful_runs == 2
            assert result.failed_runs == 2
            
            # Check that error results are properly included
            error_results = [r for r in result.results if r.error is not None]
            assert len(error_results) == 2
            
            success_results = [r for r in result.results if r.error is None]
            assert len(success_results) == 2