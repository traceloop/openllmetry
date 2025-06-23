import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from httpx import Response

from traceloop.sdk.guardrails import (
    GuardrailsClient,
    GuardrailsDecorator,
    GuardrailResult,
    GuardrailAction,
    get_guardrails_client,
    get_guardrails_decorator
)


class TestGuardrailResult:
    def test_init_with_defaults(self):
        result = GuardrailResult(GuardrailAction.PASS)
        assert result.action == GuardrailAction.PASS
        assert result.result is None
        assert result.reason is None
        assert result.score is None
        assert result.metadata == {}

    def test_init_with_values(self):
        metadata = {"key": "value"}
        result = GuardrailResult(
            action=GuardrailAction.BLOCK,
            result="blocked",
            reason="Inappropriate content",
            score=0.95,
            metadata=metadata
        )
        assert result.action == GuardrailAction.BLOCK
        assert result.result == "blocked"
        assert result.reason == "Inappropriate content"
        assert result.score == 0.95
        assert result.metadata == metadata

    def test_properties(self):
        pass_result = GuardrailResult(GuardrailAction.PASS)
        assert pass_result.pass_through is True
        assert pass_result.blocked is False
        assert pass_result.retry_required is False

        block_result = GuardrailResult(GuardrailAction.BLOCK)
        assert block_result.pass_through is False
        assert block_result.blocked is True
        assert block_result.retry_required is False

        retry_result = GuardrailResult(GuardrailAction.RETRY)
        assert retry_result.pass_through is False
        assert retry_result.blocked is False
        assert retry_result.retry_required is True


class TestGuardrailsClient:
    def setup_method(self):
        with patch('traceloop.sdk.guardrails.Config') as mock_config:
            mock_config.get_api_key.return_value = "test-key"
            mock_config.get_api_url.return_value = "https://api.test.com"
            mock_config.get_project_id.return_value = "test-project"
            
            with patch('traceloop.sdk.guardrails.HttpClient') as mock_http_client:
                self.client = GuardrailsClient()
                self.mock_http_client = mock_http_client.return_value

    def test_init_without_credentials(self):
        with patch('traceloop.sdk.guardrails.Config') as mock_config:
            mock_config.get_api_key.return_value = None
            mock_config.get_api_url.return_value = "https://api.test.com"
            mock_config.get_project_id.return_value = "test-project"
            
            with pytest.raises(ValueError, match="API key is required"):
                GuardrailsClient()

    def test_init_with_credentials(self):
        with patch('traceloop.sdk.guardrails.HttpClient') as mock_http_client:
            client = GuardrailsClient(
                api_key="custom-key",
                api_url="https://custom.api.com",
                project_id="custom-project"
            )
            assert client.api_key == "custom-key"
            assert client.api_url == "https://custom.api.com"
            assert client.project_id == "custom-project"

    @pytest.mark.asyncio
    async def test_execute_guardrail_success(self):
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.json.return_value = {
            "execution_id": "exec-123",
            "stream_url": "https://api.test.com/stream"
        }
        self.mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch.object(self.client, '_wait_for_result') as mock_wait:
            mock_wait.return_value = {"pass": True, "score": 0.85}
            
            result = await self.client.execute_guardrail(
                "test-evaluator",
                {"input": "test data"}
            )
            
            assert result.action == GuardrailAction.PASS
            assert result.score == 0.85

    @pytest.mark.asyncio
    async def test_execute_guardrail_timeout(self):
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.json.return_value = {
            "execution_id": "exec-123",
            "stream_url": "https://api.test.com/stream"
        }
        self.mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch.object(self.client, '_wait_for_result') as mock_wait:
            mock_wait.side_effect = asyncio.TimeoutError()
            
            result = await self.client.execute_guardrail(
                "test-evaluator",
                {"input": "test data"}
            )
            
            assert result.action == GuardrailAction.PASS
            assert "timed out" in result.reason

    @pytest.mark.asyncio
    async def test_execute_guardrail_error(self):
        self.mock_http_client.post = AsyncMock(side_effect=Exception("Network error"))
        
        result = await self.client.execute_guardrail(
            "test-evaluator",
            {"input": "test data"}
        )
        
        assert result.action == GuardrailAction.PASS
        assert "Network error" in result.reason

    def test_execute_guardrail_sync(self):
        with patch.object(self.client, 'execute_guardrail') as mock_async:
            mock_async.return_value = GuardrailResult(GuardrailAction.PASS)
            
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = GuardrailResult(GuardrailAction.PASS)
                
                result = self.client.execute_guardrail_sync(
                    "test-evaluator",
                    {"input": "test data"}
                )
                
                assert result.action == GuardrailAction.PASS

    @pytest.mark.asyncio
    async def test_start_execution_success(self):
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.json.return_value = {"execution_id": "exec-123"}
        self.mock_http_client.post = AsyncMock(return_value=mock_response)
        
        result = await self.client._start_execution(
            "test-evaluator",
            {"input": "test"},
            {"config": "value"},
            30
        )
        
        assert result["execution_id"] == "exec-123"
        self.mock_http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_execution_failure(self):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        self.mock_http_client.post = AsyncMock(return_value=mock_response)
        
        with pytest.raises(Exception, match="Failed to start execution"):
            await self.client._start_execution(
                "test-evaluator",
                {"input": "test"},
                None,
                30
            )

    @pytest.mark.asyncio
    async def test_wait_for_result_success(self):
        mock_lines = [
            "data: " + json.dumps({
                "type": "status_update",
                "data": {"status": "running"}
            }),
            "data: " + json.dumps({
                "type": "completed",
                "data": {"result": {"pass": True, "score": 0.9}}
            })
        ]
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.aiter_lines.return_value = mock_lines
            
            mock_client.stream.return_value.__aenter__.return_value = mock_response
            
            result = await self.client._wait_for_result(
                "exec-123",
                "https://api.test.com/stream",
                30,
                None
            )
            
            assert result["pass"] is True
            assert result["score"] == 0.9

    def test_parse_result_pass(self):
        result_data = {"pass": True, "score": 0.85, "reason": "Good content"}
        result = self.client._parse_result(result_data)
        
        assert result.action == GuardrailAction.PASS
        assert result.score == 0.85
        assert result.reason == "Good content"

    def test_parse_result_block(self):
        result_data = {"pass": False, "reason": "Blocked content"}
        result = self.client._parse_result(result_data)
        
        assert result.action == GuardrailAction.BLOCK
        assert result.reason == "Blocked content"

    def test_parse_result_retry(self):
        result_data = {"pass": False, "retry": True, "reason": "Needs retry"}
        result = self.client._parse_result(result_data)
        
        assert result.action == GuardrailAction.RETRY
        assert result.reason == "Needs retry"


class TestGuardrailsDecorator:
    def setup_method(self):
        self.mock_client = Mock(spec=GuardrailsClient)
        self.decorator = GuardrailsDecorator(self.mock_client)

    @pytest.mark.asyncio
    async def test_validate_input_async_pass(self):
        self.mock_client.execute_guardrail.return_value = GuardrailResult(GuardrailAction.PASS)
        
        @self.decorator.validate_input("test-evaluator")
        async def test_func(input_text):
            return f"Processed: {input_text}"
        
        result = await test_func("test input")
        assert result == "Processed: test input"
        self.mock_client.execute_guardrail.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_input_async_block(self):
        self.mock_client.execute_guardrail.return_value = GuardrailResult(
            GuardrailAction.BLOCK, reason="Inappropriate content"
        )
        
        @self.decorator.validate_input("test-evaluator")
        async def test_func(input_text):
            return f"Processed: {input_text}"
        
        with pytest.raises(Exception, match="Input blocked by guardrail"):
            await test_func("bad input")

    def test_validate_input_sync_pass(self):
        self.mock_client.execute_guardrail_sync.return_value = GuardrailResult(GuardrailAction.PASS)
        
        @self.decorator.validate_input("test-evaluator")
        def test_func(input_text):
            return f"Processed: {input_text}"
        
        result = test_func("test input")
        assert result == "Processed: test input"
        self.mock_client.execute_guardrail_sync.assert_called_once()

    def test_validate_input_sync_block(self):
        self.mock_client.execute_guardrail_sync.return_value = GuardrailResult(
            GuardrailAction.BLOCK, reason="Inappropriate content"
        )
        
        @self.decorator.validate_input("test-evaluator")
        def test_func(input_text):
            return f"Processed: {input_text}"
        
        with pytest.raises(Exception, match="Input blocked by guardrail"):
            test_func("bad input")

    @pytest.mark.asyncio
    async def test_validate_output_async_pass(self):
        self.mock_client.execute_guardrail.return_value = GuardrailResult(GuardrailAction.PASS)
        
        @self.decorator.validate_output("test-evaluator")
        async def test_func():
            return "Safe output"
        
        result = await test_func()
        assert result == "Safe output"
        self.mock_client.execute_guardrail.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_output_async_retry_success(self):
        call_count = 0
        
        def mock_execute_guardrail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GuardrailResult(GuardrailAction.RETRY, reason="Try again")
            return GuardrailResult(GuardrailAction.PASS)
        
        self.mock_client.execute_guardrail.side_effect = mock_execute_guardrail
        
        func_call_count = 0
        
        @self.decorator.validate_output("test-evaluator")
        async def test_func():
            nonlocal func_call_count
            func_call_count += 1
            return f"Output {func_call_count}"
        
        result = await test_func()
        assert result == "Output 2"
        assert call_count == 2
        assert func_call_count == 2

    def test_validate_output_sync_pass(self):
        self.mock_client.execute_guardrail_sync.return_value = GuardrailResult(GuardrailAction.PASS)
        
        @self.decorator.validate_output("test-evaluator")
        def test_func():
            return "Safe output"
        
        result = test_func()
        assert result == "Safe output"
        self.mock_client.execute_guardrail_sync.assert_called_once()

    def test_validate_input_with_custom_extractor(self):
        self.mock_client.execute_guardrail_sync.return_value = GuardrailResult(GuardrailAction.PASS)
        
        def custom_extractor(*args, **kwargs):
            return {"custom_input": args[0]}
        
        @self.decorator.validate_input("test-evaluator", input_extractor=custom_extractor)
        def test_func(input_text):
            return f"Processed: {input_text}"
        
        result = test_func("test input")
        assert result == "Processed: test input"
        
        call_args = self.mock_client.execute_guardrail_sync.call_args
        assert call_args[0][1] == {"custom_input": "test input"}

    def test_validate_input_with_on_block_handler(self):
        self.mock_client.execute_guardrail_sync.return_value = GuardrailResult(
            GuardrailAction.BLOCK, reason="Blocked"
        )
        
        def on_block_handler(result):
            return f"Blocked: {result.reason}"
        
        @self.decorator.validate_input("test-evaluator", on_block=on_block_handler)
        def test_func(input_text):
            return f"Processed: {input_text}"
        
        result = test_func("bad input")
        assert result == "Blocked: Blocked"


def test_get_guardrails_client():
    with patch('traceloop.sdk.guardrails.GuardrailsClient') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        client = get_guardrails_client(
            api_key="test-key",
            api_url="https://test.com",
            project_id="test-project"
        )
        
        assert client == mock_client
        mock_client_class.assert_called_once_with(
            api_key="test-key",
            api_url="https://test.com",
            project_id="test-project",
            timeout=30
        )


def test_get_guardrails_decorator():
    mock_client = Mock(spec=GuardrailsClient)
    
    with patch('traceloop.sdk.guardrails.GuardrailsDecorator') as mock_decorator_class:
        mock_decorator = Mock()
        mock_decorator_class.return_value = mock_decorator
        
        decorator = get_guardrails_decorator(mock_client)
        
        assert decorator == mock_decorator
        mock_decorator_class.assert_called_once_with(mock_client)


def test_get_guardrails_decorator_default_client():
    with patch('traceloop.sdk.guardrails.get_guardrails_client') as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        with patch('traceloop.sdk.guardrails.GuardrailsDecorator') as mock_decorator_class:
            mock_decorator = Mock()
            mock_decorator_class.return_value = mock_decorator
            
            decorator = get_guardrails_decorator()
            
            assert decorator == mock_decorator
            mock_get_client.assert_called_once()
            mock_decorator_class.assert_called_once_with(mock_client) 