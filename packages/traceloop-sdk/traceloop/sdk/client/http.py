from typing import Any, Dict, Optional

import requests
from colorama import Fore


class HTTPClient:
    base_url: str
    api_key: str
    version: str

    def __init__(self, base_url: str, api_key: str, version: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.version = version

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Traceloop-SDK-Version": self.version,
        }

    def post(self, path: str, data: Dict[str, Any]) -> Any:
        """
        Make a POST request to the API
        """
        try:
            response = requests.post(
                f"{self.base_url}/v2/{path.lstrip('/')}",
                json=data,
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_msg = self._format_error_message(path, e)
            print(Fore.RED + error_msg + Fore.RESET)
            return None

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Make a GET request to the API
        """
        try:
            response = requests.get(
                f"{self.base_url}/v2/{path.lstrip('/')}",
                params=params,
                headers=self._headers(),
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if "text/csv" in content_type or "application/x-ndjson" in content_type:
                return response.text
            else:
                return response.json()
        except requests.exceptions.RequestException as e:
            error_msg = self._format_error_message(path, e)
            print(Fore.RED + error_msg + Fore.RESET)
            return None

    def delete(self, path: str) -> bool:
        """
        Make a DELETE request to the API
        """
        try:
            response = requests.delete(
                f"{self.base_url}/v2/{path.lstrip('/')}", headers=self._headers()
            )
            response.raise_for_status()
            return response.status_code == 204 or response.status_code == 200
        except requests.exceptions.RequestException as e:
            error_msg = self._format_error_message(path, e)
            print(Fore.RED + error_msg + Fore.RESET)
            return False

    def put(self, path: str, data: Dict[str, Any]) -> Any:
        """
        Make a PUT request to the API
        """
        try:
            response = requests.put(
                f"{self.base_url}/v2/{path.lstrip('/')}",
                json=data,
                headers=self._headers(),
            )
            response.raise_for_status()
            if response.content:
                return response.json()
            else:
                return {}
        except requests.exceptions.RequestException as e:
            error_msg = self._format_error_message(path, e)
            print(Fore.RED + error_msg + Fore.RESET)
            return None

    def _format_error_message(self, path: str, exception: requests.exceptions.RequestException) -> str:
        """
        Format a detailed error message including server response if available
        """
        error_parts = [f"Error making request to {path}: {str(exception)}"]

        # Try to extract error details from response
        if hasattr(exception, 'response') and exception.response is not None:
            response = exception.response

            # Try to parse JSON error from response
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    # Check common error fields
                    if 'error' in error_data:
                        error_parts.append(f"Server error: {error_data['error']}")
                    elif 'message' in error_data:
                        error_parts.append(f"Server message: {error_data['message']}")
                    elif 'msg' in error_data:
                        error_parts.append(f"Server message: {error_data['msg']}")
                    else:
                        # Include the entire JSON if no standard field found
                        error_parts.append(f"Server response: {error_data}")
            except (ValueError, AttributeError):
                # Not JSON, try to get text
                try:
                    error_text = response.text
                    if error_text and len(error_text) < 500:  # Only include short error messages
                        error_parts.append(f"Server response: {error_text}")
                except Exception:
                    pass

        return "\n".join(error_parts)
