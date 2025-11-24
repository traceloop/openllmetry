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

    def _headers(self):
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
            print(Fore.RED + f"Error making request to {path}: {str(e)}" + Fore.RESET)
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
            print(Fore.RED + f"Error making request to {path}: {str(e)}" + Fore.RESET)
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
            print(Fore.RED + f"Error making request to {path}: {str(e)}" + Fore.RESET)
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
            print(Fore.RED + f"Error making request to {path}: {str(e)}" + Fore.RESET)
            return None
