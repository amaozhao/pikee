import base64
import hashlib
import hmac
import logging
import os
import socket
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .settings import ApolloSettingsConfig

logger = logging.getLogger(__name__)


class ApolloClientBase:
    """Base class for Apollo clients with common functionality."""

    def __init__(
        self,
        meta_server_address: Optional[str] = None,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        cluster: str = "default",
        env: str = "DEV",
        namespaces: Optional[List[str]] = None,
        ip: Optional[str] = None,
        timeout: int = 10,
        cycle_time: int = 30,
        cache_file_dir_path: Optional[str] = None,
        settings: Optional[ApolloSettingsConfig] = None,
    ):
        """
        Initialize the base client with common configuration.

        Args:
            meta_server_address: Apollo meta server address
            app_id: Application ID
            app_secret: Application secret, optional
            cluster: Cluster name, default value is 'default'
            env: Environment, default value is 'DEV'
            namespaces: Namespace list to get configuration
            ip: Deploy IP for grey release
            timeout: HTTP request timeout in seconds
            cycle_time: Configuration refresh cycle time in seconds
            cache_file_dir_path: Local cache file directory path
            settings: ApolloSettingsConfig instance
        """
        # Initialize cache directory path first
        self._cache_file_dir_path = None

        # Load configuration
        if settings is not None:
            self._meta_server_address = settings.meta_server_address
            self._app_id = settings.app_id
            self._app_secret = settings.app_secret if settings.using_app_secret else None
            self._cluster = settings.cluster
            self._timeout = settings.timeout
            self._env = settings.env
            self._cycle_time = settings.cycle_time
            self._cache_file_dir_path = settings.cache_file_dir_path
            self.ip = self._get_local_ip_address(settings.ip)
            # Convert namespaces to list if it's a string
            ns = settings.namespaces
            if isinstance(ns, str):
                namespaces = [ns]
            else:
                namespaces = ns  # type: ignore
        else:
            # Use direct parameters
            self._meta_server_address = meta_server_address
            self._app_id = app_id
            self._app_secret = app_secret
            self._cluster = cluster
            self._timeout = timeout
            self._env = env
            self._cycle_time = cycle_time
            self._cache_file_dir_path = cache_file_dir_path
            self.ip = self._get_local_ip_address(ip)
            if namespaces is None:
                namespaces = ["application"]

        # Initialize notification map
        self._notification_map = {namespace: -1 for namespace in namespaces}

        # Initialize other attributes
        self._cache: Dict = {}
        self._hash: Dict = {}
        self._config_server_url = None
        self._config_server_host = None
        self._config_server_port = None

        # Initialize cache directory path if not set
        self._init_cache_file_dir_path(self._cache_file_dir_path)

    @staticmethod
    def _sign_string(string_to_sign: str, secret: str) -> str:
        """
        Sign the string with the secret using HMAC-SHA1.

        Args:
            string_to_sign: The string to sign
            secret: The secret key

        Returns:
            Base64-encoded signature
        """
        signature = hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1).digest()
        return base64.b64encode(signature).decode("utf-8")

    @staticmethod
    def _url_to_path_with_query(url: str) -> str:
        """
        Convert URL to path with query string.

        Args:
            url: The URL to convert

        Returns:
            Path with query string
        """
        parsed = urlparse(url)
        path = parsed.path or "/"
        query = f"?{parsed.query}" if parsed.query else ""
        return path + query

    def _build_http_headers(self, url: str, app_id: str, secret: str) -> Dict[str, str]:
        """
        Build HTTP headers for Apollo API request with signature.

        Args:
            url: The request URL
            app_id: The application ID
            secret: The application secret

        Returns:
            Dictionary of HTTP headers
        """
        import time

        timestamp = str(int(time.time() * 1000))
        path_with_query = self._url_to_path_with_query(url)
        string_to_sign = f"{timestamp}\n{path_with_query}"
        signature = self._sign_string(string_to_sign, secret)

        AUTHORIZATION_FORMAT = "Apollo {}:{}"
        HTTP_HEADER_AUTHORIZATION = "Authorization"
        HTTP_HEADER_TIMESTAMP = "Timestamp"

        return {
            HTTP_HEADER_AUTHORIZATION: AUTHORIZATION_FORMAT.format(app_id, signature),
            HTTP_HEADER_TIMESTAMP: timestamp,
        }

    @staticmethod
    def _get_local_ip_address(ip: Optional[str]) -> str:
        """
        Get the local IP address for this machine.

        Args:
            ip: Explicit IP address to use, or None to detect

        Returns:
            IP address as string
        """
        if ip is not None:
            return ip
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 53))
            local_ip: str = s.getsockname()[0]
            s.close()
            return local_ip
        except BaseException:
            return "127.0.0.1"

    def _init_cache_file_dir_path(self, cache_file_dir_path: Optional[str] = None) -> None:
        """
        Initialize the cache file directory path.

        Args:
            cache_file_dir_path: Custom cache file directory path
        """
        if cache_file_dir_path is None and self._cache_file_dir_path is None:
            self._cache_file_dir_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config")
        elif cache_file_dir_path is not None:
            self._cache_file_dir_path = cache_file_dir_path

        # Ensure the cache directory exists
        if self._cache_file_dir_path and not os.path.isdir(self._cache_file_dir_path):
            os.makedirs(self._cache_file_dir_path, exist_ok=True)

    @staticmethod
    def _write_file(file_path: str, content: str) -> None:
        """
        Write content to file synchronously.

        Args:
            file_path: Path to the file
            content: Content to write
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _read_file(file_path: str) -> str:
        """
        Read content from file synchronously.

        Args:
            file_path: Path to the file

        Returns:
            File content as string
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
