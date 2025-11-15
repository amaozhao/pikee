import json
import time
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pikee.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)


class BaseLLMClient(object):
    NAME = "BaseLLMClient"

    def __init__(
        self,
        location: Optional[str] = None,
        auto_dump: bool = True,
        max_attempt: int = 5,
        exponential_backoff_factor: Optional[int] = None,
        unit_wait_time: int = 60,
        **kwargs,
    ) -> None:
        self._cache_auto_dump: bool = auto_dump
        if location is not None:
            self.update_cache_location(location)

        self._max_attempt: int = max_attempt
        assert max_attempt >= 1, f"max_attempt should be no less than 1 (but {max_attempt} was given)!"

        self._exponential_backoff_factor: Optional[int] = exponential_backoff_factor
        self._unit_wait_time: int = unit_wait_time
        if self._exponential_backoff_factor is None:
            assert self._unit_wait_time > 0, (
                f"unit_wait_time should be positive (but {unit_wait_time} was given) "
                f"if exponential backoff is disabled ({exponential_backoff_factor} was given)!"
            )
        else:
            assert isinstance(exponential_backoff_factor, int) and exponential_backoff_factor > 1, (
                "To enable the exponential backoff mode, the factor should be greater than 1 "
                f"(but {exponential_backoff_factor} was given)!"
            )

    def warning(self, warning_message: str) -> None:
        if logger:
            print(warning_message)
        return

    def debug(self, debug_message: str) -> None:
        if logger:
            logger.debug(msg=debug_message)
        return

    def _wait(self, num_attempt: int, wait_time: Optional[int] = None) -> None:
        if wait_time is None:
            if self._exponential_backoff_factor is None:
                wait_time = self._unit_wait_time * num_attempt
            else:
                wait_time = self._exponential_backoff_factor**num_attempt

        time.sleep(wait_time)  # type: ignore
        return

    def _generate_cache_key(self, messages: List[dict], llm_config: dict) -> str:
        assert isinstance(messages, List) and len(messages) > 0

        if isinstance(messages[0], Dict):
            return json.dumps((messages, llm_config))

        else:
            raise ValueError(f"Messages with unsupported type: {type(messages[0])}")

    def generate_content_with_messages(self, messages: List[dict], **llm_config) -> str:
        # TODO: utilize self.llm_config if None provided in call.
        # TODO: add functions to get tokens, logprobs.
        start_time = time.time()
        response = self._get_response_with_messages(messages, **llm_config)

        if logger:
            time_used = time.time() - start_time
            result = "receive response" if response is not None else "request failed"
            logger.debug(msg=f"{datetime.now()} {result}, time spent: {time_used} s.")

        if response is None:
            self.warning("None returned as response")
            if messages is not None and len(messages) >= 1:
                self.debug(f"  -- Last message: {messages[-1]}")
            content = ""
        else:
            content = self._get_content_from_response(response, messages=messages)

        return content

    @abstractmethod
    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _get_content_from_response(self, response: Any, messages: List[dict] = [{}]) -> str:
        raise NotImplementedError

    def update_cache_location(self, new_location: str) -> None:
        assert new_location is not None, "A valid cache location must be provided"

        self._cache_location = new_location

    def close(self):
        """Close the active memory, connections, ...
        The client would not be usable after this operation."""
