"""Exception types for benchmark runner."""


class FatalBenchmarkError(Exception):
    """Fatal error that should stop the entire benchmark run immediately.

    Used for configuration errors, missing API keys, and other issues
    that cannot be resolved by retrying. Raise this error to stop
    the entire benchmark run immediately.
    """

    pass


class NonRetryableError(Exception):
    """Error originating from our code that should NOT trigger API retries.

    Used for errors in post-processing (JSON parsing, response extraction, etc.)
    where the API call succeeded but our code failed to process the response.
    Retrying would waste API calls and money since the original response was valid.

    Examples:
        - JSON parsing failures from extract_json_from_response()
        - Response format validation errors
        - Data extraction errors from valid API responses
    """

    pass
