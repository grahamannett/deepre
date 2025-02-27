# DeepRe Tests

This directory contains tests for the DeepRe project.

## WebServices Tests (`test_web_services.py`)

The `test_web_services.py` file contains tests for the caching functionality of the `WebServices` class. The tests verify that:

1. Search results and page content are correctly cached
2. Cached content is retrieved instead of making API calls when available
3. The JSON response from search requests is correctly converted to `PageResult` objects
4. The `fetch_page_result` method correctly updates `PageResult` objects with page content

### Test Cases

- `test_perform_search_with_cache_not_cached`: Verifies that search results are fetched from the API when not in cache
- `test_perform_search_with_cache_cached`: Verifies that cached search results are used when available
- `test_fetch_page_with_cache_not_cached`: Verifies that page content is fetched from the API when not in cache
- `test_fetch_page_with_cache_cached`: Verifies that cached page content is used when available
- `test_get_search_results`: Verifies that API responses are correctly converted to `PageResult` objects
- `test_fetch_page_result`: Verifies that `PageResult` objects are correctly updated with page content
- `test_cache_saving_and_loading`: Integration test that verifies real file operations for caching

### Running the Tests

Run the tests with pytest:

```bash
cd deepre
python -m pytest tests/test_web_services.py -v
```

## Mock Fixtures

The tests use several pytest fixtures to set up the testing environment:

- `mock_cache_dir`: Creates a temporary directory for cache testing
- `web_services`: Creates a `WebServices` instance configured for testing
- `mock_client`: Creates a mock HTTP client that returns predefined responses

The tests use the `unittest.mock` library to patch methods like `get_from_cache` and `save_to_cache` to simulate cache hits and misses without actually reading from or writing to the file system in most tests.

## Integration Test

The final test, `test_cache_saving_and_loading`, performs actual file operations to test the end-to-end caching functionality. It creates a real cache file and verifies that it's correctly loaded when the same search query is made again.
