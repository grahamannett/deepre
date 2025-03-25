#!/usr/bin/env python3
"""
Ollama API utility script for managing and extending models.

This script provides functionality to create extended context models,
retrieve model information, and verify model extensions through Ollama's API.
Uses a functional approach for easy embedding in other code.
"""

import argparse
import json
import sys
from functools import wraps
from typing import Any, Callable, TypeVar

import httpx

from deepre.app_config import DEFAULT_TIMEOUT, OLLAMA_URL

OLLAMA_API_URL = f"{OLLAMA_URL}/api"

# Type variables for command handling
T = TypeVar("T")
CommandFunction = Callable[..., T]

# Command registry
commands: dict[str, tuple[CommandFunction, str]] = {}


def command(name: str, help_text: str = "") -> Callable[[CommandFunction], CommandFunction]:
    """
    Decorator to register a command.

    Args:
        name: Name of the command
        help_text: Help text for the command

    Returns:
        Decorated function that registers the command
    """

    def decorator(func: CommandFunction) -> CommandFunction:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        commands[name] = (wrapper, help_text)
        return wrapper

    return decorator


def make_api_request(
    endpoint: str,
    payload: dict[str, Any],
    base_url: str = OLLAMA_API_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Make a request to the Ollama API.

    Args:
        endpoint: API endpoint to call
        payload: Request payload
        base_url: Base URL for the Ollama API
        timeout: Request timeout in seconds

    Returns:
        Response JSON as dictionary

    Raises:
        httpx.HTTPError: If the request fails
        ValueError: If response is not valid JSON
    """
    url = f"{base_url}/{endpoint}"

    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        print(f"HTTP request failed: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"Failed to decode response as JSON: {e}")
        raise ValueError("Invalid JSON response from API") from e


@command("create", "Create a new model with extended context")
def create_model(model_name: str, new_model: str, num_ctx: int, base_url: str = OLLAMA_API_URL) -> bool:
    """
    Create a new model based on an existing one with a specified context size.

    Args:
        model_name: Source model name
        new_model: New model name
        num_ctx: Context size for the new model
        base_url: Base URL for the Ollama API

    Returns:
        True if model creation was successful
    """
    payload = {
        "model": new_model,
        "from": model_name,
        "parameters": {"num_ctx": num_ctx},
    }

    try:
        make_api_request("create", payload, base_url)
        print(f"Model creation initiated successfully: {new_model} (based on {model_name}, ctx={num_ctx})")
        return True
    except (httpx.HTTPError, ValueError):
        return False


def get_model_info(model_name: str, base_url: str = OLLAMA_API_URL) -> dict[str, Any]:
    """
    Get information about a model.

    Args:
        model_name: Name of the model
        base_url: Base URL for the Ollama API

    Returns:
        Dictionary containing model information

    Raises:
        ValueError: If model information cannot be retrieved
    """
    payload = {"model": model_name}
    return make_api_request("show", payload, base_url)


def get_model_params(model_name: str, base_url: str = OLLAMA_API_URL) -> dict[str, str | list[str]]:
    """
    Get model parameters.

    Args:
        model_name: Name of the model
        base_url: Base URL for the Ollama API

    Returns:
        Dictionary containing model parameters

    Raises:
        ValueError: If parameters cannot be retrieved or are invalid
    """
    model_info = get_model_info(model_name, base_url)
    if not model_info:
        raise ValueError(f"Failed to retrieve information for model {model_name}")

    params = parse_parameters(model_info.get("parameters", ""))
    num_ctx = params.get("num_ctx")
    if not num_ctx or (isinstance(num_ctx, list)):
        raise ValueError(f"Context size (num_ctx) not found or invalid for model {model_name}")

    return params


@command("info", "Show model information")
def show_model_info(model_name: str, base_url: str = OLLAMA_API_URL) -> bool:
    """Print model parameters and return True if successful."""
    try:
        params = get_model_params(model_name, base_url)
        print(json.dumps(params, indent=2))
        return True
    except ValueError as e:
        print(f"Error: {e}")
        return False


def parse_parameters(params_str: str) -> dict[str, str | list[str]]:
    """
    Parse parameter string into a dictionary.

    Args:
        params_str: Parameter string in format "key1 value1 key2 value2..."
                   For repeated keys, values are collected into a list.

    Returns:
        Dictionary mapping parameter keys to either a single value string
        or a list of strings for repeated keys.

    Example:
        >>> parse_parameters("stop <|eot_id|> stop <|stop|> num_ctx 32768")
        {'stop': ['<|eot_id|>', '<|stop|>'], 'num_ctx': '32768'}
    """
    if not isinstance(params_str, str):
        return {}

    # Split parameters into pairs
    params = params_str.split()
    if len(params) % 2 != 0:
        return {}

    # Group values by key
    result: dict[str, str | list[str]] = {}
    for key, value in zip(params[::2], params[1::2]):
        if key in result:
            current_value = result[key]
            if isinstance(current_value, list):
                current_value.append(value)
            else:
                result[key] = [current_value, value]
        else:
            # First occurrence of key
            result[key] = value

    return result


@command("check", "Check if model has extended context")
def is_model_extended(model_name: str, target_ctx: int = 131072, base_url: str = OLLAMA_API_URL) -> bool:
    """
    Check if a model has been extended beyond a target context size.

    Args:
        model_name: Name of the model to check
        target_ctx: Target context size to compare against
        base_url: Base URL for the Ollama API

    Returns:
        True if model's context size exceeds the target, False otherwise
    """
    try:
        model_info = get_model_info(model_name, base_url)
        params = parse_parameters(model_info.get("parameters", ""))
        num_ctx = params.get("num_ctx")
        if not num_ctx or isinstance(num_ctx, list):
            return False
        ctx_value = int(num_ctx)
        print(f"Model {model_name} extended: {ctx_value > target_ctx}")
        return ctx_value > target_ctx
    except (ValueError, TypeError):
        print(f"Failed to check context size for model {model_name}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ollama model management utility")

    parser.add_argument(
        "command",
        choices=list(commands.keys()),
        help="Command to execute: " + ", ".join(f"{cmd} ({help})" for cmd, (_, help) in commands.items()),
    )

    parser.add_argument(
        "--base-model",
        default="llama3.2:latest",
        help="Base model name (default: llama3.2:latest)",
    )

    parser.add_argument(
        "--new-model",
        default="llama3.2:latest-extended",
        help="New model name (default: llama3.2:latest-extended)",
    )

    parser.add_argument("--ctx", type=int, default=64000, help="Context size for new model (default: 64000)")

    parser.add_argument(
        "--base-url", default=OLLAMA_API_URL, help=f"Base URL for Ollama API (default: {OLLAMA_API_URL})"
    )

    return parser.parse_args()


def main() -> int:
    """
    Parse command line arguments and execute the specified command.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_args()

    # Get the command function from the registry
    command_func = commands[args.command][0]

    # Execute command with appropriate arguments
    if args.command == "create":
        result = command_func(args.base_model, args.new_model, args.ctx, args.base_url)
    else:
        result = command_func(args.base_model, args.base_url)

    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
