import json

from deepre.utils.logger import logger


def validate_json(data: str | dict) -> bool | dict:
    if isinstance(data, dict):
        return data
    try:
        data = json.loads(data)
    except json.JSONDecodeError:
        return False
    return data


def get_client_json(data: dict):
    return json.dumps(data, indent=2)


def set_client_json(self, field: str, data: str):
    try:
        # Parse the JSON string to validate and convert to dict
        json_data = json.loads(data)
        if not isinstance(json_data, dict):
            logger.error("Invalid format: JSON must be an object/dictionary")
            return

        setattr(self, field, json_data)
        logger.info(f"Successfully updated config: {json_data}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")


def json_property(field: str):
    def getter(self):
        return json.dumps(getattr(self, field), indent=2)

    def setter(self, value: str):
        try:
            json_data = json.loads(value)
            if not isinstance(json_data, dict):
                logger.error("Invalid format: JSON must be an object/dictionary")
                return
            setattr(self, field, json_data)
            logger.info(f"Successfully updated config: {json_data}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")

    return property(getter, setter)


def make_json_property(field: str):
    def getter(self):
        return json.dumps(getattr(self, field), indent=2)

    def setter(self, value: str):
        try:
            json_data = json.loads(value)
            if not isinstance(json_data, dict):
                return
            setattr(self.configs, field, json_data)
            setattr(self.clients, field, LLMProvider["ollama"](**json_data))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    return property(getter, setter)
