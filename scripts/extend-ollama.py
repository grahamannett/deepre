import httpx

base_url = "http://localhost:11434/api"

"""
this whole script is going to be redone into actual functionality of the system later (based on the ollama provider),
just including it for now for posterity
"""


def create_model(base_model: str, new_model: str, num_ctx: int):
    url = f"{base_url}/create"
    payload = {
        "model": new_model,
        "from": base_model,
        "parameters": {"num_ctx": num_ctx},
    }

    response = httpx.post(url, json=payload)

    if response.status_code == 200:
        print("Model creation initiated successfully.")
    else:
        print(f"Error: {response.status_code}, {response.text}")


def get_model_info(model_name: str):
    url = f"{base_url}/show"
    payload = {"model": model_name}

    response = httpx.post(url, json=payload)

    if response.status_code == 200:
        model_info = response.json()
        num_ctx = model_info.get("parameters", {}).get("num_ctx", "Not available")
        print(f"num_ctx for {model_name}: {num_ctx}")
    else:
        print(f"Error: {response.status_code}, {response.text}")


_cmds = {
    "create_model": create_model,
    "get_model_info": get_model_info,
}

if __name__ == "__main__":
    NUM_CTX = 64_000
    CMD = "create_model"
    BASE_MODEL = "llama3.2:latest"
    NEW_MODEL = "llama3.2:latest-extended"

    _cmds[CMD](BASE_MODEL, NEW_MODEL, NUM_CTX)
