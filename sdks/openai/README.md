# MeshPool OpenAI-compatible SDK usage

MeshPool does not require a custom SDK for the core inference path. Use the
official OpenAI clients unchanged and point `base_url` at the MeshPool control
plane `/v1` endpoint.

## Python

```python
from openai import OpenAI

client = OpenAI(
    api_key="your MeshPool session token",
    base_url="http://127.0.0.1:8080/v1",
)

stream = client.chat.completions.create(
    model="nano-gpt",
    messages=[{"role": "user", "content": "Hello from MeshPool"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Create a local session token with:

```powershell
meshpool login
```

Supported OpenAI-compatible endpoints in this build:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Other OpenAI API resources are not implemented by MeshPool yet.
