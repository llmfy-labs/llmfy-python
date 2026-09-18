![](LLMFY-LOGO.webp)

<div align="center">

  <a href="https://img.shields.io/github/actions/workflow/status/llmfy-labs/llmfy-python/release.yml">![llmfy](https://img.shields.io/github/actions/workflow/status/llmfy-labs/llmfy-python/release.yml?style=for-the-badge&logo=pypi&logoColor=blue&label=publish
  )</a>
  <a href="https://pypi.org/project/llmfy/0.11.0">![llmfy](https://img.shields.io/badge/llmfy-v0.11.0-31CA9C.svg?style=for-the-badge&logo=pypi&logoColor=yellow)</a>
  <a href="https://pypi.org/project/llmfy/">![llmfy](https://img.shields.io/pypi/v/llmfy?style=for-the-badge&label=latest&labelColor=691DC6&color=B77309)</a>
  <a href="">![python](https://img.shields.io/badge/python->=3.11-4392FF.svg?style=for-the-badge&logo=python&logoColor=4392FF)</a>

</div>

`LLMfy` is a flexible and developer-friendly framework designed to streamline the creation of applications powered by large language models (LLMs). It provides essential tools and abstractions that simplify the integration, orchestration, and management of LLMs across various use cases, enabling developers to focus on building intelligent, context-aware solutions without getting bogged down in low-level model handling. With support for modular components, prompt engineering, and extensibility, LLMfy accelerates the development of AI-driven applications from prototyping to production.

See complete documentation at [https://llmfy.readthedocs.io/](https://llmfy.readthedocs.io/)

## How to install

- Optional Library:
  - Install [anthropic](https://pypi.org/project/anthropic) to use Anthropic Claude models (native Messages API) — 🔸 optional.
  - Install [openai](https://pypi.org/project/openai) to use OpenAI models — 🔸 optional.
  - Install [boto3](https://pypi.org/project/boto3/) to use AWS Bedrock models — 🔸 optional.
  - Install [google-genai](https://pypi.org/project/google-genai) to use Google AI (Gemini) models — 🔸 optional.
  - Install [numpy](https://pypi.org/project/numpy/) to use Embedding — 🔸 optional.
  - Install [typing_extensions](https://pypi.org/project/typing-extensions/) to use state in `FlowEngine` — 🔸 optional.
  - Install [redis](https://pypi.org/project/redis/) to use `RedisCheckpointer` — 🔸 optional.
  - Install [SQLAlchemy](https://pypi.org/project/SQLAlchemy/) to use `SQLCheckpointer` — 🔸 optional. `SQLCheckpointer` supports both sync and async drivers for multiple databases:
      - PostgreSQL (async: [asyncpg](https://pypi.org/project/asyncpg/), sync: [psycopg2](https://pypi.org/project/psycopg2/)) — 🔸 optional.
      - MySQL (async: [aiomysql](https://pypi.org/project/aiomysql/), sync: [pymysql](https://pypi.org/project/PyMySQL/)) — 🔸 optional.
      - SQLite (async: [aiosqlite](https://pypi.org/project/aiosqlite/), sync: built-in) — 🔸 optional.

### Using UV
```sh
uv add llmfy
```

### Using pip
```sh
pip install llmfy
```

### Using github 

#### From a specific branch
```sh
# main
uv add git+https://github.com/llmfy-labs/llmfy-python.git@main
# or
pip install git+https://github.com/llmfy-labs/llmfy-python.git@main

# dev
uv add git+https://github.com/llmfy-labs/llmfy-python.git@dev
# or
pip install git+https://github.com/llmfy-labs/llmfy-python.git@dev
```

#### From a tag
```sh
# example tag version 0.4.3
uv add git+https://github.com/llmfy-labs/llmfy-python.git@v0.4.3
# or
pip install git+https://github.com/llmfy-labs/llmfy-python.git@v0.4.3
```

#### Github in requirements.txt

```txt
git+https://github.com/llmfy-labs/llmfy-python.git@dev
```

## How to use
Model class names follow `<Vendor><APIVariant>Model` — e.g. `OpenAIChatModel` vs `OpenAIResponsesModel` for OpenAI's two APIs, `GoogleAIGenerateModel` for Google's `generate_content` API — so the class name always tells you which API it talks to.

### Anthropic models
To use `AnthropicMessagesModel` (native Messages API), requires install `"llmfy[anthropic]"` and add below config to your env (or pass `api_key=` to the model instead):
- `ANTHROPIC_API_KEY`

### OpenAI models
To use `OpenAIChatModel` (Chat Completions) or `OpenAIResponsesModel` (Responses API), requires install `"llmfy[openai]"` and add below config to your env (or pass `api_key=` to the model instead):
- `OPENAI_API_KEY`

### AWS Bedrock models
To use `BedrockConverseModel`, requires install `"llmfy[boto3]"` and add below config to your env (or pass `aws_access_key_id=`, `aws_secret_access_key=`, `aws_bedrock_region=` to `BedrockConverseModel` instead):
- `AWS_ACCESS_KEY_ID` 
- `AWS_SECRET_ACCESS_KEY` 
- `AWS_BEDROCK_REGION`

### Google AI models
To use `GoogleAIGenerateModel`, requires install `"llmfy[google-genai]"` and add below config to your env (or pass `api_key=` to `GoogleAIGenerateModel` instead):
- `GOOGLE_API_KEY`

## Example
### LLMfy Example
```python
from llmfy import (
    OpenAIChatModel,
    OpenAIChatConfig,
    LLMfy,
    Message,
    Role,
    LLMfyException,
)

def sample_prompt():
    info = """Irufano adalah seorang software engineer.
    Dia berasal dari Indonesia.
    Kamu bisa mengunjungi websitenya di https:://irufano.github.io"""

    # Configuration
    config = OpenAIChatConfig(temperature=0.7)
    llm = OpenAIChatModel(model="gpt-4o-mini", config=config)

    SYSTEM_PROMPT = """Answer any user questions based solely on the data below:
    <data>
    {info}
    </data>
    
    DO NOT response outside context."""

    # Initialize framework
    framework = LLMfy(llm, system_message=SYSTEM_PROMPT, input_variables=["info"])

    try:
        messages = [Message(role=Role.USER, content="apa ibukota china")]
       
        response = framework.invoke(messages, info=info)
        print(f"\n>> {response.result.content}\n")

    except LLMfyException as e:
        print(f"{e}")


if __name__ == "__main__":
    sample_prompt()
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit message format, the automatic version-bump/release process, local package/docs development commands, and the required testing policy for every change.