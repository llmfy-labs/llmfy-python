# How to install

## Installation
=== "UV"

    ```shell
    uv add llmfy
    ```

=== "pip"

    ```shell
    pip install llmfy
    ```

## Prerequisites Package
 
| Package                                                          | Required/Optional | Purpose                                                 | How to Install                          |
| ---------------------------------------------------------------- | ----------------- | ------------------------------------------------------- | --------------------------------------- |
| [pydantic](https://pypi.org/project/pydantic)                    | ✅ Required        | Core dependency       | Auto-installed with `llmfy`             |
| [anthropic](https://pypi.org/project/anthropic)                  | 🔸 Optional        | Use Anthropic Claude models (native Messages API)       | `pip install "llmfy[anthropic]"`        |
| [openai](https://pypi.org/project/openai)                        | 🔸 Optional        | Use OpenAI models                                       | `pip install "llmfy[openai]"`           |
| [boto3](https://pypi.org/project/boto3/)                         | 🔸 Optional        | Use AWS Bedrock models                                  | `pip install "llmfy[boto3]"`            |
| [google-genai](https://pypi.org/project/google-genai/)           | 🔸 Optional        | Use Google AI (Gemini) models                           | `pip install "llmfy[google-genai]"`     |
| [numpy](https://pypi.org/project/numpy/)                         | 🔸 Optional        | Use Embedding                                            | `pip install "llmfy[numpy]"`            |
| [typing_extensions](https://pypi.org/project/typing-extensions/) | 🔸 Optional        | Use state in `FlowEngine`                               | `pip install "llmfy[typing_extensions]"` |
| [redis](https://pypi.org/project/redis/)                         | 🔸 Optional        | Use `RedisCheckpointer`                                 | `pip install "llmfy[redis]"`            |
| [SQLAlchemy](https://pypi.org/project/SQLAlchemy/)               | 🔸 Optional        | Use `SQLCheckpointer` (supports sync and async drivers) | `pip install "llmfy[SQLAlchemy]"`       |

- SQLCheckpointer

    | Package                                          | Required/Optional | Purpose                                       | How to Install              |
    | ------------------------------------------------ | ----------------- | --------------------------------------------- | --------------------------- |
    | [asyncpg](https://pypi.org/project/asyncpg/)     | 🔸 Optional        | PostgreSQL async driver for `SQLCheckpointer` | `pip install asyncpg`       |
    | [psycopg2](https://pypi.org/project/psycopg2/)   | 🔸 Optional        | PostgreSQL sync driver for `SQLCheckpointer`  | `pip install psycopg2`      |
    | [aiomysql](https://pypi.org/project/aiomysql/)   | 🔸 Optional        | MySQL async driver for `SQLCheckpointer`      | `pip install aiomysql`      |
    | [pymysql](https://pypi.org/project/PyMySQL/)     | 🔸 Optional        | MySQL sync driver for `SQLCheckpointer`       | `pip install pymysql`       |
    | [aiosqlite](https://pypi.org/project/aiosqlite/) | 🔸 Optional        | SQLite async driver for `SQLCheckpointer`     | `pip install aiosqlite`     |

    Any async driver also needs SQLAlchemy's asyncio extension: `pip install "sqlalchemy[asyncio]"` (pulls in `greenlet`, no longer installed by default since SQLAlchemy 2.1). Sync drivers don't need it.


## Provider-specific extras

Install only the provider packages you need:

=== "UV"

    ```shell
    # Anthropic (Claude, native Messages API) support
    uv add "llmfy[anthropic]"

    # OpenAI support
    uv add "llmfy[openai]"

    # AWS Bedrock support
    uv add "llmfy[boto3]"

    # Google AI (Gemini) support
    uv add "llmfy[google-genai]"

    # Install all optional dependencies
    uv add "llmfy[all]"
    ```

=== "pip"

    ```shell
    # Anthropic (Claude, native Messages API) support
    pip install "llmfy[anthropic]"

    # OpenAI support
    pip install "llmfy[openai]"

    # AWS Bedrock support
    pip install "llmfy[boto3]"

    # Google AI (Gemini) support
    pip install "llmfy[google-genai]"

    # Install all optional dependencies
    pip install "llmfy[all]"
    ```
