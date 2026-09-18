# LLMfy

![](assets/logo-horizontal.webp)

<a href="https://img.shields.io/github/actions/workflow/status/irufano/llmfy/release.yml">![llmfy](https://img.shields.io/github/actions/workflow/status/irufano/llmfy/release.yml?style=for-the-badge&logo=pypi&logoColor=blue&label=publish
)</a>
<a href="https://pypi.org/project/llmfy/0.11.0">![llmfy](https://img.shields.io/badge/llmfy-v0.11.0-31CA9C.svg?style=for-the-badge&logo=pypi&logoColor=yellow)</a>
<a href="https://pypi.org/project/llmfy/">![llmfy](https://img.shields.io/pypi/v/llmfy?style=for-the-badge&label=latest&labelColor=691DC6&color=B77309)</a>
<a href="">![python](https://img.shields.io/badge/python->=3.11-4392FF.svg?style=for-the-badge&logo=python&logoColor=4392FF)</a>


`LLMfy` is a flexible and developer-friendly framework designed to streamline the creation of applications powered by large language models (LLMs). It provides essential tools and abstractions that simplify the integration, orchestration, and management of LLMs across various use cases, enabling developers to focus on building intelligent, context-aware solutions without getting bogged down in low-level model handling. With support for modular components, prompt engineering, and extensibility, LLMfy accelerates the development of AI-driven applications from prototyping to production.

```python linenums="1" hl_lines="11 12 14"
def example():
    info = """
	LLMfy is framework for integrating LLM-powered applications.
	"""

    SYSTEM_PROMPT = """
    Answer any user questions based on the data:
    {{info}}
    Answer only relevant questions, otherwise, say I don't know."""

    llm = BedrockConverseModel(model="amazon.nova-lite-v1:0", config=BedrockConverseConfig(temperature=0.7))
    framework = LLMfy(llm, system_message=SYSTEM_PROMPT, input_variables=["info"])
    content = "What is LLMfy?"
    response = framework.invoke(content, info=info)
    print(f">> {response.result.content}\n")
```

