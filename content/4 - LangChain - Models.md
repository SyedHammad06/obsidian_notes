---
title: 4 - LangChain - Models
description: A compact guide to using LangChain chat models, covering invocation, streaming, structured outputs, and advanced configuration.
tags:
  - Model
  - ModelParameters
  - Invoke
  - Stream
  - Batch
  - WithStructuredOutput
  - ModelProfiles
  - LocalModels
  - RateLimiter
  - BaseURL
  - TokenUsage
  - InvocationConfig
  - ConfigurableModels
date: 2025-12-05
---
Large Language Models (LLMs) act as the core reasoning engine behind most modern AI applications. They can generate and interpret natural language, move between languages, summarize content, and answer questions - all without custom training for each task. 

Their versatility comes from the broad set of capabilities they support, including:
- *Tool calling* - letting models trigger external systems such as APIs or databases.
- *Structured outputs* - returning responses that follow a well-defined schema.
- *Multimodal inputs/outputs* - working with text, audio, images, or video.
- *Reasoning* - performing multi-step analysis before producing a final answer.

Since models guide how agents think, plan, and react, choosing the right model directly affects the reliability and performance of your application. Some models are optimized for reasoning, others for instruction-following, and some for handling much larger context windows.

LangChain defines a unified model interface, making it easy to switch providers and experiment across many different engines without rewriting your application logic.

---
# Basic Usage

Models in LangChain can operate in two modes:
1. **With agent** : where they guide tool selection and decision-making.
2. **Standalone**: where you call them directly for tasks like text generation or classification.
Both modes use the **same interface**, so you can begin simple and scale to more complex behavior later.
## Initialize a model

The simplest way to get started is by using `init_chat_model`, which constructs a chat model using the provider you choose.

```python
# ! pip install langchain-groq
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Initialization via init_chat_model
model = init_chat_model("groq:llama-3.1-8b-instant")

# Initialization via chat model provider
model = ChatGroq(model="llama-3.1-8b-instant")
```
## Key methods

LangChain chat models expose several important ways to interact with them:
- **invoke** - send one message or a list of messages and get back a complete response.
- **stream** - receive output incrementally as the model generates it.
- **batch** - process multiple independent requests in parallel.

---
# Parameters

Every model accepts a set of configuration parameters. While each provider may offer additional options, the common ones include:
- **model** - the identifier of the model you want to use.
- **api_key** - your authentication token for the provider.
- **temperature** - controls randomness (lower = predictable, higher = creative).
- **max_tokens** - sets the maximum length of the response.
- **timeout** - how long LangChain waits before abandoning the request.
- **max_retries** - how many retries should be attempted after transient failures.

```python
import os

model = init_chat_model(
    model="groq:llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,  # 0 = deterministic, 1 = creative
    max_tokens=1000,
    timeout=30,
    max_retries=3,
    # Provider-specific additional parameters can also be passed
)
```

---
# Invocation

There are three primary ways to invoke a chat model.
## Invoke

The simplest interaction: pass a single string or a chat history.

```python
response = model.invoke("Why do parrots have colorful feathers?")
response.content
```

A list of messages can be provided to a chat model to represent conversation history. Each message has a role that models use to indicate who sent the message in the conversation.

```python
conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant that translates French to English.",
    },
    {
        "role": "user",
        "content": "Translate: J'adore la programmation.",
    },
    {"role": "assistant", "content": "I love programming. "},
    {"role": "user", "content": "Translate: J'adore créer des applications."},
]

response = model.invoke(conversation)
print(response.content)
```
## Stream

Most models can stream their output content while it is being generated. By displaying output progressively, streaming significantly improves user experience, particularly for longer responses.

Streaming provides partial results as they are generated, making the user experience smoother for long responses.

```python
for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="", flush=True)
```

Some models emit richer metadata, for example reasoning traces, tool-call fragments, or text blocks.

```python
for chunk in model.stream("What color is the sky?"):
    for block in chunk.content_blocks:
        if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
            print(f"Reasoning: {reasoning}", end="", flush=True)
        elif block["type"] == "tool_call_chunk":
            print(f"Tool call chunk: {block}", end="", flush=True)
        elif block["type"] == "text":
            print(block["text"], end="", flush=True)
```

LangChain treats streamed fragments as additive: they can be combined to produce the full message.

```python
full = None  # None | AIMessageChunk
for chunk in model.stream("What color is the sky?"):
    full = chunk if full is None else full + chunk
    print(full.text)
```

As opposed to `invoke()`, which returns a single `AIMessage` after the model has finished generating it's full response, `stream()` returns multiple `AIMessageChunks` objects, each containing a portion of the output text. Importantly, each chunk in a stream is designed to be gathered into a full message via summation.

```python
full = None  # None | AIMessageChunk
for chunk in model.stream("What color is the sky?"):
    full = chunk if full is None else full + chunk
    print(full.text)

# The
# The sky
# The sky is
# The sky is typically
# The sky is typically blue
# ...

print(full.content_blocks)
# [{"type": "text", "text": "The sky is typically blue..."}]
```

When you `invoke()` a chat model, LangChain will automatically switch to an internal streaming mode if it detects that you are trying to stream the overall application.

```python
async for event in model.astream_events("Hello"):
    if event["event"] == "on_chat_model_start":
        print(f"Input: {event['data']['input']}")
    elif event["event"] == "on_chat_model_stream":
        print(f"Token: {event['data']['chunk'].text}")
    elif event["event"] == "on_chat_model_end":
        print(f"Full message: {event['data']['output'].text}")
```
## Batch

Batching a collection of independent requests to a model can significantly improve performance and reduce costs, as the processing can be done in parallel.

```python
responses = model.batch(
    [
        "Why do parrots have colorful feathers?",
        "How do airplans fly?",
        "What is quantum computing",
    ]
)
for response in responses:
    print(response)
```

The `batch()` method helps parallelize model calls client side. It is distinct from batch APIs supported by inference providers, such as OpenAI or Anthropic. By default, `batch()` will only return the final output for the entire batch. If you want to receive the output for each individual input as it finishes generating, you can stream results with `batch_as_completed()`.

```python
for response in model.batch_as_completed(
    [
        "Why do parrots have colorful feathers?",
        "How do airplanes fly?",
        "what is quantum comuting",
        "Why is the sky blue in color",
    ]
):
    print(response)
```

When using `batch_as_completed()`, results may arrive out of order. Each includes the input index for matching to reconstruct the original order as needed. When processing a large number of inputs using `batch()` or `batch_as_completed()`, you may want to control the maximum number of parallel calls. This can be done by setting the `max_concurrency` attribute in the `RunnableConfig` dictionary.

```python
model.batch(
    [],  # list of inputs
    config={
        "max_concurrency": 5,  # Limit to 5 parallel calls
    },
)
```

---
# Structured Output (via `with_structured_output`)

Models can produce responses that conform to a specified schema. This is crucial when you need predictable and machine parse-able results. LangChain supports multiple schema types and methods for enforcing structured output.

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide me details about the Inception movie")
response
```

Use `include_raw=True` to get both the parsed output and the raw AI message. *Pydantic* models provide automatic validation, while `TypedDict` and JSON Schema require manual validation.

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide me details about the Inception movie")
response
```

---
# Advanced Topics
## Model profiles

Each chat model can expose metadata about its supported features via `.profile`.  
This information is largely sourced from **models.dev**, supplemented with LangChain-specific details.

```python
model.profile
```
## Local models

LangChain supports running models entirely on your machine. This is beneficial when:
- You require strong data privacy
- You want to run custom fine-tuned models
- You want to avoid cloud inference costs
*Ollama* is one of the easiest ways to run local models.
## Rate Limiting

Many chat model providers impose a limit on the number of invocations that can be made in a given time period. If you hit a rate limit, you will typically receive a rate limit error response from the provider and will need to wait before making more requests.

To help manage rate limits, chat model integrations accept a `rate_limiter` parameter that can be provided during initialization to control the rate at which requests are made. LangChain comes with (an optional) built in `InMemoryRateLimiter`. This limiter is thread safe and can be shared by multiple threads in the same process.

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,  # 5 requests every 10s
    check_every_n_seconds=0.1,  # Assess every 100ms
    max_bucket_size=10,  # Maximum burst size
)

model = init_chat_model(
    model="llama-3.1-8b-instant",
    model_provider="groq",
    rate_limiter=rate_limiter
)
```
## Base URL or proxy

For many chat model integrations, you can configure the base URL for API requests, which allows you to use model providers that have OpenAI-compatibility APIs to use a proxy server.

Many model providers offer OpenAI-compatible APIs (e.g., Together AI, vLLM). You can use `init_chat_model` with these providers by specifying the appropriate `base_url` parameter:

```python
# model = init_chat_model(
#     model="MODEL_NAME",
#     model_provider="openai",
#     base_url="BASE_URL",
#     api_key="YOUR_API_KEY",
# )
```

For deployments requiring HTTP proxies, some model integrations support proxy configuration:

```python
# from langchain_openai import ChatOpenAI

# model = ChatOpenAI(model="gpt-4o", openai_proxy="http://proxy.example.com:8080")
```
## Token usage

A number of model providers return token usage information as part of the invocation response. When available, this information will be included on the `AIMessage` objects produced by the corresponding model. You can track aggregate token counts across models in an application using either a callback or context manager.

```python
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

model_1 = init_chat_model(model="groq:openai/gpt-oss-20b")
model_2 = init_chat_model(model="groq:llama-3.1-8b-instant")

callback = UsageMetadataCallbackHandler()
result_1 = model_1.invoke("Hello", config={"callbacks": [callback]})
result_2 = model_2.invoke("Hello", config={"callbacks": [callback]})
callback.usage_metadata
```
## Invocation config

When invoking a model, you can pass additional configuration through the `config` parameter using a `RunnableConfig` dictionary. This provides run-time control over execution behavior, callbacks and metadata tracking.

```python
response = model.invoke(
    "Tell me a joke",
    config={
        "run_name": "joke_generation",  # Custom name for this run
        "tags": ["humor", "demo"],  # Tags for categorization
        "metadata": {"user_id": "123"},  # Custom metadata
        "callbacks": [],  # Callback handlers
    },
)
response.content
```

These configuration values are particularly useful when:
- Debugging with LangSmith tracing
- Implementing custom logging or monitoring
- Controlling resource usage in production
- Tracking invocations across complex pipelines
## Configurable models

You can also create a runtime-configurable model by specifying configurable fields. If you don’t specify a model value, then *model* and *model_provider* will be configurable by default.

```python
from langchain.chat_models import init_chat_model

configurable_model = init_chat_model(temperature=0)

configurable_model.invoke(
    "what's your name",
    config={
        "configurable": {"model": "groq:llama-3.1-8b-instant"}
    },  # Run with llama-3.1-8b-instant
)
configurable_model.invoke(
    "what's your name",
    config={
        "configurable": {"model": "groq:openai/gpt-oss-20b"}
    },  # Run with openai/gpt-oss-20b
)
```

We can create a configurable model with default model values, specify which parameters are configurable, and add prefixes to configurable params

```python
first_model = init_chat_model(
    model="groq:llama-3.1-8b-instant",  # Default model
    temperature=0,
    configurable_fields=("model", "model_provider", "temperature", "max_tokens"),
    config_prefix="first",  # Useful when you have a chain with multiple models
)

first_model.invoke("what's your name")
```

```python
first_model.invoke(
    "what's your name",
    config={
        "configurable": {
            "first_model": "groq:openai/gpt-oss-20b",
            "first_temperature": 0.5,
            "first_max_tokens": 100,
        }
    },
)
```

---
# Key Takeaways

- LangChain provides a unified interface for interacting with chat models, whether you use them directly or inside agents.
- `invoke`, `stream`, and `batch` cover the core ways of generating responses, each suited to different performance and UX needs.
- Structured output via Pydantic schemas ensures predictable, machine-usable responses.
- Advanced configuration options - rate limiting, proxies, token usage tracking, and runtime-configurable models - make LangChain suitable for both experimentation and production systems.
- Switching between model providers becomes trivial, allowing rapid prototyping and comparison of different LLMs.