---
title: Langchain Quicks Starters (v1)
description: Quick Starter Guide for the new version of Langchain (v1).
tags:
  - Basics
  - Langchain
  - Agents
  - Prompts
  - Memory
  - Context
  - Tool
  - Model
  - ResponseFormatter
  - StructuredOutput
date: 2025-12-01
---
Modern AI agents are no longer simple text generators. They maintain state, call external tools, accept runtime context and return structured response that the downstream process can trust. In this blog, we'll build a small but complete example using the latest version of *Langchain's* components.
The agent that we'll be building will act as a pun-loving weather forecaster capable of determining a user's location and returning schema-validated weather information.

---
# Weather Forecaster Agent
## System Prompt
The very first thing we are going to define is the `SYSTEM_PROMPT`. The prompt defines the agent's role, it's behavior, and the rules for when the tool must be used. It is the first major control point in an LLM-based agent. Remember to keep it specific and actionable.

**Code**:
```python
SYSTEM_PROMPT = """
    You are an expert weather forecaster, who speaks in puns.

    You have access to two tools:
        - get_weather_for_location: use this to get the weather for a specific location.
        - get_user_location: use this to get the user's locatio.

    If a user asks you for the weather, make sure you knkow the location. If you can tell from the question that they wherever they are, use the get_user_location tool to find their location.
"""
```

---
## Context
`Context` is the `dataclass` that you pass into `agent.invoke(...)` every time you call the agent. It represents the runtime information for that specific request, similar to a request envelope in an API call. It usually contains things like:
- `user_id`: which user the request belongs to
- `request_id`: a tracking identifier
- etc...
This is *NOT* the agent's memory. Context doesn't automatically carry over requests; you supply it each time. If you want some parts of it to persist, you have to explicitly store them in memory or another backend.

**Code**:
```python
from dataclasses import dataclass

@dataclass
class Context:
    """Schema describing per-request runtime context."""
    user_id: str
```

---
## Tools
Tools let a agent interact with external systems by calling functions you define. Registering a function with the `@tool` gives the agent metadata it can use to decide when/what to call. 

**Code**:
```python
from langchain.tools import ToolRuntime, tool

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user Id."""
    user_id = runtime.context.user_id  # access context securely, via ToolRuntime
    return "Florida" if user_id == "1" else "SF"
```
`ToolRuntime` is the execution wrapper that LangChain injects into tools where they are executed. It provides:
- `runtime.context`: access to *Context* dataclass you pass at invocation.
- Hooks to read/write checkpoints or logs.
- Execution metadata (who called, timestamps, etc.) in some implementations.
Tools should not pull sensitive metadata from prompts. Use `ToolRuntime` to access request-scoped data (auth, user-id).

---
## Model
I use the [Groq](https://console.groq.com/home) platform to access APIs for several different LLM providers. One practical advantage is that Groq offers a free tier, which makes development and testing much easier. To integrate Groq models with LangChain, I use the `langchain-groq` package. Just make sure to place your `GROQ_API_KEY` in a `.env` file so it can be loaded securely at runtime.

**Code**:
```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

# Loading the environment variables from .env file.
load_dotenv()

# Setting up the language model with the right parameters
base_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0, max_retries=3)
```

---
## Memory (Checkpointing)
Let's use `InMemorySaver` to persist conversation state across calls.
- It stores agent state in RAM, keyed by `thread_id`.
- When the same thread ID is used again, previous state is restored.
- This allows multi-turn dialogue with memory.
- The data disappears when the process is complete.

**Code**:
```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
```

---
## Response Formatter (Structured Output)
This is where `ToolStrategy` and `dataclass` schema come in.
- A dataclass defines the schema the agent must follow.
- `ToolStrategy(ResponseFormat)` enforces JSON output that matches the schema.
- Without `ToolStrategy`, the LLM would produce raw text, inconsistent fields or malformed JSON.

**Code**:
```python
from dataclasses import dataclass
from langchain.agents.structured_output import ToolStrategy

@dataclass
class ResponseFormat:
    """Structured response returned by the agent."""
    punny_response: str
    weather_conditions: str | None = None

response_format = ToolStrategy(ResponseFormat)
```

---
## Create & Run the Agent
`create_agent(...)` wires together model, system prompt, tools, response formatting strategy, context schema, and checkpointer. When invoked:
1. The runtime composes system prompt + conversation messages.
2. The model generates output or an action (e.g., call a tool).
3. If a tool is required, the agent executes the tool and provides `ToolRuntime` (so the tool sees `runtime.context`).
4. The model is instructed (by ToolStrategy) to return JSON matching `ResponseFormat`.
5. LangChain parses & validates the JSON and returns the parsed dataclass in `response["structured_response"]`.
6. The checkpointer saves a snapshot keyed by `thread_id`.

**Code**:
```python
from langchain.agents import create_agent

agent = create_agent(
    model=base_model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=response_format,
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "1"}}

# First turn
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather outside"}]},
    config=config,
    context=Context(user_id="1"),
)

print(response["structured_response"])

# Second turn (same thread_id → memory persists)
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1"),
)

print(response["structured_response"])
```

---

### Full Code
```python
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime, tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

# Loading the environment variables
load_dotenv()

# Checkpointer
checkpointer = InMemorySaver()

# Model Initialization
base_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0, max_retries=3)

# System Prompt
SYSTEM_PROMPT = """
    You are an expert weather forecaster, who speaks in puns.
    You have access to two tools:
        - get_weather_for_location: use this to get the weather for a specific location.
        - get_user_location: use this to get the user's locatio.
    If a user asks you for the weather, make sure you knkow the location. If you can tell from the question that they wherever they are, use the get_user_location tool to find their location.
"""

# Runtime context
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# Structured response format.
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always reqeuired)
    punny_response: str
    # Any interesting information about the weather if required.
    weather_conditions: str | None = None

# Tools 
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user Id."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

# Agent
agent = create_agent(
    model=base_model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer,
)

# thread_id is a unique identifier for a given conversation
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather outside"}]},
    config=config,
    context=Context(user_id="1"),
)
print(response["structured_response"])

# continuing the conversation with the same thread_id
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1"),
)
print(response["structured_response"])
```

---
# Final Thoughts
With LangChain, Groq, and structured output enforcement, you can build powerful, stateful, tool-driven AI agents that behave predictably and integrate cleanly into larger systems.
Key takeaways:
- **System Prompt** defines behavior
- **Tools** extend capabilities
- **Model** handles generation
- **ResponseFormat + ToolStrategy** enforce strict output schemas
- **Memory** preserves state across turns
- **create_agent** wires everything together
	This foundation is enough to build production-grade agents that drive workflows, applications, or automated assistants.