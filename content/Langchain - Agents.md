---
title: Langchain - Agents
description: This blog discusses about one of the core components of Langchain, that being Agents.
tags:
  - Agents
  - Tool
  - Memory
  - StructuredOutput
  - Langchain
  - Prompts
  - Model
  - ResponseFormatter
  - DynamicModel
  - ToolErrorHandling
  - DynamicSystemPrompt
date: 2025-12-03
---
Agents combine large language models with tools to reason, act, and iterate until they reach a goal. This blog walks through the core pieces - model choices, tools, system prompts, middleware, etc.
![[agent flow.png]]
An agent created using `create_agent` is a runtime that runs an LLM in a loop, alternating between short reasoning steps and targeted tool calls (the ReAct pattern) until it either emits a final answer or reaches an iteration limit. 
The runtime is graph-based under the hood (Langgraph): nodes represent steps (model calls, tool execution, middleware) and edges represent how data flows between steps.

---
# Model

The model is the heart of an agent. It's the reasoning engine. Langchain support multiple ways to wire up models.
## Static model

Static models are configured once when creating the agent and remain unchanged throughout the execution. This is the most common and straightforward approach.

```python
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

base_model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.1,
    max_tokens=10000,
    timeout=30,
    # ... (other params)
)

agent = create_agent(model=base_model, tools=[])
```

Model instances give you complete control over configuration. Use them when you need to set specific parameters like `temperature`, `max_tokens`, `timeouts`, `base_url`, and other provider specific settings.
## Dynamic model

Dynamic models are selected at runtime based on the current state and context. This enables sophisticated routing logic and cost optimization. To use a dynamic model, create middleware using the `@wrap_model_call` decorator that modifies the model in request.

```python
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ChatGroq(model="llama-3.1-8b-instant")
advanced_model = ChatGroq(model="openai/gpt-oss-20b")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity"""
    messages_count = len(request.state["messages"])
    
    if messages_count > 10:
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model, # Default model
    tools=[], 
    middleware=[dynamic_model_selection]  
)
```

You can have your own logic of when to use the advanced model and when to use the basic model. For this simple example I went ahead with choosing the advanced model if the `messages_count` is greater than 10.

---
# Tools

Tools are the agent's actionable capabilities: APIs, web search, calculators, DB queries, etc.
Tools can be:
- Plain Python functions / coroutines.
- Decorated with `@tool` to add metadata (name, description, argument schema).
Key agent-level capabilities around tools:
- multiple sequential tool calls per user request,
- parallel tool calls when appropriate,
- tool retry logic and error handling,
- state persistence across tool calls.

## Defining tools

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def search(query: str) -> str:
    """Search for information"""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location"""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model=base_model, tools=[search, get_weather])
```

If an empty tool list is provided, the agent will consist of a single LLM node without tool-calling capabilities.
## Tool error handling

You can customize tool error handling with tool middleware (e.g., `@wrap_tool_call`). That middleware can catch exceptions and return structured `ToolMessage` objects with meaningful content and the original `tool_call_id`, enabling the agent to incorporate tool failures into its reasoning rather than crashing silently. 

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom message."""
    try:
        return handler(request)
    except Exception as e:
        # Return a cusom error message to the model.
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )

agent = create_agent(
    model=base_model,
    tools=[search, get_weather],
    middleware=[handle_tool_errors],
)
```

---
# System Prompts

A `system_prompt` shapes agent behavior (concise, verbose, role-specific). You can shape how your agent approaches tasks by providing a prompt. The `system_prompt` parameter can be provided as a string.

```python
agent = create_agent(
    model=base_model,
    tools=[],
    system_prompt="You are a helpful assistant. Be concise and accurate.",
)
```

When no `system_prompt` is provided, the agent will infer it's task from the messages. But that is not a good practice. Always provide a concise and detailed system prompt.
## Dynamic system prompt

For more advanced use cases where you need to modify the system prompt based on runtime context or agent state, you can use middleware.

The `@dynamic_prompt` decorator creates middleware that generates system prompts based on the model request.

```python
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompts(request: ModelRequest) -> str:
    """Generate system prompt based on user role"""
    user_role = request.runtime.context.get(
        "user_role", "user"
    )  # user is the default value in case it's not able to get user_role
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed techincal response."
    elif user_role == "begineer":
        return f"{base_prompt} Explain concepts simple and avoid jargon."

    return base_prompt

agent = create_agent(
    model=base_model, 
    tools=[], 
    middleware=[user_role_prompts], 
    context_schema=Context
)

# The system prompt will be set dynamically based on context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"},
)
result["messages"][-1].content
```

---
# Invocation

To run the agent you call `agent.invoke(...)` with a payload (messages + optional structured context/state). The agent processes the messages, uses the ReAct loop to call tools as needed, and returns a response structure that may include final messages, intermediate tool messages, and if configured structured outputs (JSON-like) for downstream consumption.

```python
result = agent.invoke(
    {"messages": [
	    {"role": "user", "content": "What's the weather in San Franciso"}
	]},
    context={"user_role": "begineer"},
)
result["messages"][-1].content
```

The agent follows the LangGraph *Graph API* and supports all associated methods such as stream and invoke.

---
# Structured Output

In some situations, you may want the agent to return an output in a specific format. LangChain provides strategies for structured output via the `response_format` parameter.
## ToolStrategy

`ToolStrategy` uses artificial tool calling to generate structured output. This works with any model that supports tool calling.

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


# response format class
class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


agent = create_agent(
    model=base_model, tools=[], response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Extract contact info from: John Doe, john@example.com, (+91 123-456-7890)",
            }
        ]
    }
)

result["structured_response"]
```
## ProviderStrategy

`ProviderStrategy` uses the model provider's native structured output generation. This is more reliable but only works with providers that support native structured output (e.g. OpenAI)

```python
from langchain.agents.structured_output import ProviderStrategy

agent_na = create_agent(
    model=base_model, response_format=ProviderStrategy(ContactInfo)
)  # Currently not working with Groq in my attempt.
```

> [!INFO] Note
> As of _langchain 1.0_, simply passing a schema (e.g. `response_format=ContactInfo`) is no longer supported. We must explicitly use `ToolStrategy` or `ProviderStrategy`.

---
# Memory

Agents maintain conversation history automatically through the message state. You can also configure the agent to use a custom state schema to remember additional information during the conversation. 

Information stored in the state can be thought of as the *short-term memory* of the agent. Custom state schemas must extend `AgentState` as a `TypedDict`.

There are two ways to define custom state:
	- Via middleware (preferred)
	- Via `state_schema` on `create_agent`
## Defining state via middleware

Use middleware to define custom state when your custom state needs to be accessed by specific middleware hooks and tools attached to said middleware.

```python
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any

class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = []

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        return super().before_model(state, runtime)

agent = create_agent(
	model=base_model, 
	tools=[], 
	middleware=[CustomMiddleware()]
)

# The agent can now track additional state beyond messages
result = agent.invoke(
    {
        "messages": [
	        {"role": "user", "content": "I prefer techincal explanations"}
	    ],
        "user_preferences": {"style": "techincal", "verbosity": "detailed"},
    }
)

result["messages"][-1]
```
## Defining state via `state_schema`

Use the `state_schema` parameter as a shortcut to define custom state that is only used in tools.

```python
from langchain.agents import AgentState


class CustomState(AgentState):
    user_preferences: dict


agent = create_agent(model=base_model, tools=[], state_schema=CustomState)

# The agent can now track additional state beyond messages
result = agent.invoke(
    {
        "messages": [
	        {"role": "user", "content": "I prefer technical explanations"}
	    ],
        "user_preferences": {"style": "technical", "verbosity": "detailed"},
    }
)
result["messages"][-1]
```

> [!INFO] Note
> As of _langchain 1.0_, custom state schemas must be `TypedDict` types. Pydantic models and dataclasses are no longer supported.

Defining custom state via middleware is preferred over defining it via `state_schema` on `create_agent` because it allows you to keep state extensions conceptually scoped to the relevant middleware and tools.

`state_schema` is still supported for backwards compatibility on `create_agent`.

---
# Key Takeaways

Agents become genuinely useful when the model, tools, prompts, and runtime all reinforce each other. With the right structure, they behave less like chatbots and more like modular problem-solving systems. Their strength comes from controlled reasoning, predictable actions, and the ability to adapt dynamically to context.
- They plan and act through iterative reasoning instead of one-shot responses.
- Tools give them real capabilities, and schemas keep those capabilities reliable.
- Middleware shapes behavior, routes models, and handles errors cleanly.
- Structured output and state make them stable building blocks for larger applications.