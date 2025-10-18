---
title: Langchain Basics
description: Here I will have notes and codes about the basics of Langchain
tags:
  - AI
  - Langchain
  - Basics
date: 2025-10-16
---
# LLM
Here we are using the `ChatGroq` package from _LangChain_ because [GROQ](https://groq.com/) offers free access to several large language models (LLMs). Once you create an account on [GROQ](https://groq.com/), you receive API keys that allow secure access to these models.

You should always store your API key as an environment variable — this prevents exposing it publicly and allows reusability across scripts.

To use `ChatGroq`, install the package:
```sh
pip install langchain-groq
```

LLM Initialization:
```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",  # LLM identifier to use from Groq's model catalog
    temperature=0,                 # Controls randomness: 0 = deterministic, 1 = creative
    max_tokens=1000,               # Upper limit on tokens generated in the response
    verbose=True,                  # Enables detailed execution logs for debugging
    timeout=600,                   # Maximum time (in seconds) before the request is aborted
    max_retries=3,                 # Number of retry attempts if the API call fails
)

```
_Documentation:_ [ChatGroq](https://python.langchain.com/api_reference/groq/chat_models/langchain_groq.chat_models.ChatGroq.html)

To the LLM in action, let's run the following code:
```python
result = llm.invoke("What is 1 + 1?")
print(result.content)
```
The `.invoke()` method runs the LLM using the given prompt or query.  
It returns a response object that includes:
- `content` → model output (main answer)
- `response_metadata`, `usage_metrics`, and other diagnostic fields
Other supported methods include `.stream()`, `.ainvoke()`, `.astream()`, and `.batch()` — each suited for different execution modes such as streaming or asynchronous use.
# Prompts
Prompts are critical in shaping how an LLM interprets and responds. Mastering prompt design is key to becoming a strong AI engineer.

In this example, we translate a Spanish sentence into English, while also defining the **style** we want the final output to follow.
```python
text = """
La innovación no consiste en tener muchas ideas, sino en hacer que una de ellas
funcione.
"""

style = """
Transform into concise, motivational startup-speak — the kind used in tech
founder keynotes or product launch presentations.
"""

prompt = f"""
Rewrite this:
{text}
  
in fluent English with the following style:
{style}.
"""

print(prompt)
```
Here we build a manual prompt using Python’s f-strings. However, for scalable or reusable systems, LangChain provides a better alternative — `ChatPromptTemplate`.
## ChatPromptTemplate
_LangChain_ provides `ChatPromptTemplate` to help create structured, reusable, and role-aware prompts.

This makes it easier to:
- Separate variables and prompt logic
- Reuse templates with different inputs
- Handle multi-role (system, human, AI) prompts cleanly
```python
from langchain.prompts import ChatPromptTemplate

# Here we don't use f string
template_string = """
Rewrite this:
{text}

in fluent English with the following style:
{style}.
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

print(prompt_template)
print(prompt_template.messages[0].prompt)
```
_Documentation_

Once the template is defined, we can reuse it with different inputs:
```python
msg = prompt_template.format_messages(
    style=style,
    text=text
)

print(msg)
  
response = llm.invoke(msg)
print(response.content)
```

### Difference between `prompt_template` `invoke` and `format_message` methods.

| Method                      | Purpose                                                                                                                                                                                                                  | Output                                         | Typical Use Case                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `format_messages(**kwargs)` | Fills variables and returns a list of structured **message objects** (`SystemMessage`, `HumanMessage`, etc.).                                                                                                            | `List[BaseMessage]`                            | Use when you need to **see or modify** messages before sending to the model. Example: in chains or debugging. |
| **`invoke(input_dict)`**    | Executes the full **Runnable** interface for prompts (LangChain `Runnable` protocol). It calls `format_messages()` internally and returns the same message list, but wrapped for seamless chaining with other runnables. | Same as `format_messages()` (list of messages) | Use when composing the prompt as part of a **LangChain pipeline** (e.g., `prompt                              |

Summary:
- `format_messages()` → plain message rendering utility.
- `invoke()` → full runnable call (standardized entry point) for pipelines.

If you are just formatting prompts, use `format_messages()`.  
If you are chaining components or using `Runnable` methods like `.stream()` or `.batch()`, use `invoke()`.
# Output Parsers
LLMs often return unstructured text in formats like JSON or Markdown. When you expect structured data, you can use _LangChain’s output parsers_ to extract and validate specific fields.

In this example, we’ll parse a product review into structured JSON fields (`gift`, `delivery_days`, and `price_value`).
```python
customer_review = """\
This leaf blower is pretty amazing. It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:
  
gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown. \

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1. \

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list. \
  
Format your answer as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

prompt_template = ChatPromptTemplate.from_template(review_template)

messages = prompt_template.format_messages(text=customer_review)

response = llm.invoke(messages)

gift_schema = ResponseSchema(name="gift", description="Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.")
delivery_schema = ResponseSchema(name="delivery_days", description="How many days did it take for the product to arrive? If this information is not found, output -1.")
price_value_schema = ResponseSchema(name="price_value", description="Extract any sentences about the value or price, and output them as a comma separated Python list.")

response_schemas = [gift_schema, delivery_schema, price_value_schema]
```
_Documentation:_ [ResponseSchema](https://python.langchain.com/api_reference/langchain/output_parsers/langchain.output_parsers.structured.ResponseSchema.html), [StructuredOutputParser](https://python.langchain.com/api_reference/langchain/output_parsers/langchain.output_parsers.structured.StructuredOutputParser.html)

Next, LangChain provides automatic instructions for the LLM to follow the expected JSON structure:
```python
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

print(format_instructions)
```

Finally, combine the format instructions with the input text and parse the LLM’s response:
```python
review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)

messages = prompt.format_messages(text=customer_review, format_instructions=format_instructions)

response = llm.invoke(messages)

output_dict = output_parser.parse(response.content)
print(output_dict)
```
Here, the `StructuredOutputParser` validates and converts the raw LLM text into a Python dictionary with clearly defined fields.

# Conclusion

This note demonstrates the foundational elements of building structured, reliable LLM workflows using *LangChain* and *Groq*:

- Initialize and run LLMs securely with API keys.
- Design flexible and maintainable prompts with `ChatPromptTemplate`.
- Parse structured outputs confidently using `ResponseSchema` and `StructuredOutputParser`.

Together, these tools form the backbone of scalable LLM applications — from text generation and translation to automated data extraction.  
Mastering them sets the stage for building robust, production-grade AI systems.