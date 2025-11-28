---
title: Langchain Basics - 2
description: Here I will have notes and codes about the basics of Langchain
tags:
  - Langchain
  - Basics
  - Memory
  - Chains
  - Conversation-Chain
date: 2025-10-20
---
# Memory and Conversation Chains
A key feature of chatbots is their ability to *remember parts of the conversation*, rather than treating every message as new. This allows the chatbot to maintain context, understand previous messages, and respond naturally. _Memory_ is what enables the chatbot to retain past messages and pass them as context to the next input.

A *Conversation Chain* is essentially a workflow consisting of:
1. An *LLM* - the model that generates responses.
2. A *Memory* object - which stores past user and model messages.
3. A simple logic layer that feeds the conversation history plus the new input back to the LLM each time.
Whenever you call `conversation.predict(input="...")`, Langchain automatically retrieves the stored history from memory, appends the latest user query, and passes the entire context to the model. The model then produces a context-aware response, and the new exchange is added back to the memory.

```python
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatGroq(temperature=0.0, model="llama-3.1-8b-instant")
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm = llm,
    memory = memory,
    verbose = True
)

print(conversation.predict(input="Hi, my name is Hammad"))
print(conversation.predict(input="How many r's are there in strawberry"))
print(conversation.predict(input="What is my name again?"))
# The LLM remembers my name from previous messages.
```
`ConversationBufferMemory` is a basic memory implementation that simply stores the entire conversation history as-is. This can sometimes become expensive as the conversation grows longer. _Documentation_: [ConversationBufferMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.buffer.ConversationBufferMemory.html). [ConversationChain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversation.base.ConversationChain.html)

We can inspect the memory to see what it contains.
```python
print(memory.buffer)
print(memory.load_memory_variables({}))
```
`memory.buffer`
- This is the *raw storage* of the conversation.
- It contains all messages in order (often as a single string or a structured list, depending on the memory type).
- raw, internal storage of messages.
`memory.load_memory_variables(inputs)`
- This *returns the memory in a format that an LLM can consume during a chain run.
- The method usually expects an `inputs` dictionary (e.g., the current user query), but many memory types ignore it, so you can pass `{}`.
- processed dictionary suitable for passing into chains or prompts.

To explicitly add messages to the memory, we can use the following system.
```python
memory.save_context({"input": "Hi"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
```
## Different Types of Memory
### 1. `ConversationBufferMemory`
`ConversationBufferMemory` is a basic memory implementation. This memory allows for storing of messages and then extracts the messages in a variable. _Code Example Above_.
### 2. `ConversationBufferWindowMemory`
This memory keeps track of the last `k` turns of a conversation. If the number of messages in the conversation is more than the maximum number of messages to keep, the oldest messages are dropped.
```python
from langchain.memory import ConversationBufferWindowMemory


memory = ConversationBufferWindowMemory(k=1) # k = number of messages to remember

memory.save_context({"input": "Hi"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})

print(memory.load_memory_variables({}))

llm = ChatGroq(temperature=0.0, model="llama-3.1-8b-instant")
memory = ConversationBufferWindowMemory(k = 1)
conversation = ConversationChain(
    llm = llm,
    memory = memory,
    verbose = True
)

print(conversation.predict(input="Hi, my name is Hammad"))
print(conversation.predict(input="How many r's are there in strawberry"))
print(conversation.predict(input="What is my name again?"))
# The LLM only remembers the last message, so it forgets the name.
```
_Documentation_: [ConversationBufferWindowMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.buffer_window.ConversationBufferWindowMemory.html)
### 3. `ConversationTokenBufferMemory`
Keeps only the most recent messages in the conversation under the constraint that the total number of tokens in the conversation does not exceed a certain limit. 
```python
from langchain.memory import ConversationTokenBufferMemory

llm = ChatGroq(temperature=0.0, model="llama-3.1-8b-instant")
memory = ConversationTokenBufferMemory(llm = llm, max_token_limit=10) # Need to specify llm because different llm's have different way to counting tokens
conversation = ConversationChain(
    llm = llm,
    memory = memory,
    verbose = True
)

memory.save_context({"input": "Hi"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
memory.save_context({"input": "AI is what?"}, {"output": "Amazing!"})

memory.load_memory_variables({})
# Only recent messages within the token limit are retained.
```
_Documentation_: [ConversationTokenBufferMemory](https://colab.research.google.com/drive/14RVajNxKC1EeICZvRjd9lA9wW0Ehg61i#scrollTo=z5UkkIp4a0oR)
### 4. `ConversationSummaryBufferMemory`
This memory module continually summarizes the conversation history. The summary is updated after each conversation turn. The implementations returns a summary of the conversation history which can be used to provide context to the model.
```python
from langchain.memory import ConversationSummaryBufferMemory

schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo." 

llm = ChatGroq(temperature=0.0, model="llama-3.1-8b-instant")
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)

memory.save_context({"input": "Hello"}, {"output": "What's up?"})
memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
memory.save_context({"input": "AI is what?"}, {"output": "Amazing!"})
memory.save_context({"input": "What is on my schedule today?"}, {"output": f"{schedule}"})

memory.load_memory_variables({})
# Returns a short summary of the entire conversation.
```
_Documentation_: [ConversationSummaryBufferMemory](https://python.langchain.com/api_reference/langchain/memory/langchain.memory.summary.ConversationSummaryMemory.html)
### Additional Memory Types
Langchain offers additional memory types, for eg. _Vector Data Memory_ and _Entity Memories_, but that is something to be explored later. Also you can use multiple memories at one time. We can also store the conversations in a conventional database such as SQL, Key-value databases, etc. and later retrieve and pass them to the conversation chain.

# Chains
Chains are sequences of actions or steps that connect LLM's, prompts, tools and memory to accomplish a specific task. A `LLMChain` is a type of chain in Langchain, it's a type of chain that connects a:
- A *prompt* (usually a `PromptTemplate` or `ChatPromptTemplate`) 
- An *LLM* (in our case `ChatGroq`)

Here's an example:
```python
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatGroq(temperature=0.8, model="llama-3.1-8b-instant")
prompt = ChatPromptTemplate.from_template(
    "What's the best name to describe a company that makes {product}?"
)

chain = LLMChain(llm=llm, prompt=prompt)

product = "Queen Size Sheet Set"
chain.run(product)
```
_Documentation_: [LLMChain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.llm.LLMChain.html)
## Sequential Chain
*Sequential Chain* is another type of chains. The is to combine multiple chains where the output of one chain is the input of the next chain. 

There is two types of sequential chains:
### 1. Simple Sequential Chain:
It takes in a single input and all parts of the chain also take in a single input and the final answer should also be a single output. 
```python
from langchain.chains import SimpleSequentialChain

first_prompt = ChatPromptTemplate.from_template(
    "What's the best name to describe a company that makes {product}?"
)
chain_1 = LLMChain(llm=llm, prompt=first_prompt)
  
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 word description for the following company: {company_name}"
)
chain_2 = LLMChain(llm=llm, prompt=second_prompt)

overall_simple_chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=True)

product = "Queen Size Sheet Set"
result = overall_simple_chain.run(product)

print(result)
```
Here the output of the `chain_1` that is company name is input for `chain_2` and finally the output of `chain_2` is a single variable that contains the 20 word description. You pass the first input to the chain within the run method. _Documentation_: [SimpleSequentialChain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.sequential.SimpleSequentialChain.html)
### 2. Sequential Chain:
An enhanced version of `SimpleSequentialChain` where the sequence of chains can handle multiple inputs and outputs collectively, allowing intermediate outputs to be passed between chains and multiple final outputs to be returned. Each individual chain still produces a single output key.
```python
from langchain.chains import SequentialChain

first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to English: \n\n{Review}"
)
chain_1 = LLMChain(llm=llm, prompt=first_prompt, output_key="English Review") # Output Key important to let the next chain know what to use as input

second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence: \n\n{Review}"
)
chain_2 = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review: \n\n{Review}"
)
chain_3 = LLMChain(llm=llm, prompt=third_prompt, output_key="language")

fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up resonses to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary} \n\nLanguage: {language}"
)
chain_4 = LLMChain(llm=llm, prompt=fourth_prompt)

overall_chain = SequentialChain(
    chains=[chain_1, chain_2, chain_3, chain_4],
    input_variables=["Review"],
    output_variables=["English Review", "summary", "language"],
    verbose=True
)

review = review = df.iloc[4]["Review"]

result = overall_chain(review)

print(result)
```
_Documentation_: [SequentialChain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.sequential.SequentialChain.html)
# **Key Takeaways**
- *Memory modules* enable chatbots to retain and manage conversation history.
    - Options include full conversation (`ConversationBufferMemory`), windowed (`ConversationBufferWindowMemory`), token-limited (`ConversationTokenBufferMemory`), or summarized (`ConversationSummaryBufferMemory`).
- *Chains* define workflows connecting LLMs, prompts, memory, and tools to accomplish tasks.
- *LLMChain* handles single-step tasks with a prompt and an LLM.
- *SequentialChains* allow chaining multiple LLMChains where outputs of one chain feed into the next.
    - SimpleSequentialChain: single input/output per chain.
    - SequentialChain: multiple intermediate and final outputs supported.
- Together, *memory and chains* enable conversational AI that is *context-aware, flexible, and capable of multi-turn, multi-step interactions*.