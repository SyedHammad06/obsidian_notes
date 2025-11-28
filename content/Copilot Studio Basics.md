---
title: Copilot Studio Basics
description: A detailed guide to Microsoft Copilot Studio, covering the creation of custom AI agents
tags:
  - Copilot-Studio
  - Basics
  - Agents
  - Low-Code
date: 2025-11-28
---
[Microsoft Copilot Studio](https://www.google.com/search?q=https://copilotstudio.microsoft.com/) is a low-code platform that allows you to build, customize, and manage AI assistants (now referred to as **"Copilot Agents"**). There is a public free version available via a trial account.
The platform has two primary functions:
- **Build Standalone Copilots:** Create custom AI agents from scratch for specific roles (e.g., IT Support, Customer Service).
- **Extend Microsoft 365 Copilot:** Teach the standard Microsoft 365 Copilot (in Teams/Word/Excel) new skills and how to access your organization's specific data.
## Key Features
1. **Custom Build:** Construct agents tailored for specific internal (employee-facing) or external (customer-facing) goals.
2. **Controlled Knowledge:** Unlike general LLMs (like ChatGPT), you define exactly what data the agent can access. You can restrict it to specific corporate data or allow it to use general internet knowledge as a fallback.
3. **Conversational Orchestration:** You can manually design critical conversation flows to ensure the agent adheres to strict business rules, legal requirements, or brand tone, rather than relying solely on generative AI.
4. **Tools & Automation:** The agent is not just a chatbot; it can perform tasks by connecting to other systems (via Power Platform) or custom-built APIs.

---
## Core Components
### 1. Knowledge
In Copilot Studio, **Knowledge** defines the specific information sources the agent uses to answer questions (often called "Grounding"). Common sources include:
- Public websites
- Uploaded documents (PDF, Word, etc.)
- SharePoint sites
- Dataverse tables
**Settings:** You can toggle whether the agent uses **General Knowledge** (General AI) or not.
- _Enabled:_ The agent answers from your data first but uses the LLM's general training to answer off-topic questions.
- _Disabled:_ The agent has "tunnel vision" and will only answer based strictly on the provided knowledge sources.
**Critical Best Practice:** You must write a **clear description** for every knowledge source. The AI uses semantic matching to read this description to decide _which_ document or website to use to answer a specific question.
**Limitations:** When using public websites, the agent reads the page source code. It struggles with websites heavily reliant on JavaScript, complex interactive elements, or visual-only tables that lack structured HTML tags.
### 2. Topics
**Topics** allow you to _orchestrate_ the conversation. Instead of letting the AI guess how to respond, you build structured workflows for specific scenarios. Topics consist of a **Trigger** (what starts the flow) and **Nodes** (actions, questions, messages).
**Triggers:**
- **Classic:** The topic starts when the user types a specific keyword or phrase (e.g., "Reset Password", "Emergency").
- **Generative (Dynamic):** You provide a description of what the topic does (e.g., "Use this topic when a user wants to file a complaint"). The AI analyzes the user's intent and automatically selects the correct topic based on that description.
### 3. Entities and Variables
- **Entities:** These allow the AI to extract structured data from unstructured conversation.
    - _Example:_ If a user says, "I need help for my **6-year-old**," the AI identifies "6" not just as text, but as a number/Age entity.
    - _Benefit:_ It prevents the bot from asking questions the user has already answered (Smart Skipping).
- **Variables:** The data extracted by entities (like the age "6") is stored in a **Variable**.
    - **Topic Variables:** Accessible only within the current conversation flow.
    - **Global Variables:** Accessible across the entire agent (useful for remembering a user's name or account ID throughout the session).
_Application:_ You use these variables to create **Conditional Branching** (logic). For example: If `VarAge` < 18, go to Child Services flow; If `VarAge` >= 18, go to Adult Services flow.
### 4. Tools (Formerly Actions/Plugins)
**Tools** represent the "hands" of the agent—they allow it to _do_ things, not just talk.
- **Pre-built Connectors:** You can add existing Microsoft connectors (e.g., MSN Weather). The agent automatically understands that to use this tool, it needs an input (like "City") and will ask the user for it if missing.
- **Power Automate Flows:** For custom business logic, you build a **Cloud Flow**.
    - The agent sends data (Variables) to the Flow.
    - The Flow performs the task (e.g., creates a ticket in Jira, updates a row in Excel, sends an email).
    - The Flow returns the result to the agent to display to the user.

---
## The "TechFix" IT Agent
To understand how these components work together in a production environment, let's walk through building a "TechFix" agent designed to help employees order new equipment.
### Phase 1: Knowledge (Grounding the AI)
_The Setup:_ We upload a PDF titled **"2025 IT Hardware Policy"** into the Knowledge section.
- **Concept Application:**
    - **Description Importance:** We give the file a description: "Contains rules regarding eligibility for laptop upgrades, budget limits, and approved hardware vendors." This ensures that when the user asks a policy question, the AI knows this is the specific document to read.
    - **Generative Answers:** If a user asks, "Am I allowed to get a Mac?", the agent scans the unstructured text of the PDF, finds the paragraph about Apple devices, and synthesizes a natural language answer: "Yes, according to the policy, employees in Creative departments are eligible for Macs."
### Phase 2: Topics & Orchestration (Defining the Flow)
_The Setup:_ We cannot rely on the Knowledge PDF to place an order; the PDF is read-only. We need a specific workflow for ordering. We create a custom Topic called **"Order Device"**.
- **Concept Application:**
    - **Generative Triggers:** Instead of trying to guess every phrase a user might say (e.g., "Buy laptop", "Get computer", "Need hardware"), we use a **Generative Trigger**. We describe the topic as: "Use this workflow when a user explicitly wants to place an order for new hardware." The AI now "listens" for this intent.
    - **Orchestration:** We don't want the AI to just chat; we want it to gather specific data. We design the topic nodes to guide the user: "I can help you order a new device. I just need a few details."
Within this topic, we must enforce the rule that **only Creative departments** can order Macs. This requires **Conditional Branching**—turning a linear conversation into a smart decision tree.
1. **Capture the Data:** The agent asks, "Which department are you in?" and stores the answer in a variable called `VarDept`.
2. **Add the Condition Node:** In the flow canvas, we add a Condition.
3. **Define the Rule:** We configure the "True" branch: `IF VarDept is equal to "Creative"`.
4. **Define the Paths:**
    - **Path A (True):** If the user matches the rule, the agent proceeds: "Great, you are eligible for Apple hardware." -> Proceed to Order Tool.
    - **Path B (False/Else):** If the user is in Finance/HR/IT, the agent stops the order: "I'm sorry, based on the IT Policy, only Creative departments are eligible for Macs. Please select a Windows device."
### Phase 3: Entities & Variables (Capturing Data)
_The Setup:_ The agent asks, "What device do you want?" The user replies, "I want a Lenovo X1."
- **Concept Application:**
    - **Entity Extraction:** The agent parses the user's sentence. It identifies "Lenovo X1" using a **DeviceType Entity**. It separates the data from the chitchat.
    - **Smart Skipping:** If the user had started the conversation by saying, "I need to order a Lenovo X1," the agent would skip the question "What device do you want?" because the Entity was already filled in the initial prompt.
    - **Variable Storage:** These values are saved into variables `VarDevice` ("Lenovo X1") and `VarDept` ("Marketing"). These variables are held in the agent's memory for the duration of this specific task.
### Phase 4: Tools & Power Automate (Executing the Task)
_The Setup:_ The agent has the data, but it needs to communicate with the company's ticketing system. We add a **Tool** connected to a **Power Automate Flow**.
- **Concept Application:**
    - **The "Handshake":** We define **Inputs** for the Tool (`VarDevice`, `VarDept`) so the agent knows what to pass to the backend.
    - **External Connection:** The Power Automate Flow runs in the background. It performs logic the agent cannot do alone:
        1. Checks a live Excel inventory sheet to see if the "Lenovo X1" is in stock.
        2. Creates a ticket in ServiceNow/Jira.
    - **Outputs:** The Flow finishes and sends a **Ticket ID** back to Copilot Studio.
    - **Final Response:** The agent takes that output and displays it to the user: "Success! Your order for a Lenovo X1 has been placed. Your tracking number is TKT-999."