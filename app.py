from __future__ import annotations as _annotations
import os
import time
import logging
import asyncio
import random
import chainlit as cl

from pydantic import BaseModel
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from openai.types.responses import ResponseTextDeltaEvent
from openai import AsyncAzureOpenAI
from azure.ai.projects.models import (
    AgentStreamEvent,
    MessageDeltaChunk,
    MessageRole,
    ThreadRun,
)
from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    function_tool,
    handoff,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    set_default_openai_client,
    set_default_openai_api
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

load_dotenv()
# Disable verbose connection logs
logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger.setLevel(logging.WARNING)
set_tracing_disabled(True)

AIPROJECT_CONNECTION_STRING = os.getenv("AIPROJECT_CONNECTION_STRING")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
FAQ_AGENT_ID = os.getenv("FAQ_AGENT_ID")

azure_client = AsyncAzureOpenAI(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("MY_OPENAI_API_KEY"),
)

set_default_openai_client(azure_client, use_for_tracing=False)
set_default_openai_api("chat_completions")

project_client = AIProjectClient.from_connection_string(
    conn_str=AIPROJECT_CONNECTION_STRING, credential=DefaultAzureCredential()
)


class MultiAgentContext(BaseModel):
    user_name: str | None = None
    image_path: str | None = None
    birth_date: str | None = None
    user_id: str | None = None


### TOOLS


@function_tool(
    name_override="faq_lookup_tool", description_override="Lookup frequently asked questions."
)
async def faq_lookup_tool(question: str) -> str:
    print(f"User Question: {question}")
    start_time = cl.user_session.get("start_time")
    print(f"Elapsed time: {(time.time() - start_time):.2f} seconds - faq_lookup_tool")
    is_first_token = None

    try:
        # create thread for the agent
        thread_id = cl.user_session.get("new_threads").get(FAQ_AGENT_ID)
        print(f"thread ID: {thread_id}")

        # Create a message, with the prompt being the message content that is sent to the model
        project_client.agents.create_message(
            thread_id=thread_id,
            role="user",
            content=question,
        )

        async with cl.Step(name="faq-agent") as step:
            step.input = question

            # Run the agent to process tne message in the thread
            with project_client.agents.create_stream(thread_id=thread_id, agent_id=FAQ_AGENT_ID) as stream:
                for event_type, event_data, _ in stream:
                    if isinstance(event_data, MessageDeltaChunk):
                        # Stream the message delta chunk
                        await step.stream_token(event_data.text)
                        if not is_first_token:
                            print(f"Elapsed time: {(time.time() - start_time):.2f} seconds - {event_data.text}")
                            is_first_token = True

                    elif isinstance(event_data, ThreadRun):
                        if event_data.status == "failed":
                            print(f"Run failed. Error: {event_data.last_error}")
                            raise Exception(event_data.last_error)

                    elif event_type == AgentStreamEvent.ERROR:
                        print(f"An error occurred. Data: {event_data}")
                        raise Exception(event_data)

        # Get all messages from the thread
        messages = project_client.agents.list_messages(thread_id)
        # Get the last message from the agent
        last_msg = messages.get_last_text_message_by_role(MessageRole.AGENT)
        if not last_msg:
            raise Exception("No response from the model.")

        # Delete the thread later after processing
        delete_threads = cl.user_session.get("delete_threads") or []
        delete_threads.append(thread_id)
        cl.user_session.set("delete_threads", delete_threads)

        # print(f"Last message: {last_msg.text.value}")
        return last_msg.text.value

    except Exception as e:
        logger.error(f"Error: {e}")
        return "I'm sorry, I encountered an error while processing your request. Please try again."


@function_tool
async def update_user_name(
    context: RunContextWrapper[MultiAgentContext], user_name: str, image_path: str, birth_date: str,
) -> str:
    """
    Update the customer user name using government ID or passport image and birth date.

    Args:
        user_name: The new customer user name.
        image_path: image file path of government ID or passport.
        birth_date: The customer birth date.
    """
    # Update the context
    context.context.user_name = user_name
    context.context.image_path = image_path
    context.context.birth_date = birth_date
    print(f"Context context: {context.context}")

    # Ensure that the user ID has been set by the incoming handoff
    assert context.context.user_id is not None, "User ID is required"
    return f"Updated user name to {user_name}. ID image saved successfully."


### HOOKS


async def on_account_management_handoff(context: RunContextWrapper[MultiAgentContext]) -> None:
    user_id = f"ID-{random.randint(100, 999)}"
    context.context.user_id = user_id


### AGENTS

faq_agent = Agent[MultiAgentContext](
    name="FAQ Agent",
    handoff_description="A helpful agent that can answer questions about BHP Operational Review.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the last question asked by the customer.
    2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.
    3. If you cannot answer the question, transfer back to the triage agent.""",
    tools=[faq_lookup_tool],
    model=OpenAIChatCompletionsModel(
        model=DEPLOYMENT_NAME,
        openai_client=azure_client,
    ),
)

account_management_agent = Agent[MultiAgentContext](
    name="Account Management Agent",
    handoff_description="A helpful agent that can update customer user name.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an account management agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Ask for their new user name, an image of their government ID or passport to be uploaded, and their birth date.
    # Very Important Notes
    - If the customer asks a question that is not related to the routine, transfer back to the triage agent.
    - Never respond to the user with any PII data such as password, ID number, etc.""",
    tools=[update_user_name],
    model=OpenAIChatCompletionsModel(
        model=DEPLOYMENT_NAME,
        openai_client=azure_client,
    ),
)

live_agent = Agent[MultiAgentContext](
    name="Live Agent",
    handoff_description="A live human agent that can handle complex issues or when a user specifically requests human assistance.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a human customer service representative named Sarah. Act as if you are a real person, not an AI.
    Use a conversational, friendly tone and occasionally add small human touches to your responses.
    
    You should handle:
    1. Complex issues that other agents couldn't resolve
    2. Situations where a user has asked the same question multiple times
    3. When a user explicitly asks to speak with a human agent
    4. Technical errors or issues within the application
    
    # Human touches you can incorporate:
    - Mention taking notes: "Let me note that down for you"
    - Reference checking systems: "Let me check our system for that information"
    - Personalize responses: "I understand how frustrating that can be"
    - Occasionally mention your "team" or "colleagues"
    
    If the customer's issue is resolved or is actually simple enough for the automated system to handle,
    you can transfer them back to the triage agent.""",
    tools=[],
    model=OpenAIChatCompletionsModel(
        model=DEPLOYMENT_NAME,
        openai_client=azure_client,
    ),
)

triage_agent = Agent[MultiAgentContext](
    name="Triage Agent",
    handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents."
        "Use the response from other agents to answer the question. Do not rely on your own knowledge."
        "Other than greetings, do not answer any questions yourself."
        "If a user explicitly asks for a human agent or live support, transfer them to the Live Agent."
        "If a user is asking the same question more than two times, transfer them to the Live Agent."
        "# Very Important Notes"
        "- Never respond to the user with any PII data such as password, ID number, etc."
    ),
    handoffs=[
        handoff(agent=account_management_agent, on_handoff=on_account_management_handoff),
        faq_agent,
        live_agent,
    ],
    model=OpenAIChatCompletionsModel(
        model=DEPLOYMENT_NAME,
        openai_client=azure_client,
    ),
)

faq_agent.handoffs.append(triage_agent)
account_management_agent.handoffs.append(triage_agent)
live_agent.handoffs.append(triage_agent)


async def main(user_input: str) -> None:
    current_agent = cl.user_session.get("current_agent")
    input_items = cl.user_session.get("input_items")
    context = cl.user_session.get("context")
    print(f"Received message: {user_input}")

    # Show thinking message to user
    msg = await cl.Message(f"thinking...", author="agent").send()
    msg_final = cl.Message("", author="agent")

    # Set an empty list for delete_threads in the user session
    cl.user_session.set("delete_threads", [])
    is_thinking = True

    try:
        input_items.append({"content": user_input, "role": "user"})
        # Run the agent with streaming
        result = Runner.run_streamed(current_agent, input_items, context=context)
        last_agent = ""

        # Stream the response
        async for event in result.stream_events():
            # Get the last agent name
            if event.type == "agent_updated_stream_event":
                if is_thinking:
                    last_agent = event.new_agent.name
                    msg.content = f"[{last_agent}] thinking..."
                    await msg.send()
            # Get the message delta chunk
            elif event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                if is_thinking:
                    is_thinking = False
                    await msg.remove()
                    msg_final.content = f"[{last_agent}] "
                    await msg_final.send()

                await msg_final.stream_token(event.data.delta)

        # Update the current agent and input items in the user session
        cl.user_session.set("current_agent", result.last_agent)
        cl.user_session.set("input_items", result.to_input_list())

    except Exception as e:
        logger.error(f"Error: {e}")
        msg_final.content = "I'm sorry, I encountered an error while processing your request. Please try again."

    # show the last response in the UI
    await msg_final.update()

    # Delete threads after processing
    delete_threads = cl.user_session.get("delete_threads") or []
    for thread_id in delete_threads:
        try:
            project_client.agents.delete_thread(thread_id)
            print(f"Deleted thread: {thread_id}")
        except Exception as e:
            print(f"Error deleting thread {thread_id}: {e}")

    # Create new thread for the next message
    new_threads = cl.user_session.get("new_threads") or {}

    for key in new_threads:
        if new_threads[key] in delete_threads:
            thread = project_client.agents.create_thread()
            new_threads[key] = thread.id
            print(f"Created new thread: {thread.id}")

    # Update new threads in the user session
    cl.user_session.set("new_threads", new_threads)


# Chainlit setup
@cl.on_chat_start
async def on_chat_start():
    # Initialize user session
    current_agent: Agent[MultiAgentContext] = triage_agent
    input_items: list[TResponseInputItem] = []

    cl.user_session.set("current_agent", current_agent)
    cl.user_session.set("input_items", input_items)
    cl.user_session.set("context", MultiAgentContext())

    # Create a thread for the agent
    thread = project_client.agents.create_thread()
    cl.user_session.set("new_threads", {
        FAQ_AGENT_ID: thread.id,
    })


@cl.on_message
async def on_message(message: cl.Message):
    cl.user_session.set("start_time", time.time())
    user_input = message.content

    for element in message.elements:
        # check if the element is an image
        if element.mime.startswith("image/"):
            user_input += f"\n[uploaded image] {element.path}"
            print(f"Received file: {element.path}")

    asyncio.run(main(user_input))

if __name__ == "__main__":
    # Chainlit will automatically run the application
    pass