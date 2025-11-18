from dotenv import load_dotenv
import os
import json
import requests

load_dotenv(override=True)

def push(text: str):
    """
    Send a Pushover notification if credentials are set.
    Otherwise just log to console.
    """
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")

    if not token or not user:
        print(f"[PUSH disabled] {text}", flush=True)
        return

    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={"token": token, "user": user, "message": text},
            timeout=5,
        )
    except Exception as e:
        print(f"[PUSH error] {e} | message={text}", flush=True)


def record_user_details(
    email: str,
    name: str = "Name not provided",
    notes: str = "not provided",
):
    """
    Tool: record that a user is interested in being in touch.
    """
    push(f"AI-Resume: Recording {name} <{email}>; notes={notes}")
    return {"recorded": "ok"}


def record_unknown_question(question: str):
    """
    Tool: record any question the AI could not answer with current context.
    """
    push(f"AI-Resume: Unknown question -> {question}")
    return {"recorded": "ok"}


record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this to record that a user is interested in being in touch and provided an email address.",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user.",
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it.",
            },
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation to give context.",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Use this tool when you could not answer a user's question with the provided context.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered.",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

TOOLS_JSON = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


def handle_tool_calls(tool_calls):
    """
    Execute tools requested by the model and return tool-result messages.
    """
    results = []
    for tc in tool_calls:
        tool_name = tc.function.name
        args = json.loads(tc.function.arguments or "{}")
        print(f"[Tool] {tool_name} args={args}", flush=True)

        fn = globals().get(tool_name)
        if callable(fn):
            result = fn(**args)
        else:
            result = {}

        results.append(
            {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tc.id,
            }
        )

    return results
