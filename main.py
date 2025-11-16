from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr

import json
import os
import requests
from pathlib import Path


# ---------------- Env & setup ----------------

load_dotenv(override=True)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
NAME = os.getenv("AI_RESUME_NAME", "User")

SUMMARY_PATH = Path("me/summary.txt")
PDF_PATH = Path("me/linkedin.pdf")

# ---------------- Push & tools ----------------

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
            data={
                "token": token,
                "user": user,
                "message": text,
            },
            timeout=5,
        )
    except Exception as e:
        print(f"[PUSH error] {e} | message={text}", flush=True)


def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided"):
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
                "description": "The email address of this user."
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it."
            },
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation to give context."
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Use this tool when you could not answer a user's question with the provided context.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered."
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = NAME

        # Load summary
        self.summary = self._load_summary()

        # Load LinkedIn text
        self.linkedin = self._load_linkedin(max_chars=20000)

    def _load_summary(self) -> str:
        if not SUMMARY_PATH.exists():
            return "(No summary.txt found in ./me)"
        try:
            return SUMMARY_PATH.read_text(encoding="utf-8").strip()
        except Exception as e:
            return f"(Error reading summary.txt: {e})"

    def _load_linkedin(self, max_chars: int = 20000) -> str:
        if not PDF_PATH.exists():
            return "(No linkedin.pdf found in ./me)"
        try:
            reader = PdfReader(str(PDF_PATH))
            chunks = []
            for page in reader.pages:
                text = page.extract_text() or ""
                text = " ".join(text.split())
                chunks.append(text)
            full = "\n".join(chunks).strip()
            if len(full) > max_chars:
                full = full[:max_chars] + "\n...[truncated]"
            return full if full else "(LinkedIn PDF extracted no text.)"
        except Exception as e:
            return f"(LinkedIn PDF parse error: {e})"

    def handle_tool_call(self, tool_calls):
        """
        Execute tools requested by the model and return tool-result messages.
        """
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            print(f"[Tool] {tool_name} args={arguments}", flush=True)

            fn = globals().get(tool_name)
            if callable(fn):
                result = fn(**arguments)
            else:
                result = {}

            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })

        return results

    def system_prompt(self) -> str:
        """
        Build a system prompt combining summary + LinkedIn and instructions for tools.
        """
        system_prompt = (
            f"You are acting as {self.name}. You answer questions on {self.name}'s personal website, "
            f"especially about {self.name}'s career, background, skills, and experience. "
            f"Represent {self.name} as faithfully and professionally as possible.\n\n"
            f"You are given a short summary and a LinkedIn profile text dump to ground your answers.\n\n"
            f"If you don't know the answer to a question, use the record_unknown_question tool to log it.\n"
            f"If the user seems interested in connecting, ask for their email and use the record_user_details tool.\n"
        )

        system_prompt += f"\n## Summary:\n{self.summary}\n\n"
        system_prompt += f"## LinkedIn Profile (text dump):\n{self.linkedin}\n\n"
        system_prompt += (
            f"With this context, chat with the user, always staying in character as {self.name}."
        )
        return system_prompt

    def chat(self, message, history):
        """
        Gradio callback: message (str), history (list[dict]) -> assistant reply (str)
        Runs an OpenAI tool loop until a final answer is produced.
        """
        messages = [{"role": "system", "content": self.system_prompt()}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": message})

        response = None
        done = False

        while not done:
            response = self.openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
            )

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                tool_message = choice.message
                tool_calls = tool_message.tool_calls or []
                tool_results = self.handle_tool_call(tool_calls)

                messages.append(tool_message)
                messages.extend(tool_results)
            else:
                done = True

        return response.choices[0].message.content

# ---------------- Entrypoint ----------------

if __name__ == "__main__":
    me = Me()

    # Initial greeting shown in the chat when it opens
    GREETING = (
        f"Hi, I'm {me.name}.\n\n"
        f"**Quick summary:**\n{me.summary}\n\n"
        f"Ask me anything about my experience, projects, or availability."
    )

    demo = gr.ChatInterface(
        fn=me.chat,
        type="messages",
        title="AI Resume",
        description="Ask about my background, projects, and experience.",
        chatbot=gr.Chatbot(
            value=[{"role": "assistant", "content": GREETING}],
            type="messages",
        ),
    )

    demo.launch()
