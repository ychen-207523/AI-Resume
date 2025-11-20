from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr

import os
from pathlib import Path

from tools import TOOLS_JSON, handle_tool_calls


# ---------------- Env & setup ----------------

load_dotenv(override=True)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
NAME = os.getenv("AI_RESUME_NAME", "User")

SUMMARY_PATH = Path("me/summary.txt")
PDF_PATH = Path("me/linkedin.pdf")


# ---------------- Core Me class ----------------

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
                tools=TOOLS_JSON,
            )

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                tool_message = choice.message
                tool_calls = tool_message.tool_calls or []
                tool_results = handle_tool_calls(tool_calls)

                messages.append(tool_message)
                messages.extend(tool_results)
            else:
                done = True

        return response.choices[0].message.content


# ---------------- Entrypoint ----------------

if __name__ == "__main__":
    me = Me()

    GREETING = (
        f"Hi, I'm {me.name}.\n\n"
        "Ask me anything about my background, experience, projects, or availability.\n"
        "I'll answer based on my resume and LinkedIn profile.\n"
        "If you want to connect, please provide your email and name, thank you!"
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