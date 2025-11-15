import os
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
from pypdf import PdfReader

load_dotenv(override=True)
NAME = os.getenv("AI_RESUME_NAME", "Your Name")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SUMMARY_PATH = Path("me/summary.txt")
PDF_PATH = Path("me/linkedin.pdf")

def load_summary() -> str:
    if SUMMARY_PATH.exists():
        return SUMMARY_PATH.read_text(encoding="utf-8").strip()
    return "(No summary found. Create me/summary.txt)"

SUMMARY = load_summary()
client = OpenAI()

def openai_reply(message: str, history: list[dict]) -> str:
    system = (
        f"You are acting as {NAME}. Answer only using the information below. "
        f"If the information is not present, say you don't have that detail yet.\n\n"
        f"## Summary\n{SUMMARY}\n"
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": message})

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content

GREETING = (
    f"Hi, I'm {NAME}.\n\n"
    f"**Quick summary:**\n{SUMMARY}\n\n"
    f"Ask me anything about my experience or projects."
)


demo = gr.ChatInterface(
    fn=openai_reply,
    type="messages",
    title="AI Resume (Milestone 3)",
    description="Answers grounded to your summary in me/summary.txt."
)

def on_startup():
    return [(None, GREETING)]

demo.chatbot.update(value=[(None, GREETING)])
demo.launch()