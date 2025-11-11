import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv(override=True)
NAME = os.getenv("AI_RESUME_NAME", "Your Name")

def reply(message, history):
    return f"Hi, I'm {NAME}. You said: {message}"

demo = gr.ChatInterface(fn=reply, type="messages", title="AI Resume (Milestone 1)")

if __name__ == "__main__":
    demo.launch()
