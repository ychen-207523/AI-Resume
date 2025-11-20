# AI Resume

AI Resume is a small personal chatbot that answers questions **as you**, using:

- Your **LinkedIn PDF** (`me/linkedin.pdf`)
- An extra **summary file** (`me/summary.txt`)
- OpenAI’s Chat + Tools API
- Optional **Pushover** notifications when:
  - Someone shares their email
  - The bot sees a question it couldn’t answer

The bot runs in a simple **Gradio** chat UI.

---

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   # or
   uv pip install -r requirements.txt
   ```

2. Create the `me` folder:

   ```bash
   mkdir me
   ```

3. Add your files:

   - `me/summary.txt`  
     Write anything you want the AI to know about you (this is hidden context, not shown in chat).

   - `me/linkedin.pdf`  
     Export your LinkedIn profile to PDF and put it here.

---

## .env File

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...your-key-here...
OPENAI_MODEL=gpt-4o-mini

# Name used in the AI persona
AI_RESUME_NAME=Your Name Here

# Optional: enable Pushover notifications
PUSHOVER_USER=
PUSHOVER_TOKEN=
```

If `PUSHOVER_USER` or `PUSHOVER_TOKEN` is not set, push notifications are disabled automatically.

---

## Run

```bash
python main.py
# or
uv run python main.py
```

Open the URL printed in the terminal (normally `http://127.0.0.1:7860`) to start chatting with your AI Resume.
