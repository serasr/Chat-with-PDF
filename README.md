```yaml
---
title: Chat with PDF
emoji: 💬
colorFrom: indigo
colorTo: violet
sdk: gradio
sdk_version: "5.35.0"
app_file: app.py
pinned: false
---
```

# Chat with PDF

An interactive app that lets you upload a PDF and ask natural language questions about its content using:

- **LangChain** (for Retrieval-Augmented Generation)
- **Hugging Face Transformers** (for open-source LLMs)
- **FAISS** (for fast, semantic document retrieval)
- **Gradio** (for a clean, interactive web UI)

---

## Features

--> Upload any PDF  
--> Ask follow-up questions (conversation memory)  
--> Displays source snippets used to answer  
--> Uses open-source models like `flan-t5-large`  
--> Ready to deploy on Hugging Face Spaces or locally

---

## Project Structure

```bash
chat-with-pdf/
├── app.py               # Entry point
├── chains.py            # Builds the LangChain RAG pipeline
├── interface.py         # Gradio UI components
├── utils.py             # Shared state + reset logic
├── requirements.txt     # Dependencies
├── README.md            # You’re here!
├── CHANGELOG.md         # Version changes log
└── LICENSE              # Open-source license
```

## Local Development

Install dependencies and run the app.

```bash
pip install -r requirements.txt
python app.py
```
Visit http://localhost:7860 in your browser.

## Deployed on Hugging Face Spaces

This app is designed for easy deployment to [Hugging Face Spaces](https://huggingface.co/spaces).  
It uses the official `Gradio` SDK with version control via `sdk_version`.

Deployment config is set using [YAML front matter](https://huggingface.co/docs/hub/spaces-overview#managing-your-space) at the top of this `README.md`:

## Deploy to Hugging Face Spaces

--> Create a new Space (SDK: Gradio)
--> Upload all project files (especially app.py, requirements.txt, README.md)
--> Done! Your app is live at: https://huggingface.co/spaces/<your-name>/<space-name>

## Roadmap

--> Add multi-PDF support
--> Cache embeddings for re-use
--> Highlight source text in answers
--> Add feedback & ratings to responses
--> Export chat history (Markdown / PDF)
--> Add multilingual PDF support

## Contributing

This is the first version and designed to evolve. PRs and discussions are welcome!

## License

This project is released under MIT license.



