import gradio as gr
from chain import build_chain, traced_query
import utils

def process_pdf(file):
    if file is None:
        return "Please upload a PDF first."
    try:
        utils.qa_chain = build_chain(file.name)
        utils.chat_history.clear()
        utils.pdf_loaded = True
        return "PDF loaded successfully! You can now ask questions."
    except ValueError as e:
        if "empty_document" in str(e):
            return "This PDF has no readable text. It may be a scanned image. Please upload a text-based PDF."
        return f"Failed to load PDF: {str(e)}"
    except Exception as e:
        return f"Unexpected error loading PDF: {str(e)}"

def chat_with_pdf(user_input, history):
    if not utils.pdf_loaded or utils.qa_chain is None:
        return history, "Load a PDF first."

    answer = traced_query(utils.qa_chain, user_input)  # ← was chain.invoke

    history = history + [[user_input, answer]]
    return history, ""

def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown("## Chat with your PDF")

        with gr.Column():
            file        = gr.File(label="Upload PDF", file_types=[".pdf"])
            load_button = gr.Button("Load PDF")
            status      = gr.Textbox(label="Status", interactive=False)

        chatbot    = gr.Chatbot()
        user_input = gr.Textbox(placeholder="Ask a question...")
        send_btn   = gr.Button("Send")

        load_button.click(process_pdf, inputs=file, outputs=status)
        send_btn.click(
            chat_with_pdf,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input]
        )

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)