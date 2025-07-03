import gradio as gr
from chains import build_chain
import utils

def process_pdf(file):
    if file is None:
        return "Please upload a PDF first."
    utils.qa_chain = build_chain(file.name)
    utils.chat_history.clear()
    utils.pdf_loaded = True
    return "PDF loaded successfully! You can now ask questions."

def chat_with_pdf(user_input):
    if not utils.pdf_loaded or utils.qa_chain is None:
        return utils.chat_history, "Load a PDF first."
    result = utils.qa_chain.invoke({"question": user_input})
    answer = result['answer']
    sources = "\n\n Sources:\n" + "\n".join(
        f"🔹 {doc.page_content[:200]}..." for doc in result['source_documents']
    )
    utils.chat_history.append((user_input, answer + sources))
    return utils.chat_history, ""

def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown("## Chat with your PDF (LangChain + HuggingFace + Gradio)")

        with gr.Column():
            file = gr.File(label="Upload PDF", file_types=[".pdf"])
            load_button = gr.Button("Load PDF")
            status = gr.Textbox(label="Status", interactive=False)

        chatbot = gr.Chatbot()
        with gr.Row():
            user_input = gr.Textbox(placeholder="Ask a question...", scale=4)
            send_button = gr.Button("Send", scale=1)

        load_button.click(process_pdf, inputs=file, outputs=status, scroll_to_output=False)
        send_button.click(chat_with_pdf, inputs=user_input, outputs=[chatbot, user_input], scroll_to_output=False)

    demo.launch()
