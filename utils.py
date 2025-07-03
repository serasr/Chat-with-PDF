chat_history = []
qa_chain = None
pdf_loaded = False

def reset_state():
    global chat_history, qa_chain, pdf_loaded
    chat_history.clear()
    qa_chain = None
    pdf_loaded = False
