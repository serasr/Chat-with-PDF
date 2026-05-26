# Golden test set - questions with known correct answers
# These should be representative of real user questions
# Add more as you learn what users actually ask

TEST_QUESTIONS = [
    {
        "question": "What is the main contribution of this paper?",
        "ground_truth": "The paper contributes systematic quantification of numerical fabrication in RAG systems and comprehensive evaluation of four detection methods spanning embedding-based, metric-based, and LLM-based paradigms."
    },
    {
        "question": "What four detection methods does the paper evaluate?",
        "ground_truth": "The paper evaluates four detection methods: semantic similarity, BERTScore, number heuristic, and an LLM-based approach."
    },
    {
        "question": "What is the dataset size used in the experiments?",
        "ground_truth": "500 examples randomly split into 400 for testing and 100 for development."
    },
    {
        "question": "What domains does the paper identify as high-stakes for numerical accuracy?",
        "ground_truth": "Finance, medicine, and science are identified as high-stakes domains where numerical accuracy is critical."
    },
    {
        "question": "What is the key finding about existing hallucination detection methods?",
        "ground_truth": "Existing methods have primarily focused on semantic correctness and potentially miss failures involving specific numerical details."
    },
]