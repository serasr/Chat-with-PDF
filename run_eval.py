# Run this script to evaluate the RAG system
# Usage: python run_eval.py path/to/your.pdf

import sys
from chain import build_chain
from evaluator import run_evaluation
from eval_set import TEST_QUESTIONS

if len(sys.argv) < 2:
    print("Usage: python run_eval.py path/to/your.pdf")
    sys.exit(1)

pdf_path = sys.argv[1]

print(f"Loading PDF: {pdf_path}")
chain = build_chain(pdf_path)

results = run_evaluation(chain, TEST_QUESTIONS)