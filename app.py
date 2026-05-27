import os
from transformers import pipeline

def main():
    # Hide intrusive TensorFlow/Hugging Face logging warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    print("Loading Question-Answering Model... (This may take a moment on first run)")
    # Initialize the default QA pipeline (DistilBERT trained on SQuAD)
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    print("Model loaded successfully!\n" + "="*50)
    
    # 1. Accept the core text context from the user
    print("Step 1: Enter or paste the context paragraph/text below.")
    print("Press Enter twice (or leave a blank line) when you are finished typing:")
    
    context_lines = []
    while True:
        line = input()
        if line == "":
            break
        context_lines.append(line)
        
    context = "\n".join(context_lines)
    
    if not context.strip():
        print("Context cannot be empty. Exiting script.")
        return

    print("\n" + "="*50)
    print("Step 2: Ask questions! (Type 'exit' or 'quit' to stop)")
    print("="*50)

    # 2. Continuous loop to accept user questions
    while True:
        question = input("\nYour Question: ").strip()
        
        # Check for exit command
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not question:
            print("Please type a valid question.")
            continue
            
        # Get response from the model
        try:
            result = qa_pipeline(question=question, context=context)
            
            # Print the response cleanly
            print(f"Answer: {result['answer']}")
            print(f"Confidence Score: {round(result['score'] * 100, 2)}%")
            
        except Exception as e:
            print(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    main()