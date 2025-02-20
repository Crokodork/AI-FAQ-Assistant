from transformers import pipeline

def answer_faq(question, context):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

if __name__ == "__main__":
    context = input("Enter the FAQ context (e.g., FAQ content): ")
    question = input("Enter your question: ")
    answer = answer_faq(question, context)
    print("Answer:", answer)
