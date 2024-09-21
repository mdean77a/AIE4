from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
]

def createResponseDataSet(runnableChain,test_questions,test_groundtruths):
    answers = []
    contexts = []

    for question in test_questions:
        response = runnableChain.invoke({"question" : question})
        answers.append(response["response"].content)
        contexts.append([context.page_content for context in response["context"]])
    
    response_dataset = Dataset.from_dict({
        "question" : test_questions,
        "answer" : answers,
        "contexts" : contexts,
        "ground_truth" : test_groundtruths
    })
    return response_dataset

def evaluateRAGAS(runnableChain,test_questions,test_groundtruths):
    response_dataset = createResponseDataSet(runnableChain,test_questions,test_groundtruths)
    results = evaluate(response_dataset, metrics)
    return results