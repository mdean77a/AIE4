import json

# Load the three data sets
def loadDataSets():

# Function to load the JSONL data
    def load_dataset(file_path):
        with open(file_path, "r") as f:
            dataset = json.load(f)
        
        # Extract the individual components
        questions = dataset.get("questions", [])
        relevant_contexts = dataset.get("relevant_contexts", [])
        corpus = dataset.get("corpus", {})
        
        return questions, relevant_contexts, corpus

# Load test dataset
    file_path = "test_dataset.jsonl"
    questions, relevant_contexts, corpus = load_dataset(file_path)
    test_dataset = {
        "questions" : questions,
        "relevant_contexts" : relevant_contexts,
        "corpus" : corpus
    }

# Load training dataset
    file_path = "training_dataset.jsonl"
    questions, relevant_contexts, corpus = load_dataset(file_path)
    training_dataset = {
        "questions" : questions,
        "relevant_contexts" : relevant_contexts,
        "corpus" : corpus
    }

# Load validation dataset
    file_path = "val_dataset.jsonl"
    questions, relevant_contexts, corpus = load_dataset(file_path)
    val_dataset = {
        "questions" : questions,
        "relevant_contexts" : relevant_contexts,
        "corpus" : corpus
    }

    return test_dataset, training_dataset, val_dataset




