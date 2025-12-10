import evaluate

available_models = ["bleu", "rouge"]

class Evaluate:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metric = evaluate.load(model_name)

    def score(self, candidate: str, reference: str):
        r = self.metric.compute(predictions=candidate, references=reference)
        
        return {
            "result": r["bleu"]
        }