from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer

class BERTScore:
    available_models = [
        "bert-base-uncased",
        "microsoft/deberta-large-mnli"
    ]

    def __init__(self, model_type):
        self.model_name = model_type
        self.scorer = BERTScorer(model_type=model_type)

    def metric(self, prediction: str, reference: str):
        if len(prediction) == 0 or len(reference) == 0:
            return 0, 0, 0
        P, R, F1 = self.scorer.score([prediction], [reference])

        # Precision, Recall, F1
        return P, R, F1
    