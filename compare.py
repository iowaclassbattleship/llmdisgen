from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer

class TextComparator():
    def __init__(self, model_type):
        self.scorer = BERTScorer(model_type=model_type)


    def score(self, candidate: str, reference: str):
        P, R, F1 = self.scorer.score([candidate], [reference])

        # Precision, Recall, F1
        return P, R, F1
    