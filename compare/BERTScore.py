from bert_score import BERTScorer

class BERTScore:
    available_models = [
        "bert-base-uncased",
        "microsoft/deberta-large-mnli"
    ]

    def __init__(self, model_type):
        self.model_name = model_type
        self.scorer = BERTScorer(
            lang="en",
            model_type=model_type,
            rescale_with_baseline=True,
        )
        self.scorer._tokenizer.model_max_length = 512

    def metric(self, prediction: str, reference: str):
        if not prediction.strip() or not reference.strip():
            return 0.0, 0.0, 0.0
        P, R, F1 = self.scorer.score([prediction], [reference])

        return P.item(), R.item(), F1.item()
    