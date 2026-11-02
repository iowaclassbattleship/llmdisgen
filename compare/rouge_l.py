import evaluate

_rouge = evaluate.load("rouge")


def metric(prediction: str, reference: str):
    if not prediction.strip() or not reference.strip():
        return 0.0, 0.0, 0.0

    result = _rouge.compute(
        predictions=[prediction],
        references=[reference],
        rouge_types=["rougeL"],
    )

    return None, None, result["rougeL"]
