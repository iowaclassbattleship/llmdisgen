from compare.BlockMatch import BlockMatch
from compare.BERTScore import BERTScore, available_models

prediction = [
    "The abstract highlights an increased interest in assessing the impact of biomedical research on clinical guidelines over the past five years. The study focuses on cancer research and analyzes the 43 UK guidelines and associated Health Technology Assessments published up to October 2006. The authors aim to determine the geographical provenance and type of research of these guidelines, comparing them to overall oncology research published in the peak years of guideline references (1999-2001).",
    "The study found that UK papers were cited nearly three times as frequently as would have been expected from their presence in world oncology research (6.5%). Edinburgh and Glasgow stood out for their unexpectedly high contributions to the guidelines' scientific base. Additionally, the cited papers from the UK acknowledged more explicit funding from all sectors than did the UK cancer research papers at the same research level.",
    "This discussion raises several questions and points of interest:",
    "1. Increased interest in impact assessment: The study notes a substantially increased interest in biomedical research impact assessment over the past five years. This highlights the growing importance of evaluating the effectiveness and efficiency of research in the biomedical field.\n2. Geographic provenance of guidelines: The study analyzes the geographical provenance of the guidelines, which can help",
]

reference = [
    "The abstract highlights an increased interest in assessing the impact of biomedical research on clinical guidelines over the past five years. The study focuses on cancer research and analyzes the 43 UK guidelines and associated Health Technology Assessments published up to October 2006. The authors aim to determine the geographical provenance and type of research of these guidelines, comparing them to overall oncology research published in the peak years of guideline references (1999-2001).",
    "The study found that UK papers were cited nearly three times as frequently as would have been expected from their presence in world oncology research (6.5%). Edinburgh and Glasgow stood out for their unexpectedly high contributions to the guidelines' scientific base. Additionally, the cited papers from the UK acknowledged more explicit funding from all sectors than did the UK cancer research papers at the same research level.",
    "This discussion raises several questions and points of interest:",
    "1. Increased interest in impact assessment: The study notes a substantially increased interest in biomedical research impact assessment over the past five years. This highlights the growing importance of evaluating the effectiveness and efficiency of research in the biomedical field.\n2. Geographic provenance of guidelines: The study analyzes the geographical provenance of the guidelines, which can help",
]

B = BlockMatch()
BERT = BERTScore(model_type=available_models[0])

P, R, F1 = B.metric(prediction, prediction, BERT.metric)

print(P)