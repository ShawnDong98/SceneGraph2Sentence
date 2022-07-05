from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    def compute_bleu_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = sent1
            sent2 = sent2
            score.append(sentence_bleu([sent1], sent2,
            weights=(self.w1, self.w2, self.w3, self.w4)))

        return score