from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor


class Scorer:
    def __init__(self, pred, gt):
        self.pred = pred 
        self.gt = gt
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Spice(), "SPICE"),
            # (Meteor(), "METEOR"),
        ]

    def evaluate(self):
        total_scores = {}
        for scorer, method in self.scorers:
            score, _ = scorer.compute_score(self.gt, self.pred)
            if isinstance(method, list):
                for sc, m in zip(score, method):
                    total_scores[m] = sc * 10
            else:
                total_scores[method] = score * 10

        return total_scores