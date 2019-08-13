import math as m
import random

import exp_scores as es


class weighter(object):  # NOQA
    """  Object whose primary function is to generate a signed edge
    weight from a verification score. This verification score is
    converted to a value from histograms of positive and negative
    scores and thence to a weight.  The weights are always integer.
    The human_prob of decision correctness is used to scale and bound
    the weights. Specifically, the human_prob -> max_weight and
    nothing can be outside the range -max_weight to max_weight
    """

    def __init__(self, scorer, human_prob=0.98, max_weight=999):
        self.scorer = scorer
        self.human_prob = human_prob
        self.max_weight = int(max_weight)   # just to be sure
        self.max_raw_weight = self.raw_wgt_(human_prob)

    def wgt(self, score):
        """  Given a verification score produce a (scalar) weight """
        w0 = self.raw_wgt_(score)
        w = self.scale_and_trunc_(w0)
        return int(w)

    def human_wgt(self, is_marked_correct):
        """  Given a human decision, produce a weight """
        if is_marked_correct:
            return self.max_weight
        else:
            return -self.max_weight

    def random_wgt(self, is_match_correct):
        """  Generate a random weight depending on whether or not the
        match is correct.
        """
        s = self.scorer.random_score(is_match_correct)
        return self.wgt(s)

    def human_random_wgt(self, is_marked_correct):
        """
        Return random weight for human decision.  Combine the correctness
        of the match with a random flip as to whether the human makes
        a correct decision.
        """
        p = random.random()
        correct_decision = p <= self.human_prob
        if (correct_decision and is_marked_correct) or \
           (not correct_decision and not is_marked_correct):
            return self.max_weight
        else:
            return -self.max_weight

    def raw_wgt_(self, s):
        """  Return a continuous-valued weight from the score, using
        the scorer to get positive and negative histogram values.
        """
        hp, hn = self.scorer.get_pos_neg(s)
        epsilon = 0.000001   # to prevent overflow
        ratio = hp / (hn + epsilon)
        wgt = m.log(ratio)
        return wgt

    def scale_and_trunc_(self, w0):
        """
        Map the weight into an integer in the range -max_weight,... max_weight.
        """
        w0 = max(-self.max_raw_weight,
                 min(self.max_raw_weight, w0))
        w = round(w0 / self.max_raw_weight * self.max_weight)
        return w


if __name__ == "__main__":
    error_frac = 0.15
    neg_pos_ratio = 5.0
    scorer = es.exp_scores.create_from_error_frac(error_frac, neg_pos_ratio)

    human_prob = 0.98
    max_weight = 999
    s2w = weighter(scorer, human_prob, max_weight)

    print("\nSampling of weights:")
    n = 100
    for i in range(n + 1):
        s = i / n
        print("score: %4.2f, wgt %d" % (s, s2w.wgt(s)))

    """
    n = 25
    print("\nGenerating", n, "positives")
    wgts = sorted([ew.random_wgt(is_match_correct=True) for i in range(n)])
    for w in wgts:
        print(w, end=' ')
    print()

    print("\nGenerating", n, "negatives")
    wgts = sorted([ew.random_wgt(is_match_correct=False) for i in range(n)])
    for w in wgts:
        print(w, end=' ')
    print()

    marked_correct = True
    print("\nPositive weight from human = %d" % ew.human_wgt(marked_correct))

    marked_correct = False
    print("Negative weight from human = %d" % ew.human_wgt(marked_correct))

    is_match_correct = True
    num_samples = 1000
    num_errors = 0
    for i in range(num_samples):
        wgt = ew.human_random_wgt(is_match_correct)
        if wgt != ew.max_weight:
            num_errors += 1
    print("Positive human_random_wgt: num errors %d, num samples %d, pct errors %1.2f"
          % (num_errors, num_samples, 100*num_errors/num_samples))

    """
