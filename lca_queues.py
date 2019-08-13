import lca_heap as lh


class LCAQueues(object):

    def __init__(self, lcas=None):
        self.Q = lh.LCAHeap()
        if lcas is None:
            self.S = set()
        else:
            self.S = set(lcas)
        self.W = set()
        self.done = set()

    def top_Q(self):
        return self.Q.top_Q()

    def pop_Q(self):
        self.Q.pop_Q()

    def add_to_Q(self, lcas):
        if type(lcas) != list and type(lcas) != set:
            lcas = [lcas]
        for a in lcas:
            self.Q.insert(a)

    def get_S(self):
        return self.S

    def clear_S(self):
        self.S.clear()

    def add_to_S(self, a):
        self.S.add(a)

    def add_to_W(self, a):
        self.W.add(a)

    def num_on_W(self):
        return len(self.W)

    def remove(self, lcas):
        if type(lcas) != list and type(lcas) != set:
            lcas = [lcas]
        for a in lcas:
            if a in self.W:
                self.W.remove(a)
            elif a in self.S:
                self.S.remove(a)
            else:
                self.Q.remove(a)

    def switch_to_splitting(self):
        """
        Clear all queues prior to the creation of singleton LCAs for splitting
        """
        self.Q.clear()
        self.S.clear()
        self.W.clear()

    def switch_to_stability(self):
        """  None of what's below is needed now that we've inserted a splitting phase
        self.S.update(self.W)
        self.W.clear()
        self.S.update(self.Q.get_all())
        self.Q.clear()
        """
        pass

    def add_to_done(self, a):
        self.done.add(a)

    def score_change(self, a, from_delta, to_delta):
        # print("LCAQueues::score_change from_delta %a to_delta %a."
        #       % (from_delta, to_delta), "Current queue is", self.which_queue(a))
        if a in self.S:
            # print("leaving lca in scoring")
            pass   # leave it for an update
        elif to_delta < 0:
            # print("placing lca in scoring")
            self.remove(a)
            self.add_to_S(a)
        else:
            # print("placing lca in main Q")
            self.remove(a)
            self.Q.insert(a)

    def which_queue(self, a):
        if a in self.S:
            return "S"
        elif a in self.W:
            return "W"
        elif a in self.Q.heap:
            return "Q"
        else:
            return None

    def is_consistent(self):
        all_ok = self.Q.is_consistent()
        q_set = set(self.Q.lca2index.keys())

        qs = q_set & self.S
        if len(qs) > 0:
            print("LCA queues, Q and S intersect")
            all_ok = False

        qw = q_set & self.W
        if len(qw) > 0:
            print("LCA queues, Q and W intersect")
            all_ok = False

        sw = self.S & self.W
        if len(sw) > 0:
            print("LCA queues, S and W intersect")
            all_ok = False

        return all_ok


def test_all():
    v = [lh.LCALite(123, 1.0),
         lh.LCALite(456, 5.3),
         lh.LCALite(827, 7.8),
         lh.LCALite(389, 8.9),
         lh.LCALite(648, 8.6),
         lh.LCALite(459, 9.4),
         lh.LCALite(628, 8.2),
         lh.LCALite(747, 4.7)]
    queues = LCAQueues(v)

    print()
    print("After initialization: lengths should be (0, %d, 0)"
          " and are (%d, %d, %d)"
          % (len(v), len(queues.Q), len(queues.S), len(queues.W)))
    queues.get_S()
    queues.clear_S()
    queues.add_to_S(v[0])
    queues.add_to_S(v[1])
    queues.add_to_Q(v[2:-2])
    queues.add_to_W(v[-2])
    queues.add_to_W(v[-1])
    lcas_on_S = v[:2]
    lcas_on_Q = v[2:-2]
    lcas_on_W = v[-2:]
    print("After moving around: lengths should be (%d, 2, 2)"
          " and are (%d, %d, %d)"
          % (len(v) - 4, len(queues.Q), len(queues.S), len(queues.W)))
    print("num_on_W should be %d, and is %d" % (len(queues.W), queues.num_on_W()))
    print("Which queue: should be S and is", queues.which_queue(lcas_on_S[0]))
    print("Which queue: should be Q and is", queues.which_queue(lcas_on_Q[0]))
    print("Which queue: should be W and is", queues.which_queue(lcas_on_W[0]))

    print("Here is Q:")
    queues.Q.print_structure()
    a = queues.top_Q()
    print("top of queue should have values (459, 9.4) and has", str(a))
    queues.pop_Q()
    a1 = queues.top_Q()
    print("popped off queue; new top should have values (389, 8.9) and has",
          str(a1))
    queues.add_to_Q(a)  # put it back on....
    print('put top back on')

    print("---------------")
    print("Testing score_change method:")
    queues.score_change(lcas_on_S[0], 4, -3)
    print("Changed on S should stay on S:", queues.which_queue(lcas_on_S[0]))
    queues.score_change(lcas_on_Q[0], 4, -3)
    print("Negative 'to' score change from Q should be on S:",
          queues.which_queue(lcas_on_Q[0]))
    queues.score_change(lcas_on_Q[1], -4, 3)
    print("Negative 'from' score change (positive to) from Q should be on Q:",
          queues.which_queue(lcas_on_Q[1]))
    queues.score_change(lcas_on_W[0], 4, -3)
    print("Negative 'to' score change from W should be on S:",
          queues.which_queue(lcas_on_W[0]))
    queues.score_change(lcas_on_W[1], -4, 3)
    print("Negative 'from' score change (positive to) from W should be on Q:",
          queues.which_queue(lcas_on_W[1]))


if __name__ == "__main__":
    test_all()
