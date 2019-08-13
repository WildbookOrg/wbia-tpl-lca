"""
Although this depends on the LCA having a hash value, it can be tested on any
object that has the following:

1. hash value
2. delta_score method

"""


class lca_heap(object):  # NOQA

    def __init__(self):
        self.heap = []
        self.lca2index = dict()

    def top_Q(self):
        return self.heap[0]

    def __len__(self):
        return len(self.heap)

    def pop_Q(self):
        if len(self.heap) == 1:
            self.clear()
        else:
            a_top = self.heap[0]
            del self.lca2index[a_top]
            a_last_leaf = self.heap.pop()
            self.heap[0] = a_last_leaf
            self.lca2index[a_last_leaf] = 0
            self.percolate_down(0)

    def remove(self, a):
        """
        a.pprint()
        print("len of index", len(self.lca2index))
        print("len of heap", len(self.heap))
        """
        assert(a in self.lca2index)

        # Special case of removing the end of list.
        # This also works for removing the last item.
        loc = self.lca2index[a]
        del self.lca2index[a]
        if loc == len(self.heap) - 1:
            self.heap.pop()
            return

        a_last_leaf = self.heap.pop()
        self.heap[loc] = a_last_leaf
        self.lca2index[a_last_leaf] = loc
        p_loc = (loc - 1) // 2
        if p_loc >= 0 and \
           self.heap[p_loc].delta_score() < self.heap[loc].delta_score():
            self.percolate_up(loc)
        else:
            self.percolate_down(loc)

    def insert(self, a):
        self.heap.append(a)
        last_index = len(self.heap) - 1
        self.lca2index[a] = last_index
        self.percolate_up(last_index)

    def clear(self):
        self.heap.clear()
        self.lca2index.clear()

    def get_all(self):
        return self.heap

    def percolate_down(self, i):
        i_lca = self.heap[i]
        i_delta = i_lca.delta_score()
        last_interior = len(self.heap) // 2 - 1

        while i <= last_interior:
            c = 2 * i + 1
            c_lca = self.heap[c]
            c_delta = c_lca.delta_score()
            rc = c + 1
            if rc < len(self.heap) and self.heap[rc].delta_score() > c_delta:
                c = rc
                c_lca = self.heap[c]
                c_delta = c_lca.delta_score()

            if i_delta >= c_delta:
                break
            else:
                self.heap[i] = c_lca
                self.lca2index[c_lca] = i
                i = c

        self.lca2index[i_lca] = i
        self.heap[i] = i_lca

    def percolate_up(self, i):
        i_lca = self.heap[i]
        i_delta = i_lca.delta_score()
        while i > 0:
            p = (i - 1) // 2
            p_lca = self.heap[p]
            p_delta = p_lca.delta_score()
            if p_delta >= i_delta:
                break
            else:
                self.heap[i] = p_lca
                self.lca2index[p_lca] = i
                i = p
        self.heap[i] = i_lca
        self.lca2index[i_lca] = i

    def print_structure(self):
        print("Here is the heap vector")
        for i, lca in enumerate(self.heap):
            print("    %d: %s" % (i, str(lca)))
        print("Here is the dictionary")
        for k, v in self.lca2index.items():
            print("    %s: %d" % (str(k), v))

    def is_consistent(self):
        is_ok = True
        if len(self.heap) != len(self.lca2index):
            is_ok = False
            print('is_consistent: heap is len', len(self.heap),
                  ' while lca2index is len', len(self.lca2index))

        # Make sure each index is in the heap
        for i, lca in enumerate(self.heap):
            if lca not in self.lca2index:
                print('lca at location %d with heap value %d not in lca2index'
                      % (i, lca.__heap))
                is_ok = False

        # Make sure each lca2index entry is unique
        if len(self.lca2index.values()) != len(set(self.lca2index.values())):
            print('Duplicated indices in lca2index values')
            is_ok = False

        # Make sure all indices are in range
        if not all([0 <= i < len(self.heap) for i in self.lca2index.values()]):
            print('At least one index out of range in self.lca2index.values')
            is_ok = False

        # finally test to see if the ordering property is maintained
        last_internal = len(self.heap) // 2 - 1
        for i in range(last_internal + 1):
            lchild = 2 * i + 1
            rchild = lchild + 1
            if self.heap[i].delta_score() < self.heap[lchild].delta_score():
                print("Heap index %d has score %1.1f less than left child %d with score %1.1f"
                      % (i, self.heap[i].delta_score(), lchild, self.heap[lchild].delta_score()))
                is_ok = False

            if rchild < len(self.heap) and \
               self.heap[i].delta_score() < self.heap[lchild].delta_score():
                print("Heap index %d has score %1.1f, less than right child %d with score %1.1f"
                      % (i, self.heap[i].delta_score(), rchild, self.heap[rchild].delta_score()))
                is_ok = False

        if not is_ok:
            print("Output of inconsistent data structure")
            self.print_structure()

        return is_ok


class lca_lite(object):  # NOQA
    def __init__(self, hash_value, delta_s):
        self.__hash = hash_value
        self.m_delta_score = delta_s

    def __hash__(self):
        return self.__hash

    def delta_score(self):
        return self.m_delta_score

    def __str__(self):
        return "hash = %d, delta_score = %1.1f" % (self.__hash, self.m_delta_score)

    def pprint(self):
        print(str(self))


def test_lca_lite():
    s = set()
    s.add(lca_lite(123, 0.78))
    print(len(s))


def test_all():
    h = lca_heap()

    v = [lca_lite(123, 1.0),
         lca_lite(456, 5.3),
         lca_lite(827, 7.8),
         lca_lite(389, 8.9),
         lca_lite(648, 8.6),
         lca_lite(459, 9.4),
         lca_lite(628, 8.2),
         lca_lite(747, 4.7)]
    remove0 = v[1]

    found_error = False
    for a in v:
        h.insert(a)
        if not h.is_consistent():
            found_error = True
            print("Breaking on inconsistency")
            break

    if not found_error:
        print("After %d successful inserts the heap looks like" % len(h))
        h.print_structure()

    print("Top of queue should be (459, 9.4) and is %s" % str(h.top_Q()))
    remove1 = lca_lite(585, 8.5)
    h.insert(remove1)

    h.pop_Q()
    print("After pop_Q")
    if not h.is_consistent():
        print("Inconsistent")
    else:
        print("Here is queue")
        h.print_structure()
        print("top value should have delta_score 8.9.  It has %s" % str(h.get_all()[0]))

    h.insert(lca_lite(183, 8.3))
    print("Trying two remove operations; one should trigger percolate up and the other percolate down")
    h.remove(remove0)
    h.remove(remove1)
    if not h.is_consistent():
        print("Inconsistent")
    else:
        print("Here is queue")
        h.print_structure()

    while not len(h) == 0:
        h.pop_Q()
        if not h.is_consistent():
            print("Inconsistent")
            break

    print("Emptied the queue")

    print("Running special inserts to trigger more percolate up calls in remove.")
    v = [lca_lite(123, 19),
         lca_lite(459, 16),
         lca_lite(628, 6),
         lca_lite(747, 13),
         lca_lite(827, 11),
         lca_lite(389, 4),
         lca_lite(456, 2),
         lca_lite(277, 12),
         lca_lite(648, 8)]
    remove0 = v[5]
    remove1 = v[6]
    for a in v:
        h.insert(a)
        if not h.is_consistent():
            print("Inconsistent during insert")
            break
    h.remove(remove0)
    h.remove(remove1)
    if not h.is_consistent():
        print("Inconsistent during remove")

    h.clear()
    print("Running a bunch more inserts and removes.")
    v = [lca_lite(123, 1.0),
         lca_lite(459, 9.4),
         lca_lite(628, 8.2),
         lca_lite(747, 8.7),
         lca_lite(827, 7.8),
         lca_lite(389, 8.9),
         lca_lite(456, 5.3),
         lca_lite(277, 6.7),
         lca_lite(648, 8.6),
         lca_lite(723, 9.9),
         lca_lite(823, 2.3),
         lca_lite(234, 6.5)]

    error_found = False
    for a in v:
        h.insert(a)
        if not h.is_consistent():
            print("Inconsistent during insert")
            error_found = True
            break
    if not error_found:
        print("No errors")

    error_found = False
    for a in v:
        h.remove(a)
        if not h.is_consistent():
            print("Inconsistent during remove")
            error_found = True
            break
    if not error_found:
        print("No errors")


if __name__ == "__main__":
    test_lca_lite()
    # test_all()
