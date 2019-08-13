
class CID2LCA(object):
    """
    Store a mapping from a CID to all LCAs that include it as the part of
    from_cid cluster set.
    """

    def __init__(self):
        self.cid2lcas = dict()

    def add(self, a):
        """
        Add an LCA to the dictionary, making sure that each CID is there
        and includes the LCA

        Super inefficent and needs to change; involves at least double,
        and sometimes triple hashing.
        """
        for c in a.from_cids():
            if c not in self.cid2lcas:
                self.cid2lcas[c] = set()
            self.cid2lcas[c].add(a)

    def clear(self):
        self.cid2lcas.clear()

    def containing_all_cids(self, cids):
        """Find all LCAs containing all the cids in the list.  If there is
        only one cid, there will be multiple LCAs, but if there are
        multiple cids there will be at most one LCA.

        If a cid is not in the dictionary, it means it is not in any
        LCAs, which can only happen if the cid is an isolated
        singleton. When this is discovered we can immediately return
        the empty set.
        """
        lca_sets = list()
        for c in cids:
            if c in self.cid2lcas:
                lca_sets.append(self.cid2lcas[c])
            else:
                return set()
        if len(lca_sets) == 0:
            return set()
        else:
            return set.intersection(*lca_sets)

    def remove_with_cids(self, cids):
        """Find and remove any LCA containing at least one of the cids.
        Return the set of all removed cids.

        Note that if the "from" cid set of an LCA (call it "a")
        contains two CIDs then a will need to be removed from the LCA
        set of the other cid.
        """
        all_lcas = set()
        for c in cids:
            if c in self.cid2lcas:
                lca_set = self.cid2lcas[c]
                all_lcas.update(lca_set)
                del self.cid2lcas[c]
                for a in lca_set:
                    for cp in a.from_cids():
                        if cp != c:
                            self.cid2lcas[cp].remove(a)
                # print("cid2lca::remove_with_cids: deleted cid", c)
        return all_lcas

    def is_consistent(self):
        all_ok = True
        for c in self.cid2lcas:
            for a in self.cid2lcas[c]:
                if c not in a.from_cids():
                    print("CID2LCA.is_consistent, error found")
                    print("LCA in set for cluster", c, "but does not contain it")
                    all_ok = False

                for cp in a.from_cids():
                    if cp != c and a not in self.cid2lcas[cp]:
                        print("CID2LCA.is_consistent, error found")
                        print("LCA with from_cids", a.from_cids,
                              "is in set for cid", c, "but not for cid", cp)
                        all_ok = False
        return all_ok

    def print_structure(self):
        for c in sorted(self.cid2lcas.keys()):
            print("cid", c, "has the following LCAs:")
            for a in self.cid2lcas[c]:
                print("    ", str(a))


# #############  Testing code  ######################

class LCALite(object):
    def __init__(self, hv, cids):
        self.__hash_value = hv
        self.m_cids = cids

    def __hash__(self):
        return self.__hash_value

    def __eq__(self, o):
        return self.__hash_value == o.__hash_value

    def from_cids(self):
        return self.m_cids

    def __str__(self):
        return "hash = %d, cids = %a" % (self.__hash_value, self.m_cids)


def test_all():
    lcas = [LCALite(747, [0, 1]),
            LCALite(692, [1, 2]),
            LCALite(381, [1]),
            LCALite(826, [2, 5]),
            LCALite(124, [7, 5]),
            LCALite(243, [7, 8]),
            LCALite(710, [2, 4]),
            LCALite(459, [9, 7])]

    c2a = CID2LCA()
    for a in lcas:
        c2a.add(a)

    print("len (should be )", len(c2a.cid2lcas))
    print("keys (should be [0, 1, 2, 4, 5, 7, 8, 9])", sorted(c2a.cid2lcas.keys()))

    lca_set = c2a.containing_all_cids([1, 2])
    print("containg_all_cids [1,2] (should be just 692):")
    for a in lca_set:
        print("    ", str(a))

    lca_set = c2a.containing_all_cids([2])
    print("containg_all_cids [2] (should be 692, 710, 826):")
    for a in lca_set:
        print("    ", str(a))

    lca_set = c2a.containing_all_cids([1, 5])
    print("containg_all_cids [1, 5] (should be len(0)):", len(lca_set))

    lca_set = c2a.containing_all_cids([1, 99])
    print("containg_all_cids [1, 99] (should be len(0)):", len(lca_set))

    print("========\nDictionary structure")
    c2a.is_consistent()
    c2a.print_structure()
    lca_set = c2a.remove_with_cids([2, 5])
    print("after removing [2, 5] returned LCAs should be 124, 692, 710, 826")
    for a in lca_set:
        print("    ", str(a))

    print("========\nDictionary structure")
    c2a.is_consistent()
    c2a.print_structure()
    lca_set = c2a.remove_with_cids([1])
    print("after removing [1] returned LCAs should be 381, 747")
    for a in lca_set:
        print("    ", str(a))

    print("========\nDictionary structure")
    c2a.is_consistent()
    c2a.print_structure()
    lca_set = c2a.remove_with_cids([4])
    print("after removing [4]; should have returned the empty set, did it? ",
          len(lca_set) == 0)

    print("========\nDictionary structure (final)")
    c2a.is_consistent()
    c2a.print_structure()


if __name__ == "__main__":
    # test_lca_lite()
    test_all()
