import collections

class CID2LCA(object):
    """
    Store a mapping from a cluster id (a hashable object) to all LCAs
    that include it as the part of its from_cid cluster set.
    """

    def __init__(self):
        self.cid2lcas = collections.defaultdict(set)

    def add(self, a):
        ''' Add an LCA to the dictionary, making sure that each CID is there
            and includes the LCA
        '''
        for c in a.from_cids():
            self.cid2lcas[c].add(a)

    def clear(self):
        self.cid2lcas.clear()

    def containing_all_cids(self, cids):
        '''
        Find all LCAs containing all the cids in the list.  If there is
        only one cid in the cids list, there will likely be multiple
        LCAs returned, but if there are multiple cids there will be at
        most one LCA.

        If a cid is not in the dictionary, it means it is not in any
        LCAs, which can only happen if the cid is an isolated
        singleton (a rare event, but we need to cover it).
        When this is discovered we can immediately return the empty set.
        '''
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
        '''
        Find and remove any LCA containing at least one of the cids.
        Return the set of all removed LCAs.

        Note that if the "from" cid set of an LCA (call it "a")
        contains two CIDs then a will need to be removed from the LCA
        set of the other cid.
        '''
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
        '''
        Check the consistency of the dictionary. Each LCA associated
        with each cid (cluster id) must have the cid in its from_id
        set, and each other cid in from_cid for each LCA must be in
        the dictionary and must store the LCA.
        '''
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


#############  Testing code  ######################

class lca_lite(object):
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
    lcas = [lca_lite(747, [0, 1]),
            lca_lite(692, [1, 2]),
            lca_lite(381, [1]),
            lca_lite(826, [2, 5]),
            lca_lite(124, [7, 5]),
            lca_lite(243, [7, 8]),
            lca_lite(710, [2, 4]),
            lca_lite(459, [9, 7])]

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
    if c2a.is_consistent():
        print('All consistent.')
    c2a.print_structure()
    lca_set = c2a.remove_with_cids([2, 5])
    print("after removing [2, 5] returned LCAs should be 124, 692, 710, 826")
    for a in lca_set:
        print("    ", str(a))

    print("========\nDictionary structure")
    if c2a.is_consistent():
        print('All consistent.')
    c2a.print_structure()
    lca_set = c2a.remove_with_cids([1])
    print("after removing [1] returned LCAs should be 381, 747")
    for a in lca_set:
        print("    ", str(a))

    print("========\nDictionary structure")
    if c2a.is_consistent():
        print('All consistent.')
    c2a.print_structure()
    lca_set = c2a.remove_with_cids([4])
    print("after removing [4]; should have returned the empty set, did it? ",
          len(lca_set) == 0)

    print("========\nDictionary structure (final)")
    if c2a.is_consistent():
        print('All consistent.')
    c2a.print_structure()


if __name__ == "__main__":
    test_all()
