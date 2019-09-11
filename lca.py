import networkx as nx
import cluster_tools as ct
import test_cluster_tools as tct

g_cluster_counter = 0


"""
Additional testing and other notes:

1. Make sure that it handles single clusters in the cid list; fails if this
single cluster has only one node.

2. Change the to_clusters to make it a dictionary as well for
internal consistency.

3. Careful to avoid issues with having the operator values reversed.

4. Pass the graph as argument to the constructor

5. Add code to remember what list a candidate is on.

clustering is a mapping from a cluster id to a set of node ids.

"""


class LCA(object):
    def __init__(self, subG, clustering, cids, score):
        self.subgraph = subG   # Restricted to the clustering
        self.from_clusters = {c: clustering[c] for c in cids}
        self.from_cids_sorted = tuple(sorted(cids))
        self.__hash_value = hash(self.from_cids_sorted)
        self.from_n2c = ct.build_node_to_cluster_mapping(self.from_clusters)
        self.from_score = score
        self.to_clusters = None
        self.to_score = None
        self.to_n2c = None

        self.inconsistent_pairs = None
        self.inconsistent_edges = None
        self.pairs_for_verification = None
        self.edges_for_manual = None

        self.num_per_verify = 1
        self.num_per_manual = 1

        self.incomparable_threshold = 3

    def __hash__(self):
        return self.__hash_value

    def from_cids(self):
        return self.from_cids_sorted

    def from_node2cid(self):
        return self.from_n2c

    def nodes(self):
        return set.union(*self.from_clusters.values())

    def set_to_clusters(self, to_clusters, to_score):
        self.to_score = to_score

        if self.to_clusters is None or \
           not ct.same_clustering(self.to_clusters, to_clusters):
            # print("lca::set_to_clusters: new 'to' clustering")
            self.to_clusters = to_clusters
            self.to_n2c = ct.build_node_to_cluster_mapping(self.to_clusters)
            self.inconsistent_pairs = None
            self.inconsistent_edges = None
            self.pairs_for_verification = None
            self.edges_for_manual = None
        else:
            # print("lca::set_to_clusters: 'to' clustering has not changed")
            pass

    def delta_score(self):
        return self.to_score - self.from_score

    def get_score_change(self, delta_w, n0_cid, n1_cid):
        if n0_cid == n1_cid:
            return delta_w
        else:
            return -delta_w

    def node_pair_inconsistent(self, m, n):
        same_in_from = self.from_n2c[m] == self.from_n2c[n]
        same_in_to = self.to_n2c[m] == self.to_n2c[n]
        return same_in_from != same_in_to

    def build_inconsistency_sets(self):
        assert(self.to_clusters is not None)
        self.inconsistent_pairs = set()
        self.inconsistent_edges = set()

        nodes = sorted(self.nodes())
        for i, m in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                n = nodes[j]
                if self.node_pair_inconsistent(m, n):
                    if n not in self.subgraph[m]:   # edge does not exist yet
                        self.inconsistent_pairs.add((m, n))
                    else:
                        self.inconsistent_edges.add((m, n))

    def get_node_pairs_for_verification(self):
        """
        Returns list of node pairs.
        """
        if self.inconsistent_pairs is None:
            self.build_inconsistency_sets()

        if self.pairs_for_verification is None:
            self.pairs_for_verification = self.inconsistent_pairs.copy()

        if len(self.pairs_for_verification) > 0:
            n = min(self.num_per_verify,
                    len(self.pairs_for_verification))
            v = {self.pairs_for_verification.pop() for i in range(n)}
            return v
        else:
            return set()

    def get_edges_for_manual(self):
        """
        Returns list of node pairs.
        """
        if self.inconsistent_edges is None:
            self.build_inconsistency_sets()

        """
        Remove the pairs that are considered "incomparable"
        """
        to_remove = set()
        for pr in self.inconsistent_edges:
            m, n = pr
            if 'incomparable' in self.subgraph[m][n] and \
               self.subgraph[m][n]['incomparable'] >= self.incomparable_threshold:
                print('lca::get_edges_fxsor_manual: pair', pr, 'will be considered incomparable')
                to_remove.add(pr)
        self.inconsistent_edges -= to_remove
        
        print("lca::get_edges_for_manual: inconsistent_edges:", self.inconsistent_edges)
        print('edges_for_manual', self.edges_for_manual)
        if len(self.inconsistent_edges) == 0:
            return set()

        """ 
        If the manual augmentation has not started or it has been
        exhausted, (re)start the process.
        """
        if self.edges_for_manual is None or \
           len(self.edges_for_manual) == 0:
            self.edges_for_manual = list(self.inconsistent_edges)
            self.edges_for_manual.sort(key=lambda e: abs(self.subgraph[e[0]][e[1]]['weight']))
        
        n = min(self.num_per_manual, len(self.edges_for_manual))
        v = self.edges_for_manual[-n:]
        del self.edges_for_manual[-n:]
        print('n=', n, 'v=', v, 'edges_for_manual', self.edges_for_manual)
        return set(v)

    def add_edge(self, e):
        """
        Do not change weight here because the subgraph aliases the
        overall graph.  Assume the calling function makes this change.
        """
        n0, n1, wgt = e
        delta_wgt = wgt
        """  The following is removed because weights are now additive """
        # if n1 in self.subgraph[n0]:
        #    delta_wgt -= self.subgraph[n0][n1]['weight']

        #  Update from score
        n0_cid = self.from_n2c[n0]
        n1_cid = self.from_n2c[n1]
        same_in_from = n0_cid == n1_cid
        from_score_change = self.get_score_change(delta_wgt, n0_cid, n1_cid)
        self.from_score += from_score_change

        """The to_clusters may not yet exist, which could occur if this LCA
        has just been created.  In this case, there is nothing more to
        do and we can safely return 0 for the to_delta score because
        this LCA will already be on the scoring queue and will be left there.
        """
        if self.to_n2c is None:
            return (from_score_change, 0)

        n0_cid = self.to_n2c[n0]
        n1_cid = self.to_n2c[n1]
        same_in_to = n0_cid == n1_cid
        to_score_change = self.get_score_change(delta_wgt, n0_cid, n1_cid)
        self.to_score += to_score_change

        #  If the edge is inconsistent and the augmentation process has
        #  already started then
        if same_in_from != same_in_to and self.inconsistent_pairs is not None:
            # Remove it from the inconsistent node pairs list if it is there
            # (This means the edge is new.)
            pr = tuple([n0, n1])
            if pr in self.inconsistent_pairs:
                self.inconsistent_pairs.remove(pr)
                if self.pairs_for_verification is not None and \
                   pr in self.pairs_for_verification:
                    self.pairs_for_verification.remove(pr)

            # Add it to the inconsistent edges list so it will
            # be picked up next time through, but remove it for
            # now from the list to ensure it is not repeatedly
            # checked (perhaps by another LCA)
            self.inconsistent_edges.add(pr)
            if self.edges_for_manual is not None and \
               pr in self.edges_for_manual:
                self.edges_for_manual.remove(pr)

        # Finally, return the score changes
        return (from_score_change, to_score_change)

    def pprint_short(self, stop_after_from=False):
        print("from:", end='')  # NOQA
        for cid in sorted(self.from_clusters.keys()):
            print(" %d: %a" % (cid, sorted(self.from_clusters[cid])), end='')
        check_score = ct.clustering_score(self.subgraph, self.from_n2c)
        if check_score != self.from_score:
            print("\nfrom score error: should be %a, but is %a"
                  % (check_score, self.from_score))
        if stop_after_from:
            print()
            return

        print("; to:", end='')
        for cid in sorted(self.to_clusters.keys()):
            print(" %a" % (sorted(self.to_clusters[cid])), end='')
        check_score = ct.clustering_score(self.subgraph, self.to_n2c)
        if check_score != self.to_score:
            print("\nto score error: should be %a, but is %a"
                  % (check_score, self.to_score))
        else:
            print("; delta", self.delta_score())

    def pprint(self, stop_after_from=False):
        print("from_n2c:", self.from_n2c)
        print("subgraph nodes", self.subgraph.nodes())
        check_score = ct.clustering_score(self.subgraph, self.from_n2c)
        print("from clusters (score = %a, checking %a):"
              % (self.from_score, check_score))
        if self.from_score != check_score:
            print("lca: SCORING ERROR in from")
        for cid in sorted(self.from_clusters.keys()):
            print("    %d: %a" % (cid, self.from_clusters[cid]))
        if stop_after_from:
            return

        check_score = ct.clustering_score(self.subgraph, self.to_n2c)
        print("to clusters (score = %a, checking = %a):"
              % (self.to_score, check_score))
        if self.to_score != check_score:
            print("SCORING ERROR in to")
        for cid in sorted(self.to_clusters.keys()):
            print("    %d: %a" % (cid, self.to_clusters[cid]))
        print("score_difference %a" % self.delta_score())

        if self.inconsistent_pairs is None:
            print("have not started augmentation")
        else:
            print("inconsistent_pairs:", self.inconsistent_pairs)
            print("inconsistent_edges:", self.inconsistent_edges)
            if self.pairs_for_verification is not None:
                print("pairs_for_verification:", self.pairs_for_verification)
            if self.edges_for_manual is not None:
                print("edges_for_manual:", sorted(self.edges_for_manual))


def build_example_LCA():
    G = tct.ex_graph_fig1()
    n2c_optimal = {'a': 0, 'b': 0, 'd': 0, 'e': 0,
                   'c': 1,
                   'h': 2, 'i': 2,
                   'f': 3, 'g': 3, 'j': 3, 'k': 3}
    clustering_opt = ct.build_clustering(n2c_optimal)
    cid0 = 2
    cid1 = 3
    nodes_in_clusters = list(clustering_opt[2] | clustering_opt[3])
    subG = G.subgraph(nodes_in_clusters)

    score = ct.cid_list_score(subG, clustering_opt, n2c_optimal, [cid0, cid1])
    a = LCA(subG, clustering_opt, [cid0, cid1], score)

    to_clusters = {0: {'f', 'h', 'i', 'j'}, 1: {'g', 'k'}}
    subG = G.subgraph(nodes_in_clusters)
    to_node2cid = {n: cid for cid in range(len(to_clusters))
                   for n in to_clusters[cid]}
    to_score = ct.clustering_score(subG, to_node2cid)
    a.set_to_clusters(to_clusters, to_score)

    return a, G


def test_LCA_class():
    print("===========================")
    print("=====  test_LCA_class =====")
    print("===========================")

    a, G = build_example_LCA()

    print("a.from_cids should return [2, 3]; returns", sorted(a.from_cids()))
    print("a.nodes should return ['f', 'g', 'h', 'i', 'j', 'k']; returns",
          sorted(a.nodes()))

    print("a.delta_score should be -18 and it is", a.delta_score())

    a.build_inconsistency_sets()
    a.pprint()

    print("-------")
    print("1st call to get_node_pairs_for_verification should return"
          " (f,h) and (h,j):")
    v = a.get_node_pairs_for_verification()
    print(v)

    print("-------")
    print("2nd call to get_node_pairs_for_verification should return []:")
    v = a.get_node_pairs_for_verification()
    print(v)

    print("------")
    print("1st call to get_edges_for_manual")
    v = a.get_edges_for_manual()
    print("Should return (j, k), (f, g):", v)
    print("and four edges should remain on list", a.edges_for_manual)

    print("------")
    print("2nd call to get_edges_for_manual")
    v = a.get_edges_for_manual()
    print("should return (f, k), (i, j) OR (g, j), (i, j):", v)
    print("and two edges should remain on list", a.edges_for_manual)

    print("------")
    print("3rd call to get_edges_for_manual")
    v = a.get_edges_for_manual()
    print("should yield (f, i), (g, j) OR (f, i), (f, k):", v)
    print("and the list should be empty", a.edges_for_manual)

    print("------")
    print("4th call to get_edges_for_manual")
    v = a.get_edges_for_manual()
    print("should restart it, resorted and yielded (j,k), (f,g):", v)
    print("and four edges should remain on list", a.edges_for_manual)


def test_LCA_add_edge_method():
    print("\n")
    print("==============================")
    print("====  test LCA.add_edge  =====")
    print("==============================")

    a, G = build_example_LCA()
    a.pprint()

    print('Changing an existing edge')
    change_edge = tuple(['i', 'j', 3])
    (from_change, to_change) = a.add_edge(change_edge)
    print("Change edge:", change_edge, "delta_wgt should be (-7, 7)"
          " and is (%d, %d)" % (from_change, to_change))
    print("a.delta_score should be -4 and it is", a.delta_score())
    G['i']['j']['weight'] = 3

    change_edge = tuple(['i', 'j', -4])
    (from_change, to_change) = a.add_edge(change_edge)
    print("Changing back by adding:", change_edge,
          "delta_wgt should be (7, -7)"
          " and is (%d, %d)" % (from_change, to_change))
    print("a.delta_score should be -18 and it is", a.delta_score())
    G['i']['j']['weight'] = -4

    print('-------')
    print('Adding a new edge')
    change_edge = tuple(['f', 'h', 4])
    (from_change, to_change) = a.add_edge(change_edge)
    print("Change edge:", change_edge, "delta_wgt should be (-4, 4)"
          " and is (%d, %d)" % (from_change, to_change))
    print("a.delta_score should be -10 and it is", a.delta_score())
    G.add_edge('f', 'h', weight=4)

    print('-------')
    print('Adding a change to an existing, consistent edge')
    change_edge = tuple(['h', 'i', 9])
    (from_change, to_change) = a.add_edge(change_edge)
    print("Change edge:", change_edge, "delta_wgt should be (3, 3)"
          " and is (%d, %d)" % (from_change, to_change))
    print("a.delta_score should still be -10 and it is", a.delta_score())
    G.add_edge('h', 'i', weight=9)

    print('-------')
    print('Restarting tests for adding edge during augmentation')
    a, G = build_example_LCA()
    a.pprint()

    a.build_inconsistency_sets()
    edge = tuple(['f', 'h', -5])
    pr = tuple([edge[0], edge[1]])
    print("Before adding (f, h), should be in inconsistent_pairs\n"
          "but should not be in inconsistent_edges; yes?",
          pr in a.inconsistent_pairs and pr not in a.inconsistent_edges)
    (from_delta, to_delta) = a.add_edge(edge)
    print('After adding (h, f, -5), delta_score should now be -28:',
          a.delta_score())
    print('(f, h) should not be in inconsistent_pairs, but should be\n'
          'in inconsistent_edges; yes?',
          pr not in a.inconsistent_pairs and pr in a.inconsistent_edges)

    print('-------')
    print('Removing from waiting list for augmentation')
    a, G = build_example_LCA()
    a.pprint()
    a.num_per_verify = 1   # to make sure there is at least one on the list
    v = a.get_node_pairs_for_verification()   # either (f, h), or (h, i)
    print(v)
    print('After running get_node_pairs_for_verification, pairs lists are:')
    print('inconsistent_pairs:', a.inconsistent_pairs)
    print('pairs_for_verification:', a.pairs_for_verification)

    pr = tuple(['h', 'j'])
    e = tuple([pr[0], pr[1], -4])
    (from_change, to_change) = a.add_edge(e)
    print('After adding', pr)
    print('It should not be in any pairs set; correct?',
          pr not in a.inconsistent_pairs and
          pr not in a.pairs_for_verification)
    print('It should be in the inconsistent edges set; correct?',
          pr in a.inconsistent_edges)
    print('It should not be on the edges_for_manual list; correct?',
          a.edges_for_manual is None or pr not in a.edges_for_manual)

    pr = tuple(['f', 'h'])
    e = tuple([pr[0], pr[1], -4])
    (from_change, to_change) = a.add_edge(e)
    print('After adding', pr)
    print('It should not be in any pairs set; correct?',
          pr not in a.inconsistent_pairs and
          pr not in a.pairs_for_verification)
    print('It should be in the inconsistent edges set; correct?',
          pr in a.inconsistent_edges)
    print('It should not be on the edges_for_manual list; correct?',
          a.edges_for_manual is None or pr not in a.edges_for_manual)

    print('-------')
    print('During augmentation; adding edges that already exist')
    a, G = build_example_LCA()
    a.pprint()
    v = a.get_edges_for_manual()
    print(v)
    pr = tuple(['f', 'g'])
    assert(pr in v)   # this would have gone out for manual
    e = tuple([pr[0], pr[1], 4])
    (from_change, to_change) = a.add_edge(e)
    print('Added pr', pr, 'which was "out for manual"')
    print('It should be in the inconsistent edges set; correct?',
          pr in a.inconsistent_edges)
    print('It should not be on the edges_for_manual list; correct?',
          pr not in a.edges_for_manual)

    pr = tuple(['i', 'j'])
    assert(pr not in v)   # this is not yet out manual
    e = tuple([pr[0], pr[1], -8])
    (from_change, to_change) = a.add_edge(e)
    print('Added pr', pr, 'which was "out for manual"')
    print('It should be in the inconsistent edges set; correct?',
          pr in a.inconsistent_edges)
    print('It should not be on the edges_for_manual list; correct?',
          pr not in a.edges_for_manual)


if __name__ == "__main__":
    test_LCA_class()
    test_LCA_add_edge_method()
