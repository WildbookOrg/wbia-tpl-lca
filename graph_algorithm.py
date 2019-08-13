import networkx as nx
import timeit

from cid_to_lca import *
import cluster_tools as ct
import lca
import lca_alg1 as alg1
import lca_alg2 as alg2

from lca_queues import *


"""
Data structures:

1. clustering:  cid -> set of nodes; each node is in exactly one set

2. node2cid:  graph node -> id of cluster that contains it

3. queues:  store all current LCAs on one of Q, S and W

4. cid2lca: manage information about association between one or
cids and the LCAs that might include them.  This is much more
than a mapping from a pair of cids to the LCA that combines them.

"""

class graph_algorithm(object):  # NOQA

    def __init__(self, G, params, clustering=None):
        self.G = G
        self.params = params

        if clustering is None:
            self.form_singletons_clustering()
        else:
            self.clustering = clustering
            self.check_clustering()
        self._next_cid = max(self.clustering.keys()) + 1

        self.node2cid = ct.build_node_to_cluster_mapping(self.clustering)
        self.score = ct.clustering_score(self.G, self.node2cid)
        self.form_initial_lcas()

        self.phase = None

        """  Need to set these callbacks to request and receive
        information from the verfication algorithm and to do the same
        from human reviewers.
        """
        # self.node_generator = None
        # self.ranker_request = None
        # self.ranker_result = None
        self.verifier_request_cb = None
        self.verifier_result_cb = None
        self.human_request = None
        self.human_result_cb = None

        self.start_splitting_cb = None
        self.next_splitting_itr_cb = None
        self.start_stability_cb = None
        self.next_stability_itr_cb = None
        self.num_verifier_results = 0
        self.num_human_results = 0

        self.is_interactive = False

    # def set_node_generator(self, node_generator):
    #     self.node_generator = node_generator

    # def set_ranker(self, requester, resulter):
    #    self.ranker_request = requester
    #    self.ranker_result = resulter

    def set_algorithmic_verifiers(self, requester, resulter):
        self.verifier_request_cb = requester
        self.verifier_result_cb = resulter

    def set_human_reviewers(self, requester, resulter):
        self.human_request_cb = requester
        self.human_result_cb = resulter

    def set_interactive(self, is_interactive=True):
        self.is_interactive = is_interactive

    def set_trace_callbacks(self, start_splitting, next_splitting_itr,
                            start_stability, next_stability_itr):
        self.start_splitting_cb = start_splitting
        self.next_splitting_itr_cb = next_splitting_itr
        self.start_stability_cb = start_stability
        self.next_stability_itr_cb = next_stability_itr

    def form_singletons_clustering(self):
        self.clustering = {cid: {n} for cid, n in enumerate(self.G.nodes())}

    def check_clustering(self):
        # Make sure no cluster is empty
        cluster_lengths = [len(c) for c in self.clustering.values()]
        assert(min(cluster_lengths) > 0)

        #  Check number of cluster nodes = number of nodes in clusters
        num_in_graph = len(self.G.nodes())
        num_in_clusters = sum(cluster_lengths)
        assert(num_in_graph == num_in_clusters)

        #  Check that the sets of nodes are the same.  Since we know
        #  there are no duplicates in the graph's nodes and since we
        #  know the sizes are the same, then we can conclude there are
        #  no repeated nodes in the clusters and so the clusters are
        #  disjoint.
        cluster_nodes = set.union(*self.clustering.values())
        graph_nodes = set(self.G.nodes())
        assert(cluster_nodes == graph_nodes)

    def form_initial_lcas(self):
        cid_pairs = ct.form_connected_cluster_pairs(self.G, self.clustering,
                                                    self.node2cid)
        self.cid2lca = CID2LCA()
        self.queues = lca_queues()
        self.create_lcas(cid_pairs)

    def run_main_loop(self):
        self.phase = "scoring"
        done = False

        num_run = 0
        max_to_run = 99999
        print("==================")
        print("Starting main loop")
        print("Min delta score", self.params['min_delta_score'])
        print("==================")

        while not done:
            if self.is_interactive:
                input("Enter to continue: ")

            # Step 4a: compute at the LCA alternative scores
            self.compute_lca_scores()

            num_run += 1
            print("=============")
            print("Iteration", num_run)
            print("=============")
            self.show_brief_state()
            # self.show_clustering()
            # self.show_queues()

            if self.phase == "splitting" and \
               self.next_splitting_itr_cb is not None:
                self.next_splitting_itr_cb(self.clustering, self.node2cid,
                                           self.num_human_results)
            elif self.phase == "stability" and \
               self.next_stability_itr_cb is not None:
                self.next_stability_itr_cb(self.clustering, self.node2cid,
                                           self.num_human_results)

            if num_run > max_to_run:
                return

            # Step 4b: get the LCA with the top delta_score
            a = self.queues.top_Q()

            # Step 4c:
            if a.delta_score() > 0:
                print("ga::run_main_loop Decision: apply LCA")
                # Note: no need to explicitly remove a from the top of
                # the heap because this is done during the replacement
                # process itself.
                self.score += a.delta_score()
                self.apply_lca(a)

            elif self.phase == "scoring":
                print("ga::run_main_loop Decision: switch phases from scoring to splitting")
                if self.start_splitting_cb is not None:
                    self.start_splitting_cb(self.clustering, self.node2cid)
                self.phase = "splitting"
                self.queues.switch_to_splitting()
                self.cid2lca.clear()
                singletons = [tuple([cid]) for cid in self.clustering]
                # print("ga::run_main_loop singletons:", singletons)
                self.create_lcas(singletons)  # each cluster id will become its own from
                print("ga::run_main_loop after switch to splitting, queues are")
                self.show_queues()

            elif self.phase == "splitting" and \
                 a.delta_score() < self.params["min_delta_score"] / 8:  # was 2
                print("ga::run_main_loop Decision: switch phases from splitting to stability")
                if self.start_stability_cb is not None:
                    self.start_stability_cb(self.clustering, self.node2cid)
                self.phase = "stability"
                self.queues.switch_to_stability()
                cid_pairs = ct.form_connected_cluster_pairs(self.G, self.clustering,
                                                            self.node2cid)
                self.create_lcas(cid_pairs)
                print("ga::run_main_loop after switch to stability, queues are")
                self.show_queues()

            elif self.params["min_delta_score"] < a.delta_score():
                print("ga::run_main_loop Decision: trying augmentation")
                self.queues.pop_Q()
                if self.augmentation(a):
                    print("ga::run_main_loop adding top of Q to waiting list")
                    self.queues.add_to_W(a)
                else:
                    print("ga::run_main_loop adding top of Q to done list")
                    self.queues.add_to_done(a)

            elif self.queues.num_on_W() == 0:
                print("ga::run_main_loop decision: all deltas too low and empty waiting queue W, so done")
                done = True
                continue

            else:
                print("ga::run_main_loop decision: all deltas too low,"
                      "but non-empty waiting queue; should wait")
                pass  # should be suspended here until new edges arrive

            edges = self.get_new_edges_and_weights()  # no new nodes allowed here...
            new_cid_pairs = set()
            temp_e = None
            for e in edges:
                cids = self.cids_for_edge(e)
                lcas_to_change = self.cid2lca.containing_all_cids(cids)

                if len(lcas_to_change) == 0:   # previously disconnected
                    print("ga::run_main_loop adding edge: e =", e, "between disconnected clusters")
                    cid_pair = (min(cids[0], cids[1]), max(cids[0], cids[1]))
                    new_cid_pairs.add(cid_pair)
                    self.G.add_weighted_edges_from([e])

                else:
                    print("ga::run_main_loop adding edge: e =", e,
                          "between connected clusters")
                    for a in lcas_to_change:
                        print("queue =", self.queues.which_queue(a))
                        (from_delta, to_delta) = a.add_edge(e)
                        self.queues.score_change(a, from_delta, to_delta)
                    if e[1] in self.G[e[0]]:
                        old_w = self.G[e[0]][e[1]]['weight']
                        self.G[e[0]][e[1]]['weight'] += e[2]
                        # print("ga::run_main_loop Old edge, added weight: old", old_w, "new",
                        #       self.G[e[0]][e[1]]['weight'])
                    else:
                        self.G.add_weighted_edges_from([e])
                        # print("ga::run_main_loop New edge with weight:",
                        #       self.G[e[0]][e[1]]['weight'])

                if len(cids) == 1:
                    self.score += e[2]
                    # print("ga::run_main_loop changed score (1) to", self.score)
                else:
                    self.score -= e[2]
                    # print("ga::run_main_loop changed score (2) to", self.score)

            if len(new_cid_pairs) > 0:
                self.create_lcas(new_cid_pairs)

            if temp_e is not None:
                n0, n1, _ = temp_e
                if n1 not in self.G[n0]:
                    print("Missing", temp_e)
                    assert(False)

    def compute_lca_scores(self):
        lcas_for_scoring = self.queues.get_S()
        print("ga::compute_lca_scores: num lcas =", len(lcas_for_scoring))
        for a in lcas_for_scoring:
            # a.pprint_short(stop_after_from=True)
            if self.phase == "scoring":
                # print("ga::compute_lca_scores: scoring with alg1")
                to_clusters, to_score = alg1.lca_alg1(a.subgraph)
            else:
                # print("ga::compute_lca_scores: scoring with alg2")
                cids = a.from_cids()
                # print("ga::cids", cids)
                to_clusters, to_score = alg2.lca_alg2(a.subgraph, a.from_cids(), a.from_node2cid())
                # print("to_clusters:", to_clusters, "delta_score", to_score - a.from_score)
            a.set_to_clusters(to_clusters, to_score)

        self.queues.add_to_Q(lcas_for_scoring)
        self.queues.clear_S()

    def apply_lca(self, a):
        # Step 1: Get the cids of the clusters to be removed and get
        # the lcas to be removed
        old_cids = a.from_cids()
        old_lcas = self.cid2lca.remove_with_cids(old_cids)
        self.queues.remove(old_lcas)

        # Step 2: Form the new clusters and replace the old ones
        new_clusters = a.to_clusters.values()  # pull out the sets from the dictionary
        new_next_cid = self._next_cid + len(new_clusters)
        new_cids = range(self._next_cid, new_next_cid)
        self._next_cid = new_next_cid
        added_clusters = {id: nodes for id, nodes in zip(new_cids, new_clusters)}
        # print("ga::apply_lca: old_cids to remove",
        #       old_cids, " added_clusters", added_clusters)

        ct.replace_clusters(old_cids, added_clusters, self.clustering, self.node2cid)
        """
        print("BEFORE replace:")
        print("old_cids", old_cids)
        print("added_clusters", added_clusters)
        print("clustering", self.clustering)
        print("node2cid", self.node2cid)
        """
        """
        print("AFTER replace:")
        print("clustering", self.clustering)
        print("node2cid", self.node2cid)
        """

        #  Step 3: Form a list of CID singleton and/or pairs involving at
        #  least one of the new clusters.  Whether singletons or pairs or both
        #  included depends on the current phase of the computation.
        if self.phase == "scoring":
            new_cid_sets = ct.form_connected_cluster_pairs(self.G, self.clustering,
                                                           self.node2cid, new_cids)
        elif self.phase == "splitting":
            new_cid_sets = [(cid,) for cid in new_cids]
        else:  # self.phase == "stability"
            new_cid_sets = ct.form_connected_cluster_pairs(self.G, self.clustering,
                                                           self.node2cid, new_cids)
            new_cid_sets.extend([(cid,) for cid in new_cids])

        # print("ga::new_cid_sets (2):", new_cid_sets)
        new_cid_sets = set(new_cid_sets)

        # Step 4: Create a new LCA from each set in new_cid_sets.
        self.create_lcas(new_cid_sets)

    def create_lcas(self, cid_tuples):
        for cids in cid_tuples:
            assert(len(cids) <= 2)  # singletons or pairs only
            if len(cids) == 1 and len(self.clustering[cids[0]]) == 1:
                # print("ga::create_lcas: skipping singleton cid", cids, "with one node")
                continue  # can't have an LCA with a singleton cluster of a single node
            elif len(cids) == 1:
                # print("ga::create_lcas: creating from singleton cid", cids)
                nodes = self.clustering[cids[0]]
            else:
                # print("ga::create_lcas: creating from multiple cids", cids)
                nodes = self.clustering[cids[0]] | self.clustering[cids[1]]
            subG = self.G.subgraph(nodes)
            from_score = ct.cid_list_score(subG, self.clustering, self.node2cid, cids)
            a = lca.LCA(subG, self.clustering, cids, from_score)
            self.cid2lca.add(a)
            self.queues.add_to_S(a)

    def augmentation(self, a):
        if self.verifier_request_cb is None and self.human_result_cb is not None:
            print("ga::augmentation: both None")
            return False

        if self.verifier_request_cb is not None:
            node_pairs = a.get_node_pairs_for_verification()
            if len(node_pairs) > 0:
                print("ga::augmentation: verify node pairs", list(node_pairs))
                self.verifier_request_cb(node_pairs)
                return True

        if self.human_request_cb is not None:
            edges = a.get_edges_for_manual()
            if len(edges) > 0:
                print("ga::augmentation: manual review edges", edges)
                self.human_request_cb(edges)
                return True

        print("ga::augmentation: no pairs or edges")
        return False

    def get_new_edges_and_weights(self):
        edges = []
        if self.verifier_result_cb is not None:
            new_edges = self.verifier_result_cb()
            self.num_verifier_results += len(new_edges)
            edges += new_edges

        if self.human_result_cb is not None:
            new_edges = self.human_result_cb()
            self.num_human_results += len(new_edges)
            edges += new_edges
            if len(new_edges) > 0:
                for n0, n1, w in new_edges:
                    cid0, cid1 = self.node2cid[n0], self.node2cid[n1]
                    if cid0 != cid1:
                        leng0, leng1 = len(self.clustering[cid0]), len(self.clustering[cid1])
                        if leng0 > leng1:
                            leng0, leng1 = leng1, leng0
                        # print("g1:augmentation manual review pair with cluster sizes", leng0, leng1,
                        #       "weight", w)
                    else:
                        leng0 = len(self.clustering[cid0])
                        # print("g1:augmentation manual review singleton with cluster size", leng0,
                        #       "weight", w)

        edges = [(min(e[0], e[1]), max(e[0], e[1]), e[2]) for e in edges]  # ensure ordering
        print("ga::get_new_edges_and_weights returning edges =", edges)
        return edges

    def new_lca_for_edge(self, e, cids):
        assert(len(cids) == 2)  # linking two clusters
        nodes = self.clustering(cids[0]) | self.clustering(cids[1])
        subG = self.G.subgraph(nodes)  # need to have the edge already added
        score = ct.cid_list_score(subG, self.clustering, self.node2cid, cids)
        a = lca.LCA(subG, self.clustering, cid, score)
        self.queues.add_to_S(a)

    def cids_for_edge(self, e):
        n0, n1, _ = e
        assert(n0 in self.node2cid)
        assert(n1 in self.node2cid)
        cid0 = self.node2cid[n0]
        cid1 = self.node2cid[n1]
        if cid0 == cid1:
            return [cid0]
        else:
            return [cid0, cid1]

    def show_clustering(self):
        # print("-----------------------")
        ct.print_structures(self.G, self.clustering, self.node2cid, self.score)

    def show_queues(self):
        print("-----------------------")
        print("Scoring queue:")
        if len(self.queues.S) == 0:
            print("empty")
        else:
            for a in self.queues.S:
                a.pprint_short(stop_after_from=True)

        print("Waiting queue:")
        if len(self.queues.W) == 0:
            print("empty")
        else:
            for a in self.queues.W:
                a.pprint_short(stop_after_from=True)

        print("Main queue:")
        if len(self.queues.Q.heap) == 0:
            print("empty")
        else:
            for i, a in enumerate(self.queues.Q.heap):
                print("heap entry:", i, end=' ')  # NOQA
                a.pprint_short(stop_after_from=False)
        print("-----------------------")

    def show_brief_state(self):
        print("Queue lengths: main Q %d, scoring %d, waiting %d"
              % (len(self.queues.Q), len(self.queues.S), self.queues.num_on_W()))

        print("Top LCA: ", end='')
        # print("Top delta score:", self.queues.top_Q().delta_score(), end='')
        self.queues.top_Q().pprint_short(stop_after_from=False)
        # print("Clustering score:", self.score)
        print("Clusters:", len(self.clustering))
        print("Verify results:", self.num_verifier_results)
        print("Human results:", self.num_human_results)

    def is_consistent(self):
        """ Each edge between two different clusters should be
        """
        all_ok = self.lca_queues.is_consistent()
        return all_ok  # Need to add a lot more here.....

"""
Testing code starts here
"""
class test_generator1(object):  # NOQA

    def __init__(self, which_graph=1):
        self.no_calls_yet = True
        self.verify_used = set()
        self.verify_requested = set()
        self.human_used = set()
        self.human_requested = set()

        self.G = nx.Graph()
        if which_graph == 1:
            self.G.add_weighted_edges_from([('a', 'b', 6), ('a', 'e', 3), ('a', 'f', -2),
                                            ('b', 'c', -4), ('b', 'e', -1), ('b', 'f', 1),
                                            ('c', 'd', 5),
                                            ('d', 'f', -3),
                                            ('f', 'g', 4), ('f', 'h', 5),
                                            ('g', 'h', -2), ('g', 'i', -2), ('g', 'j', 2),
                                            ('h', 'j', -3), ('i', 'j', 6)])
            self.no_calls_yet = True
            self.first_edges = None

            self.verify_available = {('a', 'g'): -3, ('a', 'h'): -4, ('b', 'g'): -2,
                                     ('b', 'h'): -3, ('e', 'f'): -4, ('e', 'g'): -6,
                                     ('e', 'h'): -5}

            self.human_available = {('a', 'f'): -3, ('b', 'f'): -2, ('f', 'g'): 6,
                                    ('g', 'h'): 5, ('a', 'e'): -5, ('b', 'e'): -4}

        elif which_graph == 2:
            self.G.add_weighted_edges_from([('a', 'b', -2), ('a', 'c', 6),
                                            ('b', 'd', 3), ('c', 'd', 1)])
            self.first_edges = None
            self.verify_available = {('a', 'd'): 3, ('b', 'c'): 4}
            self.human_available = dict()

        elif which_graph == 3:
            self.G.add_weighted_edges_from([('a', 'b', -2), ('a', 'c', 6),
                                            ('b', 'd', 3), ('c', 'd', 1)])
            self.first_edges = None
            self.verify_available = {('a', 'd'): 3, ('b', 'c'): -4}
            self.human_available = {('a', 'b'): -5, ('a', 'd'): 8,
                                    ('b', 'd'): 1, ('b', 'c'): -8}

        elif which_graph == 4:
            self.G.add_weighted_edges_from([('a', 'b', 2), ('c', 'd', 1)])
            self.first_edges = [('a', 'c', 5), ('a', 'd', -4),
                                ('b', 'c', -3), ('b', 'd', 2)]
            self.verify_available = dict()
            self.human_available = dict()

    def verify_request(self, node_prs):
        # print("Entering verify_request: node_prs =", node_prs)
        node_prs = set(node_prs)
        node_prs -= self.verify_used
        self.verify_requested |= node_prs

    def verify_result(self):
        # print("Entering verify_result")
        if self.no_calls_yet:
            self.no_calls_yet = False
            if self.first_edges is not None:
                return self.first_edges

        #  Only return those that have not already been returned
        edges = []
        for pr in self.verify_requested:
            self.verify_used.add(pr)
            if pr in self.verify_available:
                e = (pr[0], pr[1], self.verify_available[pr])
                del self.verify_available[pr]
            else:
                e = (pr[0], pr[1], 0)
            if pr not in self.human_available:
                self.human_available[pr] = 2 * e[2]
            edges.append(e)

        self.verify_requested.clear()

        # print("Returning edges", edges)
        return edges

    def human_request(self, node_prs):
        # print("Entering human_request with node_prs", node_prs)
        node_prs = set(node_prs)
        self.human_requested |= node_prs
        # print("Leaving human_request behind:", self.human_requested)

    def human_result(self):
        # print("Entering human_result")
        edges = []

        if self.human_requested.issubset(self.human_used):
            # print('cleared requested from used to start over')
            self.human_used -= self.human_requested

        for pr in self.human_requested:
            if pr in self.human_used:
                continue

            if pr in self.human_available:
                e = (pr[0], pr[1], self.human_available[pr])
                # print("human available e=", e)
            elif pr[1] in self.G[pr[0]]:
                e = (pr[0], pr[1], 2 * self.G[pr[0]][pr[1]]['weight'])
                # print("taking from graph, e=", e)
            else:
                e = (pr[0], pr[1], 0)
                # print("making up, e=", e)
            edges.append(e)
            self.human_used.add(pr)

        self.human_requested.clear()

        # print("Returning edges", edges)
        return edges

if __name__ == "__main__":
    tg = test_generator1(which_graph=4)
    params = {'min_delta_score': -10}

    ga_instance = graph_algorithm(tg.G, params)
    ga_instance.set_algorithmic_verifiers(tg.verify_request, tg.verify_result)
    ga_instance.set_human_reviewers(tg.human_request, tg.human_result)
    ga_instance.run_main_loop()
