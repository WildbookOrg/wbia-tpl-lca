import networkx as nx
import sys
import time

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


class GraphAlgorithm(object):

    def __init__(self, edges, params, clustering=None):
        self.graph_from_edges(edges)

        self.params = params
        assert('min_delta_score' in self.params)
        self.assign_parameter_defaults()

        if clustering is None:
            self.form_singletons_clustering()
        else:
            self.clustering = clustering
            self.check_clustering()
        self._next_cid = max(self.clustering.keys()) + 1

        self.node2cid = ct.build_node_to_cluster_mapping(self.clustering)
        self.score = ct.clustering_score(self.G, self.node2cid)
        self.form_initial_lcas()

        #  Use these to keep track of what's out for review.
        #  All waiting LCAs must have a node pair in one of
        #  these sets.
        self.pairs_waiting_verify = set()
        self.pairs_waiting_human = set()
        self.num_verifier_results = 0
        self.num_human_results = 0
        self.wait_since_edge = 0

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

    def assign_parameter_defaults(self):
        if 'seconds_to_sleep' not in self.params:
            self.params['seconds_to_sleep'] = 1.0
        if 'max_before_sleep' not in self.params:
            self.params['max_before_sleep'] = 50
        if 'seconds_before_restart_queues' not in self.params:
            self.params['seconds_before_restart_queues'] = 1200

    def graph_from_edges(self, edges):
        # Form an edge dict mapping prs to weights
        # Also form a dict mapping prs to the number of 0 wgts
        d = dict()   
        zc = dict()
        for e in edges:
            if e[0] < e[1]:
                pr = (e[0], e[1])
            else:
                pr = (e[1], e[0])
            if pr in d:
                d[pr] += e[2]
            else:
                d[pr] = e[2]

            if e[2] == 0:
                if pr in zc:
                    zc[pr] += 1
                else:
                    zc[pr] = 1

        combined = [e + (d[e],) for e in d]
        self.G = nx.Graph()
        self.G.add_weighted_edges_from(combined)
        
        for pr in zc:
            self.G[pr[0]][pr[1]]['incomparable'] = zc[pr]

    def form_singletons_clustering(self):
        self.clustering = {cid: {n} for cid, n in enumerate(self.G.nodes())}

    def check_clustering(self):
        # Make sure no cluster is empty
        cluster_lengths = [len(c) for c in self.clustering.values()]
        assert(min(cluster_lengths) > 0)

        #  Check number of cluster nodes = number of nodes in clusters
        num_in_graph = len(self.G.nodes())
        num_in_clusters = sum(cluster_lengths)
        if num_in_graph != num_in_clusters:
            raise GraphFormationError

        #  Check that the sets of nodes are the same.  Since we know
        #  there are no duplicates in the graph's nodes and since we
        #  know the sizes are the same, then we can conclude there are
        #  no repeated nodes in the clusters and so the clusters are
        #  disjoint.
        cluster_nodes = set.union(*self.clustering.values())
        graph_nodes = set(self.G.nodes())
        if cluster_nodes != graph_nodes:
            raise GraphFormationError

    def form_initial_lcas(self):
        cid_pairs = ct.form_connected_cluster_pairs(self.G, self.clustering,
                                                    self.node2cid)
        self.cid2lca = CID2LCA()
        self.queues = LCAQueues()
        self.create_lcas(cid_pairs)

    def run_main_loop(self):
        self.phase = "scoring"
        done = False

        num_run = 0
        max_to_run = 200
        print("==================")
        print("Starting main loop")
        print("Min delta score", self.params['min_delta_score'])
        print("==================")

        self.pairs_waiting_verify.clear()
        self.pairs_waiting_human.clear()

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
                return num_run

            # Step 4b: get the LCA with the top delta_score
            a = self.queues.top_Q()

            # Step 4c:
            if a is not None and a.delta_score() > 0:
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
                 (a is None or
                  a.delta_score() < self.params["min_delta_score"] / 8):  # was 2
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

            elif a is not None and self.params["min_delta_score"] < a.delta_score():
                print("ga::run_main_loop Decision: trying augmentation")
                self.queues.pop_Q()
                if self.augmentation(a):
                    print("ga::run_main_loop adding top of Q to waiting list")
                    self.queues.add_to_W(a)
                else:
                    print("ga::run_main_loop adding top of Q to done list")
                    self.queues.add_to_done(a)

            # At this point, we are in the stability phase and nothing on the
            # main queue is productive.
            elif self.queues.num_on_W() == 0:
                print("ga::run_main_loop decision: all deltas too low and empty waiting queue W, so done")
                done = True
                continue

            else:
                print("ga::run_main_loop decision: all deltas too low but waiting for edges")

            wait_since_edge_or_resend = 0
            while True:
                verify_edges, human_edges = self.get_new_edges_and_weights()
                num_new_edges = len(verify_edges) + len(human_edges)
                self.incorporate_new_edges(verify_edges, is_human_result=False)
                self.incorporate_new_edges(human_edges, is_human_result=True)

                if not self.should_wait_for_edges(num_new_edges):
                    break

                seconds_to_sleep = self.params['seconds_to_sleep']
                print("ga::run_main_loop: sleeping "
                      "for %0.2f seconds" % seconds_to_sleep)
                time.sleep(seconds_to_sleep)
                wait_since_edge_or_resend += seconds_to_sleep

                print("ga::run_main_loop: wait since edge or resend is now %0.2f seconds"
                      % wait_since_edge_or_resend)
                if wait_since_edge_or_resend > self.params['seconds_before_restart_queues']:
                    self.resend_waiting()
                    wait_since_edge_or_resend = 0

            if not self.is_consistent():
                assert(False)
    
        return num_run

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
        print("ga::apply_lca: old_cids to remove",
              old_cids, " added_clusters", added_clusters)

        ct.replace_clusters(old_cids, added_clusters, self.clustering, self.node2cid)

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

        print("ga::new_cid_sets (2):", new_cid_sets)
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
        """
        Ask the LCA for some of the node pairs it needs for
        verification and send them via the verifier callback.  If
        there are no pairs, do the same for human review. If no pairs
        are available, the LCA will be considered done.
        """

        if self.verifier_request_cb is None and self.human_result_cb is not None:
            print("ga::augmentation: both None")
            return False

        if self.verifier_request_cb is not None:
            node_pairs = a.get_node_pairs_for_verification()
            if len(node_pairs) > 0:
                # Only send pairs that aren't already out for verification
                node_pairs -= self.pairs_waiting_verify
                print("ga::augmentation: verify node pairs", list(node_pairs))
                self.verifier_request_cb(node_pairs)
                self.pairs_waiting_verify |= node_pairs
                return True

        if self.human_request_cb is not None:
            node_pairs = a.get_edges_for_manual()
            if len(node_pairs) > 0:
                node_pairs -= self.pairs_waiting_human
                print("ga::augmentation: manual review edges", node_pairs)
                self.human_request_cb(node_pairs)
                self.pairs_waiting_human |= node_pairs
                return True

        print("ga::augmentation: no pairs or edges, so this LCA will be considered 'done'")
        return False

    def get_new_edges_and_weights(self):
        verifier_edges = []
        if self.verifier_result_cb is not None:
            new_edges = self.verifier_result_cb()
            self.num_verifier_results += len(new_edges)
            verifier_edges = [(min(e[0], e[1]), max(e[0], e[1]), e[2])
                              for e in new_edges]  # ensure ordering
            self.pairs_waiting_verify -= {(e[0], e[1]) for e in verifier_edges}

        human_edges = []
        if self.human_result_cb is not None:
            new_edges = self.human_result_cb()
            self.num_human_results += len(new_edges)
            human_edges = [(min(e[0], e[1]), max(e[0], e[1]), e[2])
                           for e in new_edges]  # ensure ordering
            self.pairs_waiting_human -= {(e[0], e[1]) for e in human_edges}

        print("ga::get_new_edges_and_weights returning verifier_edges",
              verifier_edges, "and human edges", human_edges)
        return verifier_edges, human_edges

    def incorporate_new_edges(self, edges, is_human_result=False):
        """
        Incorporate each new edge, doing something different
        depending on the state of the LCAs involved, whether or not
        the edge already exists, if it is a verification or human
        manual weight, and if it is an incomparable edge.  Phew!
        Maybe I need to simplify here...
        """
        new_cid_pairs = set()

        for e in edges:
            cids = self.cids_for_edge(e)
            lcas_to_change = self.cid2lca.containing_all_cids(cids)
            edge_added = False

            # Handle previously disconnected LCAs
            if len(lcas_to_change) == 0:   # previously disconnected
                print("ga::incorporate_new_edges: adding edge: e =",
                      e, "between disconnected clusters")
                cid_pair = (min(cids[0], cids[1]), max(cids[0], cids[1]))
                new_cid_pairs.add(cid_pair)
                self.G.add_weighted_edges_from([e])
                edge_added = True

            # Handle verification result that is a repeated edge
            elif not is_human_result and e[1] in self.G[e[0]]:
                print("ga::incorporate_new_edges: verify edge: e =",
                      e, "is already in the graph")
                edge_added = False

            # Handline verification result that is not already in the
            # graph
            elif not is_human_result:
                print("ga::incorporate_new_edges: verify edge: e =",
                      e, "being added to the graph")
                edge_added = True
                self.G.add_weighted_edges_from([e])

            # Handle human result that is an 'incomparable'
            elif is_human_result and e[2] == 0:
                print("ga::incorporate_new_edges manual incomparable edge: e =", e,
                      "between connected clusters")
                edge_added = False
                if 'incomparable' not in self.G[e[0]][e[1]]:
                    self.G[e[0]][e[1]]['incomparable'] = 1
                else:
                    self.G[e[0]][e[1]]['incomparable'] += 1
                print('incomparable value:', self.G[e[0]][e[1]]['incomparable'])

            # Handle normal result: a new edge for verification or a
            # non-zero weight for human results.
            else:
                print("ga::incorporate_new_edges adding edge: e =", e,
                      "between connected clusters")
                edge_added = True
                self.G[e[0]][e[1]]['weight'] += e[2]

            # If the edge was added, change the score and decide how
            # to handle the affected LCAs
            if edge_added:
                for a in lcas_to_change:
                    (from_delta, to_delta) = a.add_edge(e)
                    self.queues.score_change(a, from_delta, to_delta)

                if len(cids) == 1: # edge is within cluster, so add weight
                    self.score += e[2]
                else: # edge is between clusters, so subtract weight
                    self.score -= e[2]

            # Since the edge was not added, the scores don't change
            # but the LCA must be moved to because an edge was
            # returned.  What's below will put all affected LCAs on
            # the main Q.
            else:
                for a in lcas_to_change:
                    self.queues.score_change(a, 0, 0)

        if len(new_cid_pairs) > 0:
            self.create_lcas(new_cid_pairs)

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

    def should_wait_for_edges(self, num_new_edges):
        '''
        Decide whether or not to sleep based on whether or not new
        results have come back and on the size of the waiting queue.
        '''
        print("ga::should_wait: num_new_edges", num_new_edges,
              "num_on_W", self.queues.num_on_W(),
              "len(Q)", len(self.queues.Q),
              "max_before_sleep", self.params['max_before_sleep'])
        """
        print('Current W:')
        if self.queues.num_on_W() == 0:
            print('- none -')
        else:
            for a in self.queues.W:
                a.pprint_short()
        """

        if num_new_edges > 0 or self.queues.num_on_W() == 0:
            # print("ga::should_wait: no sleep because new edges or W empty")
            return False
        elif self.queues.num_on_W() > len(self.queues.Q): # could add percentage
            print("ga::should_wait: sleep because W is bigger than Q")
            return True
        elif self.queues.num_on_W() >= self.params['max_before_sleep']:
            print("ga::should_wait: sleep because W is too large")
            return True
        else:
            # print("ga::should_wait: no sleep because W is too small")
            return False

    def resend_waiting(self):
        if self.verifier_request_cb is not None:
            print("ga::resend_waiting: %d pairs to verifier"
                  % len(self.pairs_waiting_verify))
            self.verifier_request_cb(list(self.pairs_waiting_verify))

        if self.human_request_cb is not None:
            print("ga::resend_waiting: %d pairs for human review"
                  % len(self.pairs_waiting_human))
            self.human_request_cb(list(self.pairs_waiting_human))

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
        if len(self.queues.Q) > 0:
            print("Top LCA: ", end='')
            # print("Top delta score:", self.queues.top_Q().delta_score(), end='')
            self.queues.top_Q().pprint_short(stop_after_from=False)
        print("Clusters:", len(self.clustering))
        print("Verify results:", self.num_verifier_results)
        print("Human results:", self.num_human_results)

    def is_consistent(self):
        #  make sure the queues are non-intersecting
        all_ok = self.queues.is_consistent()

        #  For each LCA on W, make sure there is at least one node pr
        #  waiting for a verification or manual review result.  Note
        #  that the reverse is not true: there could be a pr waiting a
        #  result with no corresponding LCA on W because the LCA could
        #  have been pulling off due to another result.
        waiting_prs = self.pairs_waiting_verify | self.pairs_waiting_human
        for a in self.queues.W:
            nodes = a.nodes()
            found = False
            for pr in waiting_prs:
                if pr[0] in nodes and pr[1] in nodes:
                    found = True
                    break
            if not found:
                print("ga::is_consistent: no waiting node pair for waiting LCA", end=' ')
                a.pprint_short(stop_after_from=True)
                all_ok = False

        return all_ok


"""
Testing code starts here
"""

def test_initial_graph():
    w_e = [('a', 'b', 3),
           ('c', 'b', 1),
           ('a', 'c', 5),
           ('b', 'd', 0),
           ('a', 'd', -4),
           ('c', 'a', -6),
           ('b', 'd', 0),
           ('b', 'a', 4),
           ('d', 'b', 0),
           ('a', 'b', 0)]
    c_e = [('a', 'b', w_e[0][2] + w_e[7][2]),
           ('a', 'c', w_e[2][2] + w_e[5][2]),
           w_e[4],
           w_e[1],
           ('b', 'd', 0)]
    params = {'min_delta_score': -10}
    gai = GraphAlgorithm(w_e, params)
    print('test_initial_graph: length of edges should be:', len(c_e),
          'and is', gai.G.number_of_edges())
    mistakes = 0
    if len(c_e) != gai.G.number_of_edges():
        mistakes += 1
    for (n0, n1, wgt) in c_e:
        if n1 not in gai.G[n0]:
            print('Error:', n1, 'not in the edges of', n0)
            mistakes += 1
        elif wgt != gai.G[n0][n1]['weight']:
            print('Error: edge (%a, %a) should have weight %a but has %a'
                  % (n0, n1, wgt, gai.G[n0][n1]['weight']))
            mistakes += 1
    if mistakes > 0:
        print('test initial_graph found', mistakes, 'edge mistakes')
    else:
        print('test_initial_graph found no mistakes')

    pairs = [('a', 'b'), ('b', 'd'), ('b', 'c')]
    counts = [1, 3, 0]
    for pr, cc in zip(pairs, counts):
        cnt = 0
        if 'incomparable' in gai.G[pr[0]][pr[1]]:
            cnt = gai.G[pr[0]][pr[1]]['incomparable']
        print("test_initial_graph: pair", pr, "should have",
              cc, "incomparable. It has", cnt)


class TestGenerator1(object):

    def __init__(self, which_graph=0):
        self.no_calls_yet = True
        self.verify_used = set()
        self.verify_requested = set()
        self.human_used = set()
        self.human_requested = set()

        self.max_delay_before_response = 0
        self.human_delay_before_response = 0
        self.verify_delay_before_response = 0

        self.params = {'min_delta_score': -12,
                       'max_before_sleep': 3,
                       'seconds_to_sleep' : 0.01,
                       'seconds_before_restart_queues' : 0.03}

        if which_graph == 0:
            """
            This simple example requires the incomparable edges to
            be eliminated from the graph before convergence. It also
            induces algorithm "sleeping" and "restarting".
            """
            self.weighted_edges = [('a', 'b', 4), ('a', 'c', 3)]
            self.first_edges = None
            self.verify_available = {('b', 'c'): 2}
            self.human_available = {('b', 'c'): 0,  # marks edges as
                                    ('a', 'c'): 0}  # incomparable
            self.gt_clustering = {0 : set(['a', 'b', 'c'])}
            self.max_delay_before_response = 6
            self.human_delay_before_response = self.max_delay_before_response
            self.verify_delay_before_response = self.max_delay_before_response

        elif which_graph == 1:
            """
            More complicated "ordinary" example
            """
            self.weighted_edges = [('a', 'b', 6), ('a', 'e', 3), ('a', 'f', -2),
                                   ('b', 'c', -4), ('b', 'e', -1), ('b', 'f', 1),
                                   ('c', 'd', 5),
                                   ('d', 'f', -3),
                                   ('f', 'g', 4), ('f', 'h', 5),
                                   ('g', 'h', -2), ('g', 'i', -2), ('g', 'j', 2),
                                   ('h', 'j', -3), ('i', 'j', 6)]
            self.first_edges = None
            self.gt_clustering = {0 : set(['a', 'b']),
                                  1 : set(['c', 'd']),
                                  2 : set(['e']),
                                  3 : set(['f', 'g', 'h', 'i', 'j'])}
            self.verify_available = {('a', 'g'): -3, ('a', 'h'): -4, ('b', 'g'): -2,
                                     ('b', 'h'): -3, ('e', 'f'): -4, ('e', 'g'): -6,
                                     ('e', 'h'): -5}
            self.human_available = {('a', 'b') : 10, ('a', 'c') : -10,
                                    ('a', 'f'): -8, ('b', 'f'): -8, ('f', 'g'): 9,
                                    ('g', 'h'): 8, ('a', 'e'): -8, ('b', 'e'): -9}
            self.max_delay_before_response = 6
            self.human_delay_before_response = self.max_delay_before_response
            self.verify_delay_before_response = self.max_delay_before_response

        elif which_graph == 2:
            """
            Example where the inconsistent edge ('a', 'b') is never
            directly removed because it becomes "incomparable", but
            convergence and the correct answer are reached anyway.
            """
            self.weighted_edges = [('a', 'b', -2), ('a', 'c', 6),
                                   ('b', 'd', 3), ('c', 'd', 1)]
            self.first_edges = None
            self.verify_available = {('a', 'd'): 3, ('b', 'c'): 4}
            self.human_available = {('a', 'b'): 0}
            self.gt_clustering = {0 : set(['a', 'b', 'c', 'd'])}

        elif which_graph == 3:
            """
            Example where a single node, 'b', is driven out of the
            clustering based on the human results.
            """
            self.weighted_edges = [('a', 'b', -2), ('a', 'c', 6),
                                   ('b', 'd', 4), ('c', 'd', 1)]
            self.first_edges = None
            self.verify_available = {('a', 'd'): 3, ('b', 'c'): -1}
            self.human_available = {('a', 'b'): -5, ('a', 'c'): 8,
                                    ('b', 'd'): 1, ('b', 'c'): -8}
            self.gt_clustering = {0 : set(['a', 'c', 'd']),
                                  1 : set(['b'])}

        else: # elif which_graph == 4:
            """
            Example where edge pairs are injected into the graph,
            connecting initial clusters that weren't previously
            connected. 
            """
            self.weighted_edges = [('a', 'b', 2), ('c', 'd', 1)]
            self.first_edges = [('a', 'c', 5), ('a', 'd', -4),
                                ('b', 'c', -3), ('b', 'd', 2)]
            self.verify_available = dict()
            self.human_available = {('b', 'd'): 12,
                                    ('a', 'c'): 11,
                                    ('a', 'b'): -10,
                                    ('a', 'd'): -9,
                                    ('b', 'c'): -8}
            self.gt_clustering = {0 : set(['a', 'c']),
                                  1 : set(['b', 'd'])}

        """Complete the set of human available edges, so that a response is
        obtained for each.  This is not needed for verify edges,
        because all edges not explicitly provided are 0
        """ 
        n2c = ct.build_node_to_cluster_mapping(self.gt_clustering)
        nodes = sorted(n2c.keys())
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                ni, nj = (nodes[i], nodes[j])
                pr = (ni, nj)
                if pr in self.human_available:
                    pass
                elif n2c[ni] == n2c[nj]:
                    self.human_available[pr] = 10
                else:
                    self.human_available[pr] = -10

        self.G = nx.Graph()
        self.G.add_weighted_edges_from(self.weighted_edges)

    def verify_request(self, node_prs):
        node_prs = set(node_prs)
        node_prs -= self.verify_used
        self.verify_requested |= node_prs

    def verify_result(self):
        if self.no_calls_yet:
            self.no_calls_yet = False
            if self.first_edges is not None:
                return self.first_edges

        if self.verify_delay_before_response > 0:
            self.verify_delay_before_response -= 1
            return []
        else:
            self.verify_delay_before_response = self.max_delay_before_response

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
        return edges

    def human_request(self, node_prs):
        node_prs = set(node_prs)
        self.human_requested |= node_prs
        # print("Leaving human_request behind:", self.human_requested)

    def human_result(self):
        if self.human_delay_before_response > 0:
            self.human_delay_before_response -= 1
            return []

        self.human_delay_before_response = self.max_delay_before_response
        edges = []

        if self.human_requested.issubset(self.human_used):
            # print('cleared requested from used to start over')
            self.human_used -= self.human_requested

        for pr in self.human_requested:
            if pr in self.human_used:
                print('skipping pair', pr)
                continue
            assert(pr in self.human_available)
            e = (pr[0], pr[1], self.human_available[pr])
            edges.append(e)
            self.human_used.add(pr)
        self.human_requested.clear()
        return edges

if __name__ == "__main__":
    which_graph = 0
    if len(sys.argv) == 2:
        which_graph = int(sys.argv[1])
        print('Testing graph', which_graph, 'from command line')
    else:
        print('Testing graph', which_graph, 'from default')
    
    tg = TestGenerator1(which_graph=which_graph)
    gai = GraphAlgorithm(tg.weighted_edges, tg.params)
    gai.set_algorithmic_verifiers(tg.verify_request, tg.verify_result)
    gai.set_human_reviewers(tg.human_request, tg.human_result)
    num = gai.run_main_loop()
    gai.show_clustering()

    num_eq = ct.count_equal_clustering(tg.gt_clustering,
                                       gai.clustering, gai.node2cid)
    if num_eq == len(gai.clustering):
        print("Successfully matches ground truth")
    else:
        print("Fails to match ground truth")

    test_initial_graph()
