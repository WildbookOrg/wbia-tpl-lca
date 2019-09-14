import math as m
import matplotlib
matplotlib.use('Qt5Agg')  # NOQA
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from scipy.special import comb
from scipy.stats import gamma
import sys

import cluster_tools as ct
import exp_scores as es
import weighter as wgtr


class Simulator(object):
    """
    Important note (2019-08-01): when human-weighted edges are added, only the
    latest edge is currently "remembered" in this graph.  This could
    diverge from the actual graph when that graph combines weights
    additively.  One solution is to make this a multigraph, but not now.
    """

    def __init__(self, params, wgtr, seed=None):
        self.params = params
        self.wgtr = wgtr
        self.orig_edges = []
        self.gt_clustering = dict()   # cid -> at first a list of nodes, then a set
        self.gt_node2cid = dict()     # node -> cid
        self.r_clustering = None      # "reachable" clustering
        self.r_node2cid = None        # "reachable" node 2 cd=ic mapping
        self.ranker_matches = dict()  # node -> set of nodes
        self.G = nx.Graph()
        self.G_orig = nx.Graph()      # generated graph without algorithmic additions
        self.verify_edges = []
        self.human_edges = []
        if seed is not None:
            random.seed(seed)
        self.max_delay_steps = 10
        self.human_steps_until_return = -1
        self.verify_steps_until_return = -1

    def create_random_clusters(self):
        """
        Generate the clusters and the nodes within the clusters according to
        a gamma distribution.  The information to control this is in the params dictionary.
        The results are the gt_node2cid mapping and the gt_clustering as a dictionary mapping
        cluster names to lists of nodes.  In the method generate_from_clusters these lists
        are turned into sets.
        :return: None
        """

        # Compute digit_per_node that controls the generation of the name strings.
        expected_nodes = 1 + self.params['gamma_shape'] * self.params['gamma_scale']
        digits_per_node = 2 + int(m.log10(expected_nodes))

        """
        Determine how many nodes will be in each cluster according to a gamma distribution
        """
        samples = np.random.gamma(self.params['gamma_shape'],
                                  self.params['gamma_scale'],
                                  self.params['num_clusters'])
        samples = 1 + np.round(samples).astype(int)

        """
        Create the nodes within each cluster and a dictionary mapping the (generated)
        cluster ids to lists of nodes.
        """
        ni = 0
        for cid in range(len(samples)):
            # print('\nNext cluster')
            n = samples[cid]
            nodes = ['n' + str(i).zfill(digits_per_node) for i in range(ni, ni+n)]
            self.gt_clustering[cid] = nodes
            for node in nodes:
                self.gt_node2cid[node] = cid
            ni += n
            print("cid %d, n %d, nodes %a, ni %d" % (cid, n, nodes, ni))

    def create_clusters_from_ground_truth(self, clustering):
        self.gt_clustering = clustering
        nodes_sets = [set(v) for v in clustering.values()]
        cluster_nodes = set.union(*nodes_sets)
        total_nodes = sum([len(s) for s in clustering.values()])
        assert(len(cluster_nodes) == total_nodes)
        self.gt_node2cid = ct.build_node_to_cluster_mapping(clustering)

    def generate_from_clusters(self):
        all_nodes = [n for v in self.gt_clustering.values() for n in v]           # list of node ids
        print(all_nodes)
        self.orig_edges = []

        num_correct_positive = 0
        num_correct_negative = 0
        num_correct_zero = 0
        num_incorrect_positive = 0
        num_incorrect_negative = 0
        num_incorrect_zero = 0

        for cid in self.gt_clustering:
            cluster = self.gt_clustering[cid]
            for nid in cluster:
                self.ranker_matches[nid] = set()

            #  Examine each pair of nodes in a cluster to determine if they should be "ranker matches"
            for i, ith_node in enumerate(cluster):
                for j in range(i + 1, len(cluster)):
                    prob = random.uniform(0, 1)
                    jth_node = cluster[j]
                    # print('i %d, ith_node %s, j %d, jth_node %s, prob %1.3f'
                    #       % (i, ith_node, j, jth_node, prob))
                    if prob < self.params['p_ranker_correct']:
                        self.ranker_matches[ith_node].add(jth_node)
                        self.ranker_matches[jth_node].add(ith_node)
                        is_match_correct = True
                        wgt = self.wgtr.random_wgt(is_match_correct)
                        if wgt > 0:
                            num_correct_positive += 1
                        elif wgt == 0:
                            num_correct_zero += 1
                        else:
                            num_correct_negative += 1

                        e = (ith_node, jth_node, wgt)
                        self.orig_edges.append(e)
                        # print("adding positive edge", e)

        num_from_ranker = self.params['num_from_ranker']
        assert(num_from_ranker > 0)
        num_nodes = len(all_nodes)

        # Change the list to a set
        self.gt_clustering = {cid: set(cluster)
                              for cid, cluster in self.gt_clustering.items()}

        """
        Step 2:
        Generate "incorrect" match edges, sufficient to have the required
        number of edges generated by the ranking algorithm.
        """
        for i, ith_node in enumerate(all_nodes):
            # print("\nAdding negative edges for node", ith_node)
            matches = self.ranker_matches[ith_node]
            cid = self.gt_node2cid[ith_node]
            cluster = self.gt_clustering[cid]

            """ 
            In the rare case that there are too many correct
            matches, i.e. for an extremely large cluster, trim them out
            """
            while len(matches) > num_from_ranker:
                matches.pop()

            """
            Generate edges between clusters
            """
            while len(matches) < num_from_ranker:
                j = random.randint(0, num_nodes - 1)
                jth_node = all_nodes[j]
                if jth_node not in matches and jth_node not in cluster:
                    matches.add(jth_node)
                    is_match_correct = False
                    wgt = self.wgtr.random_wgt(is_match_correct)
                    if wgt > 0:
                        num_incorrect_positive += 1
                    elif wgt == 0:
                        num_incorrect_zero += 1
                    else:
                        num_incorrect_negative += 1

                    if ith_node < jth_node:
                        e = (ith_node, jth_node, wgt)
                    else:
                        e = (jth_node, ith_node, wgt)
                    # print("adding negative edge", e)
                    self.orig_edges.append(e)

        self.G.add_weighted_edges_from(self.orig_edges)
        print("simulator::generate: adding %d edges" % len(self.orig_edges))
        print("%d correct match edges have positive weight" % num_correct_positive)
        print("%d correct match edges have zero weight" % num_correct_zero)
        print("%d correct match edges have negative weight" % num_correct_negative)
        print("%d incorrect match edges have positive weight" % num_incorrect_positive)
        print("%d incorrect match edges have zero weight" % num_incorrect_zero)
        print("%d incorrect match edges have negative weight" % num_incorrect_negative)

        self.G_orig.add_nodes_from(self.G)
        self.G_orig.add_weighted_edges_from(self.orig_edges)

        """
        Step 3: Generate the "reachable" ground truth, the obtainable
        result given simulated failures to match that could disconnect
        a correct match.
        """
        self.r_clustering = dict()
        k = 0
        for cc in self.gt_clustering.values():
            H = self.G.subgraph(cc)
            prev_k = k
            for new_cc in nx.connected_components(H):
                self.r_clustering[k] = new_cc
                k += 1
            if k - prev_k > 1:
                print("GT cluster", cc, "split into", k - prev_k, "...")
                for i in range(prev_k, k):
                    print("   ", self.r_clustering[i])
            else:
                print("GT cluster", cc, "is intact")
        self.r_node2cid = ct.build_node_to_cluster_mapping(self.r_clustering)

    def print_clusters(self, lca_clustering):
        print("LCA's clusters:")
        for cid, c in lca_clustering.items():
            print("%a:" % cid, c)

        print("\nGround truth clusters:")
        for cid, c in self.gt_clustering.items():
            print("%a:" % cid, c)

    def save_graph(self):
        pass

    def restore_graph(self):
        pass

    def verify_request(self, node_pairs):
        print("Start of verify_request: node_pairs", node_pairs)
        """
        The result of verification is deterministic and symmetric.
        So we need to keep and return the same result if called multiple
        times. To do this, we keep it in the graph.  By contrast, the
        result of each human decision is generated randomly for each
        request.
        """
        if self.verify_steps_until_return < 0:
            self.verify_steps_until_return = random.randint(0, self.max_delay_steps)
            print("sim::verify_request: first delay will be %d steps" %
                  self.verify_steps_until_return)

        new_edges = []
        for pr in set(node_pairs):              # make sure no repeats
            if pr[1] not in self.G[pr[0]]:      # no edge already
                assert(pr[0] < pr[1])
                cid0 = self.gt_node2cid[pr[0]]
                cid1 = self.gt_node2cid[pr[1]]
                is_match_correct = cid0 == cid1
                wgt = self.wgtr.random_wgt(is_match_correct)
                e = tuple([pr[0], pr[1], wgt])
                new_edges.append(e)
                self.verify_edges.append(e)
                # print("Adding verifier edge", e)

        self.G.add_weighted_edges_from(new_edges)

    def verify_result(self):
        if self.verify_steps_until_return == 0:
            edges_returned = self.verify_edges.copy()
            self.verify_edges.clear()
            self.verify_steps_until_return = random.randint(0, self.max_delay_steps)
            print("sim::verify_request: next delay will be %d steps" %
                  self.verify_steps_until_return)
            return edges_returned
        else:
            self.verify_steps_until_return -= 1
            return []

    def gen_human_wgt(self, pr):
        cid0 = self.gt_node2cid[pr[0]]
        cid1 = self.gt_node2cid[pr[1]]
        is_match_correct = cid0 == cid1
        wgt = self.wgtr.human_random_wgt(is_match_correct)
        return wgt

    def human_request(self, node_pairs):
        if self.human_steps_until_return < 0:
            self.human_steps_until_return = random.randint(0, self.max_delay_steps)
            print("sim::human_request: delay will be %d steps" %
                  self.human_steps_until_return)

        new_edges = []
        for pr in set(node_pairs):
            if pr[0] > pr[1]:
                pr = (pr[1], pr[0])
            wgt = self.gen_human_wgt(pr)
            e = tuple([pr[0], pr[1], wgt])
            new_edges.append(e)
            self.human_edges.append(e)
        self.G.add_weighted_edges_from(new_edges)

    def human_result(self):
        """
        """
        if self.human_steps_until_return == 0:
            edges_returned = self.human_edges.copy()
            self.human_edges.clear()
            self.human_steps_until_return = random.randint(0, self.max_delay_steps)
            print("sim::human_result: next delay will be %d steps" %
                  self.human_steps_until_return)
            return edges_returned
        else:
            self.human_steps_until_return -= 1
            return []

    @staticmethod
    def incremental_stats(num_human, clustering, node2cid,
                          true_clustering, true_node2cid):
        frac, prec, rec = ct.percent_and_PR(clustering, node2cid,
                                            true_clustering, true_node2cid)
        result = {"num human": num_human,
                  "num clusters": len(clustering),
                  "num true clusters": len(true_clustering),
                  "frac correct": frac,
                  "precision": prec,
                  "recall": rec}
        return result

    def trace_start_splitting(self, clustering, node2cid):
        """
        Record information about the number of human decisions vs. the
        accuracy of the current clustering.  The comparison is made
        against both the ground truth clustering and the "reachable"
        ground truth clustering. For each new number of
        human decisions, we record (1) this number, (2) the number of
        ground truth clusters (fixed value), (3) the number of current
        clusters, (4) the fraction of current clusters that are
        exactly correct, (5) the precision and (6) the recall.  The
        same thing will be done for the "reachable" clusters.
        """
        result = self.incremental_stats(0, clustering, node2cid,
                                        self.gt_clustering, self.gt_node2cid)
        self.gt_results = [result]
        result = self.incremental_stats(0, clustering, node2cid,
                                        self.r_clustering, self.r_node2cid)
        self.r_results = [result]
        self.prev_num_human = 0

    def trace_start_stability(self, clustering, node2cid):
        pass

    def trace_compare_to_gt(self, clustering, node2cid, num_human):
        if num_human <= self.prev_num_human:
            return
        result = self.incremental_stats(num_human, clustering, node2cid,
                                        self.gt_clustering, self.gt_node2cid)
        self.gt_results.append(result)
        result = self.incremental_stats(num_human, clustering, node2cid,
                                        self.r_clustering, self.r_node2cid)
        self.r_results.append(result)
        self.prev_num_human = num_human

    @staticmethod
    def csv_output(out_file, results):
        f = open(out_file, "w")
        f.write("human decisions, num clusters, num true clusters, " +
                "frac eq true, precision, recall\n")
        for res in results:
            f.write("%d, %d, %d, %.4f, %.4f %.4f\n" %
                    (res["num human"], res["num clusters"],
                     res["num true clusters"], res["frac correct"],
                     res["precision"], res["recall"]))
        f.close()

    @staticmethod
    def plot_convergence(results, filename=None):
        num_human_decisions = [r["num human"] for r in results]
        num_actual_clusters = [r["num clusters"] for r in results]
        num_true_clusters = [r["num true clusters"] for r in results]
        # frac_correct = [r["frac correct"] for r in results]

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Number of human decisions')
        ax1.set_ylabel('Number of clusters')

        color = 'blue'
        ax1.plot(num_human_decisions, num_actual_clusters, color=color)

        # ADD FRAC CORRECT TO PLOT ON RIGHT SIDE!!!!

        color = 'green'
        ax1.plot(num_human_decisions, num_true_clusters, color=color)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            print("Saved plot to", filename)
            plt.close()

    def generate_plots(self, out_prefix):
        out_name = out_prefix + "_gt.pdf"
        self.plot_convergence(self.gt_results, out_name)
        out_name = out_prefix + "_reach_gt.pdf"
        self.plot_convergence(self.r_results, out_name)
        """
        print('\nReachable GT:')
        self.plot_convergence(self.r_results)
        """


def find_np_ratio(gamma_shape, gamma_scale, ranker_per_node,
                  prob_match):
    print("=====")
    print("find_np_ratio: gamma_shape", gamma_shape,
          "gamma_scale", gamma_scale, "ranker_per_node",
          ranker_per_node, " prob_match", prob_match)
    print("from these we get (factoring at least one per node)")
    mean = 1 + gamma_shape * gamma_scale
    mode = 1 + (gamma_shape - 1) * gamma_scale
    std_dev = m.sqrt(gamma_shape) * gamma_scale

    print("mean", mean, " mode", mode, " std_dev %.2f" % std_dev)

    limit = int(mean + 10 * std_dev)

    pos_per_node = 0
    for k in range(2, limit + 1):
        prob_k1 = gamma.cdf(k - 0.5, gamma_shape, scale=gamma_scale)
        prob_k2 = gamma.cdf(k - 1.5, gamma_shape, scale=gamma_scale)
        prob_k = prob_k1 - prob_k2

        max_correct = min(ranker_per_node, k - 1)
        exp_correct = 0
        sum_prob_i = 0
        for i in range(max_correct + 1):
            prob_i = comb(k - 1, i) * prob_match**i * (1 - prob_match)**(k - 1 - i)
            term = i * prob_i
            sum_prob_i += prob_i
            exp_correct += term
        exp_correct /= sum_prob_i
        # print("k", k, "prob_k", prob_k, "exp_correct", exp_correct)

        pos_per_node += prob_k * exp_correct

    neg_per_node = ranker_per_node - pos_per_node
    ratio = neg_per_node / pos_per_node
    print("leaving find_np_ratio: neg_per_node %1.2f" % neg_per_node,
          " pos_per_node %1.2f" % pos_per_node,
          " np_ratio %1.2f" % ratio)
    return ratio


def find_np_ratio_gt(cluster_sizes, ranker_per_node, prob_match):
    num_nodes = sum(cluster_sizes)
    total_matches = num_nodes*ranker_per_node
    expected_positive = 0
    for ai in cluster_sizes:
        ei = min(ai*(ai-1)*prob_match, ranker_per_node*ai)
        print("ai %d, ei %.4f" % (ai, ei))
        expected_positive += ei
    np_ratio = total_matches / expected_positive - 1
    return np_ratio


# def find_np_ratio(num_clusters, nodes_per_cluster, ranker_per_node,
#                   prob_match):
#     """
#     Given: (a) the number of clusters, (b) the expected number of
#     nodes per cluster, (c) the number of matching produced by the
#     ranking algorithm for each node, and (d) the probability that a
#     correct match is formed between a pair of nodes that should match,
#     find the expected ratio of negative edges to positive edges.
#     """
#     print('nodes_per_cluster', nodes_per_cluster)
#     lmbd = 1 / (nodes_per_cluster - 1)
#     limit = 1 + round(15 / lmbd)

#     pos_per_cluster = 0
#     for i in range(2, limit+1):
#         p = m.exp(-lmbd * (i-1.5)) - m.exp(-lmbd * (i-0.5))
#         prs = i*(i-1) / 2 * prob_match
#         prs = min(prs, i*ranker_per_node)
#         pos = p * prs
#         pos_per_cluster += pos
#         print(i, prs, pos, pos_per_cluster)

#     neg_per_cluster = nodes_per_cluster * ranker_per_node - pos_per_cluster
#     print("pos_per_cluster:", pos_per_cluster, ", neg_per_cluster:", neg_per_cluster)
#     ratio = neg_per_cluster / pos_per_cluster
#     return ratio


def test_find_np_ratio():
    gamma_shape = 2
    gamma_scale = 3
    ranker_per_node = 10
    prob_match = 0.85
    ratio = find_np_ratio(gamma_shape, gamma_scale, ranker_per_node, prob_match)
    print("Final np ratio is %1.4f" % ratio)

    gamma_shape = 1
    gamma_scale = 3
    ranker_per_node = 15
    prob_match = 0.85
    ratio = find_np_ratio(gamma_shape, gamma_scale, ranker_per_node, prob_match)
    print("Final np ratio is %1.4f" % ratio)


def test_find_np_ratio_gt():
    cluster_sizes = [4, 8, 2, 1, 6, 6]
    ranker_per_node = 8
    prob_match = 0.9
    np_ratio = find_np_ratio_gt(cluster_sizes, ranker_per_node, prob_match)
    exp_np_ratio = .846154
    print("find_np_ratio_gt with cluster sizes", cluster_sizes,
          "ranker_per_node", ranker_per_node,
          "prob_match", prob_match,
          "yields np_ratio %.3f" % np_ratio,
          "expected np_ratio %.3f" % exp_np_ratio)

def test_after_gen(sim):
    print("\n================================\n"
          "Testing the generated simulator:\n"
          "================================")
    print("Number of clusters:  should be %d is %d"
          % (sim.params['num_clusters'], len(sim.gt_clustering)))

    num_nodes_in_graph = len(sim.G.nodes())
    num_nodes_in_node2cid = len(sim.gt_node2cid)
    num_nodes_in_clusters = sum([len(c)
                                 for c in sim.gt_clustering.values()])

    print("All three numbers of nodes should be equal:\n "
          "    %d in graph\n" % num_nodes_in_graph,
          "    %d in node2cid\n" % num_nodes_in_node2cid,
          "    %d in cluster" % num_nodes_in_clusters)

    """
    All node_matches sets are the same length, and that length is the params values.
    """
    num_from_ranker = params['num_from_ranker']
    print("Each node should have %d matches from the ranker" % num_from_ranker)
    all_same = True
    for node, matches in sim.ranker_matches.items():
        if len(matches) != num_from_ranker:
            all_same = False
            print("Error: node %s has %d matches" % (node, len(matches)))
    if all_same:
        print("Yes. All have the correct number of ranker matches")

    """
    Examine the distribution of true and false edges
    """
    tp = fp = tn = fn = zero_pos = zero_neg = 0
    for n0 in sim.G.nodes():
        cid0 = sim.gt_node2cid[n0]
        for n1 in sim.G[n0]:
            if n0 < n1:
                cid1 = sim.gt_node2cid[n1]
                wgt = sim.G[n0][n1]['weight']
                if cid0 == cid1:
                    if wgt > 0:
                        tp += 1
                    elif wgt < 0:
                        fn += 1
                    else:
                        zero_pos += 1
                else:
                    if wgt > 0:
                        fp += 1
                    elif wgt < 0:
                        tn += 1
                    else:
                        zero_neg += 1

    print("tp %d    fp %d\nfn %d    tn %d\nzero_pos %d  zero_neg %d"
          % (tp, fp, fn, tn, zero_pos, zero_neg))
    print("actual ratio = %1.2f" % ((fp + tn + zero_neg) / (tp + fn + zero_pos)))
    print("actual number of xx positive pairs", (tp + fn + zero_pos))
    print("actual number of xx negative pairs", (tn + fp + zero_neg))
    """
    exp_neg, exp_pos = sim.wgtr.eval_overlap()
    print("Correct match fraction with positive weight = %5.3f, expected = %5.3f"
          % ((tp / (tp+fn)), exp_pos))
    print("Incorrect match fraction with negative weight = %5.3f, expected = %5.3f"
          % ((tn / (tn+fp)), exp_neg))
    """

    """
    Did we get the right fraction of correct matches becoming real matches?
    """
    num_correct_edges = tp + fn + zero_pos
    num_possible = 0
    for c in sim.gt_clustering.values():
        nc = len(c)
        num_possible += nc * (nc - 1) // 2
    print("Fraction of true edges generated = %1.3f, expected = %1.3f"
          % (num_correct_edges / num_possible, sim.params['p_ranker_correct']))

    avg_per_cluster = len(sim.G.nodes()) / len(sim.gt_clustering)
    exp_per_cluster = 1 + sim.params['gamma_shape'] * sim.params['gamma_scale']
    print("Average cluster size is %1.2f" % avg_per_cluster)
    print("Expected cluster size is %1.2f" % exp_per_cluster)


def get_positive_missing(sim, num_requested):
    '''
    Retrieve a number of node pairs that are not edge-connected within a cluster.
    Results are restricted to at most one edge from each node.
    :param sim: The simulation
    :param num_requested: The target number of node pairs.
    :return: List of node pair tuples, with each tuple in increasting order.  Length of list is <= num_requested
    '''
    pos_missing = []
    nodes = list(sim.gt_node2cid.keys())
    random.shuffle(nodes)
    for n0 in nodes:
        cid0 = sim.gt_node2cid[n0]
        for n1 in sim.gt_clustering[cid0]:
            if n0 != n1 and n1 not in sim.G[n0]:
                pr = (min(n0, n1), max(n0, n1))
                if pr not in pos_missing:
                    pos_missing.append(pr)
                    break
        if len(pos_missing) == num_requested:
            break
    return pos_missing


def get_negative_missing(sim, num_requested):
    neg_missing = []
    max_tries = 50   # avoid infinite loop
    tries = 0
    nodes = list(sim.gt_node2cid.keys())
    num_nodes = len(nodes)
    while len(neg_missing) < num_requested and tries < max_tries:
        tries += 1
        n0 = nodes[random.randint(0, num_nodes - 1)]
        n1 = nodes[random.randint(0, num_nodes - 1)]
        cid0 = sim.gt_node2cid[n0]
        cid1 = sim.gt_node2cid[n1]
        if n0 != n1 and cid0 != cid1 and \
           n1 not in sim.G[n0]:
            pr = (min(n0, n1), max(n0, n1))
            neg_missing.append(pr)
    return neg_missing


def test_verify(sim):
    """
    1. Test abilty to ignore edges already in the graph.
    """
    pairs = []
    nodes = list(sim.gt_node2cid.keys())
    num_nodes = len(nodes)
    num_already_there = 2
    for _ in range(num_already_there):
        n0 = nodes[random.randint(0, num_nodes - 1)]
        n1 = next(iter(sim.G[n0].keys()))  # first nbr in n1 dictionary
        pr = (min(n0, n1), max(n0, n1))
        pairs.append(pr)
    sim.verify_request(pairs)
    edges = sim.verify_result()
    print("test_verify:")
    print("After adding edges already in the graph, num returned should be 0. "
          "It is", len(edges))

    """
    2. Test ability to gather missing edges.  Do it 2x
    """
    for tries in range(2):
        # Get missing edges
        target_pos = tries+1
        target_neg = tries+2
        pos_missing = get_positive_missing(sim, target_pos)
        neg_missing = get_negative_missing(sim, target_neg)
        missing = pos_missing + neg_missing
        print("Verify request / results try", tries)
        print("Num pos_missing (should be %d) is %d" % (target_pos, len(pos_missing)))
        print("Num neg_missing (should be %d) is %d" % (target_neg, len(neg_missing)))

        #  Make the request for weights for these edges. This does not get them
        #  yet.  Make sure though that it has set the delay.
        sim.verify_request(missing)
        steps_until = sim.verify_steps_until_return
        print("Delay steps until return should be non-negative and is",
              steps_until)

        # For the given number of steps, the result should be empty
        has_error = False
        while steps_until > 0 and not has_error:
            steps_until -= 1
            edges = sim.verify_result()
            if len(edges) > 0:
                has_error = True
                print("Error: with ", steps_until, "iterations remaining",
                      "returned", len(edges), "edges.")

        if not has_error:
            print("Successfully iterated through the delay without returning edges.")

        #  This time it should return edges.
        edges = sim.verify_result()
        print("Requested missing edges: positive", pos_missing,
              " negative:", neg_missing)
        print("returned are", edges)
        if len(missing) != len(edges):
            print("Error: incorrect number returned")
        else:
            print("Correct number returned")
        print("sim.verify_edges should be len 0, and it is", len(sim.verify_edges))


def get_positive_for_human(sim, num_pos):
    """
    get several that are known positive and several that are known negative
    """
    pos = []
    cluster_ids = list(sim.gt_clustering.keys())
    random.shuffle(cluster_ids)
    for cid in cluster_ids:
        c = sim.gt_clustering[cid]
        if len(c) == 1:
            continue
        c = list(c)
        pr = (min(c[0], c[1]), max(c[0], c[1]))
        pos.append(pr)
        if len(pos) >= num_pos:
            break
    return pos


def get_negative_for_human(sim, num_neg):
    nodes = list(sim.gt_node2cid.keys())
    num_nodes = len(nodes)
    neg = []
    max_tries = 30   # avoid infinite loop
    tries = 0
    while len(neg) < num_neg and tries < max_tries:
        tries += 1
        n0 = nodes[random.randint(0, num_nodes - 1)]
        n1 = nodes[random.randint(0, num_nodes - 1)]
        cid0 = sim.gt_node2cid[n0]
        cid1 = sim.gt_node2cid[n1]
        if n0 != n1 and cid0 != cid1:
            pr = (min(n0, n1), max(n0, n1))
            neg.append(pr)
    return neg


def test_human(sim):
    print("Test simulation of human")
    for tries in range(2):
        print("try", tries)
        print("Test human request / result try", tries)
        pos_for_human = get_positive_for_human(sim, tries + 2)
        neg_for_human = get_negative_for_human(sim, tries + 1)
        for_human = pos_for_human + neg_for_human
        print(for_human)
        sim.human_request(for_human)

        print("Requested edges from human: positive", pos_for_human,
              " negative:", neg_for_human)
        steps_until = sim.human_steps_until_return
        print("Delay steps until return should be non-negative and is",
              steps_until)

        # For the given number of steps, the result should be empty
        has_error = False
        while steps_until > 0 and not has_error:
            steps_until -= 1
            edges = sim.human_result()
            if len(edges) > 0:
                has_error = True
                print("Error: with ", steps_until, "iterations remaining",
                      "returned", len(edges), "edges.")

        if not has_error:
            print("Successfully iterated through the delay without returning edges.")

        #  This time it should return edges.
        edges = sim.human_result()
        print("Requested human edges: positive", pos_for_human,
              " negative:", neg_for_human)
        print("returned are", edges)
        if len(for_human) != len(edges):
            print("Error: incorrect number returned")
        else:
            print("Correct number returned")
        print("sim.human_edges should be len 0, and it is", len(sim.human_edges))


if __name__ == "__main__":
    test_find_np_ratio()
    test_find_np_ratio_gt()

    params = dict()
    params['pos_error_frac'] = 0.1
    params['num_clusters'] = 16

    # The following are parameters of the gamma distribution.
    # Recall the following properties:
    #      mean is shape*scale,
    #      mode is (shape-1)*scale
    #      variance is shape*scale**2 = mean*scale
    # So when these are both 2 the mean is 4, the mode is 2
    # and the variance is 4 (st dev 2).  And, when shape = 1,
    # we always have the mode at 0.
    #
    # The mean and the mode must be offset by 1 because every cluster
    # has at least one node.
    #
    params['gamma_shape'] = 1.5
    params['gamma_scale'] = 3
    # num_per_cluster= params['gamma_scale'] * params['gamma_shape'] + 1
    params['p_ranker_correct'] = 0.9
    params['p_human_correct'] = 0.98
    params['num_from_ranker'] = 10

    np_ratio = find_np_ratio(params['gamma_shape'],
                             params['gamma_scale'],
                             params['num_from_ranker'],
                             params['p_ranker_correct'])
    print('np_ratio = %1.3f' % np_ratio)

    scorer = es.ExpScores.create_from_error_frac(params['pos_error_frac'],
                                                  np_ratio)
    wgtr = wgtr.Weighter(scorer, human_prob=params['p_human_correct'])

    sim = Simulator(params, wgtr)
    sim.create_random_clusters()
    sim.generate_from_clusters()
    r_leng = len(sim.r_clustering)
    gt_leng = len(sim.gt_clustering)
    print("gt length", gt_leng, "reachable length", r_leng)
    test_after_gen(sim)
    test_verify(sim)
    test_human(sim)

    prior_clusters = {"cl00": ['n01', 'n02', 'n03'],
                      "cl01": ['n10', 'n11', 'n12', 'n13', 'n17'],
                      "cl02": ['n05', 'n07'],
                      "cl03": ['n41', 'n42', 'n27', 'n31'],
                      "cl04": ['n29', 'n22', 'n50'],
                      "cl05": ['n60'],
                      "cl06": ['n55'],
                      "cl07": ['n71', 'n75', 'n79', 'n80'],
                      "cl08": ['n81', 'n82', 'n83', 'n84']}
    params['num_clusters'] = 10
    params['num_from_ranker'] = 6
    sim = Simulator(params, wgtr)
    sim.create_clusters_from_ground_truth(prior_clusters)
    sim.generate_from_clusters()
    r_leng = len(sim.r_clustering)
    gt_leng = len(sim.gt_clustering)
    print("gt length", gt_leng, "reachable length", r_leng)
    test_after_gen(sim)
    test_verify(sim)
    test_human(sim)

    """
    print("\nClusters")
    for cid in sim.gt_clustering:
        print("%3d: %a" % (cid, sim.gt_clusters[cid]))

    print("Length gt_node2cid %d, should be equal to graph nodes %d" \
          % (len(sim.gt_node2cid), len(sim.G.nodes())))

    print("\nRanker matches\n")
    for nid in sim.ranker_matches:
        print("Node %s has matches:" %nid, end=' ')
        for m in sim.ranker_matches[nid]:
            print(m, end=' ')
        print()

    print("\nEdges for each node (ranker matches in each direction!)")
    for nid in sorted(sim.G.nodes()):
        print("Node ", nid)
        for m in sim.G[nid]:
            n0, n1 = min(nid, m), max(nid, m)
            print("(%a, %a, %1.4f)" % (n0, n1, sim.G[nid][m]['weight']))
    """
