
import networkx as nx
import cluster_tools as ct

"""
Given a simulator, a lower threshold and an upper threshold.

0. Copy the nodes of the graph from the simulator into the new graph.

1. For each edge of the old graph

a. if its weight is below the lower threshold, eliminate it

b. if its weight is above the upper threshold, keep it

c. otherwise, ask the human verifier to make a decision about it, and
   insert into the new graph if it is positive
"""


class Baseline(object):

    def __init__(self, G, human_request, human_result):
        self.G = G
        self.human_request = human_request
        self.human_result = human_result

    def generate_clustering(self, num_to_review):
        H = nx.Graph()
        H.add_nodes_from(self.G.nodes)
        kept_edges = []
        request_pairs = []
        edges = [e for e in self.G.edges.data('weight')]
        by_abs_wgt = sorted(edges, key=lambda e: abs(e[2]))
        request_pairs = [e[:2] for e in by_abs_wgt[:num_to_review]]
        kept_edges = [e for e in by_abs_wgt[num_to_review:] if e[2] > 0]
        self.human_request(request_pairs)
        human_edges = self.human_result()
        kept_edges += [e for e in human_edges if e[2] > 0]
        H.add_edges_from(kept_edges)

        idx = 0
        clustering = dict()
        for cc in nx.connected_components(H):
            clustering[idx] = cc

        node2cid = ct.build_node_to_cluster_mapping(clustering)
        return clustering, node2cid

    def evaluate_baseline(self, sim, out_prefix, n_min, n_max, n_inc=10):
        gt_results = []
        r_results = []
        for n in range(n_min, n_max + 1, n_inc):
            clustering, node2cid = self.generate_clustering(n)
            result = sim.incremental_stats(n, clustering, node2cid,
                                           sim.gt_clustering, sim.gt_node2cid)
            gt_results.append(result)
            result = sim.incremental_stats(n, clustering, node2cid,
                                           sim.r_clustering, sim.r_node2cid)
            r_results.append(result)

        out_name = out_prefix + "_gt.pdf"
        sim.plot_convergence(gt_results, out_name)
        out_name = out_prefix + "_reach_gt.pdf"
        sim.plot_convergence(r_results, out_name)
        # sim.csv_output(out_prefix + "_gt.csv", sim_i.gt_results)
        # sim.csv_output(out_prefix + "_r.csv", sim_i.r_results)

        # ##  ADD CSV OUTPUT BOTH HERE AND IN SIMULATOR.


"""

1. For each edge, generate the simulation result as though it is
human review and add to a dictionary.

2. Sort the edges by abs weight.

3. Make the decision about human labeling

4. For each number of human decisions in the incremented list, form
the graph, do connected components labeling, and analyze the
structure.  Add to the accumulated results.

5. Generate final statistics and plots

"""


class Baseline(object):

    def __init__(self, sim):
        self.sim = sim
        self.nodes = sim.G_orig.nodes
        # print("\n========")
        # print("In baseline.__init__:")
        edges = [e for e in sim.G_orig.edges.data('weight')]
        edges = [(min(e[0], e[1]), max(e[0], e[1]), e[2]) for e in edges]
        self.edges_by_abs_wgt = sorted(edges, key=lambda e: abs(e[2]))
        # print("edges_by_abs_wgt:", self.edges_by_abs_wgt)
        prs = [(min(e[0], e[1]), max(e[0], e[1])) for e in edges]
        self.dict_human = {pr: sim.gen_human_wgt(pr) for pr in prs}
        # print("dict_human:", self.dict_human)
        self.gt_results = []
        self.r_results = []

    def one_iteration(self, num_human):
        orig_edges = [e for e in self.edges_by_abs_wgt[num_human:] if e[2] > 0]
        human_prs = [(e[0], e[1]) for e in self.edges_by_abs_wgt[:num_human]]
        human_edges = [(pr[0], pr[1], self.dict_human[pr]) for pr in human_prs]
        human_edges = [e for e in human_edges if e[2] > 0]
        # print("\n--------")
        # print("orig_edges:", orig_edges)
        # print("human_edges:", human_edges)
        edges = orig_edges + human_edges
        new_G = nx.Graph()
        new_G.add_nodes_from(self.nodes)
        new_G.add_weighted_edges_from(edges)

        idx = 0
        clustering = dict()
        for cc in nx.connected_components(new_G):
            # print("idx =", idx, "cc =", list(cc))
            clustering[idx] = set(cc)
            idx += 1

        node2cid = ct.build_node_to_cluster_mapping(clustering)
        return clustering, node2cid

    def all_iterations(self, n_min, n_max, n_inc):
        for n in range(n_min, n_max + 1, n_inc):
            clustering, node2cid = self.one_iteration(n)
            result = self.sim.incremental_stats(n, clustering, node2cid,
                                                self.sim.gt_clustering,
                                                self.sim.gt_node2cid)
            self.gt_results.append(result)
            result = self.sim.incremental_stats(n, clustering, node2cid,
                                                self.sim.r_clustering,
                                                self.sim.r_node2cid)
            self.r_results.append(result)

    def generate_plots(self, out_prefix):
        out_name = out_prefix + "_gt.pdf"
        self.sim.plot_convergence(self.gt_results, out_name)
        out_name = out_prefix + "_reach_gt.pdf"
        self.sim.plot_convergence(self.r_results, out_name)
