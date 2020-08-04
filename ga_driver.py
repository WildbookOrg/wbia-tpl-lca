import collections
import datetime as dt
import logging
import networkx as nx
import os

import cluster_tools as ct
import compare_clusterings
import db_interface_sim
import edge_generator
import exp_scores as es
import graph_algorithm as ga
import weighter

'''

This file contains functionality needed to prepare and run one or more
instances of the LCA graph algorithm derived from the query request, the
database, and the configuration.  It contains functionality needed for
all currently-planned ways of initiating a run of the graph
algorithms.

What's needed?  

There are two primary ways that a run of the graph algorithm is
triggered:

(1) Addition of new edges from the verification algorithm following
match queries for new nodes (annotations or, eventually,
encounters).

(2) Addition of new edges from human decisions.

(3) A list of one or more cluster ids that need to be checked.

These latter two usually go together and may be the result of
reviewing a change indicated after new nodes and queries from (1).
What happens is that in the final review a mistake is found. This
requires at least one new edges specifying two nodes that are in the
same cluster and show different animals, or two nodes that are in the
different clusters that are show the same animal. These are human
decisions, but they generally also require the context of the affected
clusters, hence (2) and (3).

Now onto specifics:

Three major data objects are created here prior to running the GA.
(1) the configuration dictionary
(2) the weighters
(3) the list of ccPIC (connected component of Potentially-Impacted Clusters):
  each is a pair containing a list of subgraph edges and a dictionary of the
  subclustering.  Note that this subclustering will be empty if the edges are
  all between nodes that have not been clustered already.

Not created by functionality here are
(1) the database interface object and
(2) the edge_generator object
Both are created separately based on the source of the data for the
graph algorithm, whether it be from similulation or from a true animal
id database and real human deciders. There are abstract base class
interfaces to these.  See overall_driver.py for examples of creating
these for simulations.

The work flow proceeds in the following stages

1. Based on the config file and based on recent ground-truth positive
and negative decisions and verification results, create the parameters
dictionary and the weighter objects.

2. Form the ga_driver. Based on the query and the current ID graph,
the constructor produces a list of "ccPICs" (connected compontent of
potentially impacted clusters). These are independent subgraphs that
could change. LCA is run on each separately. Each ccPIC contains the
list of edges and the list of current clusters.

3. Method ga_driver.run_all_ccPIC runs the LCA graph algorithm
(calling the function ga_driver.run_ga_on_ccPIC) on each ccPIC.

4. The main result of the graph algorithm is a list of cluster_change
objects (see compare_clusterings.py). At the point at which the graph
algorithm runs end, no change in the clusterings will have yet been
commmitted to the database. (However, during the running of LCA new
edges are created.)

5. The last step is committing the cluster changes. These depend on
the type of change and the requirements imposed from outside for
reviewing. For example, working with species like zebras and rally
events, no final reviews at all may be needed. On the other hand, it
may be that only merge and split type operations require review and
additions to existing clusters or formation of new clusters do not.
'''


logger = logging.getLogger()


def params_and_weighters(config_ini, verifier_gt):
    ga_params = dict()

    phc = float(config_ini['EDGE_WEIGHTS']['prob_human_correct'])
    assert 0 < phc < 1
    ga_params['prob_human_correct'] = phc
    s = config_ini['EDGE_WEIGHTS']['augmentation_names']
    ga_params['aug_names'] = s.strip().split()

    p = float(config_ini['ITERATIONS']['min_delta_prob_converge'])
    assert 0 < p <= 1
    ga_params['min_delta_prob_converge'] = p

    s = float(config_ini['ITERATIONS']['min_delta_stability_ratio'])
    assert s > 1
    ga_params['min_delta_stability_ratio'] = s

    n = int(config_ini['ITERATIONS']['num_per_augmentation'])
    assert n >= 1
    ga_params['num_per_augmentation'] = n

    n = int(config_ini['ITERATIONS']['tries_before_edge_done'])
    assert n >= 1
    ga_params['tries_before_edge_done'] = n

    i = int(config_ini['ITERATIONS']['ga_iterations_before_return'])
    assert i >= 1
    ga_params['ga_iterations_before_return'] = i

    log_level = config_ini['LOGGING']['log_level']
    ga_params['log_level'] = log_level
    log_file = config_ini['LOGGING']['log_file']
    ga_params['log_file'] = log_file
    try:
        os.remove(log_file)
    except Exception:
        pass
    log_format = '%(levelname)-6s [%(filename)21s:%(lineno)3d] %(message)s'
    logging.basicConfig(
        filename=log_file, level=log_level, format=log_format
    )

    ga_params['draw_iterations'] = config_ini['DRAWING'].getboolean('draw_iterations')
    ga_params['drawing_prefix'] = config_ini['DRAWING']['drawing_prefix']

    wgtrs = generate_weighters(ga_params, verifier_gt)

    wgtr = wgtrs[0]
    ga_params['min_delta_score_converge'] = -2 * wgtr.wgt(ga_params['min_delta_prob_converge'])
    ga_params['min_delta_score_stability'] = ga_params['min_delta_score_converge'] / \
        ga_params['min_delta_stability_ratio']

    return ga_params, wgtrs


def generate_weighters(ga_params, verifier_gt):
    wgtrs = []
    for aug in ga_params['aug_names']:
        if aug == 'human':
            continue
        assert aug in verifier_gt
        probs = verifier_gt[aug]
        logger.info('Building scorer and weighter for verifier %s' % aug)
        scorer = es.exp_scores.create_from_samples(probs['gt_positive_probs'],
                                                   probs['gt_negative_probs'])
        wgtr = weighter.weighter(scorer, ga_params['prob_human_correct'])
        wgtrs.append(wgtr)
    return wgtrs


class ga_driver(object):
    def __init__(self, verifier_results, human_decisions, cluster_ids_to_check,
                 db, edge_gen, ga_params):
        logger.info('=============================================')
        logger.info('Start of graph algorithm overall driver which')
        logger.info('creates one or more graph algorithm instances.')
        logger.info(dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info('Parameters, input and derived:')
        for k, v in ga_params.items():
            logger.info("    %s: %a" % (k, v))

        assert len(verifier_results) > 0 or len(human_decisions) > 0
        self.db = db
        self.edge_gen = edge_gen
        self.ga_params = ga_params

        self.edge_quads = \
            self.edge_gen.new_edges_from_verifier(verifier_results) + \
            self.edge_gen.new_edges_from_human(human_decisions)
        logger.info('Formed incoming graph edge quads to initiate LCA:')
        for q in self.edge_quads:
            logger.info('   (%a, %a, %d, %s)' % (q[0], q[1], q[2], q[3]))
        if len(cluster_ids_to_check) == 0:
            logger.info('No particular clusters to check using LCA (this is typical)')
        else:
            logger.info('Checking the following clusters using LCA: %a' % cluster_ids_to_check)

        self.temp_count = 0
        self.temp_cids = set()
        self.temp_node_to_cid = dict()
        self.temp_cid_to_node = dict()
        self.direct_cids = set(cluster_ids_to_check)
        self.cid_pairs = set()
        self.find_direct_cids_and_pairs()
        self.find_indirect_cid_pairs()
        self.form_ccPICs()

        self.changes_to_review = []

    def get_cid(self, n):
        cid = self.db.get_cid(n)
        if cid is None:
            if n in self.temp_node_to_cid:
                cid = self.temp_node_to_cid[n]
            else:
                self.temp_count += 1
                cid = -self.temp_count
                self.temp_node_to_cid[n] = cid
                self.temp_cid_to_node[cid] = n
        return cid

    def add_cid_pair(self, cid0, cid1):
        if cid0 < cid1:
            self.cid_pairs.add((cid0, cid1))
        else:
            self.cid_pairs.add((cid1, cid0))

    def find_direct_cids_and_pairs(self):
        for n0, n1, _, _ in self.edge_quads:
            cid0 = self.get_cid(n0)
            cid1 = self.get_cid(n1)
            if cid0 != cid1:
                self.add_cid_pair(cid0, cid1)
                self.direct_cids.add(cid0)
                self.direct_cids.add(cid1)
            else:
                self.direct_cids.add(cid0)

    def find_indirect_cid_pairs(self):
        for cid in self.direct_cids:
            if self.is_temp(cid):
                node = self.temp_cid_to_node[cid]
                outgoing_edges = self.db.edges_from_node(node)
            else:
                outgoing_edges = self.db.edges_leaving_cluster(cid)
            aggregate_edges = collections.defaultdict(int)
            for n0, n1, w, _ in outgoing_edges:
                aggregate_edges[(n0, n1)] += w
            for (n0, n1), sum_w in aggregate_edges.items():
                if sum_w > 0:
                    cid0 = self.get_cid(n0)
                    cid1 = self.get_cid(n1)
                    self.add_cid_pair(cid0, cid1)

    def is_temp(self, cid):
        return -self.temp_count <= cid < 0

    def form_ccPICs(self):
        self.ccPICs = []
        cid_graph = nx.Graph()
        cid_graph.add_edges_from(self.cid_pairs)
        for cc in nx.connected_components(cid_graph):
            cids = list(cc)
            clustering = {}
            nodes = set()
            for cid in cids:
                if self.is_temp(cid):
                    nodes.add(self.temp_cid_to_node[cid])
                else:
                    nodes_in_c = set(self.db.get_nodes_in_cluster(cid))
                    clustering[cid] = nodes_in_c
                    nodes |= nodes_in_c
            edges = self.db.edges_between_nodes(nodes)
            # print('final set of nodes:', nodes)
            # print('resulting edges', edges)
            # print('clustering', clustering)
            ccPIC = (edges, clustering)
            self.ccPICs.append(ccPIC)

        logger.info("Formed %d ccPIC edge and clustering pairs, having" % len(self.ccPICs))
        for e, c in self.ccPICs:
            logger.info("    %d edges involving %d current clusters"
                         % (len(e), len(c)))

    def run_ga_on_ccPIC(self, ccPIC_edges, ccPIC_clustering):
        gai = ga.graph_algorithm(ccPIC_edges,
                                 ccPIC_clustering.values(),
                                 self.ga_params['aug_names'],
                                 self.ga_params,
                                 self.edge_gen.edge_request_cb,
                                 self.edge_gen.edge_result_cb)

        """
        Add call backs for removing nodes, pausing, getting intermediate
        results, and getting the status.
        """
        gai.set_remove_nodes_cb(self.edge_gen.remove_nodes_cb)

        """
        Could add other callbacks, such as
        gai.set_status_check_cbs(...)  # Get GA status. Details TBD
        gai.set_result_cbs(...)  # Get current clustering
        gai.set_log_contents_cbs(...)  #
        """

        """
        This runs the main loop 10 iterations at a time in a while
        loop. Currently, it is written to run synchronously, but of course
        it will eventually run asychronously and therefore the callbacks
        will be used to feed it informationa and get intermediate results.
        """
        iter_num = 0
        converged = False
        paused = False
        while not converged:
            num_iter_to_run = 10
            paused, iter_num, converged = gai.run_main_loop(
                iter_num, iter_num + num_iter_to_run
            )

        """
        Compute and then return the final information - the changes to
        the clusters.
        """
        ccPIC_n2c = ct.build_node_to_cluster_mapping(ccPIC_clustering)
        changes = compare_clusterings.find_changes(
            ccPIC_clustering,
            ccPIC_n2c,
            gai.clustering,
            gai.node2cid,
        )

        logger.info('After LCA convergence on ccPIC, here are the cluster changes:')
        for i, cc in enumerate(changes):
            logger.info("Change %d" % i)
            cc.log_change()

        return changes

    def run_all_ccPICs(self):
        for edges, clustering in self.ccPICs:
            changes = self.run_ga_on_ccPIC(edges, clustering)
            self.changes_to_review.append(changes)
        return self.changes_to_review


if __name__ == "__main__":
    ga_params = {'aug_names': ['vamp', 'human'],
                 'prob_human_correct': 0.97,
                 'log_level': logging.DEBUG}

    log_file = 'test_ga_driver.log'
    try:
        os.remove(log_file)
    except Exception:
        print("FAILED")
    log_format = '%(levelname)-6s [%(filename)18s:%(lineno)3d] %(message)s'
    logging.basicConfig(
        filename=log_file, level=ga_params['log_level'], format=log_format
    )
    logging.info('=================================')
    logging.info('Start of example to test ga_driver')

    db_quads = [('a', 'b', 45, 'vamp'),
                ('a', 'd', 50, 'vamp'),
                ('a', 'd', -100, 'human'),
                ('b', 'd', -85, 'vamp'),
                ('b', 'd', 100, 'human'),
                ('d', 'f', 45, 'vamp'),
                ('d', 'f', -100, 'human'),
                ('f', 'h', 4, 'vamp'),
                ('f', 'i', 6, 'vamp'),
                ('f', 'i', -100, 'human'),
                ('h', 'i', 85, 'vamp'),
                ('h', 'j', 80, 'vamp'),
                ('i', 'j', 75, 'vamp'),
                ('j', 'k', -100, 'human'),
                ('k', 'l', 80, 'vamp'),
                ('l', 'm', -50, 'vamp'),
                ('l', 'm', 100, 'human')]
    db_clusters = {100: ('a', 'b'),
                   101: ('d'),
                   102: ('h', 'i', 'j'),
                   103: ('k', 'l')}
    db = db_interface_sim.db_interface_sim(db_quads, db_clusters)

    verifier_results = [('b', 'e', 0.9, 'vamp'),
                        ('f', 'g', 0.15, 'vamp')]
    human_decisions = [('a', 'c', True)]
    cluster_ids_to_check = ([103])

    gt_probs = {
        'vamp':
        {'gt_positive_probs': [0.90, 0.98, 0.60,
                               0.80, 0.93, 0.97,
                               0.45, 0.83, 0.92,
                               0.85, 0.79, 0.66],
         'gt_negative_probs': [0.01, 0.55, 0.24, 0.16, 0.05,
                               0.02, 0.60, 0.04, 0.32, 0.25,
                               0.43, 0.01, 0.02, 0.33, 0.23,
                               0.04, 0.23]
        }
        }

    weighters = generate_weighters(ga_params, gt_probs)
    print(weighters)
    edge_gen = edge_generator.edge_generator(db, weighters[0])

    gad = ga_driver(verifier_results, human_decisions, cluster_ids_to_check,
                    db, edge_gen, ga_params)

    ccp0 = [[('a', 'b', 45, 'vamp'),
             ('a', 'c', 100, 'human'),
             ('a', 'd', 50, 'vamp'),
             ('a', 'd', -100, 'human'),
             ('b', 'd', -85, 'vamp'),
             ('b', 'd', 100, 'human'),
             ('b', 'e', 83, 'vamp')],
            {100: {'b', 'a'}, 101: {'d'}}]

    ccp1 = [[('f', 'g', -95, 'vamp'),
             ('f', 'h', 4, 'vamp'),
             ('f', 'i', 6, 'vamp'),
             ('f', 'i', -100, 'human'),
             ('h', 'i', 85, 'vamp'),
             ('h', 'j', 80, 'vamp'),
             ('i', 'j', 75, 'vamp')],
            {102: {'j', 'h', 'i'}}]

    ccp2 = [[('k', 'l', 80, 'vamp'),
             ('l', 'm', -50, 'vamp'),
             ('l', 'm', 100, 'human')],
            {103: {'k', 'l'}}]

    corr_ccPIC = [ccp0, ccp1, ccp2]
    if len(corr_ccPIC) == len(gad.ccPICs):
        print('Correct number of ccPICs found:', len(corr_ccPIC))
    else:
        print('Incorrect length of ccPICs: found %d, expected %d'
              % (len(gad.ccPICs), len(corr_ccPIC)))

    i = 0
    for est, exp in zip(corr_ccPIC, gad.ccPICs):
        print('----------------')
        print("Testing ccPIC", i)
        if est[0] == exp[0]:
            print('Edge list is correct:')
            print(exp[0])
        else:
            print('Error in edge list:')
            print('Estimated', est[0])
            print('Expected', exp[0])

        if est[1] == exp[1]:
            print('Cluster dictionary is correct:')
            print(exp[1])
        else:
            print('Error in cluster dictionary:')
            print('Estimated', est[1])
            print('Expected', exp[1])
        i += 1
        
