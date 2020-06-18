# -*- coding: utf-8 -*-
"""
2020-06-15:

An example showing how the graph algorithm code can be set up and
used, including use of callbacks. It does not use real weight but
instead uses hand-specified weights. These come out of the
edge_generator object.
"""

import logging
import os

import cluster_tools as ct
import compare_clusterings as cc
import exp_scores as es
import graph_algorithm as ga
import weighter


def generate_weighter(gt_pos_probs, gt_neg_probs, prob_human_correct, max_weight):
    """
    Create the object that will generate weights. Input parameters:

    . gt_pos_probs, gt_neg_probs are lists of probabilities from
      positive and negative verification algorithm decisions (not
      human decisions - output from the verification algorithms)

    . prob_human_correct: the probability that a human has made a
      correct decision about whether or not a pair of annotations
      shows the same animal; it is recommended that this probability
      not be set over 0.99 and perhaps should be much smaller.

    . max_weight - this just gives a scale factor on the logit
      function.  Rethinking the idea of this - maybe force it to just
      be 1.0.  (See the start of the edge_generator)
    """
    assert max_weight > 0
    scorer = es.exp_scores.create_from_samples(gt_pos_probs, gt_neg_probs)
    wgtr = weighter.weighter(scorer, prob_human_correct, max_weight)
    return wgtr


class edge_generator(object):
    def __init__(self, wgtr):
        """
        Hmmm....  The configuration of the wgtr is crucial and should
        not be tampered with...  If the wgtr is changed then earlier
        weights --- even from a year or more in the past --- will not
        be comparable to new weights. Need to think about this, but
        for now, let's skip this question.
        """
        self.wgtr = wgtr

        """
        The edges to start the graph algorithm come from two sources:
        Source 1: Pre-existing edges should come from the database and
        they should likely be stored as weights. These edges can be
        either from VAMP or from human decisions. They should be
        stored in the database as weights, and so come out of the
        database as weights. In this example, however, I start with
        probabilities and human decisions and convert them to weights.
        """
        self.pre_existing_vamp_edges = {
            ('a', 'b'): self.wgtr.wgt(0.85),
            ('b', 'e'): self.wgtr.wgt(0.2),
            ('e', 'f'): self.wgtr.wgt(0.35),
        }
        self.pre_existing_human_edges = {
            ('a', 'b'): self.wgtr.human_wgt(True),
            ('e', 'f'): self.wgtr.human_wgt(False),
        }

        """
        Source 2: New edges for the new nodes at the start of the
        computation should come from VAMP or some other verification
        algorithm. They should be converted from the VAMP
        probabilities to weights using the weighting function. While
        it is not done here, these weights should be stored in the
        database after they are generated.
        """
        self.initial_vamp_probs = {
            ('a', 'c'): 0.65,
            ('a', 'd'): 0.1,
            ('c', 'e'): 0.75,
            ('d', 'e'): 0.7,
        }

        """
        Subsequently, during its computation, the graph algorithm will
        ask for edges either from VAMP or from a human. These will come
        back as probabilities for VAMP and binary (boolean) (or
        trinary, if we are considering "incomparable", decisions from
        the human. (I did not do this here.) These are converted to
        weights and these weights  are sent to the graph algorithm by
        the callbacks. In addition, these weights must be stored in
        the database. Again, this is not done in this example here.
        """
        self.initial_human_decisions = {('a', 'b'): True}
        self.vamp_probs = {
            ('a', 'e'): 0.35,
            ('a', 'f'): 0.2,
            ('b', 'c'): 0.8,
            ('b', 'd'): 0.25,
            ('b', 'f'): 0.05,
            ('c', 'd'): 0.1,
            ('c', 'f'): 0.1,
            ('d', 'f'): 0.05,
        }
        self.human_decisions = {
            ('a', 'b'): True,
            ('a', 'c'): True,
            ('a', 'd'): False,
            ('a', 'e'): False,
            ('a', 'f'): False,
            ('b', 'c'): True,
            ('b', 'd'): False,
            ('b', 'e'): False,
            ('b', 'f'): False,
            ('c', 'd'): False,
            ('c', 'e'): False,
            ('c', 'f'): False,
            ('d', 'e'): True,
            ('d', 'f'): False,
            ('e', 'f'): False,
        }

        """
        Information to demo the removal of nodes. Removal only occurs
        once in this demo.
        """
        self.has_remove_been_called = False
        self.nodes_to_remove = ['f']

        """
        As a separate note: this object introduces a delay between
        when edges are requested and when they are returned. This is
        just for demo purposes and should be ignored in the real
        system.
        """
        self.max_delay_steps = 4
        self.steps_until_return = self.max_delay_steps

        """
        List to store requests
        """
        self.edge_requests = []

    def initial_edges(self):
        """
        Generate the initial edges for the graph algorithm from the
        pre-existing edges in the database and from the initial VAMP
        (verification algorithm) probabilities.
        """
        edges = []
        for (n0, n1), w in self.pre_existing_vamp_edges.items():
            e = (n0, n1, w, 'vamp')
            edges.append(e)

        for (n0, n1), w in self.pre_existing_human_edges.items():
            e = (n0, n1, w, 'human')
            edges.append(e)

        for (n0, n1), p in self.initial_vamp_probs.items():
            w = self.wgtr.wgt(p)
            e = (n0, n1, w, 'vamp')
            edges.append(e)

        return edges

    def edge_request_cb(self, req_list):
        """
        Here comes the request from the graph algorithm. This is where
        the calls to VAMP and to human reviewers will be launched.  In this
        example, however, the requests are simply stored because we
        have hand-generated results to result in the result
        function.
        """
        self.edge_requests += req_list

    def edge_result_cb(self):
        """
        Check the similuted delay and only continue if the delay has
        reached 0.
        """
        if self.steps_until_return > 0:
            self.steps_until_return -= 1
            return []
        self.steps_until_return = self.max_delay_steps

        """
        Grab the hand-generated results from either the vamp or the
        human-result dictionary.
        """
        edge_list = []
        for n0, n1, aug_name in self.edge_requests:
            if aug_name == 'vamp':
                p = self.vamp_probs[(n0, n1)]
                w = self.wgtr.wgt(p)
            else:
                b = self.human_decisions[(n0, n1)]
                w = self.wgtr.human_wgt(b)
            e = (n0, n1, w, aug_name)
            edge_list.append(e)

        self.edge_requests.clear()
        return edge_list

    def remove_nodes_cb(self):
        """
        Demonstrate the removal of a node.
        """
        to_remove = []
        if len(self.nodes_to_remove) > 0:
            to_remove = self.nodes_to_remove
            self.to_remove = []
        return to_remove


if __name__ == '__main__':
    """
    Assign the default parameterts. These are modified below.  Need a
    better way to handle this.
    """
    ga_params = ga.default_params()

    """
    Start the logging process by assigning the file and clearing it if
    it is there. Then set the format. This format is reset at the
    start of the graph algorithm, but that's fine.
    """
    log_file = 'example.log'
    try:
        os.remove(log_file)
    except Exception:
        pass
    log_format = '%(levelname)-6s [%(filename)18s:%(lineno)3d] %(message)s'
    logging.basicConfig(
        filename=log_file, level=ga_params['log_level'], format=log_format
    )
    logging.info('=================================')
    logging.info('Start of example to demo the GA.')

    """
    Here is a hand-generated set of probabilities that might be
    generated from a trained VAMP. These should be replaced by real
    probabilities produced on training pairs that were not used to
    train VAMP. There should be many more of these, of course. There
    should also be a set of these for each verification algorithm
    (assuming we move past VAMP). Note to future self: if we have
    multiple verification algorithms, the weighter will need to be
    revised. This should not be too hard.
    """
    gt_pos_probs = [0.98, 0.60, 0.80, 0.93, 0.97, 0.45, 0.83, 0.92, 0.85, 0.79, 0.66]
    gt_neg_probs = [
        0.01,
        0.55,
        0.24,
        0.16,
        0.05,
        0.02,
        0.60,
        0.04,
        0.32,
        0.25,
        0.43,
        0.01,
        0.02,
        0.33,
        0.41,
        0.23,
        0.04,
        0.23,
    ]

    """
    Form the parameters, and the weighter.
    """
    wgtr = generate_weighter(
        gt_pos_probs,
        gt_neg_probs,
        ga_params['prob_human_correct'],
        ga_params['max_edge_weight'],
    )

    """
    Form the edge generator. This is the object to pay closest
    attention to.
    """
    eg = edge_generator(wgtr)

    """
    The initial clustering is formed from 'a', 'b', 'e' and 'f'. These
    are analogous to database annotations.  The queury nodes are 'c'
    and 'c'. The dictionary keys here are just placeholders for what
    will a UUID, perhaps that of the marked individual.
    """
    old_clustering = {0: ['a', 'b'], 1: ['e'], 2: ['f']}
    query_nodes = ['c', 'd']

    """
    Update the configuration parameters. We need something better with
    the parameters here. Need a cleaner way to get them all set
    consistently.  TBD
    """
    min_converge = -0.9 * (wgtr.human_wgt(True) - wgtr.human_wgt(False))
    ga_params['min_delta_score_converge'] = min_converge
    ga_params['min_delta_score_stability'] = (
        min_converge / ga_params['min_delta_stability_ratio']
    )
    ga_params['compare_to_ground_truth'] = False

    """
    Build the graph algorithm object
    """
    aug_names = ['vamp', 'human']
    gai = ga.graph_algorithm(
        eg.initial_edges(),
        old_clustering.values(),
        aug_names,
        ga_params,
        eg.edge_request_cb,
        eg.edge_result_cb,
        log_file,
    )

    """
    Add call backs for removing nodes, pausing, getting intermediate
    results, and getting the status.
    """
    gai.set_remove_nodes_cb(eg.remove_nodes_cb)

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
    Output final information --- clusters:  This will be in the form
    of the changes to the clusters.  Singletons, all new, extended,
    merge, split and merge/split.  This is what I'm working on net
    """
    changes = cc.compare_clusterings(
        old_clustering,
        ct.build_node_to_cluster_mapping(old_clustering),
        ga.clustering,
        ga.node2cid,
    )
