# -*- coding: utf-8 -*-
"""
Manage the request and return of weights for edges based on the
results from verification algorithms and human reviewers. Together,
the algorithms and the review are called "augmentation methods". There
may be no verification algorithm or there may be multiple such
algorithms. Results are used once for each combination of edge
node pair and verification algorithm. Results may be obtained from
multiple human reviewers. The names of the verification algorithm are
provided in the order they are to be called, with the human reviewer
being last.

The manager keeps a record of the requests it makes, allowing only one
request per node pair at once. This avoids redundant requests, but can
lead to indefinite waiting for dropped requests. These requests can be
reissued by resetting the list of node pairs it is waiting for. From
outside the graph algorithm this is done through calling the
graph_algorithm.reset_waiting method.

Information is kept in a dictionary that associates each node pair
with a list of booleans.  A boolean at location i for a given node
pair indicates whether or not the i-th augmentation method has
returned a wgt for the node pair. This prevents return of redundant
weights. A simple set called waiting_for keeps the triples of node,
node and augmentation method that has been called for.
"""

import collections
import logging
import os

from graph_algorithm import default_params

logger = logging.getLogger()


def empty_gen(n):
    # Create a function to generate a list with n False values.
    def gen():
        return [0] * n

    return gen


class weight_manager(object):
    def __init__(self, aug_names, params, request_cb, result_cb):
        """
        Initialize with a single request callback, a single result
        callback and the names of the augmentation methods. This list
        must end with "human".  Initialize the augment_count dict
        and the set of edges that we are waiting for.
        """
        self.aug_names = aug_names
        self.tries_before_done = params['tries_before_edge_done']
        self.request_cb = request_cb
        self.result_cb = result_cb
        assert self.aug_names[-1] == 'human'
        self.num_names = len(self.aug_names)
        self.augment_count = collections.defaultdict(empty_gen(self.num_names))
        self.waiting_for = set()
        self.counts = [0] * self.num_names

    def _name_to_index(self, name):
        """
        Find the index of the augmentation name in the list. It is an
        error if it is not there.
        """
        try:
            return self.aug_names.index(name)
        except ValueError:
            print('Error:  Augmentation function name', name, 'is unknown')
            assert False

    def get_initial_edges(self, labeled_edges):
        """
        Given a sequence of labeled edges, each in the form
          (n0, n1, w, name)
        Return all 3-tuples of (n0, n1, w) where (n0, n1) are part of
        at least one edge and w is the combined weight across all
        edges for the node pair. The return is in the form of
        generator. In addition, for each (n0, n1) record which
        augmentation methods produced edge results.  If a (non-human)
        verification method is repeated for a node pair, only the
        first weight is used.
        """
        ret_edges = collections.defaultdict(float)
        for e in labeled_edges:
            if e[0] < e[1]:
                pr = (e[0], e[1])
            else:
                pr = (e[1], e[0])
            w = e[2]
            i = self._name_to_index(e[3])
            ac = self.augment_count[pr]
            if (ac[i] == 0) or (i == self.num_names - 1):
                ac[i] += 1
                ret_edges[pr] += w

        # Return results through a generator
        for pr, w in ret_edges.items():
            triple = (pr[0], pr[1], w)
            yield triple

    def request_new_weights(self, node_pairs):
        """
        Make a request for the next weight for each node pair.  For
        each pair, the first augmentation method used corresponds to
        the first False entry in the list. If there are no False's then
        the human review augmentation method is used. If the object is
        already waiting for the node_pair / augmentation combination
        then the node pair is skipped.
        """
        req_list = []
        for pr in node_pairs:
            assert pr[0] < pr[1]
            try:
                i = self.augment_count[pr].index(0)
            except ValueError:
                i = -1
            req = (pr[0], pr[1], self.aug_names[i])
            if req not in self.waiting_for:
                self.waiting_for.add(req)
                req_list.append(req)
        logger.info('Request edges: %s' % str(req_list))
        self.request_cb(req_list)

    def get_weight_changes(self):
        """
        Return the results of the augmentation methods in the form of
           (n0, n1, delta_wgt)
        where delta_wgt is the weight returned by the method. It is
        called a delta_wgt because it is a change to the weight of the
        edge in the LCA clustering graph.

        Note, this should be able to handle edges that it has not
        requested and handle nodes that do not yet exist.
        """
        new_results = self.result_cb()
        for (n0, n1, new_wgt, a_name) in new_results:
            if n0 > n1:
                n0, n1 = n1, n0
            pr = (n0, n1)
            req = (n0, n1, a_name)
            if req in self.waiting_for:
                self.waiting_for.remove(req)
            ac = self.augment_count[pr]  # creates list if not there
            i = self._name_to_index(a_name)

            if i == len(self.aug_names) - 1 or ac[i] == 0:
                e = (n0, n1, new_wgt)
                ac[i] += 1
                self.counts[i] += 1
                yield e

    def edge_counts(self):
        s = ''
        for i in range(self.num_names):
            s += ' ' + self.aug_names[i] + ' ' + str(self.counts[i])
        return s

    def num_human_decisions(self):
        return self.counts[-1]

    def awaiting_requests(self):
        return self.waiting_for

    def reset_waiting(self):
        self.waiting_for.clear()

    def futile_tester(self, n0, n1):
        ac = self.augment_count[(n0, n1)]
        return ac[-1] >= self.tries_before_done


########################################################


class test_callbacks(object):
    def __init__(self, edges_to_add, num_to_return, unexpected_edges):
        self.edges_to_return = edges_to_add
        self.edges_requested = []
        self.unexpected_edges = unexpected_edges
        self.num_to_return = num_to_return
        self.ntr_index = -1
        self.prev_index = -1

    def request_cb(self, new_req_edges):
        self.edges_requested += new_req_edges

    def result_cb(self):
        self.ntr_index = (self.ntr_index + 1) % len(self.num_to_return)
        k = min(self.num_to_return[self.ntr_index], len(self.edges_requested))
        ret_edges = []
        if len(self.unexpected_edges) > 0:
            ret_edges += self.unexpected_edges
            self.unexpected_edges = []
        while k > 0:
            n0, n1, a_name = self.edges_requested.pop(0)
            k -= 1
            for i, ae in enumerate(self.edges_to_return):
                if n0 == ae[0] and n1 == ae[1] and a_name == ae[3]:
                    self.edges_to_return.pop(i)
                    ret_edges.append(ae)
                    break
        return ret_edges


if __name__ == '__main__':
    log_fname = './test.log'
    try:
        os.remove(log_fname)
    except Exception:
        pass

    p = default_params()

    aug_names = ['vamp', 'siamese', 'human']
    log_format = '%(levelname)-6s [%(filename)18s:%(lineno)d] %(message)s'
    log_level = logging.INFO
    logging.basicConfig(filename=log_fname, level=p['log_level'], format=log_format)
    init_edges = [
        (0, 1, 0.5, 'vamp'),
        (0, 1, -0.3, 'siamese'),
        (0, 1, 0.7, 'human'),
        (1, 2, -0.8, 'vamp'),
        (0, 2, 0.8, 'siamese'),
        (1, 2, -0.4, 'human'),
        (1, 2, 0.7, 'vamp'),  # should be ignored...
        (1, 2, -1.0, 'siamese'),
        (0, 1, 0.8, 'human'),
    ]  # should not be ignored...
    aug_names = ['vamp', 'siamese', 'human']

    unrequested_edges = [(0, 4, 0.5, 'vamp'), (6, 7, -0.1, 'vamp')]
    edges_to_add = [
        (0, 1, 0.9, 'human'),
        (0, 1, 0.7, 'human'),
        (0, 2, 0.4, 'human'),
        (0, 2, 0.5, 'vamp'),
        (0, 2, 0.7, 'siamese'),  # should be ignored
        (0, 2, -1.5, 'human'),
        (1, 2, 1.0, 'vamp'),  # should be ignored
        (1, 2, 1.4, 'human'),  # should be used
        (1, 3, 0.1, 'siamese'),
        (1, 3, 1.2, 'vamp'),
        (1, 3, 1.3, 'human'),
        (1, 2, 1.6, 'human'),
        (6, 7, 1.0, 'siamese'),
    ]

    num_to_return = [1, 2, 2, 3]
    tc = test_callbacks(edges_to_add, num_to_return, unrequested_edges)

    wm = weight_manager(aug_names, p, tc.request_cb, tc.result_cb)
    print('Output should be (0, 1, 1.7), (0, 2, -0.8), (1, 2, -2.2)')
    for e in wm.get_initial_edges(init_edges):
        print(e)
    print('Aug count', wm.augment_count)
    print(
        'After get_initial_edges:\n' '  len(wm.augment_count) should be 3 and is',
        len(wm.augment_count),
    )
    print(
        '  wm.augment_count((0, 1)) should be [1, 1, 2] and is', wm.augment_count[(0, 1)]
    )
    print(
        '  wm.augment_count((0, 2)) should be [0, 1, 0] and is', wm.augment_count[(0, 2)]
    )
    print(
        '  wm.augment_count((1, 2)) should be [1, 1, 1] and is', wm.augment_count[(1, 2)]
    )

    pairs = [(0, 1), (1, 3), (1, 3)]
    wm.request_new_weights(pairs)

    print('==========')
    print('First set of changes:')
    print('Should get (0, 1, 0.9) plus unrequested' '(0, 4, 0.5) and (6, 7, -0.1)')
    for res in wm.get_weight_changes():
        print(res)
    print(
        'After first set of changes:\n' '  len(wm.augment_count) should be 6 and is',
        len(wm.augment_count),
    )
    print(
        '  wm.augment_count((0, 1)) should be [1, 1, 3] and is', wm.augment_count[(0, 1)]
    )
    print(
        '  wm.augment_count((1, 3)) should be [0, 0, 0] and is', wm.augment_count[(1, 3)]
    )
    print(
        '  wm.augment_count((6, 7)) should be [1, 0, 0] and is', wm.augment_count[(6, 7)]
    )

    print('==========')
    print('Second set of changes, with new edge (1,3).')
    print('Should get (1, 3) only once, with weight 1.2, despite two requests.')
    for res in wm.get_weight_changes():
        print(res)

    pairs = [(1, 3)]  # obtain the result from siamese
    wm.request_new_weights(pairs)
    pairs = [(0, 2), (0, 1)]  # obtain the result from vamp and then human
    wm.request_new_weights(pairs)
    print('==========')
    print('Third set of changes. Should be (1, 3, 0.1) and (0, 2, 0.5).')
    for e in wm.get_weight_changes():
        print(e)

    print('==========')
    print('Fourth set of changes.')
    print('Before: futile_tester(0, 1) should be False. It is', wm.futile_tester(0, 1))
    print('Should be (0, 1, 0.7) - the fourth human result for (0,1)')
    for e in wm.get_weight_changes():
        print(e)
    print('After: futile_tester(0, 1) should be True. It is', wm.futile_tester(0, 1))

    pairs = [(1, 3), (0, 2)]
    wm.request_new_weights(pairs)
    print('==========')
    print('Fifth set of changes.')
    print('Should be (1, 3, 1.3)')
    for e in wm.get_weight_changes():
        print(e)

    print('==========')
    print('Sixth set of changes.')
    print('Should be (0, 2, 0.4)')
    for e in wm.get_weight_changes():
        print(e)

    pairs = [(1, 2), (1, 2)]  # both human, but only one returned.
    wm.request_new_weights(pairs)
    print('==========')
    print('Duplicate human reviews.')
    print('Should only get (1, 2, 1.4)')
    for e in wm.get_weight_changes():
        print(e)

    pairs = [(6, 7)]
    wm.request_new_weights(pairs)
    print('==========')
    print('Finally, testing that we can work with the unrequested edges')
    print('Should (6, 7, 1.0)')
    for e in wm.get_weight_changes():
        print(e)
