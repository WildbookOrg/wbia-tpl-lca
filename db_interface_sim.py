# -*- coding: utf-8 -*-
import networkx as nx
import logging

from wbia_lca import cluster_tools as ct
from wbia_lca import compare_clusterings
from wbia_lca import db_interface


logger = logging.getLogger('wbia_lca')


"""
Init edges
Init clusters
Correct clusters
Query edges
"""


class db_interface_sim(db_interface.db_interface):  # NOQA
    def __init__(self, edges, clustering):
        super(db_interface_sim, self).__init__(edges, clustering)
        self.edge_graph = nx.Graph()
        self.add_edges(edges)
        self.clustering = clustering
        self.node_to_cid = ct.build_node_to_cluster_mapping(self.clustering)

    def add_edges(self, quads):
        """
        See base class for requirements
        """
        self.edge_graph.add_edges_from([(n0, n1) for n0, n1, _, _ in quads])
        for n0, n1, w, aug_name in quads:
            attrib = self.edge_graph[n0][n1]
            if aug_name == 'human':
                if 'human' in attrib:
                    attrib['human'].append(w)
                else:
                    attrib['human'] = [w]
            else:
                attrib[aug_name] = w

    def get_weight(self, triple):
        n0, n1, aug_name = triple
        try:
            attrib = self.edge_graph[n0][n1]
            if aug_name == 'human':
                return sum(attrib['human'])
            else:
                return attrib[aug_name]
        except KeyError:
            return None

    def cluster_exists(self, cid):
        return cid in self.clustering

    def get_cid(self, node):
        """
        Get the cluster id associated with a node. Returns None if the
        node is not part of a current cluster.
        """
        try:
            return self.node_to_cid[node]
        except KeyError:
            return None

    def get_nodes_in_cluster(self, cid):
        """
        Find all the nodes the cluster referenced by cid.
        """
        try:
            return self.clustering[cid]
        except Exception:
            return None

    def edges_from_attributes(self, n0, n1):
        quads = []
        attrib = self.edge_graph[n0][n1]
        for a, wgt in attrib.items():
            if a == 'human':
                quads.extend([(n0, n1, w, 'human') for w in wgt])
            else:
                quads.append((n0, n1, wgt, a))
        return quads

    def edges_within_cluster(self, cid):
        """
        Find the multigraph edges that are within a cluster.
        Edges must be returned with n0<n1.
        """
        quads = []
        cluster = sorted(self.clustering[cid])
        for i, ni in enumerate(cluster):
            for j in range(i + 1, len(cluster)):
                nj = cluster[j]
                if nj in self.edge_graph[ni]:
                    quads.extend(self.edges_from_attributes(ni, nj))
        return quads

    def edges_leaving_cluster(self, cid):
        quads = []
        cluster = self.clustering[cid]
        for ni in cluster:
            for nj in self.edge_graph[ni]:
                if nj not in cluster:
                    quads.extend(self.edges_from_attributes(ni, nj))
        return quads

    def edges_between_clusters(self, cid0, cid1):
        """
        Find the multigraph edges that connect between cluster cid0
        and cluster cid1
        """
        assert cid0 != cid1
        quads = []
        cluster1 = self.clustering[cid1]
        for ni in self.clustering[cid0]:
            ei = self.edge_graph[ni]
            for nj in cluster1:
                if nj in ei:
                    quads.extend(self.edges_from_attributes(ni, nj))
        return quads

    def edges_between_nodes(self, node_set):
        """
        Find all edges between any pair of nodes in the node set.
        """
        quads = []
        node_list = sorted(node_set)
        for i, ni in enumerate(node_list):
            for j in range(i + 1, len(node_list)):
                nj = node_list[j]
                if nj in self.edge_graph[ni]:
                    quads.extend(self.edges_from_attributes(ni, nj))
        return quads

    def edges_from_node(self, node):
        quads = []
        for nj in self.edge_graph[node]:
            quads.extend(self.edges_from_attributes(node, nj))
        return quads

    def remove_nodes(self, nodes):
        for n in nodes:
            self.remove_node(n)

    def remove_node(self, n):
        if n not in self.edge_graph.nodes:
            return

        self.edge_graph.remove_node(n)
        cid = self.node_to_cid[n]
        del self.node_to_cid[n]
        self.clustering[cid].remove(n)
        if len(self.clustering[cid]) == 0:
            del self.clustering[cid]

    def commit_cluster_change(self, cc):
        """
        Commit the cluster changes to the database. This involves
        updating the node to cluster id dictionary and the clustering
        dictionary. One way to do this would be to have special
        operations for each type of change. Instead, this function
        works generically except for the single case of no changes at
        all.
        """
        if cc.change_type == 'Unchanged':
            return

        # 1. Add new clusters
        self.clustering.update(cc.new_clustering)

        # 2. Remove old clusters
        removed_cids = set(cc.old_clustering.keys()) - set(cc.new_clustering.keys())
        for old_c in removed_cids:
            del self.clustering[old_c]

        # 3. Update the node to clusterid mapping
        new_node_to_cid = ct.build_node_to_cluster_mapping(cc.new_clustering)
        self.node_to_cid.update(new_node_to_cid)

        # 4. Removed nodes should have already been removed from the
        #    db through the call to self.remove_nodes.
        for n in cc.removed_nodes:
            assert n not in self.node_to_cid


def test_example():
    quads = [
        ('a', 'b', 4, 'vamp'),
        ('a', 'b', 10, 'human'),
        ('a', 'd', 11, 'embed'),
        ('d', 'e', -5, 'vamp'),
        ('a', 'e', 2, 'embed'),
        ('b', 'c', -8, 'human'),
        ('c', 'e', 1, 'vamp'),
        ('c', 'e', -4, 'embed'),
        ('a', 'b', 12, 'human'),
        ('c', 'f', 2, 'vamp'),
        ('d', 'e', -6, 'embed'),
        ('d', 'g', -19, 'human'),
        ('e', 'g', 12, 'human'),
        ('e', 'h', 4, 'embed'),
        ('c', 'f', 9, 'human'),
        ('e', 'h', 5, 'vamp'),
        ('c', 'f', 11, 'human'),
        ('c', 'f', 6, 'vamp'),
        ('f', 'c', 7, 'embed'),
        ('a', 'd', 12, 'embed'),
        ('g', 'h', 12, 'vamp'),
        ('f', 'h', -6, 'embed'),
    ]
    clustering = {1000: ['a', 'b', 'd'], 1001: ['c', 'f'], 1002: ['e', 'g', 'h']}
    return quads, clustering


def print_edge(g, n0, n1):
    logger.info('(%s, %s):' % (n0, n1))
    for k, v in g[n0][n1].items():
        logger.info(
            '%s %s'
            % (
                k,
                v,
            )
        )


def test_db_interface_sim():
    quads, clustering = test_example()

    db = db_interface_sim(quads, clustering)
    logger.info(db.clustering)

    for n0 in db.edge_graph:
        for n1 in db.edge_graph[n0]:
            if n0 < n1:
                logger.info('')
                print_edge(db.edge_graph, n0, n1)

    logger.info('\nTest get weight')
    triples = [
        ('a', 'b', 'human'),  # 22
        ('d', 'e', 'vamp'),  # -5
        ('d', 'e', 'embed'),  # -6
        ('d', 'e', 'human'),  # None
        ('a', 'c', 'human'),  # None
        ('a', 'q', 'embed'),  # None
        ('m', 'z', 'vamp'),
    ]  # None
    answers = [22, -5, -6, None, None, None, None]
    num_err = 0
    for t, a in zip(triples, answers):
        res = db.get_weight(t)
        logger.info(
            'Expected %s and got %s'
            % (
                a,
                res,
            )
        )
        if res != a:
            logger.info('Error')
            num_err += 1
    if num_err == 0:
        logger.info('no errors')

    logger.info('\nTest get_cid')
    nodes = ['a', 'e', 'z']
    answers = [1000, 1002, None]
    num_err = 0
    for n, a in zip(nodes, answers):
        res = db.get_cid(n)
        logger.info(
            'Expected %s and got %s'
            % (
                a,
                res,
            )
        )
        if res != a:
            logger.info('Error')
            num_err += 1
    if num_err == 0:
        logger.info('no errors')

    logger.info('\nTest get_nodes_in_cluster')
    cids = [1000, 1003]
    answers = [['a', 'b', 'd'], None]
    num_err = 0
    for cid, a in zip(cids, answers):
        res = db.get_nodes_in_cluster(cid)
        logger.info(
            'Expected %s and got %s'
            % (
                a,
                res,
            )
        )
        if res != a:
            logger.info('Error')
            num_err += 1
    if num_err == 0:
        logger.info('no errors')

    logger.info('\nTest edges_within_cluster')
    logger.info(db.edges_within_cluster(1000))

    logger.info('\nTest edges_leaving_cluster')
    logger.info(db.edges_leaving_cluster(1002))

    logger.info('\nTest edges_between_clusters')
    logger.info(db.edges_between_clusters(1002, 1000))

    logger.info('\nTest edges_between_nodes')
    logger.info(db.edges_between_nodes(set(['a', 'b', 'e', 'g'])))

    logger.info('\nTest edges_from_node a (should have edges to b, d, e')
    logger.info(db.edges_from_node('a'))
    logger.info('\nTest edges_from_node e (should have edges to a, c, d, g, h)')
    logger.info(db.edges_from_node('e'))

    logger.info('\nTesting remove_nodes and commit_cluster_change')
    logger.info('Output the current state:')
    logger.info(db.clustering)
    logger.info(db.node_to_cid)

    old_clustering = {1001: set(['c', 'f']), 1002: set(['e', 'g', 'h'])}
    new_clustering = {
        1002: set(['c', 'e', 'g']),
        1003: set(['j']),
        1004: set(
            [
                'h',
                'k',
            ]
        ),
    }
    cc = compare_clusterings.clustering_change(old_clustering, new_clustering)
    logger.info('Here is the cluster change object')
    cc.print_it()

    logger.info('Removing the node(s)')
    db.remove_nodes(cc.removed_nodes)

    logger.info('Here is the new state')
    logger.info(db.clustering)
    logger.info(db.node_to_cid)

    db.commit_cluster_change(cc)
    logger.info('After committing the change, here is the final state')
    logger.info(db.clustering)
    logger.info(db.node_to_cid)


if __name__ == '__main__':
    test_db_interface_sim()
