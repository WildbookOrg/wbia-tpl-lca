# -*- coding: utf-8 -*-
import logging

from wbia_lca import db_interface


logger = logging.getLogger('wbia_lca')


class db_interface_sim(db_interface.db_interface):  # NOQA
    def __init__(self, edges, clustering):
        super(db_interface_sim, self).__init__(edges, clustering)

    def add_edges(self, quads):
        """
        Add edges of the form (n0, n1, w, aug_name). This can be a
        single quad or a list of quads. For each, if the combinatin of
        n0, n1 and aug_name already exists and aug_name is not 'human'
        then the new edge replaces the existing edge. Otherwise, this
        edge quad is added as though the graph is a multigraph.
        """
        raise NotImplementedError()

    def get_weight(self, triple):
        """
        Return the weight if the combination of n0, n1 and aug_name.
        If the aug_name is 'human' the summed weight is
        returned. Returns None if triple is unknown.
        """
        raise NotImplementedError()

    def cluster_exists(self, cid):
        """
        Return True iff the cluster id exists in the clustering
        """
        raise NotImplementedError()

    def get_cid(self, node):
        """
        Get the cluster id associated with a node. Returns None if
        cluster does not exist
        """
        raise NotImplementedError()

    def get_nodes_in_cluster(self, cid):
        """
        Find all the nodes the cluster referenced by cid.  Returns
        None if cluster does not exist.
        """
        raise NotImplementedError()

    def edges_within_cluster(self, cid):
        """
        Find the multigraph edges that are within a cluster.
        Edges must be returned with n0<n1
        """
        raise NotImplementedError()

    def edges_leaving_cluster(self, cid):
        """
        Find the multigraph edges that connect between cluster cid and
        a different cluster.
        """
        raise NotImplementedError()

    def edges_between_clusters(self, cid0, cid1):
        """
        Find the multigraph edges that connect between cluster cid0
        and cluster cid1
        """
        raise NotImplementedError()

    def edges_node_to_cluster(self, n, cid):
        """
        Find all edges between a node and a cluster.
        """
        raise NotImplementedError()

    def edges_between_nodes(self, node_set):
        """
        Find all edges between any pair of nodes in the node set.
        """
        raise NotImplementedError()

    def commit_cluster_change(cluster_change):
        """
        Commit the changes according to the type of change.  See
        compare_clusterings.py
        """
        raise NotImplementedError()
