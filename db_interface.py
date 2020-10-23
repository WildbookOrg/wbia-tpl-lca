# -*- coding: utf-8 -*-
"""
The interface to the database asks for properties like nodes, edges,
clusters and cluster ids. These are deliberately abstracted from the
design of Wildbook. Their current Wildbook analogs are annotations,
edges, marked individuals and their uuids. Since this could change in
the future and we also hope that the algorithm could be used for other
applications, we keep the terms of interface abstract.

All edges are communicated as "quads" where each quad is of the form
  (n0, n1, w, aug_name)
Here, n0 and n1 are the nodes, w is the (signed!) weight and aug_name
is the augmentation method --- a verfication algorithm or a human
annotator -- that produced the edge.  Importantly, n0 < n1.

As currently written, nodes are not added to the database through this
interface. It is assumed that these are entered using a separate
interface that runs before the graph algorithm.

I suspect we can implement this as a class hierarchy where examples and
simulation are handled through one subclass and the true database
interface is handled through another.
"""
import logging


logger = logging.getLogger('wbia_lca')


class db_interface(object):  # NOQA
    def __init__(self, edges, clustering):
        super(db_interface, self).__init__()

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
