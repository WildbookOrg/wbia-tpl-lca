# -*- coding: utf-8 -*-
import networkx as nx

'''Old clustering:
list of tuples, where each tuple itself contains two tuples:
. cluster id
. tuple of nodes

New nodes:
. list of new nodes, not assigned to an old clustering, yet

New clusters:
. list of tuples, where each tuple contains all the node ids of one
cluster.

Goal is to return:

From the new cluster viewpoint:

Unchanged:  A new cluster is an old cluster, unchanged.

Extended: A new cluster contains nodes from a single old cluster plus
new annotations

New: A new cluster is formed only from new nodes

Merge: A new cluster is formed from a combination of all nodes
in two or more previous clusters plus zero or more new nodes

Split: A new cluster is formed from a proper subset of nodes from
exactly one old cluster, plus, perhaps, new nodes:

Merge/split: A new cluster is formed from a combation of nodes from at
least two old clusters, where at least one of them contains a proper
subset of nodes from a previous cluster. New nodes may be added.


'''


def bipartite_cc(
    from_visited, from_nbrs, to_visited, to_nbrs, from_nodes
):  # on this, marked visited
    to_nodes = set()
    for v in from_nodes:
        for t in from_nbrs[v]:
            if not to_visited[t]:
                to_visited[t] = True
                to_nodes.add(t)
    if len(to_nodes) == 0:
        return from_nodes, set()
    else:
        to_rec, from_rec = bipartite_cc(
            to_visited, to_nbrs, from_visited, from_nbrs, to_nodes
        )
        return from_rec + from_nodes, to_rec


def compare_clusterings(old_clustering, old_n2c, new_clustering, new_n2c):
    # build graph....
    # n
    pairs = set()
    for cid, nodes in old_clustering.values():
        new_nbrs = {new_n2c[n] for n in nodes}
        pairs += {(cid, new_cid) for new_cid in new_nbrs}

    g = nx.Graph()
    g.add_edges(pairs)
    # for cc in nx.connected_components(g):
    #     old_cids = {c for c in cc if True}

    old_to_new = {cid: set([]) for cid in old_clustering}
    new_to_old = {cid: set([]) for cid in new_clustering}
    for cid, nodes in old_clustering.values:
        for n in nodes:
            if n in new_n2c:  # might have been dropped
                old_to_new[cid].add(new_n2c[n])

    for cid, nodes in new_clustering.values():
        for n in nodes:
            if n in old_n2c:  # might be new
                new_to_old[cid].add(old_n2c[n])

    old_visited = {cid: False for cid in old_clustering}
    # new_visited = {cid: False for cid in new_clustering}

    for cid, b in old_visited.values():
        if b:
            continue
