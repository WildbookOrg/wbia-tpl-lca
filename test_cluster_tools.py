import networkx as nx
import cluster_tools as ct


def ex_graph_fig1():
    G = nx.Graph()
    G.add_weighted_edges_from([('a', 'b', 8), ('a', 'd', 4), ('a',  'e', -2),
                               ('b', 'c', -1), ('b', 'e', 4),
                               ('b', 'f', -4), ('b', 'i', -4),
                               ('c', 'f', 2), ('c', 'g', -3),
                               ('d', 'e', -1), ('d', 'h', 3),
                               ('e', 'i', -5),
                               ('f', 'g', 8), ('f', 'i', 2), ('f', 'j', 2),
                               ('f', 'k', -3),
                               ('g', 'j', -3), ('g', 'k', 7),
                               ('h', 'i', 6),
                               ('i', 'j', -4),
                               ('j', 'k', 5)])
    return G


def ex_graph_fig4():
    G = nx.Graph()
    G.add_weighted_edges_from([('a', 'b', 1), ('a', 'd', 6), ('a', 'e', -8),
                               ('a', 'f', -8),
                               ('b', 'c', 1), ('b', 'd', -9), ('b', 'e', 4),
                               ('b', 'f', -7),
                               ('c', 'd', -8), ('c', 'e', -7), ('c', 'f', 5),
                               ('d', 'e', 1),
                               ('e', 'f', 1)])
    return G


'''
def ex_graph_fig5():
    G = nx.Graph()
    G.add_weighted_edges_from([('a', 'b', 9), ('a', 'e', -2),
                               ('b', 'c', -6), ('b', 'e', 5), ('b', 'f', -2),
                               ('c', 'd', 8), ('c', 'e', 4),
                               ('d', 'e', 2), ('d', 'f', -1),
                               ('e', 'f', 6)])
    return G
'''


def test_build_clustering_and_mapping():
    print("==================")
    print("Testing build_clustering")
    empty_n2c = {}
    empty_clustering = ct.build_clustering(empty_n2c)
    print("Empty node 2 cluster mapping should produce empty "
          "clustering {}.  Result is", empty_clustering)

    # G = ex_graph_fig1()
    n2c_optimal = {'a': 0, 'b': 0, 'd': 0, 'e': 0,
                   'c': 1,
                   'h': 2, 'i': 2,
                   'f': 3, 'g': 3, 'j': 3, 'k': 3}

    clustering = ct.build_clustering(n2c_optimal)
    print("Cluster 0 should be ['a', 'b', 'd', 'e']. It is",
          sorted(clustering[0]))
    print("Cluster 1 should be ['c']. It is",
          sorted(clustering[1]))
    print("Cluster 2 should be ['h', 'i']. It is",
          sorted(clustering[2]))
    print("Cluster 3 should be ['f', 'g', 'j', 'k']. It is",
          sorted(clustering[3]))

    print("==================")
    print("Testing build_node_to_cluster_mapping")
    empty_clustering = {}
    empty_n2c = ct.build_node_to_cluster_mapping(empty_clustering)
    print("Empty clustering should produce empty node-to-cluster mapping {} "
          "Result is", empty_n2c)

    n2c_rebuilt = ct.build_node_to_cluster_mapping(clustering)
    print("After rebuilding the node2cid mapping should be the same.  Is it?",
          n2c_optimal == n2c_rebuilt)


def test_build_clustering_from_lists():
    print("================================")
    print("test_build_clustering_from_lists")
    clist = [['h', 'i', 'j'], ['k', 'm'], ['p']]
    n = len(clist)
    clustering = ct.build_clustering_from_lists(range(n), clist)
    print("Returned clustering:")
    print(clustering)
    correct = len(clustering) == 3
    print("Correct number of clusters", correct)
    correct = set(clist[0]) == clustering[0] and \
        set(clist[1]) == clustering[1] and \
        set(clist[2]) == clustering[2]
    print("Clusters are correct:", correct)

    #  Catching error from repeated entry
    clist = [['h', 'i', 'j'], ['k', 'm'], ['p', 'p']]
    n = len(clist)
    try:
        clustering = ct.build_clustering_from_lists(range(n), clist)
    except AssertionError:
        print("Caught error from having repeated entry in one cluster")

    #  Catching error from intersecting lists
    clist = [['h', 'i', 'k'], ['k', 'm'], ['p', 'q']]
    n = len(clist)
    try:
        clustering = ct.build_clustering_from_lists(range(n), clist)
    except AssertionError:
        print("Caught error from having intersecting lists")    

  
def test_cluster_scoring_and_weights():
    G = ex_graph_fig1()

    print("=====================")
    print("Testing cid_list_score")
    n2c_random = {'a': 0, 'b': 0, 'f': 0,
                  'c': 1, 'g': 1,
                  'd': 2, 'e': 2, 'i': 2,
                  'h': 3, 'j': 3, 'k': 3}
    clustering_random = ct.build_clustering(n2c_random)
    score = ct.cid_list_score(G, clustering_random, n2c_random, [0, 2, 3])
    print('Score between clusters [0, 2, 3] should be -5 and is', score)

    print("=====================")
    print("Testing clustering_score")
    ''' First clustering:  all together '''
    n2c_single_cluster = {n: 0 for n in G.nodes}
    print('Score with all together should be 21.  Score =',
          ct.clustering_score(G, n2c_single_cluster))

    ''' Second clustering:  all separate '''
    n2c_all_separate = {n: i for i, n in enumerate(G.nodes)}
    print('Score with all together should be -21.  Score =',
          ct.clustering_score(G, n2c_all_separate))

    ''' Third clustering: optimal, by hand '''
    n2c_optimal = {'a': 0, 'b': 0, 'd': 0, 'e': 0,
                   'c': 1,
                   'h': 2, 'i': 2,
                   'f': 3, 'g': 3, 'j': 3, 'k': 3}
    print('Optimal score should be 49. Score =',
          ct.clustering_score(G, n2c_optimal))

    negatives, positives = ct.get_weight_lists(G, sort_positive=True)
    print('Length of negatives should be 10.  It is', len(negatives))
    print('Length of positives should be 11.  It is', len(positives))
    print('0th positive should be 8.  It is', positives[0])
    print('Last positive should be 2.  It is', positives[-1])


def test_has_edges_between():
    G = ex_graph_fig1()
    c0 = {'a', 'd'}
    c1 = {'c', 'f'}
    c2 = {'b', 'i', 'j'}
    print("========================")
    print("Testing has_edges_between_them")
    res01 = ct.has_edges_between_them(G, c0, c1)
    print("c0 to c1 should be False. is", res01)
    res02 = ct.has_edges_between_them(G, c0, c2)
    print("c0 to c2 should be True. is", res02)
    res12 = ct.has_edges_between_them(G, c1, c2)
    print("c1 to c2 should be True. is", res12)
    res10 = ct.has_edges_between_them(G, c1, c0)
    print("c1 to c0 should be False. is", res10)


def test_merge():
    print("===========================")
    print("test_merge")
    G = ex_graph_fig1()
    n2c_optimal = {'a': 0, 'b': 0, 'd': 0, 'e': 0,
                   'c': 1,
                   'h': 2, 'i': 2,
                   'f': 3, 'g': 3, 'j': 3, 'k': 3}
    clustering = ct.build_clustering(n2c_optimal)

    print('-------------')
    print('score_delta_after_merge')
    delta = ct.score_delta_after_merge(2, 3, G, clustering)
    print('possible merge of 2, 3; delta should be -4, and is', delta)

    print('-------------')
    print('merge_clusters')
    score_before = ct.clustering_score(G, n2c_optimal)
    delta = ct.merge_clusters(0, 2, G, clustering, n2c_optimal)
    score_after = ct.clustering_score(G, n2c_optimal)
    print('delta =', delta, "should be", score_after - score_before)
    print('---')
    for c in clustering:
        print('%a: %a' % (c, clustering[c]))
    print('---')
    for n in G.nodes:
        print('%a: %a' % (n, n2c_optimal[n]))

    print('--------')
    print('Retesting merge with order of clusters reversed')
    n2c_optimal = {'a': 0, 'b': 0, 'd': 0, 'e': 0,
                   'c': 1,
                   'h': 2, 'i': 2,
                   'f': 3, 'g': 3, 'j': 3, 'k': 3}
    clustering = ct.build_clustering(n2c_optimal)

    print('-------------')
    print('score_delta_after_merge')
    delta = ct.score_delta_after_merge(3, 2, G, clustering)
    print('possible merge of 3, 2; delta should be -4, and is', delta)

    print('-------------')
    print('merge_clusters')
    score_before = ct.clustering_score(G, n2c_optimal)
    delta = ct.merge_clusters(2, 0, G, clustering, n2c_optimal)
    score_after = ct.clustering_score(G, n2c_optimal)
    print('delta =', delta, "should be", score_after - score_before)
    print('---')
    for c in clustering:
        print('%a: %a' % (c, clustering[c]))
    print('---')
    for n in G.nodes:
        print('%a: %a' % (n, n2c_optimal[n]))


def test_shift_between_clusters():
    print("===========================")
    print("test_shift_between_clusters")
    n2c_optimal = {'a': 0, 'b': 0, 'd': 0, 'e': 0,
                   'c': 1,
                   'h': 2, 'i': 2,
                   'f': 3, 'g': 3, 'j': 3, 'k': 3}
    clustering = ct.build_clustering(n2c_optimal)

    n0_cid, n1_cid = 3, 2
    n0_nodes_to_move = {'f', 'j'}
    print('Shifting from cluster %d to %d:' % (n0_cid, n1_cid))
    print('Nodes to move:', sorted(n0_nodes_to_move))
    print('Cluster %d: %a' %
          (n0_cid, sorted(clustering[n0_cid])))
    print('Cluster %d: %a' %
          (n1_cid, sorted(clustering[n1_cid])))

    ct.shift_between_clusters(n0_cid, n0_nodes_to_move, n1_cid,
                              clustering, n2c_optimal)
    print('After shift, cluster %d: %a' %
          (n0_cid, sorted(clustering[n0_cid])))
    print('After shift, cluster %d: %a' %
          (n1_cid, sorted(clustering[n1_cid])))
    print("n2c['f'] =", n2c_optimal['f'])
    print("n2c['j'] =", n2c_optimal['j'])
    print("n2c['h'] =", n2c_optimal['h'])
    print("n2c['i'] =", n2c_optimal['i'])
    print("n2c['g'] =", n2c_optimal['g'])
    print("n2c['k'] =", n2c_optimal['k'])


def test_replace_clusters():
    print("===========================")
    print("test replace_clusters")
    n2c = {'a': 0, 'b': 0, 'd': 0, 'e': 0,
           'c': 1,
           'h': 2, 'i': 2,
           'f': 3, 'g': 3,
           'j': 4, 'k': 4}
    clustering = ct.build_clustering(n2c)
    old_cids = [2, 4]
    added_clusters = {5: set(['j']), 7: set(['h', 'i', 'k'])}
    ct.replace_clusters(old_cids, added_clusters, clustering, n2c)
    print('Cluster ids, should be 0, 1, 3, 5, 7.  Are:',
          clustering.keys())
    print("clustering[5] should be {'j'}!! and is", clustering[5])
    print("clustering[7] should be {'h', 'i', 'k'} and is", clustering[7])
    print("n2c['h'] should be 7 and is", n2c['h'])
    print("n2c['j'] should be 5 and is", n2c['j'])


def test_form_connected_cluster_pairs():
    print("=================================")
    print("test form_connected_cluster_pairs")
    G = ex_graph_fig1()
    n2c = {'a': 0, 'b': 0, 'd': 0, 'e': 0,
           'c': 1,
           'h': 2, 'i': 2,
           'f': 3, 'g': 3,
           'j': 4, 'k': 4}
    clustering = ct.build_clustering(n2c)

    cid_pairs = ct.form_connected_cluster_pairs(G, clustering, n2c)
    print("form_connected_cluster_pairs(G, clustering, n2c)")
    print("result: ", cid_pairs)
    print("expecting: ", [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3),
                          (2, 4), (3, 4)])

    new_cids = [1, 4]
    cid_pairs = ct.form_connected_cluster_pairs(G, clustering, n2c, new_cids)
    print("form_connected_cluster_pairs(G, clustering, n2c, new_cids)")
    print("result: ", cid_pairs)
    print("expecting: ", [(0, 1), (1, 3), (2, 4), (3, 4)])


def test_same_clustering():
    '''
    '''
    clustering0 = {0: {'a', 'b'},
                   3: {'c'},
                   4: {'d', 'e'},
                   6: {'f', 'g', 'h'},
                   8: {'i', 'j', 'k', 'l', 'm'}}
    clustering1 = {6: {'d', 'e'},
                   8: {'c'},
                   16: {'f', 'g', 'h'},
                   19: {'i', 'k', 'l', 'm', 'j'},
                   20: {'b', 'a'}}
    clustering2 = {6: {'d', 'c', 'e'},
                   16: {'f', 'g', 'h'},
                   22: {'i', 'j', 'k', 'l', 'm'},
                   25: {'b', 'a'}}

    print("====================")
    print("test_same_clustering")
    print("first test should generate no output and then return True")
    print(ct.same_clustering(clustering0, clustering1, True))
    print("second test should generate no output and then return False")
    print(ct.same_clustering(clustering0, clustering2, False))
    print("third test should generate mismatch output and then return False")
    print("Expected:")
    print("['c'] not in 2nd")
    print("['d', 'e'] not in 2nd")
    print("['c', 'd', 'e'] not in 1st")
    result = ct.same_clustering(clustering0, clustering2, True)
    print("It returned", result)

    
def test_comparisons():
    '''
    '''
    gt = {0: {'a', 'b'},
          3: {'c'},
          4: {'d', 'e'},
          6: {'f', 'g', 'h'},
          8: {'i', 'j', 'k', 'l', 'm'},
          10: {'o'},
          13: {'p', 'q'},
          15: {'r', 's', 't'},
          16: {'u', 'v', 'w'},
          19: {'y', 'z', 'aa'}}
    gt_n2c = ct.build_node_to_cluster_mapping(gt)

    est = {25: {'y', 'z', 'aa'},
           29: {'u', 'v'},
           31: {'w', 'r', 's', 't'},
           37: {'p'},
           41: {'q', 'o', 'm'},
           43: {'i', 'j', 'k', 'l'},
           47: {'a', 'b'},
           53: {'c'},
           59: {'d', 'e'},
           61: {'f', 'g', 'h'}}
    est_n2c = ct.build_node_to_cluster_mapping(est)

    print("================")
    print("test_comparisons")
    print("ct.compare_by_lengths")

    ct.compare_by_lengths(est, est_n2c, gt)

    print("Output for this example should be:\n"
          "1, 2, 1, 0.50, 0.667\n"
          "2, 3, 2, 0.67, 0.833\n"
          "3, 4, 2, 0.50, 0.854\n"
          "5, 1, 0, 0.00, 0.800")

    print("------")
    print("ct.pairwise_eval")
    # result = ct.compare_to_ground_truth(est, est_n2c, gt, gt_n2c)
    result = ct.percent_and_PR(est, est_n2c, gt, gt_n2c)
    print("Result is [%1.3f, %1.3f, %1.3f]"
          % tuple(result))
    num_clusters = len(est)
    num_correct = 5
    tp, fp, fn = 18, 6, 7
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Should be [%1.3f, %1.3f, %1.3f]"
          % (num_correct / num_clusters, precision, recall))


def test_count_equal():
    '''
    '''
    gt = {0: {'a', 'b'},
          3: {'c'},
          4: {'d', 'e'},
          6: {'f', 'g', 'h'},
          8: {'i', 'j', 'k', 'l', 'm'},
          10: {'o'},
          13: {'p', 'q'},
          15: {'r', 's', 't'},
          16: {'u', 'v', 'w'},
          19: {'y', 'z', 'aa'}}

    est = {25: {'y', 'z', 'aa'},
           29: {'u', 'v'},
           31: {'w', 'r', 's', 't'},
           37: {'p'},
           41: {'q', 'o', 'm'},
           43: {'i', 'j', 'k', 'l'},
           47: {'a', 'b'},
           53: {'c'},
           59: {'d', 'e'},
           61: {'f', 'g', 'h'}}

    est_n2c = ct.build_node_to_cluster_mapping(est)
    n = ct.count_equal_clustering(gt, est, est_n2c)
    print("test_count_equal: should be 5 and is", n)


if __name__ == '__main__':
    test_build_clustering_and_mapping()
    test_build_clustering_from_lists()
    test_cluster_scoring_and_weights()
    test_has_edges_between()
    test_merge()
    test_shift_between_clusters()
    test_replace_clusters()
    test_form_connected_cluster_pairs()
    test_comparisons()
    test_same_clustering()
    test_count_equal()
