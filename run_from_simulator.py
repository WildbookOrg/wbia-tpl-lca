import sys

import cluster_tools as ct
import exp_scores as es
import graph_algorithm as ga
import json
import simulator as sim
import weighter as wgtr

import baseline


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: %s [gt_json_file] output_prefix")
        sys.exit()
    elif len(sys.argv) == 2:
        has_gt = False
        gen_prefix = sys.argv[1]
    else:
        has_gt = True
        json_file = sys.argv[1]
        gen_prefix = sys.argv[2]
        with open(json_file) as f:
            s = f.read()
            gt_clusters = json.loads(s)

    """
    Specify the parameters for the simulator
    """
    if has_gt:
        sim_params = dict()
        sim_params['pos_error_frac'] = 0.15
        sim_params['num_from_ranker'] = 6  # 10
        sim_params['p_ranker_correct'] = 0.85
        sim_params['p_human_correct'] = 0.98
        sim_params['num_clusters'] = len(gt_clusters)
        cluster_sizes = [len(c) for c in gt_clusters.values()]
        np_ratio = sim.find_np_ratio_gt(cluster_sizes, sim_params['num_from_ranker'],
                                     sim_params['p_ranker_correct'])
        print('np_ratio (from ground truth) = %1.3f' % np_ratio)
    else:
        sim_params = dict()
        sim_params['pos_error_frac'] = 0.15
        sim_params['num_clusters'] = 16  # 256
        sim_params['num_from_ranker'] = 4  # 10
        sim_params['p_ranker_correct'] = 0.85
        sim_params['p_human_correct'] = 0.98

        # The following are parameters of the gamma distribution.
        # Recall the following properties:
        #      mean is shape*scale,
        #      mode is (shape-1)*scale
        #      variance is shape*scale**2 = mean*scale
        # So when these are both 2 the mean is 4, the mode is 2
        # and the variance is 4 (st dev 2).  And, when shape = 1,
        # the mode is at 0 and we have an exponential distribution
        # with the beta parameter of that distribution = scale.
        #
        # The mean and the mode must be offset by 1 because every cluster
        # has at least one node.
        #
        sim_params['gamma_shape'] = 1  # 2   # 1
        sim_params['gamma_scale'] = 2  # 1.5 # 2
        num_per_cluster = sim_params['gamma_scale'] * sim_params['gamma_shape'] + 1

        """
        Build the exponential weight generator
        """
        np_ratio = sim.find_np_ratio(sim_params['gamma_shape'],
                                     sim_params['gamma_scale'],
                                     sim_params['num_from_ranker'],
                                     sim_params['p_ranker_correct'])
        print('np_ratio (from generated) = %1.3f' % np_ratio)

    num_sim = 1  # 10
    for i in range(num_sim):
        print("===================================")
        print("Starting simulation", i)
        file_prefix = gen_prefix + ("_%02d" % i)
        scorer = es.ExpScores.create_from_error_frac(sim_params['pos_error_frac'],
                                                      np_ratio)
        wgtr_i = wgtr.Weighter(scorer, human_prob=sim_params['p_human_correct'])

        """
        Build the simulator
        """
        # seed = 9314
        sim_i = sim.Simulator(sim_params, wgtr_i)  # , seed=seed)

        if has_gt:
            sim_i.create_clusters_from_ground_truth(gt_clusters)
        else:
            sim_i.create_random_clusters()
        sim_i.generate_from_clusters()

        """
        Specify parameters for the graph algorithm
        """
        gr_params = dict()
        print(wgtr_i.human_wgt(True), wgtr_i.human_wgt(False))
        gr_params['min_delta_score'] = -0.9 * (wgtr_i.human_wgt(True) - wgtr_i.human_wgt(False))  # NOQA

        gai = ga.GraphAlgorithm(sim_i.orig_edges, gr_params)
        gai.set_algorithmic_verifiers(sim_i.verify_request, sim_i.verify_result)
        gai.set_human_reviewers(sim_i.human_request, sim_i.human_result)
        gai.set_trace_callbacks(sim_i.trace_start_splitting, sim_i.trace_compare_to_gt,
                                sim_i.trace_start_stability, sim_i.trace_compare_to_gt)
        is_interactive = False
        gai.set_interactive(is_interactive=is_interactive)
        gai.run_main_loop()



        print("SUMMARY")
        print("\nHere are the final clusters")
        sim_i.print_clusters(gai.clustering)

        print("\nCompare to ground truth")
        print("By GT cluster length:")
        ct.compare_by_lengths(gai.clustering, gai.node2cid, sim_i.gt_clustering)
        pct, pr, rec = ct.percent_and_PR(gai.clustering, gai.node2cid,
                                         sim_i.gt_clustering, sim_i.gt_node2cid)
        print("Pct equal %.3f, Precision %.3f, Recall %.3f" % (pct, pr, rec))

        print("\nCompare to reachable ground truth")
        print("By reachable cluster length:")
        ct.compare_by_lengths(gai.clustering, gai.node2cid, sim_i.r_clustering)
        pct, pr, rec = ct.percent_and_PR(gai.clustering, gai.node2cid,
                                         sim_i.r_clustering, sim_i.r_node2cid)
        print("Pct equal %.3f, Precision %.3f, Recall %.3f" % (pct, pr, rec))

        sim_i.csv_output(file_prefix + "_gt.csv", sim_i.gt_results)
        sim_i.csv_output(file_prefix + "_r.csv", sim_i.r_results)
        sim_i.generate_plots(file_prefix)

        b = baseline.BaselineSimulator(sim_i)
        max_human_baseline = 10 * sim_params['num_clusters']

        b.all_iterations(0, max_human_baseline, 5)
        b.generate_plots(file_prefix + "_base")
