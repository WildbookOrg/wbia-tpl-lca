# -*- coding: utf-8 -*-
import logging
import sys

from wbia_lca import baseline
from wbia_lca import cluster_tools as ct

from wbia_lca import exp_scores as es
from wbia_lca import graph_algorithm as ga
from wbia_lca import simulator as sim
from wbia_lca import weighter as wgtr


logger = logging.getLogger('wbia_lca')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        logger.info('Usage: %s output_prefix')
        sys.exit()
    gen_prefix = sys.argv[1]

    """
    Specify the parameters for the simulator
    """
    sim_params = dict()
    sim_params['pos_error_frac'] = 0.15
    sim_params['num_clusters'] = 128  # 256
    sim_params['num_from_ranker'] = 4  # 10
    sim_params['p_ranker_correct'] = 0.85
    sim_params['p_human_correct'] = 0.97

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
    # num_per_cluster = sim_params['gamma_scale'] * sim_params['gamma_shape'] + 1

    """
    Build the exponential weight generator
    """
    np_ratio = sim.find_np_ratio(
        sim_params['gamma_shape'],
        sim_params['gamma_scale'],
        sim_params['num_from_ranker'],
        sim_params['p_ranker_correct'],
    )

    ga_params = {}
    ga_params['prob_human_correct'] = sim_params['p_human_correct']
    ga_params['min_delta_converge_multiplier'] = 0.9
    ga_params['min_delta_stability_ratio'] = 8
    ga_params['augmentation_names'] = ['vamp', 'human']
    ga_params['num_per_augmentation'] = 2
    ga_params['tries_before_edge_done'] = 4
    ga_params['ga_iterations_before_return'] = 100000  # convergence
    ga_params['ga_max_num_waiting'] = 50
    ga_params['log_level'] = logging.INFO
    ga_params['draw_iterations'] = False
    ga_params['drawing_prefix'] = 'drawing_lca'

    num_sim = 1  # 10
    for i in range(num_sim):
        """ Get the graph algorithm parameters """
        logger.info('===================================')
        logger.info('Starting simulation %s' % (i,))
        file_prefix = gen_prefix + ('_%02d' % i)
        log_file = file_prefix + '.log'

        """Configure the log file. This is repeated in the __init__ function
        for the graph_algorithm class, something that is only done here
        simulation information into the log file. It should not be done
        when running with "live" data.
        """
        from wbia_lca import formatter

        handler = logging.FileHandler(log_file, mode='w')
        handler.setLevel(ga_params['log_level'])
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # To do: output information to log file about simulation.
        logger.info('Negative / positive prob ratio %1.3f' % np_ratio)
        logger.info('Simulation parameters %a', sim_params)

        scorer = es.exp_scores.create_from_error_frac(
            sim_params['pos_error_frac'], np_ratio
        )
        wgtr_i = wgtr.weighter(scorer, human_prob=sim_params['p_human_correct'])

        """
        Build the simulator
        """
        # seed = 9314
        sim_i = sim.simulator(sim_params, wgtr_i)  # , seed=seed)
        init_edges, aug_names = sim_i.generate()
        init_clusters = []

        """
        Specify parameters for the graph algorithm
        """
        min_converge = -ga_params['min_delta_converge_multiplier'] * (
            wgtr_i.human_wgt(True) - wgtr_i.human_wgt(False)
        )
        ga_params['min_delta_score_converge'] = min_converge
        ga_params['min_delta_score_stability'] = (
            min_converge / ga_params['min_delta_stability_ratio']
        )

        gai = ga.graph_algorithm(
            init_edges,
            init_clusters,
            aug_names,
            ga_params,
            sim_i.augmentation_request,
            sim_i.augmentation_result,
        )

        gai.set_trace_compare_to_gt_cb(
            sim_i.trace_start_human, sim_i.trace_iter_compare_to_gt
        )

        max_iterations = int(1e5)
        should_pause = converged = False
        iter_num = 0
        while iter_num < max_iterations and not converged:
            should_pause, iter_num, converged = gai.run_main_loop(iter_num)

        logger.info('')
        logger.info('Compare to ground truth')
        logger.info('By GT cluster length:')
        ct.compare_by_lengths(gai.clustering, gai.node2cid, sim_i.gt_clustering)
        pct, pr, rec = ct.percent_and_PR(
            gai.clustering, gai.node2cid, sim_i.gt_clustering, sim_i.gt_node2cid
        )
        logger.info('Pct equal %.3f, Precision %.3f, Recall %.3f' % (pct, pr, rec))

        logger.info('')
        logger.info('Compare to reachable ground truth')
        logger.info('By reachable cluster length:')
        ct.compare_by_lengths(gai.clustering, gai.node2cid, sim_i.r_clustering)
        pct, pr, rec = ct.percent_and_PR(
            gai.clustering, gai.node2cid, sim_i.r_clustering, sim_i.r_node2cid
        )
        logger.info('Pct equal %.3f, Precision %.3f, Recall %.3f' % (pct, pr, rec))

        sim_i.csv_output(file_prefix + '_gt.csv', sim_i.gt_results)
        sim_i.csv_output(file_prefix + '_r.csv', sim_i.r_results)
        sim_i.generate_plots(file_prefix)

        b = baseline.baseline(sim_i)
        max_human_baseline = 10 * sim_params['num_clusters']

        b.all_iterations(0, max_human_baseline, 5)
        b.generate_plots(file_prefix + '_base')
