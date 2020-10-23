# -*- coding: utf-8 -*-
from wbia.control import controller_inject
from wbia.constants import CONTAINERIZED, PRODUCTION  # NOQA
from wbia.web.graph_server import GraphActor
import logging
import utool as ut

import configparser
import json
import sys

from wbia_lca import ga_driver
from wbia_lca import overall_driver

from wbia_lca import _plugin_db_interface
from wbia_lca import _plugin_edge_generator


logger = logging.getLogger('wbia_lca')


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)

register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']


@register_ibs_method
@register_api('/api/plugin/lca/sim/', methods=['GET'])
def wbia_plugin_lca_sim(ibs, ga_config, verifier_gt, request, db_result=None):
    r"""
    Create an LCA graph algorithm object

    Args:
        ibs (IBEISController): wbia controller object
        ga_config (str): graph algorithm config INI file
        verifier_gt (str): json file containing verification algorithm ground truth
        request (str): json file continain graph algorithm request info
        db_result (str, optional): file to write resulting json database

    Returns:
        object: changes_to_review

    CommandLine:
        python -m wbia_lca._plugin wbia_plugin_lca_sim
        python -m wbia_lca._plugin wbia_plugin_lca_sim:0
        python -m wbia_lca._plugin wbia_plugin_lca_sim:1
        python -m wbia_lca._plugin wbia_plugin_lca_sim:2

    RESTful:
        Method: GET
        URL:    /api/plugin/lca/overall/driver/

    Doctest:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> import utool as ut
        >>> import random
        >>> random.seed(1)
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> ga_config = 'examples/default/config.ini'
        >>> verifier_gt = 'examples/default/verifier_probs.json'
        >>> request = 'examples/default/request_example.json'
        >>> db_result = 'examples/default/result.json'
        >>> changes_to_review = ibs.wbia_plugin_lca_sim(ga_config, verifier_gt, request, db_result)
        >>> results = []
        >>> for cluster in changes_to_review:
        >>>     lines = []
        >>>     for change in cluster:
        >>>         line = []
        >>>         line.append('query nodes %s' % (sorted(change.query_nodes),))
        >>>         line.append('change_type %s' % (change.change_type,))
        >>>         line.append('old_clustering %s' % (sorted(change.old_clustering), ))
        >>>         line.append('len(new_clustering) %s' % (len(sorted(change.new_clustering)), ))
        >>>         line.append('removed_nodes %s' % (sorted(change.removed_nodes),))
        >>>         lines.append('\n'.join(line))
        >>>     results.append('\n-\n'.join(sorted(lines)))
        >>> result = '\n----\n'.join(sorted(results))
        >>> print('----\n%s\n----' % (result, ))
        ----
        query nodes ['c', 'e']
        change_type Merge
        old_clustering ['100', '101']
        len(new_clustering) 1
        removed_nodes []
        ----
        query nodes ['f']
        change_type Extension
        old_clustering ['102']
        len(new_clustering) 1
        removed_nodes []
        -
        query nodes ['g']
        change_type New
        old_clustering []
        len(new_clustering) 1
        removed_nodes []
        ----
        query nodes ['m']
        change_type Extension
        old_clustering ['103']
        len(new_clustering) 1
        removed_nodes []
        ----

    Doctest:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> import utool as ut
        >>> import random
        >>> random.seed(1)
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> ga_config = 'examples/merge/config.ini'
        >>> verifier_gt = 'examples/merge/verifier_probs.json'
        >>> request = 'examples/merge/request_example.json'
        >>> db_result = 'examples/merge/result.json'
        >>> changes_to_review = ibs.wbia_plugin_lca_sim(ga_config, verifier_gt, request, db_result)
        >>> results = []
        >>> for cluster in changes_to_review:
        >>>     lines = []
        >>>     for change in cluster:
        >>>         line = []
        >>>         line.append('query nodes %s' % (sorted(change.query_nodes),))
        >>>         line.append('change_type %s' % (change.change_type,))
        >>>         line.append('old_clustering %s' % (sorted(change.old_clustering), ))
        >>>         line.append('len(new_clustering) %s' % (len(sorted(change.new_clustering)), ))
        >>>         line.append('removed_nodes %s' % (sorted(change.removed_nodes),))
        >>>         lines.append('\n'.join(line))
        >>>     results.append('\n-\n'.join(sorted(lines)))
        >>> result = '\n----\n'.join(sorted(results))
        >>> print('----\n%s\n----' % (result, ))
        ----
        query nodes []
        change_type Merge/Split
        old_clustering ['100', '101']
        len(new_clustering) 2
        removed_nodes []
        -
        query nodes []
        change_type Unchanged
        old_clustering ['102']
        len(new_clustering) 1
        removed_nodes []
        ----

    Doctest:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> import utool as ut
        >>> import random
        >>> random.seed(1)
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> ga_config = 'examples/zero/config.ini'
        >>> verifier_gt = 'examples/zero/verifier_probs.json'
        >>> request = 'examples/zero/request_example.json'
        >>> db_result = 'examples/zero/result.json'
        >>> changes_to_review = ibs.wbia_plugin_lca_sim(ga_config, verifier_gt, request, db_result)
        >>> results = []
        >>> for cluster in changes_to_review:
        >>>     lines = []
        >>>     for change in cluster:
        >>>         line = []
        >>>         line.append('query nodes %s' % (sorted(change.query_nodes),))
        >>>         line.append('change_type %s' % (change.change_type,))
        >>>         line.append('old_clustering %s' % (sorted(change.old_clustering), ))
        >>>         line.append('len(new_clustering) %s' % (len(sorted(change.new_clustering)), ))
        >>>         line.append('removed_nodes %s' % (sorted(change.removed_nodes),))
        >>>         lines.append('\n'.join(line))
        >>>     results.append('\n-\n'.join(sorted(lines)))
        >>> result = '\n----\n'.join(sorted(results))
        >>> print('----\n%s\n----' % (result, ))
        ----
        query nodes ['a', 'b', 'c', 'd', 'e']
        change_type New
        old_clustering []
        len(new_clustering) 1
        removed_nodes []
        -
        query nodes ['f', 'h', 'i', 'j']
        change_type New
        old_clustering []
        len(new_clustering) 1
        removed_nodes []
        -
        query nodes ['g']
        change_type New
        old_clustering []
        len(new_clustering) 1
        removed_nodes []
        -
        query nodes ['k', 'l', 'm']
        change_type New
        old_clustering []
        len(new_clustering) 1
        removed_nodes []
        ----
    """
    # 1. Configuration
    config_ini = configparser.ConfigParser()
    config_ini.read(ga_config)

    # 2. Recent results from verification ground truth tests. Used to
    # establish the weighter.
    with open(verifier_gt, 'r') as fn:
        verifier_gt = json.loads(fn.read())

    # 3. Form the parameters dictionary and weight objects (one per
    # verification algorithm).
    ga_params, wgtrs = ga_driver.params_and_weighters(config_ini, verifier_gt)
    if len(wgtrs) > 1:
        logger.info('Not currently handling more than one weighter!!')
        sys.exit(1)
    wgtr = wgtrs[0]

    # 4. Get the request dictionary, which includes the database, the
    # actual request edges and clusters, and the edge generator edges
    # and ground truth (for simulation).
    with open(request, 'r') as fn:
        request = json.loads(fn.read())

    db = overall_driver.form_database(request)
    edge_gen = overall_driver.form_edge_generator(request, db, wgtr)
    verifier_req, human_req, cluster_req = overall_driver.extract_requests(request, db)

    # 5. Form the graph algorithm driver
    driver = ga_driver.ga_driver(
        verifier_req, human_req, cluster_req, db, edge_gen, ga_params
    )

    # 6. Run it. Changes are logged.
    changes_to_review = driver.run_all_ccPICs()

    # 7. Commit changes. Record them in the database and the log
    # file.
    # TBD

    return changes_to_review


def form_database(request):
    """
    From the request json object extract the database if it is there.
    If not, return an empty database. The json includes edge quads
    (n0, n1, w, aug_name) and a clustering dictionary.
    """
    edge_quads = []
    clustering_dict = dict()

    if 'database' not in request:
        return edge_quads, clustering_dict

    req_db = request['database']
    if 'quads' in req_db:
        edge_quads = req_db['quads']
    if 'clustering' in req_db:
        clustering_dict = {str(cid): c for cid, c in req_db['clustering'].items()}

    db = _plugin_db_interface.db_interface(edge_quads, clustering_dict)
    return db


def form_edge_generator(request, db, wgtr):
    """
    Form the edge generator object. Unlike the database, the generator
    must be there for the small example / simulator to run.
    """
    try:
        gen_dict = request['generator']
    except KeyError:
        logger.info('Information about the edge generator must be in the request.')
        sys.exit(1)

    # Get hand-specified results from the verifier that aren't in the
    # database yet. These are prob_quads of the form (n0, n1, prob,
    # aug_name).  The weighter will be used to turn the prob into a
    # weight.
    prob_quads = []
    if 'verifier' in gen_dict:
        prob_quads = gen_dict['verifier']

    # Get human decisions of the form (n0, n1, bool). These will be
    # returned as new edges when first requested
    human_triples = []
    if 'human' in gen_dict:
        human_triples = gen_dict['human']

    # Get the ground truth clusters - used to generate edges that
    # aren't listed explicitly
    gt_clusters = []
    if 'gt_clusters' in gen_dict:
        gt_clusters = gen_dict['gt_clusters']

    # Get the nodes to be removed early in the computation.
    nodes_to_remove = []
    if 'nodes_to_remove' in gen_dict:
        nodes_to_remove = gen_dict['nodes_to_remove']

    # Get the number of steps between returning edge generation
    # results. If this value is 0 then they are returned immediately
    # upon request.
    delay_steps = 0
    if 'delay_steps' in gen_dict:
        delay_steps = gen_dict['delay_steps']

    edge_gen = _plugin_edge_generator.edge_generator(
        db, wgtr, prob_quads, human_triples, gt_clusters, nodes_to_remove, delay_steps
    )
    return edge_gen


def extract_requests(request, db):
    try:
        req_dict = request['query']
    except KeyError:
        logger.info('Information about the GA query itself must be in the request JSON.')
        sys.exit(1)

    # 1. Get the verifier result quads (n0, n1, prob, aug_name).
    verifier_results = []
    if 'verifier' in req_dict:
        verifier_results = req_dict['verifier']

    # 2. Get the human decision result triples (n0, n1, bool)
    # No error checking is used
    human_decisions = []
    if 'human' in req_dict:
        human_decisions = req_dict['human']

    # 3. Get the list of existing cluster ids to check
    cluster_ids_to_check = []
    if 'cluster_ids' in req_dict:
        cluster_ids_to_check = req_dict['cluster_ids']

    for cid in cluster_ids_to_check:
        logger.info(cid)
        logger.info(cluster_ids_to_check)
        if not db.cluster_exists(cid):
            logger.info('GA request cluster id %s does not exist' % cid)
            raise ValueError('Cluster id does not exist')

    return verifier_results, human_decisions, cluster_ids_to_check


class LCAActor(GraphActor):
    """

    CommandLine:
        python -m wbia_lca._plugin LCAActor
        python -m wbia_lca._plugin LCAActor:0
        python -m wbia_lca._plugin LCAActor:1

    Doctest:
        >>> # DISABLE_DOCTEST
        >>> from wbia.web.graph_server import *
        >>> actor = LCAActor()
        >>> payload = testdata_start_payload()
        >>> # Start the process
        >>> start_resp = actor.handle(payload)
        >>> print('start_resp = {!r}'.format(start_resp))
        >>> # Respond with a user decision
        >>> user_request = actor.handle({'action': 'continue_review'})
        >>> # Wait for a response and the LCAActor in another proc
        >>> edge, priority, edge_data = user_request[0]
        >>> user_resp_payload = _testdata_feedback_payload(edge, 'match')
        >>> content = actor.handle(user_resp_payload)
        >>> actor.infr.dump_logs()

    Doctest:
        >>> # xdoctest: +REQUIRES(module:wbia_cnn, --slow)
        >>> from wbia.web.graph_server import *
        >>> import wbia
        >>> actor = LCAActor()
        >>> config = {
        >>>     'manual.n_peek'   : 1,
        >>>     'manual.autosave' : False,
        >>>     'ranking.enabled' : False,
        >>>     'autoreview.enabled' : False,
        >>>     'redun.enabled'   : False,
        >>>     'redun.enabled'   : False,
        >>>     'queue.conf.thresh' : 'absolutely_sure',
        >>>     'algo.hardcase' : True,
        >>> }
        >>> # Start the process
        >>> dbdir = wbia.sysres.db_to_dbdir('PZ_MTEST')
        >>> payload = {'action': 'start', 'dbdir': dbdir, 'aids': 'all',
        >>>            'config': config, 'init': 'annotmatch'}
        >>> start_resp = actor.handle(payload)
        >>> print('start_resp = {!r}'.format(start_resp))
        >>> # Respond with a user decision
        >>> user_request = actor.handle({'action': 'continue_review'})
        >>> print('user_request = {!r}'.format(user_request))
        >>> # Wait for a response and  the LCAActor in another proc
        >>> edge, priority, edge_data = user_request[0]
        >>> user_resp_payload = _testdata_feedback_payload(edge, 'match')
        >>> content = actor.handle(user_resp_payload)
        >>> actor.infr.dump_logs()
        >>> actor.infr.status()
    """

    def __init__(actor, *args, **kwargs):
        super(LCAActor, actor).__init__(*args, **kwargs)
        actor.db = None

    def start(actor, dbdir, aids='all', config={}, **kwargs):
        import wbia

        assert dbdir is not None, 'must specify dbdir'
        assert actor.db is None, 'LCA database already running'
        ibs = wbia.opendb(dbdir=dbdir, use_cache=False, web=False, force_serial=True)

        # 1. Configuration
        # fmt: off
        ga_params = {
            'prob_human_correct': 0.97,
            'aug_names': [
                'vamp',
                'human'
            ],
            'min_delta_converge_multiplier': 0.95,
            'min_delta_stability_ratio': 8.0,
            'num_per_augmentation': 2,
            'tries_before_edge_done': 4,
            'ga_iterations_before_return': 10,
            'ga_max_num_waiting': 50,
            'log_level': 'INFO',
            'log_file': './lca.log',
            'draw_iterations': False,
            'drawing_prefix': 'drawing_lca'
        }
        # fmt: on

        # 2. Recent results from verification ground truth tests. Used to
        # establish the weighter.
        # fmt: off
        verifier_gt = {
            'vamp': {
                'gt_positive_probs': [
                    0.96,
                    0.45,
                    0.98,
                    0.67,
                    0.87,
                    0.38,
                    0.92,
                    0.83,
                    0.77,
                    0.91
                ],
                'gt_negative_probs': [
                    0.11,
                    0.04,
                    0.61,
                    0.33,
                    0.25,
                    0.05,
                    0.12,
                    0.51,
                    0.15,
                    0.25,
                    0.31,
                    0.06,
                    0.34,
                    0.22
                ]
            }
        }
        # fmt: on

        # 3. Form the parameters dictionary and weight objects (one per
        # verification algorithm).
        wgtrs = ga_driver.generate_weighters(ga_params, verifier_gt)

        wgtr = wgtrs[0]
        ga_params['min_delta_score_converge'] = -ga_params[
            'min_delta_converge_multiplier'
        ] * (
            wgtr.human_wgt(is_marked_correct=True)
            - wgtr.human_wgt(is_marked_correct=False)
        )

        ga_params['min_delta_score_stability'] = (
            ga_params['min_delta_score_converge'] / ga_params['min_delta_stability_ratio']
        )

        # 4. Get the request dictionary, which includes the database, the
        # actual request edges and clusters, and the edge generator edges
        # and ground truth (for simulation).
        # fmt: off
        request = {
            'database': {
                'quads': [
                    ['a', 'b', 45, 'vamp'],
                    ['a', 'd', 50, 'vamp'],
                    ['a', 'd', -100, 'human'],
                    ['b', 'd', -85, 'vamp'],
                    ['b', 'd', 100, 'human'],
                    ['d', 'f', 45, 'vamp'],
                    ['d', 'f', -100, 'human'],
                    ['f', 'h', 34, 'vamp'],
                    ['f', 'i', 16, 'vamp'],
                    ['f', 'i', -100, 'human'],
                    ['h', 'i', 85, 'vamp'],
                    ['h', 'j', 80, 'vamp'],
                    ['i', 'j', 75, 'vamp'],
                    ['j', 'k', -100, 'human'],
                    ['k', 'l', 80, 'vamp'],
                    ['l', 'm', -50, 'vamp'],
                    ['k', 'm', 100, 'human'],
                ],
                'clustering': {
                    '100': ['a', 'b'],
                    '101': ['d'],
                    '102': ['h', 'i', 'j'],
                    '103': ['k', 'l'],
                },
            },
            'generator': {
                'verifier': [
                    ['a', 'e', 0.88, 'vamp'],
                    ['c', 'e', 0.83, 'vamp'],
                    ['k', 'm', 0.9, 'vamp'],
                    ['c', 'd', 0.75, 'vamp'],
                ],
                'human': [
                    ['f', 'h', True],
                    ['f', 'i', True],
                    ['l', 'm', True],
                ],
                'gt_clusters': [
                    ['a', 'b', 'c', 'd', 'e'],
                    ['f', 'h', 'i', 'j'],
                    ['g'],
                    ['k', 'l', 'm'],
                ],
                'delay_steps': 5,
                'nodes_to_remove': [],
            },
            'query': {
                'verifier': [
                    ['b', 'e', 0.9, 'vamp'],
                    ['f', 'g', 0.15, 'vamp'],
                ],
                'human': [
                    ['a', 'c', True],
                ],
                'cluster_ids': [
                    '103',
                ],
            }
        }
        # fmt: on

        ut.embed()

        actor.db = form_database(request)
        edge_gen = form_edge_generator(request, actor.db, wgtr)
        verifier_req, human_req, cluster_req = extract_requests(request, actor.db)

        # 5. Form the graph algorithm driver
        driver = ga_driver.ga_driver(
            verifier_req, human_req, cluster_req, actor.db, edge_gen, ga_params
        )

        # 6. Run it. Changes are logged.
        changes_to_review = driver.run_all_ccPICs()
        print(ibs, changes_to_review)

        # 7. Commit changes. Record them in the database and the log
        # file.
        # TBD

        # # Create the AnnotInference
        # logger.info('starting via actor with ibs = %r' % (ibs,))
        # actor.infr = wbia.AnnotInference(ibs=ibs, aids=aids, autoinit=True)
        # actor.infr.print('started via actor')
        # actor.infr.print('config = {}'.format(ut.repr3(config)))
        # # Configure query_annot_infr
        # for key in config:
        #     actor.infr.params[key] = config[key]
        # # Initialize
        # # TODO: Initialize state from staging reviews after annotmatch
        # # timestamps (in case of crash)

        # actor.infr.print('Initializing infr tables')
        # table = kwargs.get('init', 'staging')
        # actor.infr.reset_feedback(table, apply=True)
        # actor.infr.ensure_mst()
        # actor.infr.apply_nondynamic_update()

        # actor.infr.print('infr.status() = {}'.format(ut.repr4(actor.infr.status())))

        # # Load random forests (TODO: should this be config specifiable?)
        # actor.infr.print('loading published models')
        # try:
        #     actor.infr.load_published()
        # except Exception:
        #     pass

        # # Start actor.infr Main Loop
        # actor.infr.print('start id review')
        # actor.infr.start_id_review()

        return 'initialized'

    def resume(actor):
        raise NotImplementedError()

    def add_feedback(actor, **feedback):
        raise NotImplementedError()

    def add_annots(actor, aids, **kwargs):
        raise NotImplementedError()

    def remove_annots(actor, aids, **kwargs):
        raise NotImplementedError()

    def get_logs(actor):
        raise NotImplementedError()

    def get_logs_latest(actor):
        raise NotImplementedError()

    def get_status(actor):
        raise NotImplementedError()

    # ##### HotSpotter ######

    def get_feat_extractor(actor):
        raise NotImplementedError()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_lca._plugin
    """
    import xdoctest

    xdoctest.doctest_module(__file__)
