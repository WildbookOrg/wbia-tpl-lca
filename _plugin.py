# -*- coding: utf-8 -*-
from wbia.control import controller_inject
from wbia.constants import CONTAINERIZED, PRODUCTION  # NOQA
import utool as ut
import logging

import argparse
import configparser
import json
import sys

from wbia_lca import ga_driver
from wbia_lca.overall_driver import form_database, form_edge_generator, extract_requests


logger = logging.getLogger('wbia_lca')


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)

register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']


@register_ibs_method
@register_api('/api/plugin/lca/', methods=['GET'])
def wbia_plugin_lca_init(ibs):
    r"""
    Create an LCA graph algorithm object

    Args:
        ibs (IBEISController):  wbia controller object

    Returns:
        object: graph_algorithm

    CommandLine:
        python -m wbia_lca._plugin --test-wbia_lca_hello_world

    RESTful:
        Method: GET

        URL:    /api/plugin/example/identification/helloworld/

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import wbia
        >>> import utool as ut
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> resp = ibs.wbia_plugin_lca_init()
        >>>
        >>> # Result is a special variable in our doctests.  If the last line
        >>> # contains a "result" assignment, then the test checks if the lines
        >>> # specified just below the test are equal to the value of result.
        >>> result = resp + '\n' + ut.repr3({
        >>>     'database'    : ibs.get_db_init_uuid(),
        >>>     'imagesets'   : len(ibs.get_valid_imgsetids()),
        >>>     'images'      : len(ibs.get_valid_gids()),
        >>>     'annotations' : len(ibs.get_valid_aids()),
        >>>     'names'       : len(ibs.get_valid_nids()),
        >>> })
        [wbia_lca] hello world with IBEIS controller <IBEISController(testdb_identification) with UUID 1654bdc9-4a14-43f7-9a6a-5f10f2eaa279>
        {
            'annotations': 70,
            'database': UUID('1654bdc9-4a14-43f7-9a6a-5f10f2eaa279'),
            'images': 69,
            'imagesets': 7,
            'names': 21,
        }
    """
    ut.embed()

    parser = argparse.ArgumentParser('overall_driver.py')
    parser.add_argument(
        '--ga_config', type=str, required=True, help='graph algorithm config INI file'
    )
    parser.add_argument(
        '--verifier_gt',
        type=str,
        required=True,
        help='json file containing verification algorithm ground truth',
    )
    parser.add_argument(
        '--request',
        type=str,
        required=True,
        help='json file continain graph algorithm request info',
    )
    parser.add_argument(
        '--db_result', type=str, help='file to write resulting json database'
    )

    # 1. Configuration
    args = parser.parse_args()
    config_ini = configparser.ConfigParser()
    config_ini.read(args.ga_config)

    # 2. Recent results from verification ground truth tests. Used to
    # establish the weighter.
    fn = open(args.verifier_gt)
    verifier_gt = json.loads(fn.read())
    fn.close()

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
    fn = open(args.request)
    request = json.loads(fn.read())
    fn.close()
    db = form_database(request)
    edge_gen = form_edge_generator(request, db, wgtr)
    verifier_req, human_req, cluster_req = extract_requests(request, db)

    # 5. Form the graph algorithm driver
    driver = ga_driver.ga_driver(
        verifier_req, human_req, cluster_req, db, edge_gen, ga_params
    )

    # 6. Run it. Changes are logged.
    changes_to_review = driver.run_all_ccPICs()
    logger.info(changes_to_review)

    # 7. Commit changes. Record them in the database and the log
    # file.
    # TBD

    graph = graph_algorithm.graph_algorithm()  # NOQA

    args = (ibs,)
    resp = '[wbia_lca] hello world with WBIA controller %r' % args
    return resp


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_lca._plugin
    """
    import xdoctest

    xdoctest.doctest_module(__file__)
