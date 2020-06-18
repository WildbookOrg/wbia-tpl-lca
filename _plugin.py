# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject
from ibeis.constants import CONTAINERIZED, PRODUCTION  # NOQA
import utool as ut
from ibeis_lca import graph_algorithm

(print, rrr, profile) = ut.inject2(__name__)


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)

register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']


@register_ibs_method
@register_api('/api/plugin/lca/', methods=['GET'])
def ibeis_plugin_lca_init(ibs):
    r"""
    Create an LCA graph algorithm object

    Args:
        ibs (IBEISController):  ibeis controller object

    Returns:
        object: graph_algorithm

    CommandLine:
        python -m ibeis_lca._plugin --test-ibeis_lca_hello_world

    RESTful:
        Method: GET

        URL:    /api/plugin/example/identification/helloworld/

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> resp = ibs.ibeis_plugin_lca_init()
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
        [ibeis_lca] hello world with IBEIS controller <IBEISController(testdb_identification) with UUID 1654bdc9-4a14-43f7-9a6a-5f10f2eaa279>
        {
            'annotations': 70,
            'database': UUID('1654bdc9-4a14-43f7-9a6a-5f10f2eaa279'),
            'images': 69,
            'imagesets': 7,
            'names': 21,
        }
    """
    ut.embed()
    graph = graph_algorithm.graph_algorithm()  # NOQA

    args = (ibs,)
    resp = '[ibeis_lca] hello world with IBEIS controller %r' % args
    return resp


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_lca._plugin --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
