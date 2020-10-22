# -*- coding: utf-8 -*-
import logging  # NOQA

logger = logging.getLogger('wbia_lca')
logger.setLevel(logging.INFO)

try:
    from wbia_lca._version import __version__
except ImportError:
    __version__ = '0.0.0'

# from wbia_lca.version import version as __version__  # NOQA
from wbia_lca import __main__  # NOQA
from wbia_lca import _plugin  # NOQA
from wbia_lca import _plugin_db_interface  # NOQA
from wbia_lca import _plugin_edge_generator  # NOQA

from wbia_lca import baseline  # NOQA
from wbia_lca import cid_to_lca  # NOQA
from wbia_lca import cluster_tools  # NOQA
from wbia_lca import compare_clusterings  # NOQA
from wbia_lca import db_interface  # NOQA
from wbia_lca import db_interface_sim  # NOQA
from wbia_lca import draw_lca  # NOQA
from wbia_lca import edge_generator  # NOQA
from wbia_lca import edge_generator_sim  # NOQA
from wbia_lca import exp_scores  # NOQA
from wbia_lca import ga_driver  # NOQA
from wbia_lca import graph_algorithm  # NOQA
from wbia_lca import lca  # NOQA
from wbia_lca import lca_alg1  # NOQA
from wbia_lca import lca_alg2  # NOQA
from wbia_lca import lca_heap  # NOQA
from wbia_lca import lca_queues  # NOQA
from wbia_lca import overall_driver  # NOQA
from wbia_lca import run_from_simulator  # NOQA
from wbia_lca import simulator  # NOQA
from wbia_lca import test_cluster_tools  # NOQA
from wbia_lca import test_graph_algorithm  # NOQA
from wbia_lca import weight_manager  # NOQA
from wbia_lca import weighter  # NOQA
