from .auction import linear_assignment, api
from .auction.api import BACKENDS
from .compiled import sinkhorn_compiled, dual_ascent_compiled, auction_compiled
from .auction import api as auction
from .sinkhorn import api as sinkhorn
from .dual_ascent import api as dual_ascent
from .routing import max_score_routing
from .sinkhorn.api import log_stabilized_sinkhorn
from .dual_ascent.api import l2_regularized_dual_ascent

# Flash (Implicit) Solvers
from . import flash_dual_ascent
from . import flash_auction
