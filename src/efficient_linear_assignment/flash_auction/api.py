
from .triton_backend import AuctionImplicitTriton

def linear_assignment_implicit(
    Q, K, epsilon=1e-2, max_iter=1000
):
    """
    Solves LAP using Flash Auction (Q@K.T).
    """
    solver = AuctionImplicitTriton(epsilon, max_iter)
    assignment, _ = solver.solve(Q, K)
    return assignment
