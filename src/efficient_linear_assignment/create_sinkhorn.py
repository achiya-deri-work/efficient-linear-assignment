import torch

# ---------------------------------------------------------------------------
# Legacy API Adapter
# ---------------------------------------------------------------------------

def log_stabilized_sinkhorn(
    C: torch.Tensor,
    mu: torch.Tensor = None,
    nu: torch.Tensor = None,
    epsilon: float = 0.1,
    num_iters: int = 20
) -> torch.Tensor:
    """
    Legacy wrapper for existing benchmarks.
    Uses _sinkhorn_core with fixed epsilon.
    """
    if C.ndim == 2: C = C.unsqueeze(0)
    B, M, N = C.shape
    device = C.device
    dtype = C.dtype
    
    if mu is None: mu = torch.ones(B, M, device=device, dtype=dtype) / M # Normalized default? Originally /N
    if mu is None: mu = torch.ones(B, N, device=device, dtype=dtype) / N
    # Original code: mu = 1/N (uniform). Wait, M sources? 1/N?
    # Original: mu = 1/N (if size N) ? No, mu is (B,N) so 1/N.
    
    # Check original code snippet in memory or file view
    if mu is None: mu = torch.ones(B, N, device=device, dtype=dtype) / N # Assuming rows N?
    # Wait, original had C shape (B, N, M)?
    # Original args: C (B, N, M).
    # New code uses (B, M, N) convention in comments?
    # Sinkhorn Universal: C (Batch, M, N). 
    # Let's check original file content I viewed.
    # Line 28: B, N, M = C.shape
    # Line 34: mu = torch.ones(B, N) / N
    # Line 36: nu = torch.ones(B, M) / M
    # So original was Rows=N, Cols=M.
    
    # My new code `sinkhorn_universal` uses B, M, N (Rows=M, Cols=N).
    # I should respect the file's convention or the user's snippet.
    # User snippet: C: (Batch, M, N).
    # Original file: C: (Batch, N, M).
    # This is a collision. "M" and "N" are swapped labels.
    # User's snippet: "Source Mass (Rows) ... M * source_usage". So Rows=M.
    # Original File: "mu: Source marginals (Batch, N)". So Rows=N.
    
    # I will adapt `log_stabilized_sinkhorn` to match the *original file's* N/M convention
    # but use `_sinkhorn_core` which expects (B, Rows, Cols).
    
    B, Rows, Cols = C.shape # N, M in original
    
    if mu is None: 
        mu = torch.ones(B, Rows, device=device, dtype=dtype) / Rows
    if nu is None: 
        nu = torch.ones(B, Cols, device=device, dtype=dtype) / Cols
        
    log_mu = torch.log(mu + 1e-8)
    log_nu = torch.log(nu + 1e-8)
    
    # Core expects log_mu, log_nu
    # And annealing. For fixed epsilon, start=end=epsilon.
    return _sinkhorn_core(
        C, log_mu, log_nu, 
        epsilon_start=epsilon, epsilon_end=epsilon, 
        scaling_steps=1, inner_iters=num_iters
    )


# ---------------------------------------------------------------------------
# Core Solver (Compiled Friendly)
# ---------------------------------------------------------------------------

def _sinkhorn_core(
    C_prime: torch.Tensor,
    log_mu: torch.Tensor,
    log_nu: torch.Tensor,
    epsilon_start: float,
    epsilon_end: float,
    scaling_steps: int,
    inner_iters: int
) -> torch.Tensor:
    """
    Pure annealing loop. No structural logic.
    """
    dtype = C_prime.dtype
    device = C_prime.device
    B, M, N = C_prime.shape

    # Annealing Schedule
    # Note: If scaling_steps is small, explicit loop is fine for dynamo.
    eps_schedule = torch.logspace(
        torch.log10(torch.tensor(epsilon_start, device=device, dtype=dtype)), 
        torch.log10(torch.tensor(epsilon_end, device=device, dtype=dtype)), 
        steps=scaling_steps, device=device, dtype=dtype
    )

    # Potentials
    f = torch.zeros(B, M, 1, device=device, dtype=dtype)
    g = torch.zeros(B, 1, N, device=device, dtype=dtype)
    
    # Annealing Loop
    for i in range(scaling_steps):
        eps = eps_schedule[i]
        M_eps = -C_prime / eps

        for _ in range(inner_iters):
            # Row Update
            # f = log(mu) - LSE(M_eps + g)
            f = log_mu.unsqueeze(-1) - torch.logsumexp(M_eps + g, dim=2, keepdim=True)
            
            # Col Update
            # g = log(nu) - LSE(M_eps + f)
            g = log_nu.unsqueeze(1) - torch.logsumexp(M_eps + f, dim=1, keepdim=True)

    # Final Plan
    log_P = (-C_prime / eps_schedule[-1]) + f + g
    P_prime = torch.exp(log_P)
    
    return P_prime

# ---------------------------------------------------------------------------
# Specialized Kernels (No conditionals on shape logic)
# ---------------------------------------------------------------------------

def sinkhorn_balanced(
    C: torch.Tensor,
    val_mu: float,
    val_nu: float,
    epsilon_start: float,
    epsilon_end: float,
    scaling_steps: int,
    inner_iters: int,
    **kwargs
) -> torch.Tensor:
    """
    Standard solver for Balanced cases (1-to-1 or Many-to-One balanced).
    """
    B, M, N = C.shape
    device = C.device
    dtype = C.dtype
    
    # Marginals (Uniform or Scaled Uniform)
    log_mu = torch.full((B, M), torch.log(torch.tensor(val_mu)), device=device, dtype=dtype)
    log_nu = torch.full((B, N), torch.log(torch.tensor(val_nu)), device=device, dtype=dtype)
    
    return _sinkhorn_core(C, log_mu, log_nu, epsilon_start, epsilon_end, scaling_steps, inner_iters)

def sinkhorn_slack_col(
    C: torch.Tensor,
    val_mu: float,
    val_nu: float,
    mass_delta: float, # Positive delta (Surplus Source)
    trash_cost: float,
    epsilon_start: float,
    epsilon_end: float,
    scaling_steps: int,
    inner_iters: int,
    **kwargs
) -> torch.Tensor:
    """
    Solver for Surplus Source (Add Slack Column).
    """
    B, M, N = C.shape
    device = C.device
    dtype = C.dtype
    
    # 1. Pads
    trash_val = torch.tensor(trash_cost, device=device, dtype=dtype) if trash_cost is not None else C.max().detach() + 1.0
    trash_col = torch.ones(B, M, 1, device=device, dtype=dtype) * trash_val
    C_prime = torch.cat([C, trash_col], dim=2)
    
    # 2. Marginals
    log_mu = torch.full((B, M), torch.log(torch.tensor(val_mu)), device=device, dtype=dtype)
    
    # Targets + Slack Bin
    log_val_nu = torch.log(torch.tensor(val_nu, device=device, dtype=dtype))
    # Note: mass_delta is the capacity of the slack bin
    log_slack = torch.log(torch.tensor(mass_delta, device=device, dtype=dtype))
    
    # Expand scalar logs to shape
    log_nu_vec = torch.full((B, N), log_val_nu, device=device, dtype=dtype)
    log_slack_vec = torch.full((B, 1), log_slack, device=device, dtype=dtype)
    log_nu_final = torch.cat([log_nu_vec, log_slack_vec], dim=1)
    
    # 3. Solve
    P_prime = _sinkhorn_core(C_prime, log_mu, log_nu_final, epsilon_start, epsilon_end, scaling_steps, inner_iters)
    
    # 4. Slice
    return P_prime[:, :, :N]

def sinkhorn_slack_row(
    C: torch.Tensor,
    val_mu: float,
    val_nu: float,
    mass_delta: float, # Negative delta (Surplus Target) -> Abs used as slack cap
    trash_cost: float,
    epsilon_start: float,
    epsilon_end: float,
    scaling_steps: int,
    inner_iters: int,
    **kwargs
) -> torch.Tensor:
    """
    Solver for Surplus Target (Add Slack Row).
    """
    B, M, N = C.shape
    device = C.device
    dtype = C.dtype
    
    # 1. Pads
    trash_val = torch.tensor(trash_cost, device=device, dtype=dtype) if trash_cost is not None else C.max().detach() + 1.0
    trash_row = torch.ones(B, 1, N, device=device, dtype=dtype) * trash_val
    C_prime = torch.cat([C, trash_row], dim=1)
    
    # 2. Marginals
    # Sources + Slack Source
    log_val_mu = torch.log(torch.tensor(val_mu, device=device, dtype=dtype))
    log_slack = torch.log(torch.tensor(abs(mass_delta), device=device, dtype=dtype))
    
    log_mu_vec = torch.full((B, M), log_val_mu, device=device, dtype=dtype)
    log_slack_vec = torch.full((B, 1), log_slack, device=device, dtype=dtype)
    log_mu_final = torch.cat([log_mu_vec, log_slack_vec], dim=1)
    
    log_nu = torch.full((B, N), torch.log(torch.tensor(val_nu)), device=device, dtype=dtype)
    
    # 3. Solve
    P_prime = _sinkhorn_core(C_prime, log_mu_final, log_nu, epsilon_start, epsilon_end, scaling_steps, inner_iters)
    
    # 4. Slice
    return P_prime[:, :M, :]

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def dispatch_sinkhorn_kernel(
    C: torch.Tensor,
    source_usage: float = 1.0,           
    target_mass_per_source: float = None
):
    """
    Returns the appropriate specialized kernel and the configured arguments for it.
    
    Returns:
        (kernel_func, kwargs_dict)
    """
    if C.ndim == 2: C = C.unsqueeze(0)
    B, M, N = C.shape
    
    # --- 1. Define Masses & Detect Imbalance ---
    val_mu = source_usage
    total_source_mass = M * val_mu

    if target_mass_per_source is None:
        # Auto-Balance Mode
        val_nu = total_source_mass / N
    else:
        # User Constraint Mode
        val_nu = target_mass_per_source * source_usage

    total_target_mass = N * val_nu
    mass_delta = total_source_mass - total_target_mass
    
    # Config Dictionary
    config = {
        'C': C,
        'val_mu': val_mu,
        'val_nu': val_nu,
        'mass_delta': mass_delta
    }
    
    # Dispatch Logic
    if mass_delta > 1e-4:
        # Case A: Surplus Source -> Slack Column
        return sinkhorn_slack_col, config
        
    elif mass_delta < -1e-4:
        # Case B: Surplus Target -> Slack Row
        return sinkhorn_slack_row, config
        
    else:
        # Case C: Balanced
        return sinkhorn_balanced, config


def sinkhorn_universal(
    C: torch.Tensor,
    source_usage: float = 1.0,
    target_mass_per_source: float = None,
    trash_cost: float = None,
    epsilon_start: float = 1.0,
    epsilon_end: float = 1e-2,
    scaling_steps: int = 10,
    inner_iters: int = 10
) -> torch.Tensor:
    """
    Universal Entry Point.
    Dispatches to the correct specialized kernel.
    """
    # 1. Get Specialized Kernel
    kernel_func, config = dispatch_sinkhorn_kernel(C, source_usage, target_mass_per_source)
    
    # 2. Add Runtime Args
    config.update({
        'trash_cost': trash_cost,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'scaling_steps': scaling_steps,
        'inner_iters': inner_iters
    })
    
    # 3. Execute
    return kernel_func(**config)
