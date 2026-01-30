import torch
from typing import Tuple

class AuctionTorch:
    def __init__(self, epsilon: float = 1e-2, max_iter: int = 1000):
        self.epsilon = epsilon
        self.max_iter = max_iter

    def solve(self, cost_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solves LAP minimizing cost.
        Auction algorithm typically maximizes Benefit.
        Benefit = -Cost.
        
        Args:
            cost_matrix: (B, N, M)
            
        Returns:
            assignment: (B, N)
            prices: (B, M)
        """
        # Convert to benefit matrix (maximize benefit)
        # Keep input dtype (e.g. F16) -> Ops will promote to F32 when interacting with 'prices' (F32)
        benefits = -cost_matrix
        
        B, N, M = benefits.shape
        prices = torch.zeros((B, M), device=benefits.device, dtype=benefits.dtype)
        assignment = torch.full((B, N), -1, device=benefits.device, dtype=torch.long)
        
        # Track unassigned agents. 
        # For simplicity in pure torch, we iterate 'max_iter' times or until conv.
        # A fully vectorized approach often iterates on *all* agents or masks unassigned.
        # Track object ownership invserse to assignment
        object_to_agent = torch.full((B, M), -1, device=benefits.device, dtype=torch.long)
        
        # Flatten everything to (TotalAgents, M)
        # This allows us to efficient mask unassigned agents across the entire batch
        flat_benefits = benefits.view(-1, M) # (B*N, M)
        flat_prices = prices.view(-1, M) # (B, M)
        
        # We need a mapping from AgentIndex -> PriceRowIndex.
        # agent_idx \in [0, B*N). batch_id = agent_idx // N.
        
        N_total = B * N
        flat_assignment = torch.full((N_total,), -1, device=benefits.device, dtype=torch.long)
        flat_obj_to_agent = torch.full((B, M), -1, device=benefits.device, dtype=torch.long).view(-1) # (B*M)
        
        # Precompute batch indices for agents
        agent_batch_ids = torch.arange(B, device=benefits.device).repeat_interleave(N)
        
        # Global Object Offsets: Object 'm' in batch 'b' has global ID 'b*M + m'
        # batch_obj_offsets = torch.arange(B, device=benefits.device).repeat_interleave(M) * M
        
        for i in range(self.max_iter):
            # 1. Identify unassigned agents (Global indices)
            unassigned_mask = (flat_assignment == -1)
            active_agent_indices = torch.nonzero(unassigned_mask).squeeze(1)
            
            num_active = active_agent_indices.numel()
            if num_active == 0:
                break
                
            # 2. Gather Data for Active Agents
            # Active Benefits: (K, M)
            active_benefits = flat_benefits[active_agent_indices]
            
            # Active Prices: Need prices for the batch each agent belongs to.
            active_batch_ids = agent_batch_ids[active_agent_indices]
            # Expanding prices is expensive if done naively? 
            # prices is (B, M). gathering (K, M) is fine.
            active_prices = prices[active_batch_ids]
            
            # 3. Compute Bids (Vectorized on K)
            # values = benefits - prices
            # (K, M)
            current_values = active_benefits - active_prices
            
            # Top 2
            # For extremely large M, topk(dim=1) is expensive?
            # It's O(M log K) or O(M). With M=4096, it's 4096 ops per agent.
            top2_values, top2_indices = torch.topk(current_values, k=2, dim=1)
            
            best_val = top2_values[:, 0]
            second_val = top2_values[:, 1]
            best_obj_idx = top2_indices[:, 0]
            
            # Increment
            increments = best_val - second_val + self.epsilon
            
            # 4. Update Prices (Conflict Resolution)
            # Each active agent wants to update 'best_obj_idx' in their batch.
            # Global Object Index = batch_offset + local_obj
            global_obj_indices = active_batch_ids * M + best_obj_idx
            
            # Bids = Price(current) + Increment
            flat_prices_view = prices.view(-1)
            target_current_prices = flat_prices_view[global_obj_indices]
            
            bids = target_current_prices + increments
            
            # Scatter Max Bid to Global Prices
            flat_prices_view.scatter_reduce_(0, global_obj_indices, bids, reduce='amax', include_self=True)
            
            # 5. Update Assignments
            # Get New Prices for targets
            new_target_prices = flat_prices_view[global_obj_indices]
            
            # Check matches (Winning condition)
            did_win = torch.abs(bids - new_target_prices) < 1e-5
            
            # 6. Transfer Ownership (Collision Resistant)
            # Identify potential winners (anyone who matched the price)
            winning_indices = torch.nonzero(did_win).squeeze(1) # Indices into 'active' arrays
            
            if winning_indices.numel() > 0:
                potential_winner_agents = active_agent_indices[winning_indices]
                potential_won_objects = global_obj_indices[winning_indices] # Global Obj IDs
                
                # Conflict Resolution: If multiple agents matched price for same obj, pick one.
                # Use a temporary buffer to elect one winner per object.
                # Initialize with -1.
                # Note: We only need to cover the objects that were bid on.
                # But creating full B*M buffer is efficient enough?
                # Or sparse? Full B*M is memory safe and fast (scatter).
                
                iter_obj_winners = torch.full((B * M,), -1, device=benefits.device, dtype=torch.long)
                
                # Scatter agents. Last writer wins (arbitrary tie-breaker).
                iter_obj_winners[potential_won_objects] = potential_winner_agents
                
                # Verify who won
                # Read back the elected winner for each potential win
                elected_winners = iter_obj_winners[potential_won_objects]
                
                # Filter: You are a winner only if you were elected
                is_actual_winner = (elected_winners == potential_winner_agents)
                
                actual_winning_agents = potential_winner_agents[is_actual_winner]
                actual_won_objects = potential_won_objects[is_actual_winner]
                
                # Now perform updates with guaranteed unique winner per object
                if actual_winning_agents.numel() > 0:
                    # Check previous owners of these objects
                    prev_owners = flat_obj_to_agent[actual_won_objects] # Global Agent IDs
                    
                    # Unassign previous owners
                    valid_prev = (prev_owners != -1)
                    if valid_prev.any():
                        agents_to_unassign = prev_owners[valid_prev]
                        flat_assignment[agents_to_unassign] = -1
                    
                    # Assign new (Inverse Map)
                    flat_obj_to_agent[actual_won_objects] = actual_winning_agents
                    
                    # Assign new (Direct Map)
                    flat_assignment[actual_winning_agents] = actual_won_objects % M # Store local obj index
                
        assignment = flat_assignment.view(B, N)
        return assignment, prices
