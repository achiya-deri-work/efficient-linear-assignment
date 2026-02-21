
import math
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

@dsl_user_op
def mbarrier_expect_tx(
    mbar_ptr: cute.Pointer,
    amount: int | cutlass.Int32,
    *,
    loc=None,
    ip=None,
):
    mbar_ptr_i32 = mbar_ptr.toint(loc=loc, ip=ip).ir_value()
    # Amount usually u32
    amount_val = cutlass.Int32(amount).ir_value(loc=loc, ip=ip)
    
    llvm.inline_asm(
        None,
        [mbar_ptr_i32, amount_val],
        "mbarrier.expect_tx.shared.b64 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# Adjust import to our refactored location
import efficient_linear_assignment.cute.utils as utils
import efficient_linear_assignment.cute.utils as utils
import efficient_linear_assignment.cute.copy_utils as copy_utils
from efficient_linear_assignment.cute import blackwell_helpers as sm100_utils

class DualAscentSm100:
    arch = 100

    def __init__(
        self,
        d_block: int = 64,  # D dimension
        m_block_size: int = 128, # BlockM (N)
        n_block_size: int = 128, # BlockN (M)
        q_stage: cutlass.Constexpr[int] = 2,
    ):
        # 0. Runtime Environment Check
        import subprocess
        try:
            nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
            if "release 13.1" not in nvcc_version:
                print(f"WARNING: V4 Kernel requires CUDA 13.1. Found: {nvcc_version}")
        except Exception:
            print("WARNING: nvcc not found. Ensure you run with 'source ~/.bashrc && use_cute'")

        self.d_block = d_block
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = q_stage
        
        # Tiling Configuration
        self.cta_tiler = (m_block_size, n_block_size, d_block)
        
        self.mma_tiler_qk = (m_block_size, n_block_size, d_block)
        self.qk_acc_dtype = Float32 # Accumulate in FP32
        
        self.cluster_shape_mn = (1, 1)
        
        # Warp Setup
        self.mma_warp_id = 0
        self.load_warp_ids = (1,)
        # self.epilogue_warp_ids = (2,) # Simplify to 2 warps for testing
        
        self.threads_per_cta = cute.arch.WARP_SIZE * 2 # 64 threads
        self.mbar_total = 1 # Simple barrier for PoC

        self.threads_per_cta = cute.arch.WARP_SIZE * 2 # 64 threads

    @cute.jit
    def __call__(self, mQ, mK, mAlpha, mBeta, mGradP, epsilon, stream=None):
        # 1. Tensor Wrapper & Layout Normalization (Python Host Code)
        # Assumed mQ, mK are cute.Tensor wrappers from `from_dlpack`
        # We need to enforce layouts that match what the kernel expects (Row Major usually)
        
        # New Stride Helper (from flash_fwd.py) to ensure alignment
        new_stride = lambda t: (
            *(s if isinstance(s, int) else cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        
        mQ = cute.make_tensor(mQ.iterator, cute.make_layout(mQ.shape, stride=new_stride(mQ)))
        mK = cute.make_tensor(mK.iterator, cute.make_layout(mK.shape, stride=new_stride(mK)))
        # Alpha/Beta [N], [M] -> Stride [1] or similar
        mAlpha = cute.make_tensor(mAlpha.iterator, cute.make_layout(mAlpha.shape, stride=(1,)))
        mBeta = cute.make_tensor(mBeta.iterator, cute.make_layout(mBeta.shape, stride=(1,)))
        mGradP = cute.make_tensor(mGradP.iterator, cute.make_layout(mGradP.shape, stride=(1,)))
        
        # 2. WGMMA Configuration
        # Use trivial tiled mma helper from blackwell_helpers
        q_dtype = mQ.element_type
        k_dtype = mK.element_type
        # Major modes: K-major for both usually for Gemm?
        # Q: [N, D] -> Row Major -> Stride (D, 1) -> Rank-2 is (N, D).
        # In TiledMMA land, we need to know if operands are K-major or MN-major.
        # If Q is RowMajor [N, D], it is K-Major (fastest dim is D/K).
        # If K is RowMajor [M, D], it is K-Major (fastest dim is D/K).
        
        # Check Layouts
        q_major_mode = cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode() 
        # Note: We need to import LayoutEnum or use raw check?
        # flash_fwd imports it. Let's assume K major for now or use helper.
        
        # mma_tiler_qk = (M, N, K) = (128, 128, 64)
        tiled_mma = sm100_utils_basic.make_trivial_tiled_mma(
            q_dtype,
            tcgen05.OperandMajorMode.K, # Q
            tcgen05.OperandMajorMode.K, # K
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_qk[:2] # (M, N)
        )
        
        # 3. Smem Layouts
        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler_qk,
            q_dtype,
            self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler_qk,
            k_dtype,
            # self.kv_stage # Assume 2 stages for now?
             2 # K stage
        )
        
        # 4. TMA Atoms
        # Load Op
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE) # CTA Group 1
        
        # Cluster parameters
        cluster_shape = (1, 1, 1) # (M, N, K) cluster?
        
        # Make Atom for Q
        # Note: flash_fwd uses make_tiled_tma_atom_A / B
        tma_atom_Q, mQ_device = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]), # Cast/Select modes for TMA
            self.mma_tiler_qk,
            tiled_mma,
            cluster_shape
        )
        
        # Make Atom for K
        tma_atom_K, mK_device = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma,
            cluster_shape
        )
        
        # 5. Launch
        # Allocate Smem size?
        # For PoC we assume ample smem or calculate it.
        # SmemAllocator logic in kernel uses `shared_storage` class which we haven't defined fully
        # but we used `smem.allocate(self.shared_storage)`.
        # We need to define `self.shared_storage` struct!
        
        # Define Shared Storage dynamically or in Init?
        # Doing it in __call__ locally or reusing class member if static.
        # Let's verify `kernel` logic: `storage = smem.allocate(self.shared_storage)`
        # `self.shared_storage` must be a type/struct.
        
        # Define Storage Type
        sQ_size = cute.cosize(sQ_layout)
        sK_size = cute.cosize(sK_layout)
        
        @cute.struct
        class SharedStorage:
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            sQ: cute.struct.Align[
                cute.struct.MemRange[q_dtype, sQ_size],
                1024 # Alignment
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[k_dtype, sK_size],
                1024
            ]
            
        self.shared_storage = SharedStorage
        smem_size = SharedStorage.size_in_bytes()
        
        # Launch Kernel
        grid_dim = (1, 1, 1) # Single CTA for PoC
        
        self.kernel(
             mQ_device, # Use the device-specialized tensor from make_tma_atom
             mK_device, 
             mAlpha, mBeta, mGradP, epsilon,
             tma_atom_Q, tma_atom_K,
             sQ_layout, sK_layout,
             tiled_mma
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            smem=smem_size,
            stream=stream
        )

    # GPU device kernel
    @cute.jit(entry_point=True, preprocess=True)
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mAlpha: cute.Tensor,
        mBeta: cute.Tensor,
        mGradP: cute.Tensor,
        epsilon: Float32,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_id = warp_idx // 4 
        
        # Shared Memory Allocation
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        
        # TMA Descriptors Prefetch
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            
        # Synchronization Barriers
        mbar_ptr = storage.mbar_ptr.data_ptr() 
        if warp_idx == 0:
            # Initialize Barrier (Expect 1 completion from Producer? Or bytes?)
            # TMA transaction usually counts bytes. 
            # But mbarrier with TMA can be transaction count or bytes.
            # Using defaults.
            cute.arch.mbarrier_init(mbar_ptr, 1) # simple init
        
        
        # Producer (Warp 1: Load)
        if warp_idx == 4: # Warp 1
            # Load Q (Resident)
            bx = cute.arch.block_idx()[0]
            
            # Helper to slice Global Tensor for current Tile
            # mQ is (N, D). Tile is (BlockM, D).
            # local_tile(tensor, tile_shape, coord)
            gQ = cute.local_tile(mQ, (self.m_block_size, self.d_block), (bx, 0))
            
            # Bytes for Q
            # Hardcoded 4 bytes for Float32
            q_bytes = self.m_block_size * self.d_block * 4
            
            # Make Copy Fn for Q
            # Pass FULL mQ (device tensor).
            # tma_get_copy_fn will partition it into tiles based on the Atom.
            # We then index into it with 'bx'.
            # Note: We assume mQ is (BlockM * Grid, D) or similar structure compatible with the Atom's tiling (128, 64).
            
            copy_Q_func, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), sQ, mQ
            )

            # Barrier Logic: Expect + Copy + Arrive
            mbarrier_expect_tx(mbar_ptr, q_bytes)
            # Copy tile 'bx' to shared '0'
            # Note: copy_Q_func(src_idx, dst_idx).
            # If tma_partition(mQ) results in Rank-1 of tiles (along N), then src_idx=bx is correct.
            # mQ was layout (N, D). Atom is (128, 64).
            # Partitioning (N, D) by (128, 64) -> (ceil(N/128), 1).
            # So linear index 'bx' should work.
            copy_Q_func(bx, 0, tma_bar_ptr=mbar_ptr)
            
            cute.arch.mbarrier_arrive(mbar_ptr)
            
            # Loop over K
            num_k_tiles = cute.ceil_div(mK.shape[0], self.n_block_size)
            k_bytes = self.n_block_size * self.d_block * 4
            
            # Pre-create Copy Fn for K (Full Tensor)
            # Need to pass mK.
            copy_K_func, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K, 0, cute.make_layout(1), sK, mK
            )
            
            for k_tile in range(num_k_tiles):
                # Issue TMA Copy for K (tile index k_tile)
                mbarrier_expect_tx(mbar_ptr, k_bytes)
                copy_K_func(k_tile, 0, tma_bar_ptr=mbar_ptr)
                cute.arch.mbarrier_arrive(mbar_ptr)
                
                cute.syncthreads() 
            
        


        # Correct Implementation: Fully inside the loop
        if warp_idx == 0:
            # Identity Layout (M, N) for coordinates
            tIdx = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
            tCrIdx = tiled_mma.partition_fragment_C(tIdx) 

            # Register Fragments
            tCrC = cute.make_tensor_like(tiled_mma.partition_fragment_C(sQ))
            tCrA = cute.make_tensor_like(tiled_mma.partition_fragment_A(sQ))
            tCrB = cute.make_tensor_like(tiled_mma.partition_fragment_B(sK))
            
            num_k_tiles = cute.ceil_div(mK.shape[0], self.n_block_size)
            for k_tile in range(num_k_tiles):
                # 1. Wait for Producer (TMA Load Complete)
                # Wait for phase 0 (single pass)
                # Note: For looped updates, we need phase flipping. 
                # But for this simple PoC we assume we wait for the barrier.
                cute.arch.mbarrier_wait(mbar_ptr, 0)
                
                # 2. Clear Accumulator for this tile
                cute.clear(tCrC)
                
                # 3. WGMMA (Compute C_sub)
                sm100_utils.gemm_ptx(
                     tiled_mma.op,
                     tCrC,
                     tCrA, tCrB,
                     sQ, sK,
                     zero_init=True # Always clear, we want just Q @ K_tile.T
                )
                
                # 4. Atomic Accumulate P (Compute P_sub and Add)
                for i in range(tCrC.size()):
                    c_val = tCrC(i)
                    
                    # Coordinates
                    coord = tCrIdx(i)
                    local_m, local_n = coord[0], coord[1]
                    
                    global_m = int(cute.arch.block_idx()[0]) * self.m_block_size + local_m
                    global_n = k_tile * self.n_block_size + local_n
                    
                    # Read Alpha, Beta (Global)
                    # NOTE: Random access to global mAlpha/mBeta per thread is slow if not coalesced.
                    # But for PoC, we do direct reads.
                    # mAlpha is [N], mBeta is [M].
                    
                    # Boundary Checks
                    if global_m < mQ.shape[0] and global_n < mK.shape[0]:
                        alpha_val = mAlpha[global_m]
                        beta_val = mBeta[global_n]
                        
                        # P_ij = exp( (alpha + beta - C_ij) / epsilon )
                        # epsilon is float
                        p_exponent = (alpha_val + beta_val - c_val) / epsilon
                        p_val = cute.exp(p_exponent)
                        
                        # Atomic Add to grad_P (Accumulator for Row Sums)
                        # grad_P[global_m] += p_val
                        ptr = sm100_utils_basic.elem_pointer(mGradP, global_m) # Or utils.elem_pointer
                        # Use our imported utils
                        ptr = utils.elem_pointer(mGradP, (global_m,))
                        utils.atomic_add_fp32(p_val, ptr)

                # 5. Wait for Producer (to overwrite buffer) -> Actually Producer waits for Consumer?
                # In simple barrier-less manual sync:
                # Producer loads to Smem. Consumer reads.
                # Consumer must finish reading before Producer overwrites.
                cute.syncthreads() 


