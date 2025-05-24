from testing import assert_equal
from gpu.host import DeviceContext

from gpu import thread_idx, block_idx, block_dim, grid_dim, warp, barrier
from gpu.memory import load
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb

from math import ceildiv
from memory import UnsafePointer
from os.atomic import Atomic
from random import randint
from time import perf_counter_ns

alias TPB = 512
alias LOG_TPB = 9
alias BATCH_SIZE = 8  # needs to be power of 2
alias SIZE = 1 << 30
alias NUM_BLOCKS = ceildiv(SIZE, TPB * BATCH_SIZE)
alias BLOCKS_PER_GRID_STAGE_1 = NUM_BLOCKS
alias BLOCKS_PER_GRID_STAGE_2 = 1

alias dtype = DType.int32
alias layout = Layout.row_major(SIZE)
alias stage_1_out_layout = Layout.row_major(NUM_BLOCKS)
alias out_layout = Layout.row_major(1)


fn warmup_kernel():
    _ = thread_idx.x


fn sum_kernel[
    size: Int, batch_size: Int
](out: UnsafePointer[Int32], a: UnsafePointer[Int32],):
    sums = tb[dtype]().row_major[TPB]().shared().alloc()
    global_tid = block_idx.x * block_dim.x + thread_idx.x
    tid = thread_idx.x
    threads_in_grid = TPB * grid_dim.x
    var sum: Int32 = 0

    for i in range(global_tid, size, threads_in_grid):
        # @parameter
        # for j in range(batch_size):
        #    idx = i * batch_size + j
        #    if idx < size:
        #        sum += a[idx]
        idx = i * batch_size
        if idx < size:
            sum += load[width=batch_size](a, idx).reduce_add()
    sums[tid] = sum
    barrier()

    # See https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-for-statement
    active_threads = TPB

    @parameter
    for power in range(1, LOG_TPB - 4):
        active_threads >>= 1
        if tid < active_threads:
            sums[tid] += sums[tid + active_threads]
        barrier()

    if tid < 32:
        var warp_sum: Int32 = sums[tid][0]
        warp_sum = warp.sum(warp_sum)

        if tid == 0:
            _ = Atomic.fetch_add(out, warp_sum)


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](1).enqueue_fill(0)
        stage_1_out = ctx.enqueue_create_buffer[dtype](NUM_BLOCKS).enqueue_fill(
            0
        )
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

        with a.map_to_host() as a_host:
            randint[dtype](a_host.unsafe_ptr(), SIZE, 0, 10)
            # print(a_host)

        out_tensor = out.unsafe_ptr()
        a_tensor = a.unsafe_ptr()

        num_warmup = 500
        for _ in range(num_warmup):
            _ = out.enqueue_fill(0)
            ctx.enqueue_function[sum_kernel[SIZE, BATCH_SIZE]](
                out_tensor,
                a_tensor,
                grid_dim=NUM_BLOCKS,
                block_dim=TPB,
            )
            ctx.synchronize()

        # TIME
        t1 = perf_counter_ns()

        # STAGE 1
        num_tries = 10000
        for i in range(num_tries):
            _ = out.enqueue_fill(0)
            ctx.enqueue_function[sum_kernel[SIZE, BATCH_SIZE]](
                out_tensor,
                a_tensor,
                grid_dim=NUM_BLOCKS,
                block_dim=TPB,
            )
            ctx.synchronize()

        # TIME
        t2 = perf_counter_ns()
        delta = (t2 - t1) / 1e9 / num_tries
        var bandwidth: Float64 = SIZE * 4 / delta / 1e9
        print("delta(s) = ", delta)
        print("GB/s = ", bandwidth)
        print("% of max = ", 3300 / bandwidth)

        expected = ctx.enqueue_create_host_buffer[dtype](1).enqueue_fill(0)

        with a.map_to_host() as a_host:
            for i in range(SIZE):
                expected[0] += a_host[i]

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            assert_equal(out_host[0], expected[0])
