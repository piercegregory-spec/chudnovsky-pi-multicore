import math
import time
import multiprocessing as mp

import gmpy2
from gmpy2 import mpz, mpfr

# Chudnovsky constants as infinite-precision integers (mpz)
C = mpz(640320)
C3_OVER_24 = (C**3) // 24


def bs_step(a, b):
    """
    Binary Splitting Core Step.
    Computes the combined components P, Q, and T for terms from a to b-1.
    """
    # Base case: Just a single term
    if b - a == 1:
        if a == 0:
            P = mpz(1)
            Q = mpz(1)
        else:
            P = mpz((6*a - 5) * (2*a - 1) * (6*a - 1))
            Q = mpz(a**3 * C3_OVER_24)

        T = mpz((545140134 * a + 13591409) * P)
        if a % 2 == 1:
            T = -T
        return P, Q, T

    # Recursive case: Split the workload in half
    mid = (a + b) // 2
    P1, Q1, T1 = bs_step(a, mid)
    P2, Q2, T2 = bs_step(mid, b)

    P = P1 * P2
    Q = Q1 * Q2
    T = Q2 * T1 + P1 * T2
    return P, Q, T


def parallel_bs_worker(args):
    """Worker function for the multiprocessing pool."""
    return bs_step(*args)


def timed_bs_worker(args):
    """Like parallel_bs_worker, but also returns how long the worker itself computed.
    This lets the parent separate real compute time from pool startup and data transfer."""
    t0 = time.perf_counter()
    P, Q, T = bs_step(*args)
    return P, Q, T, time.perf_counter() - t0


def merge_pair(left, right, need_P=True):
    """Combine two adjacent (P, Q, T) results (left covers earlier terms than right).
    The final result only needs Q and T, so P can be skipped for the last merge."""
    P1, Q1, T1 = left
    P2, Q2, T2 = right
    T = Q2 * T1 + P1 * T2
    Q = Q1 * Q2
    P = P1 * P2 if need_P else None
    return P, Q, T


def merge_chain(results):
    """Original left-to-right merge: a growing result is repeatedly multiplied by a small one."""
    P, Q, T = results[0]
    for next_P, next_Q, next_T in results[1:]:
        T = next_Q * T + P * next_T
        P = P * next_P
        Q = Q * next_Q
    return P, Q, T


def merge_tree(results):
    """Pairwise (binary tree) merge: operands stay similar in size, which is much cheaper
    for big-integer multiplication. Order is preserved, so the result is identical."""
    level = list(results)
    while len(level) > 1:
        last_merge = len(level) == 2
        nxt = [merge_pair(level[i], level[i + 1], need_P=not last_merge)
               for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def calculate_pi_parallel(digits, timings=None, verbose=True, merge="tree"):
    """Manages the CPU cores to calculate Pi to a specific digit length.

    If a dict is passed as `timings`, it is filled with checkpoint durations (seconds):
      setup          - precision, term count and chunk planning
      pool_wall      - whole multiprocessing phase (pool start, compute, transfer, teardown)
      worker_max     - slowest worker's own compute time (the critical path)
      worker_sum     - total compute time summed over all workers
      pool_overhead  - pool_wall - worker_max (process startup and pickling results)
      merge          - sequential merge of the per-core results (single core)
      divide         - final sqrt and high-precision division (single core)
      workers        - number of worker processes used
    """
    # 1. Determine how many bits of precision we need
    # Roughly 3.322 bits per decimal digit + safety buffer
    t_setup = time.perf_counter()
    DIGITS_PER_TERM = 14.181647462725477
    terms = max(1, math.ceil(digits / DIGITS_PER_TERM))
    precision_bits = math.ceil(digits * 3.322) + 128
    gmpy2.get_context().precision = precision_bits

    # 2. Detect your M3 Ultra cores
    cores = mp.cpu_count()
    if verbose:
        print(f"🚀 Deploying calculation across {cores} CPU cores...")
        print(f"🎯 Target: {digits} digits using {terms} terms.")

    # 3. Chop the massive workload into equal chunks for the cores
    chunk_size = math.ceil(terms / cores)
    chunks = []
    for i in range(cores):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, terms)
        if start < end:
            chunks.append((start, end))
    setup_time = time.perf_counter() - t_setup

    # 4. Spin up the parallel process pool across Apple Silicon
    t_pool = time.perf_counter()
    with mp.Pool(processes=len(chunks)) as pool:
        results = pool.map(timed_bs_worker, chunks)
    pool_wall = time.perf_counter() - t_pool
    worker_times = [r[3] for r in results]
    results = [r[:3] for r in results]

    # 5. Merge the independent results from each core
    if verbose:
        print("🌲 Merging results from individual cores...")
    t_merge = time.perf_counter()
    P, Q, T = merge_tree(results) if merge == "tree" else merge_chain(results)
    merge_time = time.perf_counter() - t_merge

    # 6. Execute final floating-point division to get Pi
    if verbose:
        print("🧮 Performing final high-precision division...")
    t_div = time.perf_counter()
    num = mpfr(Q * 426880) * gmpy2.sqrt(mpfr(10005))
    pi = num / mpfr(T)
    divide_time = time.perf_counter() - t_div

    if timings is not None:
        timings.update(
            setup=setup_time,
            pool_wall=pool_wall,
            worker_max=max(worker_times),
            worker_sum=sum(worker_times),
            pool_overhead=max(pool_wall - max(worker_times), 0.0),
            merge=merge_time,
            divide=divide_time,
            workers=len(chunks),
        )
    return pi
