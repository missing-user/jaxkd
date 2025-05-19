import argparse
import time
import jax
import jax.numpy as jnp
import jaxkd as jk
import numpy as np
import psutil
import os

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

def run_trial(points_n, queries_n, neighbors=4, trials=1):
    dims = 3
    key = jax.random.PRNGKey(83)
    results = []

    for trial in range(trials):
        print(f'\nTrial {trial+1} — points={points_n}, queries={queries_n}')
        mem0 = get_memory_usage_mb()
        key, subkey = jax.random.split(key)
        points = jax.random.normal(subkey, shape=(points_n, dims))
        queries = points[:queries_n]

        # Build KD-tree
        start = time.time()
        tree = jk.build_tree(points)
        tree[1].block_until_ready()
        kd_build_time = time.time() - start

        # KD-tree query
        start = time.time()
        neighbors_kd, distances_kd = jk.query_neighbors(tree, queries, neighbors)
        neighbors_kd.block_until_ready()
        kd_query_time = time.time() - start
        kd_mem = get_memory_usage_mb() - mem0

        # Naive query
        start = time.time()
        dists = jnp.linalg.norm(points[None, :, :] - queries[:, None, :], axis=-1)
        naive_distances = jnp.sort(dists, axis=1)[:, :neighbors]
        naive_time = time.time() - start
        naive_mem = get_memory_usage_mb() - mem0 - kd_mem

        results.append({
            "points": points_n,
            "queries": queries_n,
            "kd_build_time": kd_build_time,
            "kd_query_time": kd_query_time,
            "kd_mem": kd_mem,
            "naive_time": naive_time,
            "naive_mem": naive_mem,
        })

    return results

def main():
    configurations = [
        (100, 10),
        (100, 100),
        (1000, 10),
        (1000, 100),
        (1000, 1000),
        (10000, 10),
        (10000, 100),
    ]

    all_results = []
    for points_n, queries_n in configurations:
        results = run_trial(points_n, queries_n, trials=1)
        all_results.extend(results)

    print("\nSummary:")
    for r in all_results:
        print(f"[{r['points']} pts, {r['queries']} q] "
              f"KD: {r['kd_query_time']:.3f}s | Naive: {r['naive_time']:.3f}s | "
              f"Mem KD: {r['kd_mem']:.1f}MB | Naive: {r['naive_mem']:.1f}MB")

if __name__ == "__main__":
    main()
