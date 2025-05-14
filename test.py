import jax
import jax.numpy as jnp
from jax.tree_util import Partial
key = jax.random.key(137)
import time
import jaxkd as jk
import argparse

# Benchmark results
# 536M points, 8M queries, d=2, k=4, on H100: 7s to build and 1s to query
# 16M points, 16M queries, d=3, k=4, on H100: 0.15s to build and 5s to query

argparse = argparse.ArgumentParser(description='Test kd-tree implementation')
argparse.add_argument('--points', type=int, default=1024, help='Number of random points to generate')
argparse.add_argument('--batches', type=int, default=1, help='Number of batches to evaluate queries in')
argparse.add_argument('--dims', type=int, default=2, help='Number of dimensions')
argparse.add_argument('--neighbors', type=int, default=4, help='Number of neighbors to query')
argparse.add_argument('--trials', type=int, default=2, help='Number of trials to run')
argparse.add_argument('--queries', type=int, default=0, help='Number of queries to run (0 for all points)')
args = argparse.parse_args()

print(args)

for i in range(args.trials):
    start = time.time()
    key, subkey = jax.random.split(key)
    points = jax.random.normal(subkey, shape=(args.points, args.dims))
    tree = jk.make_kd_tree(points)
    tree[0].block_until_ready()
    print(f'Built kd-tree in {time.time() - start:.3f} seconds')

    start = time.time()
    queries = points[:args.queries] if args.queries > 0 else points
    if args.batches == 1: neighbors, distances = jax.vmap(lambda query: jk.query_neighbors(query, points, *tree, k=args.neighbors))(queries)
    else: neighbors, distances = jax.lax.map(lambda query: jk.query_neighbors(query, points, *tree, k=args.neighbors), queries, batch_size=len(points)//args.batches)
    neighbors.block_until_ready()
    print(f'Found query neighbors in {time.time() - start:.3f} seconds')

    del points, tree, queries, neighbors, distances