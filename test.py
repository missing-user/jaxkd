import jax
import jax.numpy as jnp
from jax.tree_util import Partial
key = jax.random.key(137)
import time
import sys
import jaxkd as jk
import argparse

argparse = argparse.ArgumentParser(description='Test kd-tree implementation')
argparse.add_argument('--n_points', type=int, default=1024, help='Number of random points to generate')
argparse.add_argument('--n_batches', type=int, default=1, help='Number of batches to evaluate queries in')
argparse.add_argument('--n_dim', type=int, default=2, help='Number of dimensions')
argparse.add_argument('--n_neighbors', type=int, default=4, help='Number of neighbors to query')
argparse.add_argument('--n_trials', type=int, default=2, help='Number of trials to run')
args = argparse.parse_args()

print(f'Using {args.n_points} random points in {args.n_batches} batches')

for i in range(args.n_trials):
    start = time.time()
    key, subkey = jax.random.split(key)
    points = jax.random.normal(subkey, shape=(args.n_points, args.n_dim))
    tree = jk.make_kd_tree(points)
    tree[0].block_until_ready()
    print(f'Built kd-tree in {time.time() - start:.3f} seconds')

    start = time.time()
    if args.n_batches == 1: neighbors, distances = jax.vmap(lambda query: jk.query_neighbors(query, points, *tree, k=args.n_neighbors))(points)
    else: neighbors, distances = jax.lax.map(lambda query: jk.query_neighbors(query, points, *tree, k=args.n_neighbors), points, batch_size=len(points)//args.n_batches)
    neighbors.block_until_ready()
    print(f'Queried all points in {time.time() - start:.3f} seconds')