import jax
import jax.numpy as jnp
from jax.tree_util import Partial
key = jax.random.key(137)
import time
import sys
import jaxkd as jk

n_points = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
print(f'Using {n_points} random points')

for i in range(3):
    start = time.time()
    key, subkey = jax.random.split(key)
    # points = jax.random.uniform(subkey, shape=(n_points, 2))
    points = jax.random.normal(subkey, shape=(n_points, 2))
    tree = jk.make_kd_tree(points)
    tree[0].block_until_ready()
    print(f'Built kd-tree in {time.time() - start:.3f} seconds')

    start = time.time()
    neighbors, distances = jax.vmap(lambda query: jk.query_neighbors(query, points, *tree, k=4))(points)
    neighbors.block_until_ready()
    print(f'Queried all points in {time.time() - start:.3f} seconds')