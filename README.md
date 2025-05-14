Minimal JAX implementation of KNN using a KD tree.

Can and should be JIT compiled. Uses this algorithm to traverse the tree: https://arxiv.org/abs/2210.12859. Much slower than `scipy.spatial.KDTree` on CPU, but can be reasonably fast when queries are vectorized on GPU. Main advantage is to avoid leaving JIT, transfering data off the GPU, and calling external libraries when a simple nearest neighbor search is needed and is not the primary computational load.

Basic usage:

```
n_points = 64 * 1024
key, subkey = jax.random.split(key)
points = jax.random.normal(subkey, shape=(n_points, 2))
tree = jk.make_kd_tree(points)
neighbors, distances = jk.query_neighbors(points[0], points, *tree, k=6)
all_neighbors, all_distances = jax.vmap(lambda q: jk.query_neighbors(q, points, *tree, k=6))(points)
```