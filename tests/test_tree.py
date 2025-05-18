import jax
import jax.numpy as jnp
import jaxkd as jk

def test_shapes():
    points = jnp.zeros((100, 3))
    queries = jnp.zeros((50, 3))
    radius = jnp.zeros((20,))

    # Tree construction
    tree = jk.build_tree(points)
    assert tree.points.shape == points.shape
    assert tree.indices.shape == points.shape[:-1]
    assert tree.split_dims.shape == points.shape[:-1]

    # Tree construction, not optimized
    tree = jk.build_tree(points, optimize=False)
    assert tree.points.shape == points.shape
    assert tree.indices.shape == points.shape[:-1]
    assert tree.split_dims == None

    # Single query
    neighbors, distances = jk.query_neighbors(tree, queries[0], 10)
    assert neighbors.shape == (10,)
    assert distances.shape == (10,)

    # Batch query
    neighbors, distances = jk.query_neighbors(tree, queries, 10)
    assert neighbors.shape == (queries.shape[0], 10)
    assert distances.shape == (queries.shape[0], 10)

    # Four possible shapes for count
    assert jk.count_neighbors(tree, queries[0], radius[0]).shape == ()
    assert jk.count_neighbors(tree, queries[0], radius).shape == (radius.shape[0],)
    assert jk.count_neighbors(tree, queries, radius[0]).shape == (queries.shape[0],)
    assert jk.count_neighbors(tree, queries, radius).shape == (queries.shape[0], radius.shape[0])


def test_small_case():
    points = jnp.array([
        [10, 46, 68, 40, 25, 15, 44, 45, 62, 53],
        [15, 63, 21, 33, 54, 43, 58, 40, 69, 67],
    ]).T

    tree = jk.build_tree(points)
    assert jnp.all(tree.points == points)
    assert jnp.all(tree.indices == jnp.array([1, 5, 9, 3, 6, 2, 8, 0, 7, 4]))
    assert jnp.all(tree.split_dims == jnp.array([ 0,  1,  1,  0,  0, -1, -1, -1, -1, -1]))


def test_full_traversal():
    kp, kq = jax.random.split(jax.random.key(83))
    points = jax.random.normal(kp, shape=(100, 3))
    queries = jax.random.normal(kq, shape=(100, 3))

    tree = jk.build_tree(points)
    neighbors, distances = jk.query_neighbors(tree, queries, 100)
    assert jnp.all(neighbors[:,-1] == jnp.array([
        3, 83,  3,  3, 53,  3,  3, 83,  3,  3, 53, 53, 34, 83,  3, 34,  3,
       34,  3, 34, 34, 83, 53, 75,  3, 83, 53, 80, 53, 53, 83, 53, 75, 83,
       53,  3, 83, 34, 53,  3, 53, 34, 83, 34, 53, 53, 34, 80, 83,  3, 25,
       34, 34, 34,  3,  3, 83,  3, 53, 83, 99, 34, 25, 75, 53, 83, 10, 99,
       34,  3, 83,  3, 53, 53,  3, 34, 53, 53, 34, 53, 34, 75, 83,  3, 80,
       83, 34,  3,  3, 80, 25, 34, 83, 53, 34, 75,  3, 53, 75, 34
    ]))

    counts = jk.count_neighbors(tree, queries, jnp.inf)
    assert jnp.all(counts == 100)


def test_random():
    kp, kq = jax.random.split(jax.random.key(83))
    points = jax.random.normal(kp, shape=(10_000, 3))
    queries = jax.random.normal(kq, shape=(1_000, 3))

    tree = jk.build_tree(points)
    neighbors, distances = jk.query_neighbors(tree, queries, 10)
    counts = jk.count_neighbors(tree, queries, 0.3)
    
    assert neighbors[0,0] == 5447
    assert jnp.allclose(distances[0,0], 0.16683109, atol=1e-5)
    assert counts[0] == 12

    assert neighbors[500, 9] == 7587
    assert jnp.allclose(distances[500, 9], 0.1562879, atol=1e-5)
    assert counts[500] == 50
    
def test_not_optimized():
    kp, kq = jax.random.split(jax.random.key(83))
    points = jax.random.normal(kp, shape=(1_000, 3))
    queries = jax.random.normal(kq, shape=(100, 3))

    tree = jk.build_tree(points, optimize=False)
    neighbors, distances = jk.query_neighbors(tree, queries, 10)
    counts = jk.count_neighbors(tree, queries, 0.3)

    assert neighbors[0,0] == 549
    assert jnp.allclose(distances[0,0], 0.2752785, atol=1e-5)
    assert counts[1] == 11