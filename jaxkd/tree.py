import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import Partial
from collections import namedtuple

tree_type = namedtuple('tree', ['points', 'indices', 'split_dims'])

@Partial(jax.jit, static_argnames=('optimized',))
def build_tree(points, optimized=True):
    """
    Build a k-d tree from points.

    Follows <https://arxiv.org/abs/2211.00120>.
    See also <https://github.com/ingowald/cudaKDTree>.
    
    Args:
        points: (N, d)
        optimized: If True (default), split along dimension with the largest range. This typically leads to faster queries. If False, cycle through dimensions in order.
        
    Returns:
        tree (namedtuple)
            - points: (N, d) Same points as input, not copied.
            - indices: (N,) Indices of points in binary tree order.
            - split_dims: (N,) Splitting dimension of each tree node, marked -1 for leaves. If `optimized=False` this is set to None.
    """
    if points.ndim != 2: raise ValueError(f'Points must have shape (N, d). Got shape {points.shape}.')
    return _build_tree_nojit(points, optimized=optimized)


def _build_tree_nojit(points, optimized=True):
    """ Non-jitted underlying implementation of `build_tree`, rarely worth it even if only building the tree once. """
    n_points = len(points)
    n_levels = n_points.bit_length()

    def step(carry, level):
        nodes, indices, split_dims = carry

        # Sort the points in each node group along the splitting dimension, either optimized or cycling
        if optimized:
            dim_range = jax.ops.segment_max(points[indices], nodes, num_segments=n_points) - jax.ops.segment_min(points[indices], nodes, num_segments=n_points)
            split_dim = jnp.argmax(dim_range, axis=-1)[nodes].astype(jnp.int8)
            points_along_dim = jnp.take_along_axis(points[indices], split_dim[:, None], axis=-1).squeeze(axis=-1)
            nodes, _, indices, split_dim, split_dims = lax.sort((nodes, points_along_dim, indices, split_dim, split_dims), dimension=0, num_keys=2) # primary sort by node, secondary sort by points
        else:
            split_dim = level % points.shape[-1]
            points_along_dim = points[indices][:, split_dim]
            nodes, _, indices = lax.sort((nodes, points_along_dim, indices), dimension=0, num_keys=2) # primary sort by node, secondary sort by points

        # Compute the branch start index
        height = n_levels - level - 1
        n_left_siblings = nodes - ((1 << level) - 1) # nodes to the left at the same level
        branch_start = (
            (1 << level) - 1 # levels above
            + n_left_siblings * ((1 << height) - 1) # left sibling internal descendants
            + jnp.minimum(n_left_siblings * (1 << height), n_points - ((1 << (n_levels-1)) - 1)) # left sibling leaf descendants
        )

        # Compute the size of the left child branch
        left_child = 2 * nodes + 1
        child_height = jnp.maximum(0, height - 1)
        first_left_leaf = ~((~left_child) << child_height) # first leaf of the left child, cryptic but just descends 2i+1 several times
        left_branch_size = (
            (1 << child_height) - 1 # internal nodes
            + jnp.minimum(1 << child_height, jnp.maximum(0, n_points - first_left_leaf)) # leaf nodes
        )

        # Split branch about the pivot
        pivot_position = branch_start + left_branch_size
        array_index = jnp.arange(n_points)
        right_child = 2 * nodes + 2
        nodes = lax.select(
            (array_index == pivot_position) | (array_index < (1 << level) - 1), # if node is pivot or in upper part of tree, keep it
            nodes,
            lax.select(array_index < pivot_position, left_child, right_child) # otherwise, put as left or right child
        )

        # Update split dimension at pivot
        if optimized: split_dims = lax.select((array_index == pivot_position) & (left_child < n_points), split_dim, split_dims)
        return (nodes, indices, split_dims), None

    # Start all points at root and sort into tree at each level
    nodes = jnp.zeros(n_points, dtype=int)
    indices = jnp.arange(n_points)
    split_dims = -1 * jnp.ones(n_points, dtype=jnp.int8) if optimized else None # technically only need for internal nodes, but this makes sorting easier at the cost of memory
    (nodes, indices, split_dims), _ = lax.scan(step, (nodes, indices, split_dims), jnp.arange(n_levels)) # nodes should equal jnp.arange(n_points) at the end
    return tree_type(points, indices, split_dims)

@Partial(jax.jit, static_argnums=(2,))
def query_neighbors(tree, query, k):
    """
    Find the k nearest neighbors of query point(s) in a k-d tree.
    
    Follows <https://arxiv.org/abs/2210.12859>.
    See also <https://github.com/ingowald/cudaKDTree>.

    Args:
        tree (namedtuple): Output of `build_tree`.
            - points: (N, d) Points to search.
            - indices: (N,) Indices of points in binary tree order.
            - split_dims: (N,) Splitting dimension of each tree node, not used for leaves. If None, assume cycle through dimensions in order.
        query: (d,) or (Q, d) Query point(s).
        k (int): Number of neighbors to return.

    Returns:
        neighbors: (k,) or (Q, k) Indices of the k nearest neighbors of query point(s).
        distances: (k,) or (Q, k) Distances to the k nearest neighbors of query point(s).
    """
    if k > len(tree.points):
        raise ValueError(f'Cannot query {k} neighbors, tree contains only {len(tree.points)} points.')
    if len(tree.points) != len(tree.indices) or len(tree.points) != len(tree.split_dims):
        raise ValueError(f'Invalid tree, got len(points)={len(tree.points)}, len(indices)={len(tree.indices)}, len(split_dims)={len(tree.split_dims)}.')
    if query.ndim == 1: return _single_query(tree, query, k)
    if query.ndim == 2: return jax.vmap(lambda q: _single_query(tree, q, k))(query)
    raise ValueError(f'Query must have shape (Q, d) or (d,). Got shape {query.shape}.')


def _single_query(tree, query, k):
    """ Single query implementation, use `query_neighbors` wrapper instead. """

    # Initialize node pointers and neighbor arrays
    current = 0
    previous = -1
    neighbors = -1 * jnp.ones(k, dtype=int)
    square_distances = jnp.inf * jnp.ones(k)
    points, indices, split_dims = tree
    n_points = len(points)

    def step(carry):
        current, previous, neighbors, square_distances = carry
        level = jnp.log2(current + 1).astype(int)
        parent = (current - 1) // 2

        # Update neighbors with the current node if necessary
        square_distance = jnp.sum(jnp.square(points[indices[current]] - query), axis=-1)
        max_neighbor = jnp.argmax(square_distances)
        replace = (current < len(points)) & (previous == parent) & (square_distance < square_distances[max_neighbor])
        neighbors = lax.select(replace, neighbors.at[max_neighbor].set(indices[current]), neighbors)
        square_distances = lax.select(replace, square_distances.at[max_neighbor].set(square_distance), square_distances)

        # Locate children and determine if far child is in range
        split_dim = (level % points.shape[-1]) if split_dims is None else split_dims[current]
        split_distance = query[split_dim] - points[indices[current], split_dim]
        near_side = (split_distance > 0).astype(int)
        near_child = 2 * current + 1 + near_side
        far_child = 2 * current + 2 - near_side
        far_in_range = (split_distance**2 <= jnp.max(square_distances))

        # Determine next node to traverse
        next = lax.select(
            (previous == near_child) | ((previous == parent) & (near_child >= n_points)), # go to the far child if we came from near child or near child doesn't exist
            lax.select((far_child < n_points) & far_in_range, far_child, parent), # only go to the far child if it exists and is in range
            lax.select(previous == parent, near_child, parent) # go to the near child if we came from the parent
        )
        return next, current, neighbors, square_distances

    # Loop until we return to root
    _, _, neighbors, square_distances = lax.while_loop(lambda carry: carry[0] >= 0, step, (current, previous, neighbors, square_distances))
    distances = jnp.linalg.norm(points[indices[neighbors]] - query, axis=-1) # recompute distances to enable VJP
    order = jnp.argsort(distances)
    return neighbors[order], distances[order]