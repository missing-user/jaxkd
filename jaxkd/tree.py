import jax
import jax.numpy as jnp
from jax.tree_util import Partial

@jax.jit
def make_kd_tree(points):
    ''' Build a balanced binary kd-tree from points, splitting along the dimension with largest range for each node.
    
    Args:
        points: (2**l, d) array of points
        
    Returns:
        leaf_indices: (2**l,) indices of points as leaves in the kd-tree
        split_values: (2**l-1,) split value for each internal node in the kd-tree
        split_dims: (2**l-1,) split dimension for each each internal node in the kd-tree
    '''
    # Initialize tree arrays
    n_levels = len(points).bit_length() - 1
    indices = jnp.arange(len(points))
    split_values = jnp.zeros(2**n_levels - 1, dtype=points.dtype)
    split_dims = jnp.zeros(2**n_levels - 1, dtype=int)
    
    for level in range(n_levels):
        # Determine optimal split dimension for current leaves
        split_dim = jnp.argmax(jnp.max(points[indices], axis=-2) - jnp.min(points[indices], axis=-2), axis=-1)
        # Partition points along the split dimension
        points_along_dim = jnp.take_along_axis(points[indices], split_dim[...,None,None], axis=-1).squeeze(-1)
        order = jnp.argsort(points_along_dim, axis=-1) # note: jnp.argpartition is not faster
        indices = jnp.take_along_axis(indices, order, axis=-1)
        indices = indices.reshape(*indices.shape[:-1], 2, -1)
        # Store the split value and dimension at the appropriate level
        medians = jnp.take_along_axis(points[indices[..., 0, -1]], split_dim[...,None], axis=-1).squeeze(-1)/2 + jnp.take_along_axis(points[indices[..., 1, 0]], split_dim[...,None], axis=-1).squeeze(-1)/2
        split_values = split_values.at[2**level-1:2**(level+1)-1].set(medians.ravel())
        split_dims = split_dims.at[2**level-1:2**(level+1)-1].set(split_dim.ravel())
    return indices.ravel(), split_values, split_dims


@Partial(jax.jit, static_argnames=('k',))
def query_neighbors(query, points, leaf_indices, split_values, split_dims, k=1):
    ''' Find the k nearest neighbors of a query point using a kd-tree.

    Args:
        query: (d,) query point
        points: (2**l, d) array of points
        leaf_indices: (2**l,) indices of points as leaves in the kd-tree
        split_values: (2**l-1,) split value for each internal node in the kd-tree
        split_dims: (2**l-1,) split dimension for each each internal node in the kd-tree
        k: number of neighbors to return

    Returns:
        neighbors: (k,) indices of the k nearest neighbors
        distances: (k,) distances to the k nearest neighbors
        n_visits: number of nodes visited in the kd-tree
    '''
    # Initialize neighbor arrays and node pointers
    curr = 0
    prev = -1
    neighbors = jnp.zeros(k, dtype=int)
    square_distances = jnp.inf * jnp.ones(k)
    max_depth = len(points).bit_length() - 1

    def step(carry):
        curr, prev, neighbors, square_distances = carry
        # If the node is a leaf, check if we need to update neighbors (note: no performance gain from doing this conditionally)
        depth = jnp.floor(jnp.log2(curr + 1)).astype(int)
        leaf = curr - (2**depth - 1)
        leaf_square_distance = jnp.sum(jnp.square(points[leaf_indices[leaf]] - query), axis=-1)
        max_neighbor = jnp.argmax(square_distances)
        replace = (depth == max_depth) & (leaf_square_distance < square_distances[max_neighbor])
        neighbors = jax.lax.select(replace, neighbors.at[max_neighbor].set(leaf_indices[leaf]), neighbors)
        square_distances = jax.lax.select(replace, square_distances.at[max_neighbor].set(leaf_square_distance), square_distances)

        # Locate children and determine if far child is in range
        split_distance = query[split_dims[curr]] - split_values[curr]
        near_side = (split_distance > 0).astype(int)
        near_child = 2*curr + 1 + near_side
        far_child = 2*curr + 2 - near_side
        far_in_range = (split_distance**2 <= jnp.max(square_distances))

        # Determine next node to traverse
        parent = (curr - 1) // 2
        next = jax.lax.select(
            prev == parent, # if we just came from the parent
            jax.lax.select(depth < max_depth, near_child, parent), # go to near child if internal node, or back to parent if leaf
            jax.lax.select(
                prev == near_child, # if we just came from the near child
                jax.lax.select(far_in_range, far_child, parent), # go to far child if in range, or back to parent if not
                parent, # if we just came from the far child, go back to parent
            )
        )
        return next, curr, neighbors, square_distances

    # Loop until we return to root
    _, _, neighbors, square_distances = jax.lax.while_loop(lambda carry: carry[0] >= 0, step, (curr, prev, neighbors, square_distances))
    order = jnp.argsort(square_distances)
    return neighbors[order], jnp.sqrt(square_distances[order])


def tree_path_to_index(path):
    ''' Convert a binary path to a tree index. '''
    path = jnp.array(path)
    depth = len(path)
    powers = 1 << jnp.arange(depth)[::-1]
    offset = (1 << depth) - 1
    local_index = jnp.sum(path * powers)
    return offset + local_index

def tree_index_to_path(index):
    ''' Convert a tree index to a binary path. '''
    depth = jnp.floor(jnp.log2(index + 1)).astype(int)
    steps = jnp.arange(depth)[::-1]
    path = ((index + 1) >> steps) & 1
    return path
