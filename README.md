Minimal JAX implementation of KNN using a KD tree.

Compatible with JIT. Uses this algorithm to traverse the tree: https://arxiv.org/abs/2210.12859. Not very fast, but can compete with `scipy.spatial.KDTree` when executed on GPU. Main use case is to avoid leaving the GPU and calling external libraries when nearest neighbor search is not the primary computational load.