import numpy as np


def cat(blocks):
    """
    Concatenates several arrays (blocks) into a single vector.

    Parameters
    ----------
    blocks: iterable yielding array-like
        The blocks to concatenate.
        
    Output
    ------
    cat: array-like
        The concatenated vector.
    """
    return np.concatenate([np.array(block, copy=False).reshape(-1)
                           for block in blocks])


def decat(vec, block_idxs, shapes):
    """
    Deconcatenates a vector of blocks back into their original shapes. Generator yielding the blocks from a concateanted vector (e.g. the output of cat()).

    Parameters
    ----------
    vec: array-like
        The concatenated vector.

    block_idxs: list of tuples
        The left and right indices of the blocks in the concatenated vector.

    shapes: list of tuples
        Shapes of the original blocks.

    Yields
    ------
    block: array-like
        The block reshaped back to its original shape.
    """
    
    for b, (left_idx, right_idx) in enumerate(block_idxs):
        yield vec[left_idx:right_idx].reshape(shapes[b])


def get_cat_block_info(blocks):
    """
    Gets information for de-concatenating a concatenated vector of blocks
    
    Parameters
    ----------
    blocks: list of array-like
        The blocks.
        
    Output
    ------
    block_idxs, shapes
    """
    shapes = [block.shape for block in blocks]
    sizes = [np.prod(shape) for shape in shapes]
    
    block_idxs = get_cat_block_idxs(sizes)
    
    return block_idxs, shapes


def get_cat_block_idxs(sizes):
    """
    Gets the left/right indices for each block in the concatenated vector.
    
    Parameters
    ----------
    sizes: iterable of ints
        The sizes of each block.
    
    Output
    ------
    block_idxs: list of tuples
        The left/right indices.
    """
    block_idxs = []
    idx_left = 0
    idx_right = 0
    for size in sizes:
        idx_right += size
        block_idxs.append((idx_left, idx_right))
        idx_left += size
        
    return block_idxs


def cat_with_info(blocks):
    """
    Concatenates a list of blocks and returns the info needed to deconcatenate the blocks.
    
    Parameters
    ----------
    blocks: list of array-like
        The blocks to concatenate.
        
    Ouptut
    -------
    vec, block_idxs, shapes
    """
    block_idxs, shapes = get_cat_block_info(blocks)
    return cat(blocks), block_idxs, shapes
     
