import numpy as np

from ya_glm.opt.base import Func
from ya_glm.opt.cat_utils import cat, decat, get_cat_block_idxs


class BlockSeparable(Func):
    """
    Represents a function of the form
    
    F(x) = sum_b f_b(x_b)
    
    where x_b denotes the bth block of x and the blocks are non-overlapping.

    This object acts on the concatenated vector of all the blocks.

    Parameters
    ----------
    funcs: list of Func
        The functions to apply to each block.

    shapes: list of tuples
        The shapes of the original blocks.
    """
    
    def __init__(self, funcs, shapes):
        self.funcs = funcs
        self.shapes = shapes
        
        # number of elements in each block
        sizes = [np.prod(shape) for shape in shapes]
        self.block_idxs = get_cat_block_idxs(sizes)
        
    def _eval(self, x):
        # de-concatenate
        blocks = decat(vec=x, block_idxs=self.block_idxs, shapes=self.shapes)
            
        # sum eval of each block
        return sum(self.funcs[b]._eval(block)
                   for (b, block) in enumerate(blocks))

    def _grad(self, x):
        # de-concatenate
        blocks = decat(vec=x, block_idxs=self.block_idxs, shapes=self.shapes)
        
        # concatenate grad of each block
        return cat(self.funcs[b]._grad(block)
                   for (b, block) in enumerate(blocks))

    def _prox(self, x, step=1):
        # de-concatenate
        blocks = decat(vec=x, block_idxs=self.block_idxs, shapes=self.shapes)
        
        # concatenate prox of each block
        return cat(self.funcs[b]._prox(block, step=step)
                   for (b, block) in enumerate(blocks))
