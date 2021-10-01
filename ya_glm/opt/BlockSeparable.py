import numpy as np
from ya_glm.opt.base import Func

# TODO: this currently assumes the union of the groups gives the
# entire input vector. Get this working when the groups may be a strict subset
# e.g. zero pad the gradient
class BlockSeparable(Func):
    """
    Represents a function of the form
    
    F(x) = sum_b f_b(x_b)
    
    where x_b denotes the bth block of x and the blocks are non-overlapping.

    This object assumes the blocks blocks are groups of indices along the first axis (i.e. a concatenation with a possible shuffling).

    Parameters
    ----------
    funcs: list of Func
        The functions to apply to each block.

    groups: list of lists
        The indices of each block. Assumes the blocks are concatenated along the first axis of the input.
    """
    def __init__(self, funcs, groups):
        self.funcs = funcs
        self.groups = groups

        # takes the consecutive concatenation of the blocks and maps
        # them back to the proper order
        self.sort_idxs = np.argsort(np.concatenate(self.groups))

    def decat(self, x):
        """
        Generator yielding the blocks

        Parameters
        ----------
        x: array-like
            The concatenated blocks.

        Yields
        ------
        block: array-like
            Each block
        """
        for grp_idxs in self.groups:
            yield x[grp_idxs]

    def cat(self, blocks):
        """
        Concatenates the blocks into a vector.

        Parameters
        ----------
        blocks: iterable
            The blocks.

        Output
        ------
        cat: array-like
            The concatenated blocks
        """
        return np.concatenate(list(blocks))[self.sort_idxs]

    def _eval(self, x):
        # de-concatenate
        split_blocks = self.decat(x)
            
        # sum eval of each block
        return sum(self.funcs[b]._eval(block)
                   for b, block in enumerate(split_blocks))

    def _grad(self, x):
        # de-concatenate
        split_blocks = self.decat(x)
        
        # concatenate grad of each block
        return self.cat(self.funcs[b]._grad(block)
                        for b, block in enumerate(split_blocks))

    def _prox(self, x, step=1):
        # de-concatenate
        split_blocks = self.decat(x)
        
        # concatenate prox of each block
        return self.cat(self.funcs[b]._prox(block, step=step)
                        for b, block in enumerate(split_blocks))

    @property
    def is_smooth(self):
        # smooth if everyone is smooth
        return all(f.is_smooth for f in self.funcs)

    @property
    def is_proximable(self):
        # proximable if everyone is proximable
        return all(f.is_proximable for f in self.funcs)

    @property
    def grad_lip(self):
        lips = [f.grad_lip for f in self.funcs]
        if any(lip is None for lip in lips):
            return None
        else:
            # L-smoothness constant given by largest eval of hessian
            return max(lips)
