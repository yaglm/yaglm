from copy import deepcopy


class ConstraintConfig:

    def __init__(self, pos=False):
        self.pos = pos

    def get_solve_kws(self):
        return deepcopy(self.__dict__)

    def validate(self): pass
