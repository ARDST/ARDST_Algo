from abc import ABC, abstractmethod

class Vtree(ABC):

    def __init__(self, index):
        self._index = index

    @property
    def index(self):
        return self._index

    @property
    def var_count(self):
        return self._var_count

    @abstractmethod
    def is_leaf(self):
        pass

    @staticmethod
    def read(file):
        with open(file, 'r') as vtree_file:
            line = 'c'
            while line[0] == 'c':
                line = vtree_file.readline()
            if line.strip().split(' ')[0] != 'vtree':
                raise ValueError('Number of vtree nodes is not specified')
            num_nodes = int(line.strip().split(' ')[1])
            nodes = [None] * num_nodes
            root = None
            for line in vtree_file.readlines():
                line_as_list = line.strip().split(' ')
                if line_as_list[0] == 'L':
                    root = VtreeLeaf(int(line_as_list[1]), int(line_as_list[2]))
                    nodes[int(line_as_list[1])] = root
                elif line_as_list[0] == 'I':
                    root = VtreeIntermediate(int(line_as_list[1]),
                                             nodes[int(line_as_list[2])], nodes[int(line_as_list[3])])
                    nodes[int(line_as_list[1])] = root
                else:
                    raise ValueError('Vtree node could only be L or I')
            return root


class VtreeLeaf(Vtree):

    def __init__(self, index, variable):
        super(VtreeLeaf, self).__init__(index)
        self._var = variable
        self._var_count = 1

    def is_leaf(self):
        return True

    @property
    def var(self):
        return self._var

    @property
    def variables(self):
        return set([self._var])


class VtreeIntermediate(Vtree):

    def __init__(self, index, left, right):
        super(VtreeIntermediate, self).__init__(index)
        self._left = left
        self._right = right
        self._variables = set()
        self._var_count = self._left.var_count + self._right.var_count
        self._variables.update(self._left.variables)
        self._variables.update(self._right.variables)

    def is_leaf(self):
        return False

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def variables(self):
        return self._variables

    @property
    def var(self):
        return 0
