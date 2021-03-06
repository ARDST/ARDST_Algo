import numpy as np


class CircuitNode(object):

    def __init__(self, index, vtree):
        self._index = index
        self._vtree = vtree
        self._num_parents = 0
        # difference between prob and feature:
        # prob is calculated in a bottom-up pass and only considers values of variables the node has
        # feature is calculated in a top-down pass using probs; equals the WMC of that node reached
        self._prob = None
        self._feature = None

    @property
    def index(self):
        return self._index

    @property
    def vtree(self):
        return self._vtree

    @property
    def num_parents(self):
        return self._num_parents

    def increase_num_parents_by_one(self):
        self._num_parents += 1

    def decrease_num_parents_by_one(self):
        self._num_parents -= 1

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, value):
        self._feature = value

    @property
    def prob(self):
        return self._prob

    @prob.setter
    def prob(self, value):
        self._prob = value


class OrGate(CircuitNode):
    """OR Gate.
       Or gates are also referred as Decision nodes."""

    def __init__(self, index, vtree, elements):
        super().__init__(index, vtree)
        self._elements = elements
        for element in elements:
            element.parent = self

    @property
    def elements(self):
        return self._elements

    def add_element(self, element):
        self._elements.append(element)
        element.parent = self

    def remove_element(self, index):
        del self._elements[index]

    def calculate_prob(self):
        if len(self._elements) == 0:
            raise ValueError("Decision nodes should have at least one elements.")
        for element in self._elements:
            element.calculate_prob()
        self._prob = np.sum([np.exp(element.prob) for element in self._elements], axis=0)
        self._prob = np.where(self._prob < 1e-5, 1e-5, self._prob)
        self._prob = np.log(self._prob)
        for element in self._elements:
            element.prob -= self._prob
        self._prob = np.where(self._prob > 0.0, 0.0, self._prob)
        self._feature = np.zeros(shape=self._prob.shape, dtype=np.float32)

    def calculate_feature(self):
        feature = np.log(self._feature)
        for element in self._elements:
            element.feature = np.exp(feature + element.prob)
            element.prime.feature += element.feature
            element.sub.feature += element.feature

    def save(self, f):
        f.write(f'D {self._index} {self._vtree.index} {len(self._elements)}')
        for element in self._elements:
            f.write(f' ({element.prime.index} {element.sub.index}')
            for parameter in element.parameter:
                f.write(f' {parameter}')
            f.write(f')')
        f.write('\n')


LITERAL_IS_TRUE = 1
LITERAL_IS_FALSE = 0

class CircuitTerminal(CircuitNode):
    """Terminal(leaf) node."""

    def __init__(self, index, vtree, var_index, var_value, parameter=None):
        super().__init__(index, vtree)
        self._var_index = var_index
        self._var_value = var_value
        self._parameter = parameter

    @property
    def var_index(self):
        return self._var_index

    @var_index.setter
    def var_index(self, value):
        self._var_index = value

    @property
    def var_value(self):
        return self._var_value

    @var_value.setter
    def var_value(self, value):
        self._var_value = value

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        self._parameter = value

    def calculate_prob(self, samples: np.array):
        if self._var_value == LITERAL_IS_TRUE:
            self._prob = np.log(samples[:, self._var_index - 1])
        elif self._var_value == LITERAL_IS_FALSE:
            self._prob = np.log(1.0 - samples[:, self._var_index - 1])
        else:
            raise ValueError('Terminal nodes should either be positive literals or negative literals.')
        self._feature = np.zeros(shape=self._prob.shape, dtype=np.float32)

    def save(self, f):
        if self._var_value == LITERAL_IS_TRUE:
            f.write(f'T {self._index} {self._vtree.index} {self._var_index}')
        elif self._var_value == LITERAL_IS_FALSE:
            f.write(f'F {self._index} {self._vtree.index} {self._var_index}')
        else:
            raise ValueError('Currently we only support terminal nodes that are either positive or negative literals.')
        for parameter in self._parameter:
            f.write(f' {parameter}')
        f.write('\n')
