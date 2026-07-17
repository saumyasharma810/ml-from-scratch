import math
from collections import deque
# Class unit with queue and indegree for backprop
class unit():
    def __init__(self, data, children = (), symbol= ''):
        self.data = data
        self.grad = 0.0
        self._backprop =  lambda: None
        self.children = children
        self.symbol = symbol
        self.ref_count = 0

    def __add__(self, other):
        if isinstance(other, (int,float)):
            other = unit(other, symbol = str(other))
        out = unit(self.data+other.data, [self, other], symbol = f'({self.symbol}+{other.symbol})')
        # adding ref_count for children
        self.ref_count += 1
        other.ref_count += 1
        def func():
            self.grad += out.grad * 1
            other.grad += out.grad * 1
        out._backprop = func
        return out

    def __mul__(self, other):
        if isinstance(other, (int,float)):
            other = unit(other, symbol = str(other))
        out = unit(self.data*other.data, [self, other], symbol = f'({self.symbol}*{other.symbol})')
        self.ref_count += 1
        other.ref_count += 1
        def func():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backprop = func
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __pow__(self, other):
        if not isinstance(other,int):
            raise ValueError("power must be an int")
        value = self.data
        out = unit(value**other, [self], symbol = f'({self.symbol}^{other})')
        self.ref_count += 1
        def func():
            self.grad += other * (value ** (other-1)) * out.grad
        out._backprop = func
        return out

    def __truediv__(self, other):
        return self * (other**-1)
        
    # For adding 2.0 + unit(-3.0)
    def __radd__(self, other):
        return self + other

    # For multiplying 2.0 * unit(-3.0)
    def __rmul__(self, other):
        return self * other

    # Non Linear functions - tanh, exp, ReLU
    def exp(self):
        e = math.exp(self.data)
        out = unit(e, [self], symbol = f'(e^{self.symbol})')
        self.ref_count += 1
        def func():
            self.grad += e * out.grad
        out._backprop = func
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = unit(t, [self], symbol = f'(tanh({self.symbol}))')
        self.ref_count += 1
        def func():
            self.grad += (1 - t**2) * out.grad
        out._backprop = func
        return out

    def ReLU(self):
        out = unit(self.data if self.data > 0 else 0, [self], symbol = f"(ReLU({self.symbol}))")
        self.ref_count += 1
        def func():
            self.grad += out.grad if self.data > 0 else 0
        out._backprop = func
        return out
        
        

    def __repr__(self):
        return f"Symbol: {self.symbol}, Data: {self.data}, Grad: {self.grad}, Ref count: {self.ref_count}, children:\n{self.children}  \n"

    def backward(self):
        self.grad = 1.0
        queue = deque()
        queue.append(self)
        while queue:
            node = queue.popleft()
            node._backprop()
            for child in node.children:
                child.ref_count -= 1
                if child.ref_count == 0:
                    queue.append(child)
        
        
            