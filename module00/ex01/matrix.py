import numpy as np
from vector import Vector
class Matrix:
    def __init__(self, data):
        print("Matrix constructor called")
        if not isinstance(data, (list, tuple)):
            print ("Matrix must be a tuple or list")
            return
        self.data = []
        self.shape = [0, 0]
        for i in data:
            if len(i) != len(data[0]):
                print("Matrix of differente dimensions")
                self.data = []
                return
            self.data.append(i)
        self.shape = [len(data), len(i)]
    def __str__(self):
        return (f"data -> {self.data} and shape -> {self.shape}")
    def __repr__(self):
        return (f"MATRIX(data -> {self.data} and shape -> {self.shape})")

    def __add__(self, other):
        if not isinstance(other, Matrix) or other.shape != self.shape:
            print ("Incorret argument type, it must be Matrix")
            return
        add = []
        for i in range(self.shape[0]):
            nl = []
            for j in range(self.shape[1]):
                nl.append(self.data[i][j] + other.data[i][j])
            add.append(nl)
        return (add)

    def _radd__(self, other):
        return (self.__add__(other))
    
    def __sub__(self, other):
        if not isinstance(other, Matrix) or other.shape != self.shape:
            print ("Incorret argument type, it must be Matrix")
            return
        sub = []
        for i in range(self.shape[0]):
            ret = []
            ret += [self.data[i][j] - other.data[i][j] for j in range(self.shape[1])]
            sub.append(ret)
        return (sub)

    def __rsub__(self, other):
        return (self.__sub__(other))

    def __truediv__(self, other):
        if not isinstance(other, int) or other <= 0:
            print ("Incorret argument type, it must be Matrix")
            return
        true = []
        true += [[self.data[i][j] / other for j in range(self.shape[1])] for i in range(self.shape[0])]
        return true
    def __rtruediv__(self, other):
        return (self.__truediv__(other))
    def __mul__(self, other):
        if isinstance(other, int):
            mul = []
            mul += [[self.data[i][j] * other for j in range(self.shape[1])] for i in range(self.shape[0])] 
        elif isinstance(other, Vector) and self.shape[1] == other.size:
            mul = []
            for i in range(self.shape[0]):
                nl = []
                suma = 0
                print(i)
                for j in range(self.shape[1]):
                    suma += self.data[i][j] * other.values[j]
                nl.append(suma)
                mul.append(nl) 
        elif isinstance(other, Matrix) and self.shape[0] == other.shape[1]:
            mul = []
            for i in range(self.shape[0]):
                nl = []
                for j in range(other.shape[1]):
                    suma = 0
                    for k in range(self.shape[1]):
                        suma += self.data[i][k] * other.data[k][j]
                    nl.append(suma)
                mul.append(nl)
        else:
            print ("Incorret argument type or shape")
            return
        return (mul)


if __name__ == "__main__":
    m1 = Matrix([[2, 3], [3, 4], [5, 0]])
    m2 = Matrix([[0, 100, 3], [0, 103, -1]])
    m3 = m1 - m2
    #Rtrue
    m4 = m2 / 2
    #True
    m4 = 2 / m2
    #Div Matrix
    m5 = m1 * m2
    #Dix scalar
    m6 = m1 * 0
    #Div Vector
    v1 = Vector([1, 4, 7])
    print(f"shape -> {m2.shape} amd vi -> {v1.size}")
    m7 = m2 * v1
    print(m2)
    print(f"M5:\n{m5}")
    print(f"M6:\n{m6}")
    print(f"M7:\n{m7}")
    
    
    mno = Matrix("lkjca")
