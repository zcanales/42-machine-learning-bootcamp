class Vector:
    def __init__(self, values):
        if not isinstance(values, (int, list, tuple)):
            print ("Values must be [int], [list] or [tuple]")
            return 
        if isinstance (values, int) and values < 0:
            print("Size of range cannot be negative")
            return
        if isinstance(values, int):
            self.values = [float(nb) for nb in range(0, values, 1)]
            self.size = values
        elif isinstance(values, tuple) and len(values) == 2:
            self.values = [float(nb) for nb in range(values[0], values[1], 1)]
            self.size = values[1] - values[0]
        elif isinstance(values, list):
            self.values = values.copy()
            self.size = len(values)
    def __str__(self):
        return "{}".format(self.values)
    def __repr__(self):
        txt = str(self.values) + " : " + str(self.shape)
        return 
    def __add__(self, other_vector): 
        if not isinstance(other_vector, Vector):
            print("Invalid argument to add must be Vector")
            return 
        ret = []
        if other_vector.size == self.size:
            ret += ([self.values[i] + other_vector.values[i]  for i in range(self.size) ])
            return (ret)
        return 
    def __radd__(self, other):
        return (self.__add__(other))

    def __sub__(self, other):
       
        if not isinstance(other, Vector) or other.size != self.size:
            print ("Invalid argument to add must be Vector and with the same size")
            return 
        ret = []
        ret += ([self.values[i] - other.values[i] for i in range(self.size)])
        return ret

    def __rsub__(self, other):
        return (other.sub(self))

    def __truediv__(self, other):
        if not isinstance(other, int) or other == 0:
            print ("Invalid argument DIV > 0")
            return 
        return [self.values[i] / other for i in range(self.size)]
    
    def __rtruediv__(self, other):
        return (self.__truediv__(other))    
    
    def __mul__(self, other):
        if not isinstance(other, Vector) or other.size != self.size:
            pritn ("Invalid argument to add must be Vector and with the same size")
        ret = []
        for i in range(self.size):
            if other.values[i] == 0:
                print("Imposible division 0")
                return
            ret += [self.values[i] * other.values[i]]
        return (ret)
    def __rmul__(self, other):
        if not isinstance(other, Vector) or other.size != self.size:
            print("Invalid argument to add must be Vector and with the same size")
            return
        ret = []
        for i in range(self.size):
            if self.values[i] == 0:
                print("Imposible division 0")
                return
            ret += [self.values[i] * other.values[i]]
        return (ret)
        

        
if __name__ == "__main__":
    v = Vector([1.0, 1.0, 2.5, 3.4])
    w = Vector((1, 5))
    print("Beginning")
    print(str(v))
    print(str(w))
    print("Suma")
    print(w + v)
    print("Resta")
    print(f"Size {v.size}")
    print(v / 2)
    print(v * w)

    

