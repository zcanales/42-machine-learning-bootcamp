


class TinyStatistician:
    def mean(self, x):
        total = 0
        if not isinstance(x, list) or len(x) == 0:
            return None
        for i in x:
            total += i
        return round(total / len(x), 2)
   
    def median(self, x):
        if not isinstance(x, list) or len(x) == 0:
            return None
        x.sort()
        l = len(x)
        if len(x) % 2 != 0:
            l_medi = int(l / 2 - 0.5)
            return float(x[l_medi])
        l_medi = int(l / 2)
        median = x[l_medi] + x[l_medi - 1] / 2
        return float(median)
    def quartiles(self, x, percentile):
        if not isinstance(x, list) or len(x) == 0:
            return None
        
        if percentile != 25 and percentile != 75:
            return None
        x.sort()
        index = int(len(x) * percentile / 100)
        quartil = float(x[index])
        return quartil
    def var(self, x):
        if not isinstance(x, list) or len(x) == 0:
            return None
        mean_res = self.mean(x)
        var = 0
        for i in x:
            var += (i - mean_res)**2
        var = var / len(x)
        return var
    def std(self, x):
        if not isinstance(x, list) or len(x) == 0:
            return None
        var = self.var(lst)
        if var != 0:
            std = var**0.5
        return std


if __name__ == "__main__":
    a = TinyStatistician()
    lst = [1, 42, 300, 10, 59]
    print(lst)
    print(f"mena -> {a.mean(lst)}")
    print(f"median -> {a.median(lst)}")
    print(f"quartiles 25 -> {a.quartiles(lst, 25)}")
    print(f"quartiles 75-> {a.quartiles(lst, 75)}")
    print(f"var-> {a.var(lst)}")
    print(f"std-> {a.std(lst)}")