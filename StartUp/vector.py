class Vector(object):
    """
    try oop
    """
    a = 0
    b = 0

    def __init__(self, a=0, b=0):
        self.a = a
        self.b = b

    def __copy__(self, other):
        self.a = other.a
        self.b = other.b

    def __le__(self, other):
        if self.a < other.a:
            return True
        elif self.a == other.a:
            if self.b < other.b:
                return True
            else:
                return False
        else:
            return False

    def __ge__(self, other):
        return not self.__le__(other)

    def __add__(self, other):
        return Vector(self.a + other.a, self.b + other.b)

    def __sub__(self, other):
        return Vector(self.a - other.a, self.b - other.b)

    def __str__(self):
        return '(%d, %d)' % (self.a, self.b)
