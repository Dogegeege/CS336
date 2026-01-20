class Test:
    def __init__(self):
        print("hello")

    def __call__(self, *args, **kwds):
        print("this is call")


class ClassFunc:
    def __init__(self):
        self.test = Test()

    def f(self):
        self.test()


rest = ClassFunc()

rest.f()
