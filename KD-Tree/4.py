"""
Author: michealowen
Last edited: 2019.7.15,Monday
"""
a = 5

class test:
    def f(self):
        global a
        print(a)
        a += 1
        return 
    
    def main(self): 
        self.f()
        self.f()
        return

t = test()
t.main()
