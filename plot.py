import matplotlib.pyplot as plt
import re


class solver(object):
    def __init__(self, path='D:\\1.txt'):
        with open(path, 'r') as f:
            self.txt = f.read()
        self.index = 0

    def get(self):
        while self.txt[self.index:self.index+13] != 'average loss:':
            if self.index + 13 > len(self.txt):
                return False
            self.index += 1
        loss = float(self.txt[self.index+13:self.index+23])
        self.index = self.index + 23
        return loss


def main():
    s = solver()
    r = []
    while True:
        t = s.get()
        if t:
            r.append(t)
        else:
            break

    plt.figure(1)
    plt.plot(range(len(r)), r)
    plt.xticks([])
    plt.show()

if __name__ == '__main__':
    main()
