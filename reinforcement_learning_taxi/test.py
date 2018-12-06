from collections import Counter
Q = [[1, 2, 3, 4], [1, 2, 2, 1], [0, 0, 0, 0], [0, 0, 1, 0], [-5, 0, 1, 10]]
Q_q = Q[4]
nA = 4
print(Q_q)
a = [Q_q[a] / sum(Q_q) if sum(Q_q) != 0 else 1 / len(Q_q) for a in range(nA)]
print(a)
print(sum(a))
