import numpy as np

# x = np.zeros((2, 3))
# print(x)
# print(x.dtype)
#
# x = np.ones((2, 3), dtype=int)
# print(x)
# print(x.dtype)
#
# x = np.full((2, 3), 5)
# print(x)
# print(x.dtype)
#
# x = np.eye(5, dtype=int)
# print(x)
# print(x.dtype)
#
# x = np.diag([1, 2, 3, 4, 5])
# print(x)
# print(x.dtype)
#
# x = np.arange(20)
# print(x)
#
# x = np.linspace(0, 25, 10)
# print(x)

# y = np.reshape(x, (4, 5))
# print(y)

# x = np.random.random((3, 3))
# print(x)
#
# x = np.random.randint(4, 15, (3, 3))
# print(x)

# x = np.array([1, 2, 3, 4, 5])
# x = np.delete(x, [0, 4])
# print(x)

# x = np.arange(9).reshape(3, 3)
# y = np.delete(x, 0, axis=0)
# w = np.delete(x, [0, 2], axis=1)
# print(y)
# print('-----')
# print(w)

# x = np.array([1, 2])
# y = np.array([[3, 4], [5, 6]])
#
# w = np.vstack((x, y))
# print(w)
#
# v = np.hstack((y, x.reshape(2, 1)))
# print(v)

# x = np.arange(1, 31).reshape((5, 6))
# z = np.diag()
# print(z)

# x = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
# z = np.unique(x)
# print(z)

# x = np.arange(25)
# x[x % 2 == 0] = 0
# print(x)

# x = np.random.randint(10, 20, 10)
# y = np.random.randint(10, 20, 10)
# print(x)
# print(y)
# print(np.intersect1d(x, y))
# print(np.union1d(x, y))
# print(np.setdiff1d(x, y))
# print(np.setdiff1d(y, x))

# x = np.random.randint(10, 20, size=(5, 5))
# print(x)
# print(np.sort(x, axis=0))
# print(np.sort(x, axis=1))

# # We create a 2 x 2 ndarray
# X = np.array([[1,2], [3,4]])
#
# # We print x
# print()
# print('X = \n', X)
# print()
#
# print('3  *X = \n', 3*  X)
# print()
# print('3 + X = \n', 3 + X)
# print()
# print('X - 3 = \n', X - 3)
# print()
# print('X / 3 = \n', X / 3)
#
# # We create a rank 1 ndarray
# x = np.array([1,2,3])
#
# # We create a 3 x 3 ndarray
# Y = np.array([[1,2,3],[4,5,6],[7,8,9]])
#
# # We create a 3 x 1 ndarray
# Z = np.array([1,2,3]).reshape(3,1)
#
# # We print x
# print()
# print('x = ', x)
# print()
#
# # We print Y
# print()
# print('Y = \n', Y)
# print()
#
# # We print Z
# print()
# print('Z = \n', Z)
# print()
#
# print('x + Y = \n', x + Y)
# print()
# print('Z + Y = \n',Z + Y)

x = np.random.randint(0, 5001, size=(1000, 20))
print(type(np.size(x)))
