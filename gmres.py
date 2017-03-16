import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt

def updataQR(A):
	size=A.shape
	if(size[1] == 1):
		x = A[:,0]
		y = 0.0 *x
		y[0] = lin.norm(x)
		u = x-y
		v= u/lin.norm(u)
		print v.reshape(2,1) * v.reshape(1,2)
		Q=np.eye(2)-2.0*v.reshape(2,1)*v.reshape(1,2)

	return Q, Q.dot(A[:,0])


def gmres(A, b, maxn):
	h=np.zeros(A.shape)
	q=np.zeros((maxn, len(b)))
	q[0] = b/lin.norm(b)
	for n in range(4):
		#arnoldi
		
		v=A.dot(q[n])
		print "iter",n, v
		for j in range(n+1):
			temp = q[j]
			h[j][n] = temp.dot(v)
			v = v - h[j][n]*q[j]
		h[n+1][n] = lin.norm(v)	
		q[n+1] = v / h[n+1][n]
		#backward solve
	return 1.0
	


a=np.zeros((5,5))

i,j=np.indices(a.shape)


a[i==j] = 2
a[i-1==j] = -1
a[i==j-1] = -1


x = np.array([1, 2, 3 ,4, 5])
b = np.array([0, 0, 0, 0, 6])
#gmres(a, b, 10)

test=np.array([[2.0],[1.0]])
print updataQR(test)


