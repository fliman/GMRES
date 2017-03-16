import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt

def oneQR(A):
	size=A.shape
	print size
	if(size[1] == 1):
		x = A[:,0]
		y = 0.0 *x
		y[0] = lin.norm(x)
		u = x-y
		v= u/lin.norm(u)
		print v.reshape(len(x),1) * v.reshape(1,len(x))
		Q=np.eye(len(x))-2.0*v.reshape(len(x),1)*v.reshape(1,len(x))

	return Q, Q.dot(A[:,0])

def updateQR(Qold, Rold, H):
	return Qold, Rold

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
		
		#update QR
		if(n == 0):
			Q,R = oneQR(h[0:n+2,0:1])
		else:
			Q,R = updateQR(Q, R, h[0:n+2,0:n+1])	
		#backward solve y	
	return 1.0
	


a=np.zeros((5,5))

i,j=np.indices(a.shape)


a[i==j] = 2
a[i-1==j] = -1
a[i==j-1] = -1


x = np.array([1, 2, 3 ,4, 5])
b = np.array([0, 0, 0, 0, 6])
gmres(a, b, 10)

# test=np.array([[2.0],[1.0],[1.0]])
# print updataQR(test)


