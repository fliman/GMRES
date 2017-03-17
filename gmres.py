import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt

def oneQR(A):
	size=A.shape
	if(size[1] == 1):
		x = A[:,0]
		y = 0.0 *x
		y[0] = lin.norm(x)
		u = x-y
		v= u/lin.norm(u)
		Q=np.eye(len(x))-2.0*v.reshape(len(x),1)*v.reshape(1,len(x))

	return Q, Q.dot(A[:,0]).reshape(size[0],1)

def updateQR(Qold, Rold, H):
	size=H.shape
	size1=Qold.shape
	#print size1
	n = size[1]-1
	
	r= Qold.dot(H[0:n+1, n])
	a=r[len(r)-1]
	b=H[n+1, n] 
	c=a/np.sqrt(a*a+b*b)
	s=b/np.sqrt(a*a+b*b)
	Qnew= np.eye(size1[0]+1)
	Qnew[0:n+1, 0:n+1] = Qold
	G=np.eye(n+2)
	G[n:n+2,n:n+2]=np.array([[c,s],[-s,c]])
	Qnew = np.matmul(G, Qnew)
	
	Rnew = Qnew.dot(H)
	return Qnew, Rnew

def BackTriangle(A, b):
	n = len(b)
	y = np.zeros(b.shape)
	for i in range(n-1,-1,-1):
		y[i] = b[i]
		for j in range(i+1, n):
			y[i] = y[i] - A[i,j] * y[j]
		y[i] = y[i] / A[i, i]
	return y	


#GMRES with restart
def gmres(A, b, x0, maxn):
	h=np.zeros((maxn+1,maxn))
	q=np.zeros((maxn+1, len(b)))
	q[0] = b/lin.norm(b)
	stop = False
	for n in range(maxn):
		#arnoldi	
		v=A.dot(q[n])
		for j in range(n+1):
			temp = q[j]
			h[j][n] = temp.dot(v)
			v = v - h[j][n]*q[j]

		h[n+1][n] = lin.norm(v)	

		if(np.abs(h[n+1][n]) < 1e-15):
			stop = True
		else:
			q[n+1] = v / h[n+1][n]	

		##Solving least square problem via updating QR	
		#update QR
		if(n == 0):
			Q,R = oneQR(h[0:n+2,0:n+1])
		else:
			Q,R = updateQR(Q, R, h[0:n+2,0:n+1])

		#backward solve
		
		y=BackTriangle(R[0:n+1,0:n+1], lin.norm(b)*Q[0:n+1,0])
		x=np.matmul(q[0:n+1,:].transpose(),y)
		if(stop): break
	return x0+x
	


a=np.zeros((11,11))

i,j=np.indices(a.shape)


a[i==j] = 3
a[i-1==j] = -1
a[i==j-1] = -1


x = np.array([1, 2, 3 ,2, 5, 6, 7, 8, 8, 7, 1])
b = np.matmul(a,x)
x0=gmres(a, b, x*0.0, 4)
print x0
x1=gmres(a, b-np.matmul(a,x0), x0, 4)
print x1
x2=gmres(a, b-np.matmul(a,x1), x1, 4)
print x2
# 

# test=np.array([[2.0],[1.0],[1.0]])
# print updataQR(test)


