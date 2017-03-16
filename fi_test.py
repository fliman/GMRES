#fourier interpolation
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
# x in [0, 1]
def fx(x):
	return np.sin(x*x*2*np.pi+x*2*np.pi)

# 
def dft(x, f, n):
	ff = 1j*np.linspace(0.0, 1.0, n)
	for i in range(n):
		phase = -np.complex(0,2)*np.pi*float(i)
		ff[i] = np.complex(0,0)
		for j in range(len(x)):
			if(j == len(x) - 1):
				w=1.0 - x[j]
			else:	
				w=(x[j+1] - x[j])
			
			w = 1.0
			ff[i] = ff[i]+ w*f[j] * np.exp(x[j]*phase)	 

	return  ff

def idft(x, ff):
	f = np.linspace(0.0, 1.0, len(x))
	for j in range(len(x)):	
		f[j] = 0.0
		for i in range(len(ff)):
			phase = -np.complex(0,2)*np.pi*float(i)
			if(j == len(x) - 1):
				w=1.0 - x[j]
			else:	
				w=(x[j+1] - x[j])
			
			w = 1.0
			f[j] = f[j]+ w*ff[i] * np.exp(-x[j]*phase)	 

	return  f

def scale(scal, y0):
	y1 = y0
	for i in range(len(scal)):
		if(np.abs(scal[i]) < 1e-15):
			y1[i] = 0.0
		else:
			y1[i] = y1[i] / scal[i]

	return y1			


nf = 20

xp=np.array([float(x)/nf for x in range(nf)])

x0=np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

y0=fx(x0)	
yp=fx(xp) 



fyp0=dft(x0, y0, len(xp))


scal = y0/idft(x0, fyp0)


print "scal1",scal


fyp1=dft(x0, y0*scal, len(xp))

print "scal2",y0/idft(x0, fyp1)



fyp = fft.fft(yp)




plt.figure()
plt.plot(xp, fft.ifft(fyp0),"r")
plt.plot(xp, fft.ifft(fyp1*20),"g")
plt.plot(xp, fft.ifft(fyp),"b")
#plt.plot(xp, fx(xp))
plt.show()