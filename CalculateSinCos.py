import numpy as np
import matplotlib.pyplot as plt

def CalcSin(x, params, share):
    [A,phi] = params
    lam = share
    y = A*np.sin(lam*x+phi)
    return y
def CalcCos(x, params, share):
    [A,phi] = params
    lam = share
    y = A*np.cos(lam*x+phi)
    return y

lam = np.pi
A_sin = 8.
A_cos = 4.7
phi_sin = np.pi/3
phi_cos = np.pi/2

x = np.arange(0.,2*np.pi,0.01)

truth_sin = np.array([A_sin,phi_sin])
truth_cos = np.array([A_cos,phi_cos])
share = np.array([lam])

ysin_mean = CalcSin(x,truth_sin,share)
ycos_mean = CalcCos(x,truth_cos,share)
'''
plt.figure(1)
plt.plot(x,ysin_mean)
plt.ylabel("Intensity")
plt.xlabel("Scattering Angle")
plt.figure(2)
plt.plot(x,ycos_mean)
plt.ylabel("Intensity")
plt.xlabel("Scattering Angle")
plt.show()
'''
