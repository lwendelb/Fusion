from __future__ import division  # Use "real" division

# Add another filepath to the Python path
import sys
#sys.path.insert(1, '/data/repos-git/quad/Functions-from-Chris')
#sys.path.insert(2, '/data/repos-git/quad/Functions-from-Chris/g2conda/GSASII')
#sys.path.insert(3, '/data/repos-git/quad')

# Import specific functions
from timeit import default_timer as timer  # Timing function
from scipy.stats import norm               # Normal distribution
from scipy.stats import multivariate_normal as mvnorm  # Multivariate normal distribution
import statsmodels.api as sm               # Library for lowess smoother
lowess = sm.nonparametric.lowess           # Lowess smoothing function
from bspline import Bspline                # Bspline function
from splinelab import augknt               # Bspline helper function

# Import entire modules
import numpy as np
import matplotlib.pyplot as plt
import CalculateSinCos as Calc
import FmcmcSC as MCMC
import csv

def par2z(par,lower,upper):
    z = norm.ppf((par-lower)/(upper-lower))
    return z
def z2par(z, lower, upper, grad=False):
    if (grad):
        d = (upper-lower)*norm.pdf(z)
        return d
    else:
        par = lower + (upper-lower)*norm.cdf(z)
        # Fudge the parameter value if we've hit either boundary
        par[np.array([par[j]==upper[j] for j in range(len(par))])] -= 1e-10
        par[np.array([par[j]==lower[j] for j in range(len(par))])] += 1e-10
        return par

#paramList = ["Lam","Asin","phisin","Acos","phicos"]
paramList = ["Lam","Phi","Asin","Acos"]
#paramList = ["Lamsin","Asin","Phisin","Lamcos","Acos","Phicos"]

truth = np.concatenate((Calc.share,Calc.truth_sin,Calc.truth_cos))
print truth
lower = np.array([0,0,0,0])
upper = np.array([2*np.pi,np.pi,20,10])
#lower = np.array([0,0,0])
#upper = np.array([2*np.pi,20,np.pi])
#lower = np.array([0,0,0,0,0,0])
#upper = np.array([2*np.pi,20,np.pi,2*np.pi,10,np.pi])
#params = np.array([np.pi/4,np.pi,1,3])
#init_z = par2z(params,lower,upper)
init_z = par2z(truth*0.9,lower,upper)

print init_z
print z2par(init_z,lower,upper)

bx = 1
bn = 1
taux = 4.
taun = 2.
#taux = 4.
#taun=2.
#yx = np.random.normal(Calc.ysin_mean,1/np.sqrt(taux))
#yn = np.random.normal(Calc.ycos_mean,1/np.sqrt(taun))
yx = np.random.normal(Calc.ysin_mean,np.sqrt(np.abs(Calc.ysin_mean/8.))/np.sqrt(taux))
yn = np.random.normal(Calc.ycos_mean,np.sqrt(np.abs(Calc.ycos_mean/4.7))/np.sqrt(taun))


'''
plt.figure(1)
plt.plot(Calc.x,Calc.ysin_mean)
plt.plot(Calc.x,yx,'r+')
plt.ylabel("Amplitude")
plt.xlabel("x")
plt.figure(2)
plt.plot(Calc.x,Calc.ycos_mean)
plt.plot(Calc.x,yn,'r+')
plt.ylabel("Amplitude")
plt.xlabel("x")
plt.show()
'''

curr= 40000
burn = 10000
results = MCMC.nlDRAM(paramList=paramList, init_z=init_z, lower=lower,
upper=upper, y_x=yx, y_n=yn, x=Calc.x, q_share=2, q_x=1, q_n=1, L=20,
shrinkage=0.007, s_p=(2.4**2),
epsilon=1e-10, m0=0, sd0=1, c_y=0.1, d_y=0.1, c_g=1, d_g=1, c_b=0.1,
d_b=0.1, adapt=200, thin=1, iters=curr, burn=burn, update=1000, plot=False, fix=False)

with open('C:\Users\Laura\Documents\Fusion-master\/temp\/results.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(results[0])

with open('C:\Users\Laura\Documents\Fusion-master\/temp\/taux.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(np.array([results[2]]))
with open('C:\Users\Laura\Documents\Fusion-master\/temp\/taun.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(np.array([results[3]]))

'''
results = MCMC.nlDRAM(paramList, init_z, lower, upper, y_x=yx, y_n=yn, x=Calc.x, L=20, shrinkage=0.2, s_p=(2.4**2), epsilon=1e-4, m0=0, sd0=1, c_y=0.1, d_y=0.1, c_g=0.1, d_g=0.1, c_b=0.1, d_b=0.1, adapt=20, thin=1, iters=100, burn=0, update=10, plot=False, fix=False)
import csv
with open('C:\Users\Laura\Documents\R\Research\sincosresults_5_10_2018_9_52.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(results[0])
with open('C:\Users\Laura\Documents\R\Research\VarS12.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(results[1])
'''

curr_keep = curr-burn
keep_params = results[0]
plt.figure(3, figsize=(20, 10))
plt.subplot(231)
plt.plot(keep_params[range(curr_keep), 0], 'k')
plt.plot(np.repeat(truth[0],curr_keep),'r')
plt.xlabel("Iteration")
plt.ylabel(paramList[0])
plt.subplot(232)
plt.plot(keep_params[range(curr_keep), 1], 'k')
plt.plot(np.repeat(truth[1],curr_keep),'r')
plt.xlabel("Iteration")
plt.ylabel(paramList[1])
plt.subplot(233)
plt.plot(keep_params[range(curr_keep), 2], 'k')
plt.plot(np.repeat(truth[2],curr_keep),'r')
plt.xlabel("Iteration")
plt.ylabel(paramList[2])
plt.subplot(234)
plt.plot(keep_params[range(curr_keep), 3], 'k')
plt.plot(np.repeat(truth[3],curr_keep),'r')
plt.xlabel("Iteration")
plt.ylabel(paramList[3])
'''
plt.subplot(235)
plt.plot(keep_params[range(curr_keep), 4], 'k')
plt.plot(np.repeat(truth[4],curr_keep),'r')
plt.xlabel("Iteration")
plt.ylabel(paramList[4])
plt.subplot(236)
plt.plot(keep_params[range(curr_keep), 5], 'k')
plt.plot(np.repeat(truth[5],curr_keep),'r')
plt.xlabel("Iteration")
plt.ylabel(paramList[5])
'''

plt.figure(4, figsize=(20,10))
plt.subplot(121)
plt.plot(results[2], 'k')
plt.xlabel("Iteration")
plt.ylabel("taux")
plt.subplot(122)
plt.plot(results[3], 'k')
plt.xlabel("Iteration")
plt.ylabel("taun")

'''
plt.subplot(236)
plt.plot(keep_params[range(curr_keep), 5], 'k')
plt.xlabel("Iteration")
plt.ylabel(paramList[5])

plt.figure(2,figsize=(20,10))
plt.subplot(231)
plt.plot(keep_params[range(curr_keep), 6], 'k')
plt.xlabel("Iteration")
plt.ylabel(paramList[6])
plt.subplot(232)
plt.plot(keep_params[range(curr_keep), 7], 'k')
plt.xlabel("Iteration")
plt.ylabel(paramList[7])
plt.subplot(233)
plt.plot(keep_params[range(curr_keep), 8], 'k')
plt.xlabel("Iteration")
plt.ylabel(paramList[8])
plt.subplot(234)
plt.plot(keep_params[range(curr_keep), 9], 'k')
plt.xlabel("Iteration")
plt.ylabel(paramList[9])
plt.subplot(235)
plt.plot(keep_params[range(curr_keep), 10], 'k')
plt.xlabel("Iteration")
plt.ylabel(paramList[10])
#plt.pause(0.1)
'''
plt.show()

## select important points
# check with sin cos
# ask about university cluster vcl.ncsu.edu
