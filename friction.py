import numpy as np
import matplotlib.pyplot as plt

def runFriction1D(N=1000, tau=0.0398, alpha=0.2069, T=2000, dt=1e-3):
    Nt = int(T/dt)

    taubar = np.ones(N, dtype=np.float64)*tau
    taubar[0] = 1.001

    x = np.zeros(N,dtype=np.float64)
    v = np.zeros(N,dtype=np.float64)
    a = np.zeros(N,dtype=np.float64)
    stuckFlag = np.ones(N, dtype=bool)

    x_new = np.zeros(N)
    v_new = np.zeros(N)
    detachTimes = np.zeros(N)
    detachArray = np.zeros(N, dtype=np.bool)

    x_1 = np.zeros(Nt)
    x_2 = np.zeros(Nt)

    #stuck_matrix = np.zeros((Nt, N))

    t = 0
    counter = 0
    for i in xrange(Nt):
        a[0] = x[1]-x[0]
        a[1:N-1] = x[0:N-2]-2*x[1:N-1]+x[2:N]
        a[-1] = x[-2]-x[-1]
        #print stuckFlag
        a = a - alpha*v + taubar*np.logical_not(stuckFlag)
        v_new = np.zeros(N)+np.logical_not(stuckFlag)*(v+a*dt)
        x_new = x+v_new*dt
        stuckFlag = np.logical_and(a < 1-taubar, stuckFlag)
        changeToStuck = v*v_new<0
        stuckFlag[changeToStuck] = True
        x_1[counter] = x_new[0]
        x_2[counter] = x_new[1]
        x = x_new
        v = v_new
        #stuck_matrix[counter, :] = stuckFlag
        detachArray = detachTimes == 0
        detachTimes[detachArray] = t*np.logical_not(stuckFlag[detachArray])
        t += dt
        counter += 1
    return detachTimes, x_1, x_2


detachTimes, x_1, x_2 = runFriction1D()

plt.plot(x_1)
plt.hold(True)
plt.plot(x_2)
plt.figure()
plt.plot(1.0/np.diff(detachTimes))
plt.show()
