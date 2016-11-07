import numpy as np

def runFriction1DAmontonsCoulombAbsoluteViscosity(N=1000, tau=0.0398, alpha=0.2069, T=2000, dt=1e-2):
    Nt = int(T/dt)

    taubar = np.ones(N, dtype=np.float64)*tau
    taubar[0] = 1.001

    x = np.zeros(N,dtype=np.float64)
    v = np.zeros(N,dtype=np.float64)
    a = np.zeros(N,dtype=np.float64)
    stuckFlag = np.ones(N, dtype=bool)

    x_new = np.zeros(N, dtype=np.float64)
    v_new = np.zeros(N, dtype=np.float64)
    detachTimes = np.zeros(N, dtype=np.float64)
    detachArray = np.zeros(N, dtype=np.bool)

    for i in xrange(Nt):
        a[0] = x[1]-x[0]
        a[1:N-1] = x[0:N-2]-2*x[1:N-1]+x[2:N]
        a[N-1] = x[N-2]-x[N-1]
        a = a - alpha*v + taubar*np.logical_not(stuckFlag)
        v_new = np.zeros(N)+np.logical_not(stuckFlag)*(v+a*dt)
        x_new = x+v_new*dt
        stuckFlag = np.logical_or(np.logical_and(a < 1-taubar, stuckFlag), v*v_new<0)
        x = x_new
        v = v_new
        detachArray = detachTimes == 0
        detachTimes[detachArray] = i*dt*np.logical_not(stuckFlag[detachArray])
    return detachTimes
