# cython: profile=True
import numpy as np
import line_profiler
cimport numpy as np
cimport cython


def runFriction(int N=1000, double tau=0.12, double alpha=0.55, int T=200, double dt=1e-2):
    cdef int Nt = int(T/dt)
    cdef np.ndarray taubar = np.ones(N, dtype=np.float64)*tau
    taubar[0] = 1.001

    cdef np.ndarray x = np.zeros(N,dtype=np.float64)
    cdef np.ndarray v = np.zeros(N,dtype=np.float64)
    cdef np.ndarray a = np.zeros(N,dtype=np.float64)
    cdef np.ndarray stuckFlag = np.ones(N, dtype=bool)

    cdef np.ndarray x_new = np.zeros(N, dtype=np.float64)
    cdef np.ndarray v_new = np.zeros(N, dtype=np.float64)
    cdef np.ndarray detachTimes = np.zeros(N, dtype=np.float64)

    cdef np.ndarray x_1 = np.zeros(Nt, dtype=np.float64)
    cdef np.ndarray x_2 = np.zeros(Nt, dtype=np.float64)

    cdef np.ndarray changeToStuck = np.zeros(N, dtype = np.bool)
    #stuck_matrix = np.zeros((Nt, N))

    cdef double t = 0.0
    cdef int counter = 0
    cdef int i
    for i in range(Nt):
        a[0] = x[1]-x[0]
        a[1:N-1] = x[0:N-2]-2*x[1:N-1]+x[2:N]
        a[N-1] = x[N-2]-x[N-1]
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
        detachTimes[detachTimes==0] = t*np.logical_not(stuckFlag[detachTimes==0])
        t += dt
        counter += 1
    #return detachTimes, x_1, x_2
