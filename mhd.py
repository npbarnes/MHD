import numpy as np

class _flux_conserve:
    """This is the abstract superclass for classes that solve
    dq/dt + d F(q)/dt = 0
    for vector q and vector valued function F
    Specific solvers must provide a step function and possibly override the
    __init__ function.
    """
    def _grid_args(self, L, **kwargs):
        """Takes in arguments
        one of: dx or m
        two of: c, u, and dt
        and computes the others then returns all five values.
        """
        k = kwargs.keys()
        if 'm' in k:
            assert 'dx' not in k
            m = kwargs['m']
            dx = L/(m+1)
        else:
            assert 'dx' in k
            dx = kwargs['dx']
            m = L/dx - 1

        return m, dx



    def __init__(self, L, dt, q0, F, **kwargs):
        """Arguments
        L: length of domain
        dt: timestep
        q0: sequence of initial functions
        F: function that takes a state vector and returns a flux vector
        kwargs: should contain one of
            dx: the grid spacing
            m: the number of gridpoints
        """
        self.L = L
        self.dt = dt
        self.q0 = q0
        self.F = F#np.vectorize(F)

        self.m, self.dx = self._grid_args(L, **kwargs)

        self.r = self.dt/self.dx
        self.grid = np.linspace(0, self.L-self.dx, self.m)
        self.q = np.empty((self.m, len(self.q0)))
        for i, init in enumerate(self.q0):
            self.q[:,i] = np.vectorize(init)(self.grid)

        self.t = 0

    def _step(self):
        """Updates self.T by one step of size self.dt"""
        raise NotImplemented()

    def step(self, N=1):
        for n in range(N):
            self._step()
            self.t += self.dt

    def stepUntil(self, time):
        if time < self.t:
            raise ValueError('Time must be in the future')

        numsteps = int((time-self.t)/self.dt)
        self.step(numsteps)

        # Take remaining partial step if needed.
        if time != self.t:
            tmp_dt = self.dt
            self.dt = time - self.t
            self.step()
            self.dt = tmp_dt

    def stepBy(self, time):
        if time < 0:
            raise ValueError('Must step forward in time')

        oldt = self.t
        numsteps = int(time/self.dt)
        self.step(numsteps)

        if self.t != oldt+time:
            tmp_dt = self.dt
            self.dt = self.t - (oldt+time)
            self.step()
            self.dt = tmp_dt

class lax_wendroff(_flux_conserve):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qs = np.empty_like(self.q)

    def _step(self):
        qs = self.qs
        r = self.r
        F = self.F
        q = self.q
        t = self.t

        qs[:-1] = 0.5*(q[:-1] + q[1:]) - 0.5*r*(F(q[1:]) - F(q[:-1]))
        qs[-1] = 0.5*(q[-1] + q[0]) - 0.5*r*(F(q[0]) - F(q[-1]))

        q[0] = q[0] - r*(F(qs[0]) - F(qs[-1]))
        q[1:] = q[1:] - r*(F(qs[1:]) - F(qs[:-1]))
