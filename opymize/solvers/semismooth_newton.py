
from opymize import Variable, LinOp
from opymize.linear import normest
from opymize.tools import truncate
from opymize.tools.util import GracefulInterruptHandler

import logging

import numpy as np
from numpy.linalg import norm
import scipy

def armijo(fun, xk, xkp1, p, p_gradf, fun_xk, eta=0.5, nu=0.9):
    """ Determine step size using backtracking

        f(xk + alpha*p) <= f(xk) + alpha*nu*<p,Df>

    Args:
        fun : objective function `f`
        xk : starting position
        xkp1 : where new position `xk + alpha*p` is stored
        p : search direction
        p_gradf : local slope along the search direction `<p,Df>`
        fun_xk : starting value `f(xk)`
        eta : control parameter (shrinkage)
        nu : control parameter (sufficient decrease)

    Returns:
        Set `xkp1[:] = xk + alpha*p` and return a tuple (f_xkp1, alpha)
        f_xkp1 : new value `f(xk + alpha*p)`
        alpha : determined step size
    """
    if p_gradf >= 0:
        raise Exception("Armijo: Not a descent direction!")

    alpha = 1.0
    while True:
        xkp1[:] = xk + alpha*p
        f_xkp1 = fun(xkp1)
        if f_xkp1 <= fun_xk + alpha*nu*p_gradf:
            break
        else:
            alpha *= eta
        if alpha < 1e-10:
            raise Exception("Armijo: Not a descent direction!")
    return f_xkp1, alpha

class SymRegOp(LinOp):
    """ Computes M^T*M + lbd*Id for the given `M` """
    def __init__(self, M, lbd=1.0, tmp=None):
        LinOp.__init__(self)
        self.M = M
        self.lbd = lbd
        self.x = M.x
        self.y = M.x
        self.tmp = M.y.new() if tmp is None else tmp
        self.adjoint = self

    def prepare_gpu(self):
        self.tmp_gpu = gpuarray.to_gpu(self.tmp)

    def _call_cpu(self, x, y=None, add=False):
        assert y is not None
        self.M(x, self.tmp)
        self.M.adjoint(self.tmp, y, add=add)
        if self.lbd is not 0:
            y += self.lbd*x

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        self.M(x, self.tmp_gpu)
        self.M.adjoint(self.tmp_gpu, y, add=add)
        if self.lbd is not 0:
            y += self.lbd*x

def CG_solve(A, b, xk, lbd=0, tmpvars=None, maxiter=int(1e3), tol=1e-5):
    """ Solve Ax + b = 0 using CG after symmetrization and regularization:

        (A^T*A + lbd*Id)*x = -A^T*b

    Args:
        A : linear operator
        b : RHS of linear equation
        xk : initial value and where the result is stored
        lbd : regularization parameter
        tmpvars : tuple (rk, rkp1, z, dk) for internal temporary variables
        maxiter : termination criterion (max. number of steps)
        tol : termination criterion (min. precision)

    Returns:
        Nothing, the result is written to `xk`
    """
    if tmpvars is not None:
        rk, rkp1, z, dk = tmpvars
    else:
        rk, rkp1, z, dk = [xk.copy() for i in range(4)]
    ATb = z
    modA = SymRegOp(A, lbd=lbd, tmp=rkp1)
    A.adjoint(b, ATb)
    modA(xk, rk)
    rk += ATb
    rk *= -1
    dk[:] = rk
    for k in range(maxiter+1):
        modA(dk, z)
        alpha = np.einsum('i,i->', rk, rk)/np.einsum('i,i->', dk, z)
        xk += alpha*dk
        rkp1[:] = rk - alpha*z
        beta = np.einsum('i,i->', rkp1, rkp1)/np.einsum('i,i->', rk, rk)
        dk[:] = rkp1 + beta*dk
        rk[:] = rkp1
        if k % 2 == 0:
            normsq_r = np.einsum('i,i->', rkp1, rkp1)
            logging.debug("CG #{:03d}: {: 9.6g}".format(k, normsq_r))
            if normsq_r < tol:
                break

class SemismoothNewtonSystem(LinOp):
    """ Block matrix of the following form:
        [[      I - K, tau*K*A^T ],
         [ -sigma*H*A,     I - H ]]
    """
    def __init__(self, A, tau, sigma):
        LinOp.__init__(self)
        self.A = A
        self.tau = tau
        self.sigma = sigma
        self.x = Variable((A.x.size,), (A.y.size,))
        self.y = self.x
        self.xtmp = self.x.new()
        self.K = None
        self.H = None
        self.adjoint = SemismoothNewtonSystemAdjoint(self)

    def _call_cpu(self, x, y=None, add=False):
        assert y is not None
        assert add is False

        x1tmp, x2tmp = self.x.vars(self.xtmp)
        x1, x2 = self.x.vars(x)
        y1, y2 = self.y.vars(y)

        self.A.adjoint(x2, x1tmp)
        self.K(x1tmp, y1)
        y1[:] = x1 + self.tau*y1
        self.K(x1, x1tmp)
        y1 -= x1tmp

        self.A(x1, x2tmp)
        self.H(x2tmp, y2)
        y2[:] = x2 - self.sigma*y2
        self.H(x2, x2tmp)
        y2 -= x2tmp

class SemismoothNewtonSystemAdjoint(LinOp):
    """ Adjoint of the given SemismoothNewtonSystem `M` """
    def __init__(self, M):
        LinOp.__init__(self)
        self.M = M
        self.x = M.y
        self.y = M.x
        self.adjoint = M

    def _call_cpu(self, x, y=None, add=False):
        assert y is not None
        assert add is False

        x1tmp, x2tmp = self.x.vars(self.M.xtmp)
        x1, x2 = self.x.vars(x)
        y1, y2 = self.y.vars(y)

        self.M.H.adjoint(x2, x2tmp)
        self.M.A.adjoint(x2tmp, y1)
        y1[:] = x1 - self.M.sigma*y1
        self.M.K.adjoint(x1, x1tmp)
        y1 -= x1tmp

        self.M.K.adjoint(x1, x1tmp)
        self.M.A(x1tmp, y2)
        y2[:] = x2 + self.M.tau*y2
        self.M.H.adjoint(x2, x2tmp)
        y2 -= x2tmp

class SemismoothNewton(object):
    def __init__(self, g, f, A):
        self.g = g
        self.f = f
        self.linop = A
        self.xy = Variable((self.g.x.size,), (self.f.x.size,))
        self.itervars = { 'xyk': self.xy.new() }
        self.constvars = { 'tau': 1.0, 'sigma': 1.0 }

    def obj_primal(self, x, Ax):
        obj, infeas = self.g(x)
        obj2, infeas2 = self.f(Ax)
        return obj + obj2, infeas + infeas2

    def obj_dual(self, ATy, y):
        obj, infeas = self.g.conj(-ATy)
        obj2, infeas2 = self.f.conj(y)
        return -obj - obj2, infeas + infeas2

    def res(self, xy, xyres, subdiff=False, norm=True):
        c = self.constvars
        x, y = self.xy.vars(xy)
        xgrad, ygrad = self.xy.vars(xyres)

        # xgrad = x - Prox[tau*g](x - tau * A^T * y)
        self.linop.adjoint(y, xgrad)
        xgrad[:] =  x - c['tau']*xgrad
        K = self.gprox(xgrad, jacobian=subdiff)
        xgrad[:] = x - xgrad

        # ygrad = y - Prox[sigma*fc](y + sigma * A * x)
        self.linop(x, ygrad)
        ygrad[:] = y + c['sigma']*ygrad
        H = self.fconjprox(ygrad, jacobian=subdiff)
        ygrad[:] = y - ygrad

        if subdiff:
            self.M.K = K
            self.M.H = H

        if norm:
            return 0.5*np.einsum('i,i->', xyres, xyres)

    def iteration_step(self, _iter):
        i = self.itervars
        c = self.constvars

        # Set up Newton system
        res_normsq = self.res(i['xyk'], i['xyres'], subdiff=True)

        # Solve modified Newton system using CG
        CG_solve(self.M, i['xyres'], i['dk'], lbd=self.lbd,
                 tmpvars=(i['cg_rk'], i['cg_rkp1'], i['xytmp'], i['cg_dk']),
                 tol=res_normsq)

        # Armijo backtracking
        self.M(i['dk'], i['xytmp'])
        p_gradf = np.einsum('i,i->', i['xyres'], i['xytmp'])
        armijo_fun = lambda xy: self.res(xy, i['xyres'], subdiff=False)
        new_normsq, alpha = armijo(armijo_fun, i['xyk'], i['xykp1'], i['dk'], \
                                   p_gradf, res_normsq)

        # Update, taken from:
        #   "Distributed Newton Methods for Deep Neural Networks"
        #   by C.-C. Wang et al. (arxiv: https://arxiv.org/abs/1802.00130)
        Md_normsq = 0.5*np.einsum('i,i->', i['xytmp'], i['xytmp'])
        rho = (new_normsq - res_normsq)/(alpha*p_gradf + alpha**2*Md_normsq)
        if rho > 0.75:
            self.lbd *= self.lbd_drop
        elif rho < 0.25:
            self.lbd *= self.lbd_boost
        logging.debug("#{:6d}: alpha = {: 9.6g}, "\
                     "rho = {: 9.6g}, lbd = {: 9.6g}"\
                     .format(_iter, alpha, rho, self.lbd))
        i['xyk'][:] = i['xykp1']

    def prepare_stepsizes(self):
        i = self.itervars
        c = self.constvars
        step_factor = 1.0

        logging.info("Estimating optimal step bound...")
        op_norm, itn = normest(self.linop)
        # round (floor) to 3 significant digits
        bnd = truncate(1.0/op_norm**2, 3) # < 1/|K|^2
        bnd *= 1.25 # boost
        fact = step_factor # tau/sigma
        c['sigma'] = np.sqrt(bnd/fact)
        c['tau'] = bnd/c['sigma']
        logging.info("Constant steps: %f (%f | %f)" % (bnd,c['sigma'],c['tau']))
        self.gprox = self.g.prox(c['tau'])
        self.fconjprox = self.f.conj.prox(c['sigma'])
        self.M = SemismoothNewtonSystem(self.linop, c['tau'], c['sigma'])
        self.lbd = 1.0
        self.lbd_drop = 0.1
        self.lbd_boost = 5.0

    def solve(self, continue_at=None, granularity=50,
                    term_relgap=1e-5, term_infeas=None, term_maxiter=int(5e2)):
        i = self.itervars
        c = self.constvars

        if continue_at is not None:
            i['xyk'][:] = continue_at

        i['xykp1'] = i['xyk'].copy()
        i['xyres'] = i['xyk'].copy()
        i['dk'] = i['xyk'].copy()
        i['xytmp'] = i['xyk'].copy()
        i['cg_dk'] = i['xyk'].copy()
        i['cg_rk'] = i['xyk'].copy()
        i['cg_rkp1'] = i['xyk'].copy()

        xk, yk = self.xy.vars(i['xyk'])
        xkp1, ykp1 = self.xy.vars(i['xykp1'])

        self.prepare_stepsizes()

        if term_infeas is None:
            term_infeas = term_relgap

        obj_p = obj_d = infeas_p = infeas_d = relgap = 0.

        logging.info("Solving (steps<%d)..." % term_maxiter)

        with GracefulInterruptHandler() as interrupt_hdl:
            _iter = 0
            while _iter < term_maxiter:
                self.iteration_step(_iter)
                _iter += 1

                if interrupt_hdl.interrupted or _iter % granularity == 0:
                    if interrupt_hdl.interrupted:
                        print("Interrupt (SIGINT) at iter=%d" % _iter)

                    self.linop(xk, ykp1)
                    self.linop.adjoint(yk, xkp1)
                    obj_p, infeas_p = self.obj_primal(xk, ykp1)
                    obj_d, infeas_d = self.obj_dual(xkp1, yk)

                    # compute relative primal-dual gap
                    relgap = (obj_p - obj_d) / max(np.spacing(1), obj_d)

                    logging.info("#{:6d}: objp = {: 9.6g} ({: 9.6g}), " \
                        "objd = {: 9.6g} ({: 9.6g}), " \
                        "gap = {: 9.6g}, " \
                        "relgap = {: 9.6g} ".format(
                        _iter, obj_p, infeas_p,
                        obj_d, infeas_d,
                        obj_p - obj_d,
                        relgap
                    ))

                    if np.abs(relgap) < term_relgap \
                       and max(infeas_p, infeas_d) < term_infeas:
                        break

                    if interrupt_hdl.interrupted:
                        break

        return {
            'objp': obj_p,
            'objd': obj_d,
            'infeasp': infeas_p,
            'infeasd': infeas_d,
            'relgap': relgap
        }

    @property
    def state(self):
        return self.xy.vars(self.itervars['xyk'])

class SemismoothQuasinewton(SemismoothNewton):
    def solve(self, continue_at=None, granularity=10,
                    term_relgap=1e-5, term_infeas=None, term_maxiter=int(1e2)):
        i = self.itervars
        c = self.constvars

        if continue_at is not None:
            i['xyk'][:] = continue_at

        i['xykp1'] = i['xyk'].copy()
        i['xykres'] = i['xyk'].copy()

        xk, yk = self.xy.vars(i['xyk'])
        xkp1, ykp1 = self.xy.vars(i['xykp1'])

        self.prepare_stepsizes()

        def f(x):
            self.res(x, i['xykres'], norm=False)
            return i['xykres']

        if term_infeas is None:
            term_infeas = term_relgap

        obj_p = obj_d = infeas_p = infeas_d = relgap = 0.

        logging.info("Solving (steps<%d)..." % term_maxiter)

        with GracefulInterruptHandler() as interrupt_hdl:
            _iter = 0
            while _iter < term_maxiter:
                i['xykp1'][:] = i['xyk']
                i['xyk'][:] = scipy.optimize.broyden1(f, i['xykp1'],
                                                      iter=granularity)
                _iter += granularity

                if interrupt_hdl.interrupted or _iter % granularity == 0:
                    if interrupt_hdl.interrupted:
                        print("Interrupt (SIGINT) at iter=%d" % _iter)

                    self.linop(xk, ykp1)
                    self.linop.adjoint(yk, xkp1)
                    obj_p, infeas_p = self.obj_primal(xk, ykp1)
                    obj_d, infeas_d = self.obj_dual(xkp1, yk)

                    # compute relative primal-dual gap
                    relgap = (obj_p - obj_d) / max(np.spacing(1), obj_d)

                    logging.info("#{:6d}: objp = {: 9.6g} ({: 9.6g}), " \
                        "objd = {: 9.6g} ({: 9.6g}), " \
                        "gap = {: 9.6g}, " \
                        "relgap = {: 9.6g} ".format(
                        _iter, obj_p, infeas_p,
                        obj_d, infeas_d,
                        obj_p - obj_d,
                        relgap
                    ))

                    if np.abs(relgap) < term_relgap \
                       and max(infeas_p, infeas_d) < term_infeas:
                        break

                    if interrupt_hdl.interrupted:
                        break

        return {
            'objp': obj_p,
            'objd': obj_d,
            'infeasp': infeas_p,
            'infeasd': infeas_d,
            'relgap': relgap
        }
