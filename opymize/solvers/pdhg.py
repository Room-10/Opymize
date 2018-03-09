
from opymize.linear import normest
from opymize.tools import truncate
from opymize.tools.util import GracefulInterruptHandler

import logging

import numpy as np
from numpy.linalg import norm

class PDHG(object):
    def __init__(self, g, f, A):
        self.g = g
        self.f = f
        self.linop = A
        self.itervars = { 'xk': self.g.x.new(), 'yk': self.f.x.new() }
        self.constvars = {}
        self.use_gpu = False

    def obj_primal(self, x, Ax):
        obj, infeas = self.g(x)
        obj2, infeas2 = self.f(Ax)
        return obj + obj2, infeas + infeas2

    def obj_dual(self, ATy, y):
        obj, infeas = self.g.conj(-ATy)
        obj2, infeas2 = self.f.conj(y)
        return -obj - obj2, infeas + infeas2

    def prepare_gpu(self):
        import opymize.tools.gpu # gpu init
        from pycuda import gpuarray
        from pycuda.elementwise import ElementwiseKernel
        from pycuda.reduction import ReductionKernel

        i = self.itervars
        c = self.constvars

        self.gprox.prepare_gpu()
        self.fconjprox.prepare_gpu()
        self.linop.prepare_gpu()

        p = ("*","[i]") if 'precond' in c else ("","")
        self.gpu_kernels = {
            'take_step': ElementwiseKernel(
                "double *zkp1, double *zk, char direction, double %sstepsize, double *zgradk" % p[0],
                "zkp1[i] = zk[i] + direction*stepsize%s*zgradk[i]" % p[1]),
            'overrelax': ElementwiseKernel(
                "double *ygradbk, double theta, double *ygradkp1, double *ygradk",
                "ygradbk[i] = (1 + theta)*ygradkp1[i] - theta*ygradk[i]"),
            'advance': ElementwiseKernel(
                "double *zk, double *zkp1, double *zgradk, double *zgradkp1",
                "zk[i] = zkp1[i]; zgradk[i] = zgradkp1[i]"),
            'residual': 0 if 'adaptive' not in c else ReductionKernel(
                np.float64, neutral="0", reduce_expr="a+b",
                map_expr="fabs((zk[i] - zkp1[i])/step - (zgradk[i] - zgradkp1[i]))",
                arguments="double step, double *zk, double *zkp1, "\
                         +"double *zgradk, double *zgradkp1"),
        }

        self.gpu_itervars = {}
        for name, val in self.itervars.items():
            if type(val) is not np.ndarray:
                continue
            self.gpu_itervars[name] = gpuarray.to_gpu(val)

        self.gpu_constvars = {}
        for name, val in self.constvars.items():
            if type(val) is not np.ndarray:
                continue
            self.gpu_constvars[name] = gpuarray.to_gpu(val)

        self.use_gpu = True
        logging.info("CUDA kernels prepared for GPU")

    def take_step(self, zkp1, zk, direction, stepsize, zgradk):
        if type(zkp1) is np.ndarray:
            zkp1[:] = zk + direction*stepsize*zgradk
        else:
            self.gpu_kernels['take_step'](zkp1, zk, np.int8(direction), stepsize, zgradk)

    def overrelax(self, ygradbk, theta, ygradkp1, ygradk):
        if type(ygradbk) is np.ndarray:
            ygradbk[:] = (1 + theta)*ygradkp1 - theta*ygradk
        else:
            self.gpu_kernels['overrelax'](ygradbk, theta, ygradkp1, ygradk)

    def residual(self, fact, zk, zkp1, zgradk, zgradkp1):
        if type(zk) is np.ndarray:
            return norm((zk - zkp1)/fact - (zgradk - zgradkp1), ord=1)
        else:
            return self.gpu_kernels['residual'](fact, zk, zkp1, zgradk, zgradkp1).get()

    def advance(self, zk, zkp1, zgradk, zgradkp1):
        if type(zk) is np.ndarray:
            zk[:], zgradk[:] = zkp1, zgradkp1
        else:
            self.gpu_kernels['advance'](zk, zkp1, zgradk, zgradkp1)

    def iteration_step(self):
        v = self.gpu_itervars if self.use_gpu else self.itervars
        xk, xkp1, xgradk, xgradkp1 = v['xk'], v['xkp1'], v['xgradk'], v['xgradkp1']
        yk, ykp1, ygradk, ygradkp1 = v['yk'], v['ykp1'], v['ygradk'], v['ygradkp1']
        ygradbk = v['ygradbk']

        i = self.itervars
        c = self.constvars
        if 'precond' in c:
            tau = self.gpu_constvars['xtau'] if self.use_gpu else c['xtau']
            sigma = self.gpu_constvars['ysigma'] if self.use_gpu else c['ysigma']
        elif 'adaptive' in c:
            tau = i['tauk']
            sigma = i['sigmak']
        else:
            tau = c['tau']
            sigma = c['sigma']

        # --- primals:
        self.take_step(xkp1, xk, -1, tau, xgradk)
        self.gprox(xkp1)
        self.linop(xkp1, ygradkp1)
        self.overrelax(ygradbk, c['theta'], ygradkp1, ygradk)

        # --- duals:
        self.take_step(ykp1, yk, 1, sigma, ygradbk)
        self.fconjprox(ykp1)
        self.linop.adjoint(ykp1, xgradkp1)

        # --- step sizes:
        if 'adaptive' in c and i['alphak'] > 1e-10:
            i['res_pk'] = self.residual(tau, xk, xkp1, xgradk, xgradkp1)
            i['res_dk'] = self.residual(sigma, yk, ykp1, ygradk, ygradkp1)

            if i['res_pk'] > c['s']*i['res_dk']*c['Delta']:
                i['tauk'] *= 1.0/(1.0 - i['alphak'])
                i['sigmak'] *= (1.0 - i['alphak'])
                i['alphak'] *= c['eta']
                self.gprox = self.g.prox(i['tauk'])
                self.fconjprox = self.f.conj.prox(i['sigmak'])

            if i['res_pk'] < c['s']*i['res_dk']/c['Delta']:
                i['tauk'] *= (1.0 - i['alphak'])
                i['sigmak'] *= 1.0/(1.0 - i['alphak'])
                i['alphak'] *= c['eta']
                self.gprox = self.g.prox(i['tauk'])
                self.fconjprox = self.f.conj.prox(i['sigmak'])

        # --- update
        self.advance(xk, xkp1, xgradk, xgradkp1)
        self.advance(yk, ykp1, ygradk, ygradkp1)

    def prepare_stepsizes(self, step_bound, step_factor, steps):
        i = self.itervars
        c = self.constvars
        if steps == "precond":
            c['precond'] = True
            c['xtau'] = i['xk'].copy()
            c['ysigma'] = i['yk'].copy()
            logging.info("Determining diagonal preconditioners...")
            self.linop.rowwise_lp(c['ysigma'])
            self.linop.adjoint.rowwise_lp(c['xtau'])
            c['ysigma'][c['ysigma'] > np.spacing(1)] = 1.0/c['ysigma'][c['ysigma'] > np.spacing(1)]
            c['xtau'][c['xtau'] > np.spacing(1)] = 1.0/c['xtau'][c['xtau'] > np.spacing(1)]
            self.gprox = self.g.prox(c['xtau'])
            self.fconjprox = self.f.conj.prox(c['ysigma'])
        else:
            bnd = step_bound
            if step_bound is None:
                logging.info("Estimating optimal step bound...")
                op_norm, itn = normest(self.linop)
                # round (floor) to 3 significant digits
                bnd = truncate(1.0/op_norm**2, 3) # < 1/|K|^2
            if steps == "adaptive":
                c['adaptive'] = True
                c['eta'] = 0.95 # 0 < eta < 1
                c['Delta'] = 1.5 # > 1
                c['s'] = 255.0 # > 0
                i['alphak'] = 0.5
                i['sigmak'] = i['tauk'] = np.sqrt(bnd)
                i['res_pk'] = i['res_dk'] = 0.0
                self.gprox = self.g.prox(i['tauk'])
                self.fconjprox = self.f.conj.prox(i['sigmak'])
                logging.info("Adaptive steps: %f" % (bnd,))
            else:
                fact = step_factor # tau/sigma
                c['sigma'] = np.sqrt(bnd/fact)
                c['tau'] = bnd/c['sigma']
                self.gprox = self.g.prox(c['tau'])
                self.fconjprox = self.f.conj.prox(c['sigma'])
                logging.info("Constant steps: %f (%f | %f)"
                             % (bnd,c['sigma'],c['tau']))


    def solve(self, continue_at=None, step_bound=None, step_factor=1.0,
                    term_relgap=1e-5, term_infeas=None, term_maxiter=int(1e4),
                    granularity=500, use_gpu=True, steps="const"):
        i = self.itervars
        c = self.constvars

        if continue_at is not None:
            i['xk'][:], i['yk'][:] = continue_at

        i.update([(s,i['xk'].copy()) for s in ['xkp1','xgradk','xgradkp1']])
        i.update([(s,i['yk'].copy()) for s in ['ykp1','ygradk','ygradkp1','ygradbk']])

        self.linop(i['xk'], i['ygradk'])
        self.linop.adjoint(i['yk'], i['xgradk'])

        if term_infeas is None:
            term_infeas = term_relgap

        obj_p = obj_d = infeas_p = infeas_d = relgap = 0.
        c['theta'] = 1.0 # overrelaxation
        self.prepare_stepsizes(step_bound, step_factor, steps)
        if use_gpu: self.prepare_gpu()

        logging.info("Solving (steps<%d)..." % term_maxiter)

        with GracefulInterruptHandler() as interrupt_hdl:
            _iter = 0
            while _iter < term_maxiter:
                self.iteration_step()
                _iter += 1

                if interrupt_hdl.interrupted or _iter % granularity == 0:
                    if interrupt_hdl.interrupted:
                        print("Interrupt (SIGINT) at iter=%d" % _iter)

                    if use_gpu:
                        for n in ['xk','xgradk','yk','ygradk']:
                            self.gpu_itervars[n].get(ary=i[n])
                    obj_p, infeas_p = self.obj_primal(i['xk'], i['ygradk'])
                    obj_d, infeas_d = self.obj_dual(i['xgradk'], i['yk'])

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

                    if np.abs(relgap) < term_relgap and max(infeas_p, infeas_d) < term_infeas:
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
        return (self.itervars['xk'], self.itervars['yk'])
