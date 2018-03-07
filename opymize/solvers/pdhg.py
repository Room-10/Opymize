
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
        self.extravars = {}

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
            'step_primal': ElementwiseKernel(
                "double *xkp1, double *xk, double %stau, double *xgradk" % p[0],
                "xkp1[i] = xk[i] - tau%s*xgradk[i]" % p[1]),
            'step_dual': ElementwiseKernel(
                "double *ykp1, double *yk, double %ssigma, double *ygradk" % p[0],
                "ykp1[i] = yk[i] + sigma%s*ygradk[i]" % p[1]),
            'overrelax': ElementwiseKernel(
                "double *ygradbk, double *ygradkp1, double *ygradk, double theta",
                "ygradbk[i] = (1 + theta)*ygradkp1[i] - theta*ygradk[i]"),
            'advance': ElementwiseKernel(
                "double *zk, double *zkp1, double *zgradk, double *zgradkp1",
                "zk[i] = zkp1[i]; zgradk[i] = zgradkp1[i]"),
            'residual': 0 if 'adaptive' not in c else ReductionKernel(
                np.float64, neutral="0", reduce_expr="a+b",
                map_expr="fabs((zk[i] - zkp1[i])/step - (zgradk[i] - zgradkp1[i]))",
                arguments="double step, double *zk, double *zkp1, "\
                         +"double *zgradk, double *zgradkp1"),
            'linop': self.linop,
            'linop_adjoint': self.linop.adjoint,
            'prox_primal': self.gprox,
            'prox_dual': self.fconjprox,
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

        logging.info("CUDA kernels prepared for GPU")

    def iteration_step_gpu(self):
        i = self.itervars
        c = self.constvars
        gi = self.gpu_itervars
        gc = self.gpu_constvars
        gk = self.gpu_kernels
        if 'precond' in c:
            tau = gc['xtau']
            sigma = gc['ysigma']
        else:
            tau = i['tauk'] if 'adaptive' in c else c['tau']
            sigma = i['sigmak'] if 'adaptive' in c else c['sigma']

        # --- primals:
        gk['step_primal'](gi['xkp1'], gi['xk'], tau, gi['xgradk'])
        gk['prox_primal'](gi['xkp1'])
        gk['linop'](gi['xkp1'], gi['ygradkp1'])
        gk['overrelax'](gi['ygradbk'], gi['ygradkp1'], gi['ygradk'], c['theta'])

        # --- duals:
        gk['step_dual'](gi['ykp1'], gi['yk'], sigma, gi['ygradbk'])
        gk['prox_dual'](gi['ykp1'])
        gk['linop_adjoint'](gi['ykp1'], gi['xgradkp1'])

        # --- step sizes:
        if 'adaptive' in c and i['alphak'] > 1e-10:
            i['res_pk'] = gk['residual'](i['tauk'],
                gi['xk'], gi['xkp1'], gi['xgradk'], gi['xgradkp1']).get()
            i['res_dk'] = gk['residual'](i['sigmak'],
                gi['yk'], gi['ykp1'], gi['ygradk'], gi['ygradkp1']).get()

            if i['res_pk'] > c['s']*i['res_dk']*c['Delta']:
                i['tauk'] *= 1.0/(1.0 - i['alphak'])
                i['sigmak'] *= (1.0 - i['alphak'])
                i['alphak'] *= c['eta']
            if i['res_pk'] < c['s']*i['res_dk']/c['Delta']:
                i['tauk'] *= (1.0 - i['alphak'])
                i['sigmak'] *= 1.0/(1.0 - i['alphak'])
                i['alphak'] *= c['eta']

        # --- update
        gk['advance'](gi['xk'], gi['xkp1'], gi['xgradk'], gi['xgradkp1'])
        gk['advance'](gi['yk'], gi['ykp1'], gi['ygradk'], gi['ygradkp1'])

    def iteration_step(self):
        i = self.itervars
        c = self.constvars
        if 'precond' in c:
            tau = c['xtau']
            sigma = c['ysigma']
        elif 'adaptive' in c:
            tau = i['tauk']
            sigma = i['sigmak']
        else:
            tau = c['tau']
            sigma = c['sigma']

        # --- primals:
        i['xkp1'][:] = i['xk'] - tau*i['xgradk']
        self.gprox(i['xkp1'], i['xkp1'])
        self.linop(i['xkp1'], i['ygradkp1'])
        i['ygradbk'][:] = (1 + c['theta'])*i['ygradkp1'] - c['theta']*i['ygradk']

        # --- duals:
        i['ykp1'][:] = i['yk'] + sigma*i['ygradbk']
        self.fconjprox(i['ykp1'], i['ykp1'])
        self.linop.adjoint(i['ykp1'], i['xgradkp1'])

        # --- step sizes:
        if 'adaptive' in c and i['alphak'] > 1e-10:
            i['res_pk'] = norm((i['xk'][:] - i['xkp1'][:])/i['tauk']
                               - (i['xgradk'][:] - i['xgradkp1'][:]), ord=1)
            i['res_dk'] = norm((i['yk'][:] - i['ykp1'][:])/i['sigmak']
                               - (i['ygradk'][:] - i['ygradkp1'][:]), ord=1)

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
        i['xk'][:] = i['xkp1']
        i['xgradk'][:] = i['xgradkp1']
        i['yk'][:] = i['ykp1']
        i['ygradk'][:] = i['ygradkp1']

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

        i['xkp1'] = i['xk'].copy()
        i['xgradk'] = i['xk'].copy()
        i['xgradkp1'] = i['xk'].copy()
        i['ykp1'] = i['yk'].copy()
        i['ygradk'] = i['yk'].copy()
        i['ygradkp1'] = i['yk'].copy()
        i['ygradbk'] = i['yk'].copy()

        self.linop(i['xk'], i['ygradk'])
        self.linop.adjoint(i['yk'], i['xgradk'])

        if term_infeas is None:
            term_infeas = term_relgap

        obj_p = obj_d = infeas_p = infeas_d = relgap = 0.
        c['theta'] = 1.0 # overrelaxation
        self.prepare_stepsizes(step_bound, step_factor, steps)

        if use_gpu:
            self.prepare_gpu()
            iteration_step = self.iteration_step_gpu
        else:
            iteration_step = self.iteration_step

        logging.info("Solving (steps<%d)..." % term_maxiter)

        with GracefulInterruptHandler() as interrupt_hdl:
            _iter = 0
            while _iter < term_maxiter:
                iteration_step()
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
