
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
        self.itervars = {} # xk, yk, etc.
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

    def pd_gap(self):
        i = self.itervars
        obj_p, infeas_p = self.obj_primal(i['xk'], i['ygradk'])
        obj_d, infeas_d = self.obj_dual(i['xgradk'], i['yk'])
        return {
            'objp': obj_p,
            'objd': obj_d,
            'infeasp': infeas_p,
            'infeasd': infeas_d,
            'relgap': (obj_p - obj_d) / max(np.spacing(1), obj_d)
        }

    def pd_res(self, eps_absp, eps_relp, eps_absd, eps_reld):
        i = self.itervars
        return {
            'resp': i['res_pk'],
            'resd': i['res_dk'],
            'epsp': np.sqrt(i['xk'].size)*eps_absp + eps_relp*i['gnorm_pk'],
            'epsd': np.sqrt(i['yk'].size)*eps_absd + eps_reld*i['gnorm_dk'],
        }


    def prepare_gpu(self, type_t="double"):
        import opymize.tools.gpu # gpu init
        from pycuda import gpuarray
        from pycuda.elementwise import ElementwiseKernel
        from pycuda.reduction import ReductionKernel
        import pycuda.driver

        i = self.itervars
        c = self.constvars

        self.gprox.prepare_gpu(type_t=type_t)
        self.fconjprox.prepare_gpu(type_t=type_t)
        self.linop.prepare_gpu(type_t=type_t)

        p = ("*","[i]") if 'precond' in c else ("","")
        self.gpu_kernels = {
            'take_step': ElementwiseKernel(
                "%s *zkp1, %s *zk, char direction, %s %sstepsize, %s *zgradk" \
                    % (type_t, type_t, type_t, p[0], type_t),
                "zkp1[i] = zk[i] + direction*stepsize%s*zgradk[i]" % p[1]),
            'overrelax': ElementwiseKernel(
                "%s *ygradbk, %s theta, %s *ygradkp1, %s *ygradk" \
                    % (type_t, type_t, type_t, type_t),
                "ygradbk[i] = (1 + theta)*ygradkp1[i] - theta*ygradk[i]"),
            'advance': ElementwiseKernel(
                "%s *zk, %s *zkp1, %s *zgradk, %s *zgradkp1" \
                    % (type_t, type_t, type_t, type_t),
                "zk[i] = zkp1[i]; zgradk[i] = zgradkp1[i]"),
            'residual': 0 if 'adaptive' not in c else ReductionKernel(
                np.float64, neutral="0", reduce_expr="a+b",
                map_expr="pow((zk[i] - zkp1[i])/a - b*zgk[i] + c*zgkp1[i], 2)",
                arguments="%s a, %s *zk, %s *zkp1, %s b, %s *zgk, %s c, %s *zgkp1"\
                            % (type_t,)*7),
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

        (free,total) = pycuda.driver.mem_get_info()
        logging.info("CUDA kernels prepared for GPU (%d/%d MB available)"
                     % (free//(1024*1024), total//(1024*1024)))

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

    def residual(self, a, zk, zkp1, b, zgk, c, zgkp1):
        if type(zk) is np.ndarray:
            return norm((zk - zkp1)/a - b*zgk + c*zgkp1, ord=2)
        else:
            return np.sqrt(gpu_kernel(a, zk, zkp1, b, zgk, c, zgkp1).get())

    def advance(self, zk, zkp1, zgradk, zgradkp1):
        if type(zk) is np.ndarray:
            zk[:], zgradk[:] = zkp1, zgradkp1
        else:
            self.gpu_kernels['advance'](zk, zkp1, zgradk, zgradkp1)

    def iteration_step(self, compute_gnorms=False):
        v = self.gpu_itervars if self.use_gpu else self.itervars
        xk, xkp1, xgradk, xgradkp1 = v['xk'], v['xkp1'], v['xgradk'], v['xgradkp1']
        yk, ykp1, ygradk, ygradkp1 = v['yk'], v['ykp1'], v['ygradk'], v['ygradkp1']
        ygradbk = v['ygradbk']

        i = self.itervars
        c = self.constvars
        theta = c['theta']

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
        self.overrelax(ygradbk, theta, ygradkp1, ygradk)

        # --- duals:
        self.take_step(ykp1, yk, 1, sigma, ygradbk)
        self.fconjprox(ykp1)
        self.linop.adjoint(ykp1, xgradkp1)

        # --- step sizes:
        if 'adaptive' in c:
            res_argsp = [  tau, xk, xkp1,     1, xgradk,     1, xgradkp1]
            res_argsd = [sigma, yk, ykp1, theta, ygradk, theta, ygradkp1]
            i['res_pk'] = self.residual(*res_argsp)
            i['res_dk'] = self.residual(*res_argsd)
            if compute_gnorms:
                res_argsp[5] -= 1
                res_argsd[5] += 1
                i['gnorm_pk'] = self.residual(*res_argsp)
                i['gnorm_dk'] = self.residual(*res_argsd)

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


    def solve(self, continue_at=None, precision="double",
                    steps="const", step_bound=None, step_factor=1.0,
                    term_pd_gap=1e-5, term_pd_res=None, term_maxiter=int(5e4),
                    granularity=5000, use_gpu=True):
        i = self.itervars
        c = self.constvars

        dtype = np.float64 if precision == "double" else np.float32
        i['xk'] = self.g.x.new(dtype=dtype)
        i['yk'] = self.f.x.new(dtype=dtype)

        if continue_at is not None:
            i['xk'][:], i['yk'][:] = continue_at

        i.update([(s,i['xk'].copy()) for s in ['xkp1','xgradk','xgradkp1']])
        i.update([(s,i['yk'].copy()) for s in ['ykp1','ygradk','ygradkp1','ygradbk']])

        self.linop(i['xk'], i['ygradk'])
        self.linop.adjoint(i['yk'], i['xgradk'])

        pd_res_mode = term_pd_res is not None
        if pd_res_mode:
            assert steps == "adaptive"
        elif type(term_pd_gap) is not tuple:
            # tolerances for relative pd-gap and infeasibilities
            term_pd_gap = (term_pd_gap, term_pd_gap)

        obj_p = obj_d = infeas_p = infeas_d = relgap = 0.
        c['theta'] = 1.0 # overrelaxation
        self.prepare_stepsizes(step_bound, step_factor, steps)
        if use_gpu: self.prepare_gpu(type_t=precision)

        logging.info("# primal variables: %d" % i['xk'].size)
        logging.info("# dual variables: %d" % i['yk'].size)

        mem = (i['xk'].nbytes*4 + i['yk'].nbytes*5) // (1024*1024)
        logging.info("Memory requirements: %d MB" % mem)

        logging.info("Solving (steps<%d)..." % term_maxiter)

        with GracefulInterruptHandler() as interrupt_hdl:
            _iter = 0
            while _iter < term_maxiter:
                check_step = (_iter % granularity == 0)
                self.iteration_step(compute_gnorms=check_step and pd_res_mode)
                _iter += 1

                if interrupt_hdl.interrupted or check_step:
                    if interrupt_hdl.interrupted:
                        print("Interrupt (SIGINT) at iter=%d" % _iter)

                    if use_gpu:
                        for n in self.gpu_itervars.keys():
                            self.gpu_itervars[n].get(ary=i[n])

                    if pd_res_mode:
                        info = self.pd_res(*term_pd_res)
                        logging.info("#{:6d}: res_p = {: 9.6g} ({: 9.6g}), " \
                            "res_d = {: 9.6g} ({: 9.6g}), ".format(
                            _iter, info['resp'], info['epsp'],
                            info['resd'], info['epsd']
                        ))
                        test_p = info['resp'] < info['epsp']
                        test_d = info['resd'] < info['epsd']
                        test_term = test_p and test_d
                    else:
                        info = self.pd_gap()
                        logging.info("#{:6d}: objp = {: 9.6g} ({: 9.6g}), " \
                            "objd = {: 9.6g} ({: 9.6g}), " \
                            "gap = {: 9.6g}, " \
                            "relgap = {: 9.6g} ".format(
                            _iter, info['objp'], info['infeasp'],
                            info['objd'], info['infeas_d'],
                            info['objp'] - info['objd'],
                            info['relgap']
                        ))
                        test_err = np.abs(info['relgap']) < term_pd_gap[0]
                        infeasp, infeasd = info['infeasp'], info['infeasd']
                        test_infeas = max(infeasp, infeasd) < term_pd_gap[1]
                        test_term = test_err and test_infeas

                    if test_term or interrupt_hdl.interrupted:
                        break

        info['iter'] = _iter
        return info

    @property
    def state(self):
        return (self.itervars['xk'], self.itervars['yk'])
