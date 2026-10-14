"""One seed-3 solve followed by matrix-free, mixed-precision refinement."""
import time
import warnings

import gmpy2 as mp
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse.linalg import LinearOperator, lobpcg, minres

from . import atomistic as atom
from .config import PARAMETERS, MEAN_TOL, PRECISION_BITS, START_NAME


MAX_LBFGS_ITERATIONS = 5000
MAX_NEWTON_CORRECTIONS = 12
# Native displacement safeguard on the fixed 2*pi period; merit backtracking
# controls each correction independently of the number of atoms.
MAX_NEWTON_NATIVE_STEP = 0.05
CURVATURE_SOFT_TOL = 1.0e-8
EIGENPAIR_RESIDUAL_TOL = 1.0e-8
EIGENPAIR_REQUEST_TOL = 2.0e-10
MAX_CURVATURE_ITERATIONS = 200
MINRES_RTOL = 1.0e-11


def _tangent_operators(u,N,cutoff,parameters,deadline):
    full=atom.hessian_operator(u,N,cutoff,parameters,deadline)
    reflector=atom.mean_gauge_reflector(N)
    def action(p):
        return atom.tangent_reduce(full@atom.tangent_lift(p,reflector),reflector)
    tangent=LinearOperator((2*N,2*N),matvec=action,dtype=float)
    _,_,h1,h2,_=atom.geometry(N)
    symbols=[4*np.sin(np.pi*np.arange(n)/n)**2/hi+hi
             for n,hi in ((N,h1),(N+1,h2))]
    def precondition(p):
        lifted=atom.tangent_lift(p,reflector)
        parts=(lifted[:N],lifted[N:])
        result=np.concatenate([np.fft.ifft(np.fft.fft(z)/symbol).real
                               for z,symbol in zip(parts,symbols)])
        return atom.tangent_reduce(result,reflector)
    preconditioner=LinearOperator((2*N,2*N),matvec=precondition,dtype=float)
    # Bound the actual rank-two Householder update on the retained rows.
    # Multiplying generic ||Q|| bounds would add an unnecessary O(N) factor.
    product=full@reflector
    retained=reflector[:-1];product_retained=product[:-1]
    correction_bound=(2*abs(retained)*np.sum(abs(product_retained))
                      +2*abs(product_retained)*np.sum(abs(retained))
                      +4*abs(np.dot(reflector,product))*abs(retained)*np.sum(abs(retained)))
    bound=full.infinity_norm_bound+np.max(correction_bound)
    return tangent,preconditioner,reflector,float(bound),full.cached_bytes


def _initial_curvature_vectors(u,N,reflector):
    """Include the continuum translation tangent, without constraining it."""
    derivatives=[]
    for z in (u[:N],u[N:]):
        modes=np.fft.fftfreq(z.size,1/z.size)
        derivatives.append(np.fft.ifft(1j*modes*np.fft.fft(z)).real)
    phase=np.r_[-derivatives[0]-.5,-derivatives[1]+.5]
    x1=2*np.pi*np.arange(N)/N; x2=2*np.pi*np.arange(N+1)/(N+1)
    vectors=[phase,np.r_[np.sin(x1),np.sin(x2)],np.r_[np.cos(x1),np.cos(x2)]]
    trial=np.column_stack([atom.tangent_reduce(v,reflector) for v in vectors])
    return np.linalg.qr(trial)[0]


def curvature_screen(u,N,cutoff,parameters=PARAMETERS,deadline=None):
    """Screen three lowest modes in the complete mean-sum tangent space."""
    tangent,preconditioner,reflector,bound,cached_bytes=_tangent_operators(
        np.asarray(u,dtype=float),N,cutoff,parameters,deadline)
    initial=_initial_curvature_vectors(np.asarray(u,dtype=float),N,reflector)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        values,vectors=lobpcg(tangent,initial,M=preconditioner,largest=False,
                              tol=EIGENPAIR_REQUEST_TOL,maxiter=MAX_CURVATURE_ITERATIONS)
    ordering=np.argsort(values);values=values[ordering];vectors=vectors[:,ordering]
    residuals=np.array([np.linalg.norm(tangent@v-value*v)
                        for value,v in zip(values,vectors.T)])
    residual_max=float(np.max(residuals))
    tolerance=max(CURVATURE_SOFT_TOL,100*np.finfo(float).eps*bound+10*residual_max)
    resolved=bool(np.all(np.isfinite(values)) and np.all(np.isfinite(residuals))
                  and residual_max<=EIGENPAIR_RESIDUAL_TOL)
    status=('unresolved' if not resolved else 'negative' if values[0]<-tolerance
            else 'positive' if values[0]>tolerance else 'soft')
    report={"curvature_min_eigenvalue":float(values[0]),
            "curvature_second_eigenvalue":float(values[1]),
            "curvature_tolerance":float(tolerance),"curvature_status":status,
            "curvature_residual_max":residual_max,"eigenvalues":values.tolist(),
            "eigenpair_residuals":residuals.tolist(),"resolved":resolved,
            "operator_infinity_norm_bound":float(bound),"cached_hessian_bytes":int(cached_bytes),
            "warnings":[str(item.message) for item in recorded]}
    return report,tangent,preconditioner,reflector,vectors,values


def _newton_step(gradient,curvature,tangent,preconditioner,reflector,vectors,values,deadline):
    soft=vectors[:,values<=curvature['curvature_tolerance']]
    def project(p):
        p=np.asarray(p).reshape(-1)
        return p-soft@(soft.T@p) if soft.shape[1] else p
    def action(p):
        atom.check_deadline(deadline)
        p=np.asarray(p).reshape(-1)
        return project(tangent@project(p))+(p-project(p))
    def precondition(p):
        p=np.asarray(p).reshape(-1)
        return project(preconditioner@project(p))+(p-project(p))
    operator=LinearOperator(tangent.shape,matvec=action,dtype=float)
    inverse=LinearOperator(tangent.shape,matvec=precondition,dtype=float)
    rhs=-project(atom.tangent_reduce(np.asarray([float(g) for g in gradient]),reflector))
    if np.linalg.norm(rhs)==0:
        return np.zeros(len(gradient)),{"minres_info":0,"linear_relative_residual":0.0}
    correction,info=minres(operator,rhs,M=inverse,rtol=MINRES_RTOL,
                            maxiter=min(2000,2*tangent.shape[0]),check=False)
    defect=float(np.linalg.norm(operator@correction-rhs)/np.linalg.norm(rhs))
    attempts=[{"rtol":MINRES_RTOL,"info":int(info),"relative_residual":defect}]
    if info!=0 or not np.isfinite(defect) or defect>1e-6:
        # MINRES's stopping estimate can precede the explicit residual gate
        # for a tiny RHS and a small positive eigenvalue. Retry more tightly;
        # the independent acceptance threshold stays unchanged.
        correction,info=minres(operator,rhs,M=inverse,rtol=1e-14,
                                maxiter=min(2000,2*tangent.shape[0]),check=False)
        defect=float(np.linalg.norm(operator@correction-rhs)/np.linalg.norm(rhs))
        attempts.append({"rtol":1e-14,"info":int(info),"relative_residual":defect})
    if info!=0 or not np.isfinite(defect) or defect>1e-6:
        raise RuntimeError(f"Newton MINRES unresolved: info={info}, relative residual={defect:.3e}")
    return atom.tangent_lift(project(correction),reflector),{
        "minres_info":int(info),"linear_relative_residual":defect,"minres_attempts":attempts,
        "excluded_mode_count":int(soft.shape[1]),
        "negative_mode_count":int(np.count_nonzero(values < -curvature['curvature_tolerance']))}


def _ratio(d):
    return max(d['full_gradient_inf_norm']/d['atomistic_full_gradient_target'],
               d['reduced_gradient_inf_norm']/d['atomistic_reduced_gradient_target'],
               d['el_residual_inf_norm']/d['atomistic_el_target'])


def _polish(values,N,cutoff,parameters,deadline,accuracy_factor,force_step=False,checkpoint=None):
    history=[];normalization={};curvature={};linear={}
    with mp.context(precision=PRECISION_BITS):
        values=atom.project_mp(atom.mp_state(values),N)
        values,normalization=atom.normalize_mp(values,N)
        for iteration in range(MAX_NEWTON_CORRECTIONS+1):
            atom.check_deadline(deadline)
            energy,gradient,d=atom.mp_value_gradient(values,N,cutoff,parameters=parameters,deadline=deadline)
            native=np.array([float(v) for v in values])
            screened=curvature_screen(native,N,cutoff,parameters,deadline)
            curvature,tangent,preconditioner,reflector,vectors,eigenvalues=screened
            if (curvature['curvature_status']=='unresolved'
                    or eigenvalues[-1]<=curvature['curvature_tolerance']):
                break
            # An underconverged iterate can have negative phase curvature.
            # Positive-subspace corrections may reduce its full residual, but
            # a stationary endpoint with negative curvature is never accepted.
            if curvature['curvature_status']=='negative' and atom.stationary(d,accuracy_factor):
                break
            if atom.stationary(d,accuracy_factor) and (iteration>0 or not force_step):
                break
            if iteration==MAX_NEWTON_CORRECTIONS:
                break
            step,linear=_newton_step(gradient,curvature,tangent,preconditioner,
                                     reflector,vectors,eigenvalues,deadline)
            maximum=float(np.max(abs(step)))
            scale=min(1.,MAX_NEWTON_NATIVE_STEP/maximum) if maximum else 1.
            accepted=False;before_ratio=_ratio(d)
            for backtrack in range(13):
                atom.check_deadline(deadline)
                candidate=atom.project_mp([v+mp.mpfr(float(scale*s)) for v,s in zip(values,step)],N)
                candidate,norm=atom.normalize_mp(candidate,N)
                trial_energy,_,trial=atom.mp_value_gradient(candidate,N,cutoff,parameters=parameters,deadline=deadline)
                allowance=mp.mpfr(64*np.finfo(float).eps)*max(mp.mpfr(1),abs(energy))
                if (trial_energy<=energy+allowance and
                    (_ratio(trial)<before_ratio or (maximum==0 and atom.stationary(trial,accuracy_factor)))):
                    values=candidate;normalization=norm;accepted=True
                    history.append({"iteration":iteration+1,"step_inf":maximum*scale,
                                    "stationarity_ratio":_ratio(trial),"backtracks":backtrack,**linear})
                    if checkpoint is not None:
                        checkpoint({"N":N,"pair_cutoff":cutoff,"u_mp_strings":atom.state_strings(values),
                                    "u":np.array([float(v) for v in values]),"start":START_NAME,
                                    "stage":"newton","solver_provenance":"seed3_mp_newton_continuation",
                                    "diagnostics":trial,"polishing_iteration":iteration+1})
                    break
                scale*=.5
            if not accepted:
                history.append({"iteration":iteration+1,"status":"no_residual_decrease",**linear})
                break
        # Re-evaluate the actual returned state. No solver status substitutes for this check.
        _,_,d=atom.mp_value_gradient(values,N,cutoff,parameters=parameters,deadline=deadline)
        d.update({key:curvature[key] for key in ('curvature_min_eigenvalue',
                   'curvature_tolerance','curvature_status','curvature_residual_max') if key in curvature})
        mean_ok=(abs(d['mean_sum'])<=MEAN_TOL and max(abs(d['mean1']),abs(d['mean2']))
                 <=atom.geometry(N)[-1]/4+1e-14)
        accepted=(atom.stationary(d,accuracy_factor) and mean_ok
                  and curvature.get('curvature_status') in ('soft','positive'))
        return {"u":np.array([float(v) for v in values]),"u_mp_strings":atom.state_strings(values),
                "diagnostics":d,"curvature":curvature,"normalization":normalization,
                "accepted":bool(accepted),"newton_history":history,"precision_bits":PRECISION_BITS,
                "accuracy_factor":accuracy_factor}


def solve_case(N,reference=None,cutoff=160,deadline=None,checkpoint=None,parameters=PARAMETERS,
               resume_state=None):
    """Solve one genuine seed-3 start; reference is deliberately unused in optimization."""
    started=time.monotonic();initial=atom.random_start(N);initial_u=atom.build_mean_gauge(initial,N)
    if resume_state is not None:
        initial_u=np.asarray(resume_state.get('initial_u',initial_u),dtype=float)
        saved=resume_state.get('u',resume_state.get('u_mp_strings'))
        if saved is None:
            raise ValueError("A resumed L-BFGS checkpoint must contain native values.")
        initial=atom.reduced_coordinates(np.asarray(saved,dtype=float),N)
    targets=atom.accuracy_targets(N)
    last_checkpoint=[time.monotonic()]
    callback_count=[0]
    def checkpoint_lbfgs(point,force=False):
        callback_count[0]+=1
        now=time.monotonic()
        if checkpoint is not None and (force or now-last_checkpoint[0]>=30.):
            checkpoint({'N':N,'pair_cutoff':cutoff,'u':atom.build_mean_gauge(point,N),
                        'initial_u':initial_u,'start':START_NAME,'stage':'lbfgs',
                        'solver_provenance':'seed3_lbfgsb_continuation' if resume_state is not None
                                             else 'seed3_lbfgsb',
                        'checkpoint_iterations':callback_count[0],
                        'elapsed_seconds':now-started})
            last_checkpoint[0]=now
    checkpoint_lbfgs(initial,force=True)
    def objective(a):
        atom.check_deadline(deadline)
        energy,gradient=atom.value_gradient(atom.build_mean_gauge(a,N),N,cutoff,parameters,deadline)
        return energy,atom.reduce_gradient(gradient,N)
    point,energy,info=fmin_l_bfgs_b(objective,initial,m=50,
        pgtol=targets['atomistic_reduced_gradient_target'],factr=1.,
        maxiter=MAX_LBFGS_ITERATIONS,maxfun=100000,maxls=100,callback=checkpoint_lbfgs)
    checkpoint_lbfgs(point,force=True)
    lbfgs_seconds=time.monotonic()-started
    finite=np.all(np.isfinite(point)) and np.isfinite(energy) and np.all(np.isfinite(info['grad']))
    task=str(info['task']);eligible=bool(finite and (info['warnflag']==0 or
        (info['warnflag']==1 and ('LIMIT' in task.upper() or 'EXCEED' in task.upper()))))
    if not eligible:
        raise RuntimeError(f"L-BFGS endpoint is ineligible for refinement: {task}")
    def checkpoint_newton(snapshot):
        if checkpoint is not None:
            checkpoint({**snapshot,'initial_u':initial_u,'elapsed_seconds':time.monotonic()-started})
    result=_polish(atom.build_mean_gauge(point,N),N,cutoff,parameters,deadline,1.,
                   checkpoint=checkpoint_newton if checkpoint is not None else None)
    result.update({"N":N,"start":START_NAME,"pair_cutoff":cutoff,"initial_u":initial_u,
                   "optimizer_task":task,"optimizer_warnflag":int(info['warnflag']),
                   "iterations":int(info['nit']),"function_calls":int(info['funcalls']),
                   "solver_provenance":("seed3_resumed_lbfgsb_then_mp_matrix_free_newton"
                                         if resume_state is not None else
                                         "seed3_lbfgsb_then_mp_matrix_free_newton"),
                   "timings":{"lbfgs_seconds":lbfgs_seconds,"total_seconds":time.monotonic()-started}})
    return result


def refine_case(state,N=None,cutoff=None,deadline=None,parameters=PARAMETERS,
                accuracy_factor=.1,checkpoint=None):
    """Refine the same saved state at a cutoff/accuracy target; never add a start."""
    N=int(state['N']) if N is None else N
    cutoff=int(state['pair_cutoff']) if cutoff is None else cutoff
    started=time.monotonic()
    values=state['u_mp_strings'] if 'u_mp_strings' in state else state['u']
    initial_u=state.get('initial_u',np.asarray(values,dtype=float).copy())
    def checkpoint_newton(snapshot):
        if checkpoint is not None:
            checkpoint({**snapshot,'initial_u':initial_u,'elapsed_seconds':time.monotonic()-started})
    result=_polish(values,N,cutoff,parameters,deadline,accuracy_factor,
                   force_step=True,checkpoint=checkpoint_newton if checkpoint is not None else None)
    result.update({"N":N,"start":state.get('start',START_NAME),"pair_cutoff":cutoff,
                   "initial_u":initial_u,
                   "solver_provenance":"mp_refinement_from_saved_endpoint",
                   "timings":{"total_seconds":time.monotonic()-started}})
    return result
