"""Native N/N+1 two-chain LJ model, gauge maps, and precision-aware forces."""
import time

import gmpy2 as mp
import numpy as np
from scipy.sparse.linalg import LinearOperator

from .config import (PARAMETERS, EL_OVER_H2_TOL, PRECISION_BITS, RANDOM_MODES,
                     RANDOM_MAX_AMPLITUDE, SEED, evaluate_parameters)
from .model import pair_potential, pair_potential_prime, pair_potential_second


TWOPI = 2.0 * np.pi
PAIR_CHUNK = 256


def check_deadline(deadline):
    """Honor an absolute time.monotonic deadline between bounded work chunks."""
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Atomistic preparation reached its time budget.")


def geometry(N):
    """Return layer sizes, native spacings, and harmonic spacing."""
    if N < 2:
        raise ValueError("N must be at least two.")
    h1, h2 = TWOPI / N, TWOPI / (N + 1)
    return N, N + 1, h1, h2, 2 * h1 * h2 / (h1 + h2)


def project_mean_gauge(u, N):
    """Use common physical translation to impose mean(u1)+mean(u2)=0."""
    u = np.asarray(u, dtype=float)
    _, _, h1, h2, _ = geometry(N)
    c = (np.mean(u[:N]) + np.mean(u[N:])) / (h1 + h2)
    return np.r_[u[:N] - h2 * c, u[N:] - h1 * c]


def build_mean_gauge(a, N):
    """Reconstruct 2N+1 native values from the original 2N coordinates."""
    a = np.asarray(a, dtype=float)
    if a.shape != (2 * N,):
        raise ValueError(f"Expected {2*N} reduced coordinates.")
    return np.r_[a, -(N + 1) * np.mean(a[:N]) - np.sum(a[N:])]


def reduced_coordinates(u, N):
    return project_mean_gauge(u, N)[:-1]


def reduce_gradient(g, N):
    """Apply the transpose of the original reconstruction Jacobian."""
    return np.r_[g[:N] - ((N + 1) / N) * g[-1], g[N:-1] - g[-1]]


def random_start(N):
    """Return the unchanged seed-3, five-mode, amplitude-0.1 reduced start."""
    _, _, h1, h2, _ = geometry(N)
    x1, x2 = h1 * np.arange(N), h2 * np.arange(N + 1)
    modes = np.arange(1, RANDOM_MODES + 1, dtype=float)
    rng = np.random.default_rng(SEED)
    offset = rng.normal()
    sine = rng.normal(size=(2, modes.size)) / modes**2
    cosine = rng.normal(size=(2, modes.size)) / modes**2
    layers = [np.full(N, offset), np.full(N + 1, -offset)]
    for i, x in enumerate((x1, x2)):
        for k, mode in enumerate(modes):
            layers[i] += sine[i, k] * np.sin(mode*x) + cosine[i, k] * np.cos(mode*x)
    full = project_mean_gauge(np.concatenate(layers), N)
    full *= RANDOM_MAX_AMPLITUDE / np.max(np.abs(full))
    return reduced_coordinates(full, N)


def normalize(u, N):
    """Apply the original Appendix-B integer shift and common translation."""
    _, _, h1, h2, h = geometry(N)
    first, second = np.asarray(u)[:N], np.asarray(u)[N:]
    m1, m2 = np.mean(first), np.mean(second)
    d = int(np.rint((h2*m2-h1*m1)/(h1*h2)))
    c = -(m1+m2+h2*d)/(h1+h2)
    result = np.r_[np.roll(first, -d)+h2*(d+c), np.roll(second, -d)+h1*c]
    return result, {"normalization_d": d, "normalization_c": float(c),
                    "normalization_mean_bound": h/4}


def reflected_state(u, N):
    """Apply the exact odd reflection representative on both native grids."""
    first, second = np.asarray(u)[:N], np.asarray(u)[N:]
    return np.r_[-first[(-np.arange(N)) % N],
                 -second[(-np.arange(N+1)) % (N+1)]]


def _chunks(u, N, cutoff, deadline=None):
    _, _, h1, h2, h = geometry(N)
    images = np.arange(-cutoff, cutoff+1)[None, :]
    layers = (u[:N], u[N:])
    for i, (hi, ho, sign) in enumerate(((h1, h2, 1.), (h2, h1, -1.))):
        own, other = layers[i], layers[1-i]
        alpha, beta = h/ho, h/hi
        for begin in range(0, own.size, PAIR_CHUNK):
            check_deadline(deadline)
            n = np.arange(begin, min(begin+PAIR_CHUNK, own.size))[:, None]
            indices = (n+images) % other.size
            arguments = sign*h*n + alpha*own[n] - beta*other[indices] - TWOPI*beta*images
            yield i, begin, indices, arguments, hi, alpha, beta


def value_gradient(u, N, cutoff, parameters=PARAMETERS, deadline=None):
    """Evaluate the exact directed finite-image energy and full gradient."""
    u = np.asarray(u, dtype=float)
    if u.shape != (2*N+1,) or cutoff < 0:
        raise ValueError("Invalid full state or image cutoff.")
    _, _, h1, h2, _ = geometry(N)
    layers = (u[:N], u[N:])
    gradients = [(2*z-np.roll(z, 1)-np.roll(z, -1))/hi
                 for z, hi in zip(layers, (h1, h2))]
    energy = sum(.5*np.sum((np.roll(z, -1)-z)**2)/hi
                 for z, hi in zip(layers, (h1, h2)))
    for i, begin, indices, arguments, hi, alpha, beta in _chunks(u, N, cutoff, deadline):
        derivative = pair_potential_prime(arguments, parameters)
        energy += hi*np.sum(pair_potential(arguments, parameters))
        gradients[i][begin:begin+indices.shape[0]] += hi*alpha*np.sum(derivative, axis=1)
        gradients[1-i] += np.bincount(indices.ravel(),
                                     weights=(-hi*beta*derivative).ravel(),
                                     minlength=layers[1-i].size)
    return float(energy), np.concatenate(gradients)


def accuracy_targets(N):
    _, _, h1, h2, h = geometry(N)
    el = EL_OVER_H2_TOL*h*h
    return {"atomistic_el_target": el, "atomistic_full_gradient_target": min(h1,h2)*el,
            "atomistic_reduced_gradient_target": .5*min(h1,h2)*el}


def diagnostics(u, N, cutoff, parameters=PARAMETERS, deadline=None):
    energy, gradient = value_gradient(u, N, cutoff, parameters, deadline)
    _, _, h1, h2, h = geometry(N)
    el = max(np.max(abs(gradient[:N]))/h1, np.max(abs(gradient[N:]))/h2)
    return {"energy": energy, "full_gradient_inf_norm": float(np.max(abs(gradient))),
            "reduced_gradient_inf_norm": float(np.max(abs(reduce_gradient(gradient,N)))),
            "el_residual_inf_norm": float(el), "el_residual_over_h2": float(el/h**2),
            "mean1": float(np.mean(u[:N])), "mean2": float(np.mean(u[N:])),
            "mean_sum": float(np.mean(u[:N])+np.mean(u[N:])), **accuracy_targets(N)}


def stationary(diagnostics, accuracy_factor=1.0):
    pairs = (("full_gradient_inf_norm", "atomistic_full_gradient_target"),
             ("reduced_gradient_inf_norm", "atomistic_reduced_gradient_target"),
             ("el_residual_inf_norm", "atomistic_el_target"))
    return all(np.isfinite(diagnostics[value]) and
               diagnostics[value] <= accuracy_factor*diagnostics[target]
               for value, target in pairs)


def mean_gauge_reflector(N):
    normal = np.r_[np.full(N, 1/N), np.full(N+1, 1/(N+1))]
    normal /= np.linalg.norm(normal)
    normal[-1] += 1
    return normal/np.linalg.norm(normal)


def tangent_lift(vector, reflector):
    value = np.r_[np.asarray(vector).reshape(-1), 0.]
    return value-2*reflector*np.dot(reflector,value)


def tangent_reduce(vector, reflector):
    value = np.asarray(vector).reshape(-1)
    return (value-2*reflector*np.dot(reflector,value))[:-1]


def hessian_operator(u, N, cutoff, parameters=PARAMETERS, deadline=None):
    """Cache O(N M) pair weights and apply H without any dense Hessian."""
    u = np.asarray(u, dtype=float)
    _, _, h1, h2, _ = geometry(N)
    blocks = []
    bound = [np.full(N, 4/h1), np.full(N+1, 4/h2)]
    for i, begin, indices, arguments, hi, alpha, beta in _chunks(u,N,cutoff,deadline):
        weights = hi*pair_potential_second(arguments, parameters)
        blocks.append((i,begin,indices,weights,alpha,beta))
        bound[i][begin:begin+indices.shape[0]] += alpha*(alpha+beta)*np.sum(abs(weights),axis=1)
        bound[1-i] += np.bincount(indices.ravel(),
                                  weights=(beta*(alpha+beta)*abs(weights)).ravel(),
                                  minlength=bound[1-i].size)
    def action(vector):
        check_deadline(deadline)
        p = np.asarray(vector).reshape(-1)
        layers = (p[:N],p[N:])
        out = [(2*z-np.roll(z,1)-np.roll(z,-1))/hi
               for z,hi in zip(layers,(h1,h2))]
        for i,begin,indices,weights,alpha,beta in blocks:
            check_deadline(deadline)
            own = layers[i][begin:begin+indices.shape[0]]
            response = weights*(alpha*own[:,None]-beta*layers[1-i][indices])
            out[i][begin:begin+indices.shape[0]] += alpha*np.sum(response,axis=1)
            out[1-i] += np.bincount(indices.ravel(),weights=(-beta*response).ravel(),
                                     minlength=layers[1-i].size)
        return np.concatenate(out)
    operator = LinearOperator((2*N+1,2*N+1),matvec=action,dtype=float)
    operator.infinity_norm_bound = max(np.max(z) for z in bound)
    operator.cached_bytes = sum(idx.nbytes+w.nbytes for _,_,idx,w,_,_ in blocks)
    return operator


def mp_state(values, bits=PRECISION_BITS):
    """Load native values without passing saved decimal strings through float."""
    with mp.context(precision=bits):
        return [mp.mpfr(value) for value in values]


def state_strings(values):
    return [format(value, '.45g') for value in values]


def project_mp(values, N):
    """Project using the exact rational ratio h2/h1=N/(N+1)."""
    mean1, mean2 = sum(values[:N])/N, sum(values[N:])/(N+1)
    common = (mean1+mean2)/(2*N+1)
    return [v-N*common for v in values[:N]]+[v-(N+1)*common for v in values[N:]]


def normalize_mp(values, N, bits=PRECISION_BITS):
    with mp.context(precision=bits):
        values = mp_state(values,bits)
        pi = mp.const_pi(); h1=2*pi/N; h2=2*pi/(N+1)
        first,second=values[:N],values[N:]
        m1,m2=sum(first)/N,sum(second)/(N+1)
        d=int(mp.rint((h2*m2-h1*m1)/(h1*h2)))
        c=-(m1+m2+h2*d)/(h1+h2)
        result=[first[(j+d)%N]+h2*(d+c) for j in range(N)]
        result += [second[(j+d)%(N+1)]+h1*c for j in range(N+1)]
        return result,{"normalization_d":d,"normalization_c":float(c),
                       "normalization_mean_bound":float(pi/(2*N+1))}


def mp_value_gradient(values,N,cutoff,bits=PRECISION_BITS,parameters=PARAMETERS,deadline=None):
    """Evaluate the same finite-cutoff model with MPFR arithmetic and exact pi."""
    with mp.context(precision=bits):
        u=mp_state(values,bits)
        if len(u)!=2*N+1:
            raise ValueError("Invalid multiprecision native-state dimension.")
        pi=mp.const_pi(); h1=2*pi/N; h2=2*pi/(N+1); h=4*pi/(2*N+1)
        epsilon,sigma,distance=evaluate_parameters(parameters,number=mp.mpfr,pi=pi)
        sigma6=sigma**6; separation2=distance**2
        layers=(u[:N],u[N:]); gradients=[]; energy=mp.mpfr(0)
        for z,hi in zip(layers,(h1,h2)):
            size=len(z)
            gradients.append([(2*z[j]-z[(j-1)%size]-z[(j+1)%size])/hi for j in range(size)])
            energy += sum((z[(j+1)%size]-z[j])**2 for j in range(size))/(2*hi)
        for i,(hi,ho,sign) in enumerate(((h1,h2,1),(h2,h1,-1))):
            own,other=layers[i],layers[1-i]; alpha=h/ho; beta=h/hi
            own_factor=hi*alpha; other_factor=hi*beta
            image_offsets=[2*pi*beta*m for m in range(-cutoff,cutoff+1)]
            for n in range(len(own)):
                if n%8==0: check_deadline(deadline)
                base=sign*h*n+alpha*own[n]
                for m,image in zip(range(-cutoff,cutoff+1),image_offsets):
                    j=(n+m)%len(other)
                    separation=base-beta*other[j]-image
                    radius2=separation2+separation*separation
                    sr6=sigma6/(radius2*radius2*radius2)
                    energy += hi*4*epsilon*(sr6*sr6-sr6)
                    derivative=24*epsilon*separation*(sr6-2*sr6*sr6)/radius2
                    gradients[i][n] += own_factor*derivative
                    gradients[1-i][j] -= other_factor*derivative
        gradient=gradients[0]+gradients[1]
        reduced=[z-mp.mpfr(N+1)/N*gradient[-1] for z in gradient[:N]]
        reduced += [z-gradient[-1] for z in gradient[N:-1]]
        el=max(max(abs(z)/h1 for z in gradient[:N]),max(abs(z)/h2 for z in gradient[N:]))
        d={"energy":float(energy),"full_gradient_inf_norm":float(max(map(abs,gradient))),
           "reduced_gradient_inf_norm":float(max(map(abs,reduced))),
           "el_residual_inf_norm":float(el),"el_residual_over_h2":float(el/h**2),
           "mean1":float(sum(u[:N])/N),"mean2":float(sum(u[N:])/(N+1)),
           "mean_sum":float(sum(u[:N])/N+sum(u[N:])/(N+1)),
           "precision_bits":bits,**accuracy_targets(N)}
        return energy,gradient,d
