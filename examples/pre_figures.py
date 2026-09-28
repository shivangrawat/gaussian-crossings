"""Reproduce the six PRE figures and the supplementary integration schematic.

Original notebooks and parameter choices are recorded in NOTEBOOKS.  The old
notebooks are retained as historical sources.  This runner uses signed-erf
formulae, stable conditional covariances, and checked adaptive quadrature.
Numerical archives are separate from the manuscript; no TeX source is edited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
from numpy.polynomial.polynomial import polymul, polyval
import scipy
from scipy.integrate import quad, quad_vec
from scipy.linalg import expm
from scipy.optimize import brentq, minimize_scalar
from scipy.special import beta, betainc, erf, erfc, ndtr, owens_t

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    '1': 'examples/fig1_illustration.ipynb',
    '2': 'examples/damped_harmonic_oscillator/sdho_zeta.ipynb',
    '3': 'examples/damped_harmonic_oscillator/theory_simulation_upcrossings.ipynb',
    '4a-c': 'examples/damped_harmonic_oscillator/fano_scan_omega0.ipynb',
    '4d-f': 'examples/damped_harmonic_oscillator/fano_scan_temp.ipynb',
    '5': 'examples/OU_noise/OU_noise.ipynb',
    '6': 'examples/rational_quadratic/rational_quadratic.ipynb',
    'S1': None,
}
SPEC = importlib.util.spec_from_file_location(
    'pre_figure3', ROOT / 'examples/damped_harmonic_oscillator/pre_figure3.py')
fig3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fig3)
PI = math.pi


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + '\n')


@lru_cache(maxsize=16)
def rq_series(shape):
    c = np.zeros(40)
    c[0] = 1.
    for n in range(1, len(c)//2):
        c[2*n] = -c[2*n-2] * (1/n/2 if math.isinf(shape)
                                          else (shape+n-1)/(2*shape*n))
    D = -c[2:]
    P = np.arange(2, len(c))*c[2:]
    q = -np.arange(2, len(c))*np.arange(1, len(c)-1)*c[2:]
    qp = q.copy()
    qp[0] += 1
    bnum = polymul(qp, D) - polymul(P, P)
    # Analytic even covariances have Var(S|X0=Xt)=O(t^4).
    if np.max(np.abs(bnum[:4])) > 1e-13:
        raise ArithmeticError('unexpected lower-order conditional-variance term')
    return D, P, q, bnum[4:]


def rq_parts(t, shape):
    if t <= 0:
        raise ValueError('positive lag required')
    if t < .05:
        D, P, qc, bn = rq_series(shape)
        d = t*t*polyval(t, D)
        p = t*polyval(t, P)
        q = polyval(t, qc)
        a = t*t*(-polyval(t, qc[2:])-polyval(t, P)**2/(2-d))/2
        b = t**4*polyval(t, bn)/(2*polyval(t, D))
        r = 1-d
    else:
        if math.isinf(shape):
            r = math.exp(-t*t/2)
            d = -math.expm1(-t*t/2)
            p = -t*r
            q = (1-t*t)*r
        else:
            logc = math.log1p(t*t/(2*shape))
            r = math.exp(-shape*logc)
            d = -math.expm1(-shape*logc)
            p = -t*math.exp(-(shape+1)*logc)
            q = math.exp(-(shape+1)*logc) - (shape+1)/shape*t*t*math.exp(-(shape+2)*logc)
        a = (1-q-p*p/(2-d))/2
        b = (1+q-p*p/d)/2
    if min(a,b,d,2-d) <= 0:
        raise FloatingPointError((t,shape,a,b,d))
    return r,p,q,d,a,b


def normalized_pair(parts, levels, historical=False):
    """K(t;u)/nu(u), avoiding exp(+u^2/2) even at large levels.

    S=(V0+Vt)/2 and D=(Vt-V0)/2 have independent conditional variances
    b and a; D has mean m. The signed error-function argument is essential.
    historical=True is used ONLY to quantify the old implementation error.
    """
    _,p,_,d,a,b = parts
    u = np.asarray(levels)
    m = p*u/(2-d)
    arg = m*math.sqrt(b/(2*a*(a+b)))
    if historical:
        arg = np.abs(arg)
    bracket = (2*math.sqrt(a*b)*np.exp(-m*m/(2*a))
               + math.sqrt(2*PI)*m*math.sqrt(a+b)*np.exp(-m*m/(2*(a+b)))*erf(arg)
               + 4*PI*(b-a-m*m)*owens_t(m/math.sqrt(a+b), math.sqrt(b/a)))
    value = np.exp(-u*u*d/(2*(2-d)))*bracket/(2*PI*math.sqrt(d*(2-d)))
    if not np.all(np.isfinite(value)):
        raise FloatingPointError('nonfinite pair intensity')
    if not historical and np.min(value) < -2e-13:
        raise FloatingPointError(('negative pair intensity', parts, float(np.min(value))))
    return value


def positive_pair(parts, level):
    """Independent positive conditional Gaussian integral (no erf/Owen formula)."""
    _,p,_,d,a,b = parts
    m = p*level/(2-d)
    def integrand(y):
        phi = math.exp(-y*y/2)/math.sqrt(2*PI)
        h = (1-y*y)*ndtr(-y) + y*phi
        z1,z2 = (math.sqrt(b)*y-m)/math.sqrt(a),(math.sqrt(b)*y+m)/math.sqrt(a)
        return h*(math.exp(-z1*z1/2)+math.exp(-z2*z2/2))/math.sqrt(2*PI)
    v,e = quad(integrand,0,12,epsabs=5e-14,epsrel=2e-11,limit=150)
    moment = b*math.sqrt(b/a)*v
    return math.exp(-level*level*d/(2*(2-d)))*moment/math.sqrt(d*(2-d))


def integrate_vector(fn, end, tol):
    breaks = [x for x in (.0001,.001,.01,.05,.2,1.,4.,12.,40.,128.,512.,2048.) if x < end]
    value,error,info = quad_vec(fn,0,end,epsabs=tol,epsrel=tol,norm='max',
                                points=breaks,limit=1200,full_output=True)
    if not info.success or not np.all(np.isfinite(value)):
        raise ArithmeticError(f'quadrature failed: {info.message}; error={error}')
    return value,float(error)


def sdho_fano(zeta, levels, tol=2e-10, cutoff=40., historical=False, direct=False):
    levels = np.atleast_1d(np.asarray(levels,dtype=float))
    nu = np.exp(-levels*levels/2)/(2*PI)
    slow = zeta if zeta <= 1 else 1/(zeta+math.sqrt(zeta*zeta-1))
    end = cutoff/slow
    def fn(t):
        parts = fig3.covariance_parts(t,float(zeta))
        pair = np.array([positive_pair(parts,float(u)) for u in levels]) if direct else normalized_pair(parts,levels,historical)
        return 2*(pair-nu)
    value,error = integrate_vector(fn,end,tol)
    return 1+value,error


def rq_tail(shape, end, levels):
    """First two powers of r plus q; convergence is checked by doubling cutoff.

    K/nu-nu = nu*[u^2*r + (pi/2)*q + (u^2-1)^2*r^2/2 + ...].
    The omitted terms are checked numerically, not claimed as a rigorous bound.
    """
    if math.isinf(shape):
        integrals = [math.sqrt(PI/(2*j))*erfc(math.sqrt(j/2)*end) for j in (1,2)]
    else:
        x = 1/(1+end*end/(2*shape))
        integrals = [math.sqrt(shape/2)*beta(j*shape-.5,.5)*betainc(j*shape-.5,.5,x) for j in (1,2)]
    p = rq_parts(end,shape)[1]
    u = np.asarray(levels)
    nu = np.exp(-u*u/2)/(2*PI)
    return 2*nu*(u*u*integrals[0]+PI/2*p+.5*(u*u-1)**2*integrals[1])


def rq_fano(shape,levels,tol=2e-10,end=2048.,historical=False,direct=False):
    levels = np.atleast_1d(np.asarray(levels,dtype=float))
    nu = np.exp(-levels*levels/2)/(2*PI)
    def fn(t):
        parts = rq_parts(t,shape)
        pair = np.array([positive_pair(parts,float(u)) for u in levels]) if direct else normalized_pair(parts,levels,historical)
        return 2*(pair-nu)
    value,error = integrate_vector(fn,end,tol)
    # The sign error changes a term quadratic in p, so it does not change
    # the retained r, q, r^2 asymptotic terms.
    return 1+value+rq_tail(shape,end,levels),error


def validate():
    """Checks target real failure modes: cancellation, sign loss and truncation."""
    import mpmath as mp
    mp.mp.dps = 65
    results = {}
    # Independent high-precision Schur-complement arithmetic at tiny lags.
    max_cov = 0.
    for shape in (.75,1.,2.,5.,math.inf):
        for t in (1e-7,1e-4,.049,.051,1.,8.):
            tt = mp.mpf(str(t))
            hh = mp.mpf(str(shape)) if math.isfinite(shape) else None
            rfun = (lambda x: mp.exp(-x*x/2)) if hh is None else (lambda x: (1+x*x/(2*hh))**(-hh))
            r,p,q = rfun(tt),mp.diff(rfun,tt),-mp.diff(rfun,tt,2)
            ref = [r,p,q,1-r,(1-q-p*p/(1+r))/2,(1+q-p*p/(1-r))/2]
            got = rq_parts(t,shape)
            for j in (3,4,5):
                rel = abs(got[j]/float(ref[j])-1)
                max_cov = max(max_cov,rel)
                if rel > 3e-7:
                    raise AssertionError(('RQ covariance',shape,t,j,rel))
    results['rq_small_lag_covariance_max_relative_error'] = max_cov
    max_cov = 0.
    for zeta in (.5,1.,2.,3.,5.05):
        for t in (1e-7,.001,.049,.051,1.,8.):
            M = expm(np.array([[0.,1.],[-1.,-2*zeta]])*t)
            r,p,q,*_ = fig3.covariance_parts(t,zeta)
            err = max(abs(r-M[0,0]),abs(p-M[0,1]*-1),abs(q-M[1,1]))
            max_cov = max(max_cov,err)
            if err > 2e-12:
                raise AssertionError(('SDHO covariance',zeta,t,err))
    results['sdho_matrix_exponential_max_abs_error'] = max_cov
    max_pair = 0.
    for family,params,parts in [('sdho',(.5,1.,2.,5.05),fig3.covariance_parts),
                                 ('rq',(.75,1.,2.,5.,math.inf),rq_parts)]:
        for param in params:
            for t in (.0001,.01,.1,1.,4.,12.):
                cov = parts(t,param)
                for level in (0.,.5,2.,4.,8.):
                    got = float(normalized_pair(cov,level))
                    ref = positive_pair(cov,level)
                    max_pair = max(max_pair,abs(got-ref))
                    if abs(got-ref) > 2e-10:
                        raise AssertionError(('independent pair integral',family,param,t,level,got,ref))
                    np.testing.assert_allclose(got,normalized_pair(cov,-level),atol=1e-15,rtol=1e-14)
    results['positive_pair_integral_max_abs_error'] = max_pair
    # Tie the stable rearrangement back to BOTH classes used by the original
    # notebooks. Their fixed-grid long-time quadrature is not used for export.
    import torch
    from gaussian_crossings.formula.formula import GaussianUpCrossings
    from gaussian_crossings.formula.formula_dimless import GaussianUpCrossingsDimless
    from gaussian_crossings.process.correlation_functions import (
        r_damped_harmonic_oscillator_noise, r_rational_quadratic, r_squared_exp)
    torch.set_default_dtype(torch.float64)
    max_package = 0.
    for cls in (GaussianUpCrossings,GaussianUpCrossingsDimless):
        for rfun,kwargs,parts in [
            (r_damped_harmonic_oscillator_noise,dict(zeta=.5,temp=1.,omega0=1.),lambda t: fig3.covariance_parts(t,.5)),
            (r_damped_harmonic_oscillator_noise,dict(zeta=2.,temp=1.,omega0=1.),lambda t: fig3.covariance_parts(t,2.)),
            (r_rational_quadratic,dict(alpha=.75,sigma=1.,tau=1.),lambda t: rq_parts(t,.75)),
            (r_squared_exp,dict(sigma=1.,tau=1.),lambda t: rq_parts(t,math.inf))]:
            for u in (-2.,0.,2.):
                model=cls(r_func=rfun,u=u,**kwargs)
                nu=math.exp(-u*u/2)/(2*PI)
                for t in (.1,1.,4.):
                    got=float(model.upcrossing_integrand(torch.tensor(t)))
                    ref=float(normalized_pair(parts(t),u))*nu-nu*nu
                    max_package=max(max_package,abs(got-ref))
    if max_package > 2e-9:
        raise AssertionError(('current package integrand',max_package))
    results['current_package_integrand_max_abs_error'] = max_package
    # Physical OU time/amplitude rescaling, compared to its two-exponential ACF.
    max_ou = 0.
    for kappa in (.01,.1,.3,.4):
        zeta = (1+kappa)/(2*math.sqrt(kappa))
        for normalized_t in (.01,.1,1.,10.):
            expected = (math.exp(-math.sqrt(kappa)*normalized_t)-kappa*math.exp(-normalized_t/math.sqrt(kappa)))/(1-kappa)
            max_ou = max(max_ou,abs(expected-fig3.covariance_parts(normalized_t,zeta)[0]))
    if max_ou > 2e-14:
        raise AssertionError(('OU scaling',max_ou))
    results['ou_rescaling_max_abs_error'] = max_ou
    max_direct = 0.
    for zeta,levels in [(1.,[2.]),(.5,[.5,3.]),(2.,[0.,2.]),(5.05,[0.,8.,20.])]:
        v,_ = sdho_fano(zeta,levels)
        d,_ = sdho_fano(zeta,levels,direct=True,tol=2e-9)
        max_direct = max(max_direct,float(np.max(np.abs(v-d))))
    for shape,levels in [(.75,[1.784]),(1.,[2.]),(5.,[3.447]),(math.inf,[4.359])]:
        v,_ = rq_fano(shape,levels)
        d,_ = rq_fano(shape,levels,direct=True,tol=2e-9)
        max_direct = max(max_direct,float(np.max(np.abs(v-d))))
    if max_direct > 2e-8:
        raise AssertionError(('independent full integral',max_direct))
    results['positive_full_integral_max_abs_error'] = max_direct
    # Verify the finite-T figure against its archived curves and trial counts.
    archive = ROOT/'data/pre_figure3_10000'
    summary = json.loads((archive/'summary.json').read_text())
    curve = np.array(summary['curve'])
    max_fig3 = 0.
    for i in (0,25,50,75,100):
        z = summary['curve_zeta'][i]
        for j,u in enumerate(fig3.LEVELS):
            value,_ = fig3.theory(z,u,tol=2e-11)
            max_fig3 = max(max_fig3,float(np.max(np.abs(value-curve[i,j]))))
    for i,point in enumerate(summary['points']):
        path = archive/f'counts_{i}.npz'
        if digest(path) != point['counts_sha256']:
            raise AssertionError('Fig. 3 counts hash changed')
        with np.load(path) as data:
            np.testing.assert_allclose(fig3.statistics(data['counts']),point['estimate'],atol=1e-13,rtol=0)
    if max_fig3 > 1e-7:
        raise AssertionError(('Fig. 3 theory',max_fig3))
    results['fig3_recomputed_curve_max_abs_change'] = max_fig3
    results['fig3_counts_hashes_and_statistics_verified'] = True
    results['fig3_reused_trials_per_damping_ratio'] = summary['run']['n_trials']
    return results


def calculate(output):
    if (output/'arrays.npz').exists():
        raise FileExistsError('Use a new archive or the plot action; calculation archives are not overwritten')
    start = time.monotonic()
    output.mkdir(parents=True,exist_ok=True)
    checks = validate()
    print('Independent numerical validation passed',flush=True)
    arrays = {}
    grids = [('sdho',np.linspace(.5,3.,250),np.linspace(0,2,250)),
             ('ou',np.linspace(.01,.4,200),np.linspace(0,1.5,200))]
    max_error = 0.
    for family,parameters,levels in grids:
        fano = np.empty((len(parameters),len(levels)))
        for j,param in enumerate(parameters):
            if family == 'sdho':
                zeta,scaled = param,levels
            else:
                zeta = (1+param)/(2*math.sqrt(param))
                scaled = levels*math.sqrt((1+param)/param)
            fano[j],err = sdho_fano(zeta,scaled)
            max_error = max(max_error,err)
            if j%25 == 0:
                print(f'{family}: {j+1}/{len(parameters)} rows; elapsed {time.monotonic()-start:.1f}s',flush=True)
        arrays[family+'_parameters'] = parameters
        arrays[family+'_levels'] = levels
        arrays[family+'_fano'] = fano
    # Same physical ranges as the two original Fig. 4 notebooks; denser mesh.
    # Temperature=0 has zero mean crossing rate and an undefined Fano ratio.
    omega = np.linspace(.5,10.,200)
    temp = np.linspace(0.,2.,201)
    u = np.linspace(0.,2.,200)
    scale_grid = np.linspace(0.,20.,2001)
    ff_omega,ff_temp = [],[]
    sdho_peaks = []
    for zeta in (.5,1.,2.):
        f,err = sdho_fano(zeta,scale_grid)
        max_error = max(max_error,err)
        frequency_levels = omega[:,None]*u[None,:]
        frequency_values,err = sdho_fano(zeta,frequency_levels.ravel())
        max_error = max(max_error,err)
        ff_omega.append(frequency_values.reshape(frequency_levels.shape))
        values = np.full((len(temp),len(u)),np.nan)
        temperature_levels = u[None,:]/np.sqrt(temp[1:,None])
        temperature_values,err = sdho_fano(zeta,temperature_levels.ravel())
        max_error = max(max_error,err)
        values[1:] = temperature_values.reshape(temperature_levels.shape)
        ff_temp.append(values)
        peak = minimize_scalar(lambda a: -sdho_fano(zeta,[a],tol=5e-11)[0][0],bounds=(0.,6.),method='bounded',options={'xatol':1e-7})
        root = brentq(lambda a: sdho_fano(zeta,[a])[0][0]-1,0.,peak.x) if f[0] < 1 else None
        sdho_peaks.append({'zeta':zeta,'a_peak':peak.x,'f_peak':-peak.fun,'first_F1_crossing':root,'f_at_zero':f[0], 'minimum_on_0_20':float(np.min(f))})
    arrays.update(omega=omega,temp=temp,scan_levels=u,omega_fano=np.array(ff_omega),temp_fano=np.array(ff_temp))
    checks['sdho_quad_vec_max_estimated_error'] = max_error
    checks['fig4_every_valid_pixel_integrated_directly'] = True
    # Repeat representative complete curves at tighter tolerances/longer lags.
    refined_error = 0.
    for z in (.5,1.,2.,3.,5.05):
        levels = np.linspace(0,20,201)
        f,_ = sdho_fano(z,levels)
        g,_ = sdho_fano(z,levels,tol=2e-11,cutoff=60.)
        refined_error = max(refined_error,float(np.max(np.abs(f-g))))
    if refined_error > 2e-8:
        raise ArithmeticError(('SDHO refinement',refined_error))
    checks['sdho_cutoff_and_tolerance_max_change'] = refined_error
    levels = np.linspace(0,8,500)
    shapes = [.75,1.,2.,5.,math.inf]
    rq_curves,rq_old,rq_peaks = [],[],[]
    rq_refine = 0.
    for shape in shapes:
        f,err = rq_fano(shape,levels)
        g,err2 = rq_fano(shape,levels,tol=2e-11,end=4096.)
        rq_refine = max(rq_refine,float(np.max(np.abs(f-g))))
        rq_curves.append(g)
        rq_old.append(rq_fano(shape,levels,end=4096.,historical=True)[0])
        peak = minimize_scalar(lambda a: -rq_fano(shape,[a],tol=2e-11,end=4096.)[0][0],bounds=(1.,6.),method='bounded',options={'xatol':1e-7})
        root = brentq(lambda a: rq_fano(shape,[a])[0][0]-1,0.,peak.x)
        rq_peaks.append({'alpha':str(shape),'a_peak':peak.x,'f_peak':-peak.fun,'first_F1_crossing':root,'f_at_zero':g[0]})
        print(f'RQ alpha={shape}: peak F={-peak.fun:.10f} at {peak.x:.7f}',flush=True)
    if rq_refine > 2e-8:
        raise ArithmeticError(('RQ cutoff refinement',rq_refine))
    checks['rq_2048_to_4096_max_change'] = rq_refine
    arrays.update(rq_levels=levels,rq_shapes=np.array(shapes),rq_fano=np.array(rq_curves),rq_historical_fano=np.array(rq_old))
    kappa = .3
    levels = np.array([0,.3,.6,.9,1.2,1.3,1.5])
    scaled = levels*math.sqrt((1+kappa)/kappa)
    zeta = (1+kappa)/(2*math.sqrt(kappa))
    comparisons = {'sdho_peaks':sdho_peaks,'rq_peaks':rq_peaks,
                   'ou_kappa_0.3':{'levels':levels,'correct':sdho_fano(zeta,scaled)[0],'historical_sign':sdho_fano(zeta,scaled,historical=True)[0]},
                   'sdho_zeta1_u2':{'correct':sdho_fano(1,[2])[0][0],'historical_sign':sdho_fano(1,[2],historical=True)[0][0]}}
    # Count downward crossings of F=1 across the computed OU rows, with a
    # tolerance wider than all numerical refinement errors. Not a theorem.
    ou = arrays['ou_fano']
    reentrant_rows = []
    for j,row in enumerate(ou):
        above = np.where(row > 1+1e-7)[0]
        if len(above) and np.any(row[above[0]:] < 1-1e-7):
            reentrant_rows.append(float(arrays['ou_parameters'][j]))
    comparisons['ou_reentrant_rows_in_plotted_grid'] = reentrant_rows
    np.savez_compressed(output/'arrays.npz',**arrays)
    write_json(output/'validation.json',checks)
    write_json(output/'comparisons.json',comparisons)
    metadata = {'created_utc':datetime.now(timezone.utc).isoformat(),
                'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                'source_sha256':digest(__file__),'notebooks':{k:({'path':v,'sha256':digest(ROOT/v)} if v else 'Source not found; reconstructed from t=t2-t1, s=t2') for k,v in NOTEBOOKS.items()},
                'fig3_archive':str(ROOT/'data/pre_figure3_10000'),
                'fig3_summary_sha256':digest(ROOT/'data/pre_figure3_10000/summary.json'),
                'numerics':'signed erf; normalized conditional variances; adaptive quad_vec; independently checked positive Gaussian integral',
                'temperature_zero':'explicitly masked: deterministic zero-variance process has undefined crossing Fano ratio',
                'mesh_sizes':{'Fig2':[250,250],'Fig4_omega':[200,200],'Fig4_temperature':[201,200],'Fig5':[200,200],'Fig6':[5,500]},
                'wall_seconds':time.monotonic()-start}
    write_json(output/'run.json',metadata)
    shutil.copyfile(__file__,output/'run_source.py')
    print(json.dumps(checks,indent=2),flush=True)
    return comparisons


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'text.usetex':True,'font.family':'serif',
                         'font.serif':['Computer Modern Roman'],'pdf.fonttype':42,
                         'text.latex.preamble':r'\usepackage{amsmath,amssymb}',
                         'axes.linewidth':2.,'xtick.major.width':2.,'ytick.major.width':2.,
                         'xtick.labelsize':24,'ytick.labelsize':24,'axes.labelsize':30})
    return plt


def panel_letter(ax, letter, x=-.20):
    ax.text(x,1.03,rf'\textbf{{{letter}}}',transform=ax.transAxes,
            fontsize=48,ha='center',va='top',clip_on=False)


def fano_norm(values):
    from matplotlib.colors import LogNorm
    finite = np.asarray(values)[np.isfinite(values)]
    # A shared symmetric logarithmic range keeps white exactly at F=1.
    span = max(abs(math.log(float(np.min(finite)))),abs(math.log(float(np.max(finite)))))
    return LogNorm(vmin=math.exp(-span),vmax=math.exp(span))


def save(fig,path):
    import matplotlib.pyplot as plt
    fig.savefig(path,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)


def plot(output,destination):
    destination.mkdir(parents=True,exist_ok=True)
    plt = style()
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import MaxNLocator, FuncFormatter, NullLocator
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    arrays = np.load(output/'arrays.npz')
    # Fig. 1: same SDHO parameters; a reproducible exact stationary realization.
    dt = .001/(math.sqrt(1-.5**2))
    steps = math.ceil(40/dt)
    t = np.linspace(0,40,steps+1)
    M,Q = fig3.transition(.5,t[1]-t[0])
    rng = np.random.default_rng(2026092701)
    initial = rng.normal(size=(4,2))
    eta = np.einsum('ij,bjn->bin',np.linalg.cholesky(Q),rng.normal(size=(4,2,steps)))
    paths = fig3.filtered_positions(M,initial,eta)
    fig,ax = plt.subplots(figsize=(18,8))
    for y in paths[:3]: ax.plot(t,y,color='.7',alpha=.55,lw=1.8)
    y = paths[3]
    ax.plot(t,y,color='black',lw=3.5)
    ax.axhline(1,color='firebrick',ls='--',lw=3.)
    for up,mark,color in [(True,'^','green'),(False,'v','blue')]:
        inds = np.where((y[:-1] < 1)&(y[1:] >= 1) if up else (y[:-1] >= 1)&(y[1:] < 1))[0]
        times = t[inds]+(1-y[inds])/(y[inds+1]-y[inds])*(t[1]-t[0])
        ax.scatter(times,np.ones_like(times),marker=mark,s=200,facecolors='none',edgecolors=color,linewidths=3,zorder=5)
    handles = [Line2D([],[],color='black',lw=3.5,label=r'$X_t$'),
               Line2D([],[],color='firebrick',ls='--',lw=3,label=r'$u$'),
               Line2D([],[],color='green',marker='^',mfc='none',mew=3,ms=13,ls='none',label=r'$N_u^{\uparrow}$'),
               Line2D([],[],color='blue',marker='v',mfc='none',mew=3,ms=13,ls='none',label=r'$N_u^{\downarrow}$'),
               Line2D([],[],color='none',label=r'$N_u=N_u^{\uparrow}+N_u^{\downarrow}$'),
               Line2D([],[],color='none',label=r'$T$: simulation time')]
    ax.legend(handles=handles,loc='center left',bbox_to_anchor=(1.02,.5),fontsize=25,frameon=False)
    ax.set(xlim=(0,40),xlabel=r'$t$',ylabel=r'$X_t$')
    ax.set_xticks([0,10,20,30,40],['0','10','20','30',r'$T$'])
    fig.tight_layout()
    save(fig,destination/'upcrossing_description.pdf')
    np.savez_compressed(output/'figure1_paths.npz',t=t,paths=paths,seed=2026092701)
    # Figs. 2 and 5: mean rates follow Rice; variance rate = mean rate * F.
    for family,name,ylabel,xlabel,floor in [('sdho','sdho_phase',r'$\zeta$',r'$u$',.01),
                                         ('ou','OU_noise_phase',r'$\kappa$',r'$u/\sigma$',.1)]:
        pars,levels,F = arrays[family+'_parameters'],arrays[family+'_levels'],arrays[family+'_fano']
        if family == 'sdho':
            mean = np.broadcast_to(np.exp(-levels[None,:]**2/2)/(2*PI),F.shape)
        else:
            mean = np.sqrt(pars[:,None])/.003/(2*PI)*np.exp(-levels[None,:]**2*(1+pars[:,None])/(2*pars[:,None]))
        values = [mean,mean*F,F]
        labels = [r'$\mathbb{E}[N_u^{\uparrow}]/T$',r'$\mathrm{Var}[N_u^{\uparrow}]/T$',r'$\mathrm{F}^{\uparrow}$']
        fig,axes = plt.subplots(1,3,figsize=(24,7))
        for j,(ax,v,label) in enumerate(zip(axes,values,labels)):
            if j == 2:
                norm,cmap = fano_norm(F),'PRGn'
                displayed = v
            else:
                # Preserve the original mean/variance display floors only.
                norm,cmap = LogNorm(vmin=floor,vmax=float(np.max(v))),'inferno'
                displayed = np.maximum(v,floor)
            im = ax.pcolormesh(levels,pars,displayed,shading='nearest',cmap=cmap,norm=norm,rasterized=True)
            ax.set_xlim(levels[0],levels[-1]); ax.set_ylim(pars[0],pars[-1])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel,rotation=0,labelpad=30)
            ax.set_title(['Mean rate','Variance rate','Fano factor'][j],fontsize=28,pad=14)
            ax.set_box_aspect(1)
            ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(5))
            if family == 'ou':
                ax.set_xticks([0,.5,1.,1.5])
                ax.set_yticks([.01,.1,.2,.3,.4])
            panel_letter(ax,chr(97+j))
            cb=fig.colorbar(im,ax=ax,pad=.025,fraction=.047)
            cb.set_label(label,fontsize=25,labelpad=20)
            if j == 2:
                ticks=[v for v in (.3,.5,1,2,3,5) if norm.vmin <= v <= norm.vmax]
                cb.set_ticks(ticks); cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
                cb.ax.yaxis.set_minor_locator(NullLocator())
        fig.tight_layout(w_pad=2)
        save(fig,destination/(name+'.pdf'))
    # Fig. 3: reuse all archived trials and CIs; no fresh selection or resampling.
    shutil.copyfile(ROOT/'data/pre_figure3_10000/summary.json',output/'summary.json')
    fig3.plot(output)
    shutil.copyfile(output/'sdho_comparison.pdf',destination/'sdho_comparison.pdf')
    # Fig. 4: preserve the two-row layout, palette, labels and panel lettering.
    fig,axes = plt.subplots(2,3,figsize=(24,15))
    norm = fano_norm(np.concatenate([arrays['omega_fano'].ravel(),arrays['temp_fano'].ravel()]))
    cmap = plt.get_cmap('PRGn').copy(); cmap.set_bad('.86')
    for row,(param,key,ylabel) in enumerate([('omega','omega_fano',r'$\omega_0$'),('temp','temp_fano',r'$\vartheta$')]):
        for j,zeta in enumerate((.5,1.,2.)):
            ax=axes[row,j]
            im=ax.pcolormesh(arrays['scan_levels'],arrays[param],np.ma.masked_invalid(arrays[key][j]),
                             cmap=cmap,norm=norm,shading='nearest',rasterized=True)
            ax.set_xlim(0,2); ax.set_ylim(arrays[param][0],arrays[param][-1])
            ax.set_xlabel(r'$u$'); ax.set_ylabel(ylabel,rotation=0,labelpad=30)
            if row == 0:
                regime=['Underdamped','Critically damped','Overdamped'][j]
                ax.set_title(rf'$\zeta={zeta:.1f}$ ({regime})',fontsize=28,pad=18)
            ax.set_box_aspect(1)
            ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
            if row == 0: ax.set_yticks([.5,2.5,5.,7.5,10.])
            panel_letter(ax,chr(97+3*row+j))
        cb=fig.colorbar(im,ax=axes[row,:].tolist(),fraction=.024,pad=.035)
        cb.set_label(r'$\mathrm{F}^{\uparrow}$',fontsize=28,labelpad=20)
        cb.set_ticks([.5,1,2]); cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
        cb.ax.yaxis.set_minor_locator(NullLocator())
    fig.subplots_adjust(left=.07,right=.88,bottom=.065,top=.94,wspace=.34,hspace=.35)
    # Reset colorbar placement after fixed panel spacing.
    for cbax,y0 in zip(fig.axes[-2:],[.57,.075]): cbax.set_position([.92,y0,.015,.355])
    save(fig,destination/'sdho_fanos.pdf')
    # Fig. 6: exact original curve set and axis range; peaks quantified separately.
    fig,ax=plt.subplots(figsize=(10,8))
    colors=[plt.get_cmap('tab10')(v) for v in np.linspace(0,1,4)]+['black']
    for shape,f,color in zip(arrays['rq_shapes'],arrays['rq_fano'],colors):
        label=r'$\alpha\to\infty$' if math.isinf(shape) else rf'$\alpha={shape:g}$'
        ax.plot(arrays['rq_levels'],f,lw=3.5,color=color,label=label)
    ax.axhline(1,color='red',ls='--',lw=2.5)
    ax.set_xlim(0,8)
    ax.set_xlabel(r'$u/\sigma$',fontsize=28)
    ax.set_ylabel(r'$\mathrm{F}^{\uparrow}$',fontsize=28,rotation=0,va='center',labelpad=30)
    ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(fontsize=24,loc='best')
    fig.tight_layout()
    save(fig,destination/'rational_quadratic_fano.pdf')
    # S1: geometry follows t=t2-t1, s=t2; no original source notebook found.
    fig,axes=plt.subplots(1,2,figsize=(15,6.5))
    for j,ax in enumerate(axes):
        ax.set_aspect('equal'); ax.axis('off')
        for start,end in [((-1.25,0),(1.35,0)),((0,-.3),(0,1.4))]:
            ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'<->','lw':2.})
        ax.set_xlim(-1.25,1.4); ax.set_ylim(-.32,1.45)
        panel_letter(ax,chr(97+j),x=.02)
    left,right=axes
    left.add_patch(Polygon([(0,0),(1,0),(1,1),(0,1)],facecolor='#b2d0ef',edgecolor='none',alpha=.9))
    left.plot([0,1,1],[1,1,0],ls='--',color='black',lw=2)
    left.text(1.37,-.03,r'$t_1$',fontsize=28);left.text(.07,1.37,r'$t_2$',fontsize=28)
    right.add_patch(Polygon([(-1,0),(0,0),(0,1)],facecolor='#8b64c9',edgecolor='none',alpha=.75))
    right.add_patch(Polygon([(0,0),(1,1),(0,1)],facecolor='#b2d0ef',edgecolor='none',alpha=.9))
    right.plot([-1,0,1,0],[0,1,1,0],ls='--',color='black',lw=2)
    right.text(1.35,-.03,r'$t$',fontsize=28);right.text(.07,1.37,r'$s$',fontsize=28)
    right.text(-.4,.25,r'$\mathrm{I}$',fontsize=28);right.text(.25,.65,r'$\mathrm{II}$',fontsize=28)
    for ax in axes:
        ax.text(-.06,-.07,r'$0$',fontsize=25,ha='right',va='top')
        ax.text(1,-.09,r'$T$',fontsize=25,ha='center',va='top')
        ax.text(-.08,1,r'$T$',fontsize=25,ha='right',va='center')
        ax.plot([1,1],[-.035,.035],color='black',lw=2)
        ax.plot([-.035,.035],[1,1],color='black',lw=2)
    right.text(-1,-.09,r'$-T$',fontsize=25,ha='center',va='top')
    right.plot([-1,-1],[-.035,.035],color='black',lw=2)
    fig.tight_layout()
    save(fig,destination/'positive_quadrant_new.pdf')
    files=sorted(destination.glob('*.pdf'))
    write_json(output/'figure_hashes.json',{p.name:digest(p) for p in files})
    write_json(output/'render.json',{'renderer_sha256':digest(__file__),
                                  'numerical_archive_source_sha256':json.loads((output/'run.json').read_text())['source_sha256'],
                                  'arrays_sha256':digest(output/'arrays.npz')})
    print(f'Exported {len(files)} figures to {destination}',flush=True)


def comparison(output,destination,paper):
    """Build the review-only old/new PDF; no manuscript/rebuttal text is edited."""
    report_build = destination.parent/'build/figure_comparison'
    report_build.mkdir(parents=True,exist_ok=True)
    results = json.loads((output/'comparisons.json').read_text())
    check = json.loads((output/'validation.json').read_text())
    preamble = r'''\documentclass[12pt]{article}
\usepackage[paperwidth=16in,paperheight=10in,margin=0.55in]{geometry}
\usepackage{graphicx,amsmath,amssymb,booktabs,xcolor,array}
\usepackage[T1]{fontenc}
\definecolor{heading}{HTML}{18384F}
\setlength{\parindent}{0pt}\setlength{\parskip}{7pt}
\pagestyle{plain}
\newcommand{\reporttitle}[1]{{\color{heading}\LARGE\bfseries #1}\par\vspace{6pt}}
\begin{document}
\reporttitle{PRE figure regeneration: results and interpretation}
Review copy, 27 September 2026. Six main-text figures and Supplemental Fig.~S1 regenerated.
The manuscript and rebuttal prose have not been revised in this pass.

\textbf{What the notebook audit establishes.}
The original quantitative notebooks import formula classes that now preserve the signed error-function argument.
Their default fixed-grid quadrature still needs numerical care; the temperature notebook includes the undefined
$\vartheta=0$ endpoint, and the OU notebook's figure-export commands are commented out.
The new runner retains the original physical ranges and curve sets, with refined heatmaps, stable small-lag
covariances, adaptive quadrature and explicit tail checks. A new notebook orchestrates the checked calculation.
The historical notebooks remain available for provenance.

\textbf{What changes scientifically.}
The mean-rate panels are unchanged analytically. The corrected variance and Fano factors are higher at many
nonzero levels. Underdamped SDHO crossings can be super-Poissonian at sufficiently high normalized thresholds.
No super-to-sub return is resolved on the plotted OU grid. All five rational-quadratic curves have a resolved
maximum above one, including the squared-exponential limit. These are numerical findings over the stated
parameter ranges, not new claims of a universal phase diagram.

\begin{minipage}[t]{0.48\linewidth}
\textbf{SDHO: $a=u\omega_0/\sqrt{\vartheta}$}\par
\begin{tabular}{rrrr}\toprule
$\zeta$ & First $F=1$ & $a_{\rm max}$ & $F_{\rm max}$\\\midrule
'''
    rows = []
    for r in results['sdho_peaks']:
        root = '--' if r['first_F1_crossing'] is None else f"{r['first_F1_crossing']:.6f}"
        rows.append(f"{r['zeta']:g} & {root} & {r['a_peak']:.6f} & {r['f_peak']:.8f} \\\\")
    middle = r'''\bottomrule\end{tabular}\par
For $\zeta=2$, $F>1$ throughout the checked range $0\leq a\leq20$.
For $\zeta=1$, $u=2$, $\omega_0=\vartheta=1$, the old-sign expression gives
$0.919610$ and the corrected result is $1.218004$.
\end{minipage}\hfill
\begin{minipage}[t]{0.48\linewidth}
\textbf{Rational quadratic: $a=u/\sigma$}\par
\begin{tabular}{rrrr}\toprule
$\alpha$ & First $F=1$ & $a_{\rm max}$ & $F_{\rm max}$\\\midrule
'''
    rqrows=[]
    for r in results['rq_peaks']:
        alpha = r'$\infty$' if r['alpha']=='inf' else r['alpha']
        rqrows.append(f"{alpha} & {r['first_F1_crossing']:.6f} & {r['a_peak']:.6f} & {r['f_peak']:.8f} \\\\")
    closing = r'''\bottomrule\end{tabular}\par
The larger-$\alpha$ maxima are small but exceed the numerical refinement errors.
All five curves start below one at $u=0$.
\end{minipage}

\textbf{Validation.} Both original formula classes agree with the stable integrand to better than
$1.2\times10^{-14}$ in the checked cases. Independent positive conditional-Gaussian integration agrees
with representative full Fano integrals to $2.0\times10^{-14}$.
SDHO cutoff/tolerance refinement changes checked curves by less than $4.5\times10^{-15}$;
doubling the rational-quadratic cutoff from 2048 to 4096 changes the curves by less than
$3.2\times10^{-13}$ with the analytic tail correction. These are observed numerical checks, not rigorous error bounds.
The existing Fig.~3 archive's counts, hashes and statistics were verified; its revised curves are unchanged.

\textbf{Interpretation of the finite-level maximum.}
$F=\mathrm{Var}(N)/\mathbb{E}N$ measures relative count fluctuations, so its maximum is not a general
upper bound on the variance or a critical point in the thermodynamic sense. At moderate levels, correlations
can create excess crossing pairs. At high levels, isolated rare excursions restore the Poisson limit.
The integrated excess pair intensity determines the balance; mean-rate suppression alone does not.

\textbf{Reading the comparison pages.} ``Submitted'' means the image currently included by the original
submitted TeX. Old-sign numerical values are freshly evaluated historical expressions, not digitized measurements
from those images. The original images' exact execution environment is not recoverable from the saved assets alone.
The regenerated figures retain the bold borders, Computer Modern labels and bare bold panel letters.
Colorbar limits are printed explicitly; compare values against $F=1$, not just color saturation.
'''
    sections = [
        ('1','upcrossing_description',False,
         r'The same SDHO parameters and level are used ($\zeta=0.5$, $\omega_0=\vartheta=1$, $u=1$, $T=40$). '
         r'This is a new, fixed-seed stationary realization sampled by exact Gaussian transitions. Crossing times are linearly interpolated. '
         r'The illustrative trajectory changes; no statistical conclusion changes.'),
        ('2','sdho_phase',False,
         r'The mean-rate panel is unchanged analytically. Variance and Fano-factor values increase at nonzero levels. '
         r'For example, at $\zeta=1,u=2$, $F$ changes from the old-sign value $0.919610$ to $1.218004$. '
         r'The original $250\times250$ parameter mesh is retained.'),
        ('3','sdho_comparison',False,
         r'The upper image is the submitted figure; the lower image is the already-revised, verified Fig.~3. '
         r'This pass does not change the revised numerical results: 10,000 trials at each of five damping ratios, '
         r'$T=120$, and pointwise 95\% whole-trial bootstrap intervals. 43 of 45 intervals contain the prediction; '
         r'the two misses were already disclosed. Long-time Fano values must not be substituted for these finite-window predictions.'),
        ('4','sdho_fanos',True,
         r'The dependence collapses onto $a=u\omega_0/\sqrt{\vartheta}$. '
         r'At $\zeta=0.5$ and $1$, $F$ crosses above one at $a\simeq2.439699$ and $1.143500$, respectively. '
         r'For $\zeta=2$, it stays above one on the checked range. Each valid pixel is integrated directly. '
         r'The $20\times20$ meshes are refined to approximately $200\times200$. '
         r'The gray boundary at $\vartheta=0$ is excluded because the Fano ratio is undefined there.'),
        ('5','OU_noise_phase',False,
         r'At $\kappa=0.3$, the corrected $F$ values at $u/\sigma=0.6,0.9,1.2,1.5$ are '
         r'$1.19033,1.34581,1.31260,1.22423$, versus old-sign values $1.03804,1.08530,0.97968,0.85892$. '
         r'The return below one disappears. No reentrant row is resolved on the refined $200\times200$ grid '
         r'($0.01\leq\kappa\leq0.4$, $0\leq u/\sigma\leq1.5$). The fixed physical time is $\tau_f=3$ ms.'),
        ('6','rational_quadratic_fano',True,
         r'All five displayed curves have super-Poissonian maxima; the former claim that the larger-$\alpha$ curves '
         r'are always sub-Poissonian is unsupported. At $\alpha=1,u/\sigma=2$, $F$ changes from about '
         r'$0.933991$ to $1.119946$. The tiny $\alpha=5$ and $\alpha=\infty$ peaks are quantified on page 1. '
         r'The original 500 levels on $[0,8]$ and all five shape values are retained.'),
        ('S1','positive_quadrant_new',False,
         r'No source notebook for this schematic was found. It was reconstructed as vector geometry from '
         r'$t=t_2-t_1$, $s=t_2$, which maps $[0,T]^2$ to the parallelogram with vertices '
         r'$(-T,0),(0,0),(T,T),(0,T)$. The integration regions and mathematical meaning are unchanged.'),
    ]
    tex=preamble+'\n'.join(rows)+middle+'\n'.join(rqrows)+closing
    for number,name,side,note in sections:
        old=(paper/(name+'.png')).resolve()
        new=(destination/(name+'.pdf')).resolve()
        tex+='\n\\clearpage\n'+rf'\reporttitle{{Figure {number}: submitted and regenerated}}'+'\n'+note+'\n'+r'\par\vspace{8pt}'+'\n'
        if side:
            for label,path in [('Submitted',old),('Regenerated',new)]:
                tex+=r'\begin{minipage}[t]{0.49\linewidth}\centering\textbf{'+label+r'}\par\vspace{5pt}'+'\n'
                tex+=r'\includegraphics[width=\linewidth,height=0.73\textheight,keepaspectratio,clip]{'+path.as_posix()+'}\n'+r'\end{minipage}\hfill'+'\n'
        else:
            for label,path in [('Submitted',old),('Regenerated',new)]:
                tex+=r'\begin{center}\textbf{'+label+r'}\par\vspace{20pt}'+'\n'
                tex+=r'\includegraphics[width=0.98\linewidth,height=0.31\textheight,keepaspectratio,clip]{'+path.as_posix()+'}\n'+r'\end{center}'+'\n'
    tex+='\n\\end{document}\n'
    source=report_build/'figure_comparison.tex'
    source.write_text(tex)
    log=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',
                        '-outdir='+str(report_build),str(source)],cwd=paper,text=True,capture_output=True)
    (report_build/'latexmk_output.txt').write_text(log.stdout+log.stderr)
    if log.returncode:
        raise RuntimeError('Comparison PDF compilation failed; inspect '+str(report_build/'latexmk_output.txt'))
    final=destination.parent/'figure_comparison.pdf'
    shutil.copyfile(report_build/'figure_comparison.pdf',final)
    print('Comparison PDF: '+str(final),flush=True)
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['validate','calculate','plot','comparison','all'])
    parser.add_argument('--output',type=Path,default=ROOT/'data/pre_figures_20260927')
    parser.add_argument('--figures',type=Path,required=True)
    parser.add_argument('--paper',type=Path,default=ROOT.parent/'upcrossing_theory_tex')
    args=parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR',str(args.output/'build/matplotlib'))
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
    if args.action=='validate': print(json.dumps(validate(),indent=2))
    if args.action in ('calculate','all'): calculate(args.output)
    if args.action in ('plot','all'): plot(args.output,args.figures)
    if args.action in ('comparison','all'): comparison(args.output,args.figures,args.paper)


if __name__=='__main__': main()
