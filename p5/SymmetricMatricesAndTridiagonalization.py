#As with question 2, first determine which method we need to use Householder/Given/Jacobi to calculate, and just comment out the others.
import cmath
import math
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import inspect
import operator
import sys
from collections import OrderedDict
from copy import deepcopy
from functools import wraps
from types import SimpleNamespace
import kwant
import scipy.constants
import scipy.sparse
import scipy.sparse.linalg as sla
from kwant.continuum.discretizer import discretize
assert sys.version_info >= (3, 6), "Use Python ≥3.6"

def householder_real(x):
    assert x.shape[0] > 0
    sigma = x[1:] @ x[1:]
    if sigma == 0:
        return (np.zeros(x.shape[0]), 0, x[0])
    else:
        norm_x = math.sqrt(x[0] ** 2 + sigma)
        v = x.copy()
        #First determine the following symbols according to the positive and negative of the first element of the vector x
        if x[0] <= 0:
            v[0] -= norm_x
            alpha = +norm_x
        else:
            v[0] += norm_x
            alpha = -norm_x
        v /= np.linalg.norm(v)
        return (v, 2, alpha)
def skew_tridiagonalize(A, overwrite_a=False, calc_q=True):
    #Is the matrix square? yes then continue
    assert A.shape[0] == A.shape[1] > 0
    #Detecting oblique symmetry
    assert abs((A + A.T).max()) < 1e-14

    n = A.shape[0]
    A = np.asarray(A)

    #Whether there is a complex element
    if np.issubdtype(A.dtype, np.complexfloating):
        householder = householder_complex
    elif not np.issubdtype(A.dtype, np.number):
        raise TypeError("pfaffian() can only work on numeric input")
    else:
        householder = householder_real
    if not overwrite_a:
        A = A.copy()
    if calc_q:
        Q = np.eye(A.shape[0], dtype=A.dtype)
    for i in range(A.shape[0] - 2):
        #Householder vector for i-th column
        v, tau, alpha = householder(A[i + 1 :, i])
        A[i + 1, i] = alpha
        A[i, i + 1] = -alpha
        A[i + 2 :, i] = 0
        A[i, i + 2 :] = 0
        #update submatrix
        w = tau * A[i + 1 :, i + 1 :] @ v.conj()
        A[i + 1 :, i + 1 :] += np.outer(v, w) - np.outer(w, v)
        if calc_q:
            y = tau * Q[:, i + 1 :] @ v
            Q[:, i + 1 :] -= np.outer(y, v.conj())
    if calc_q:
        return (np.asmatrix(A), np.asmatrix(Q))
    else:
        return np.asmatrix(A)
def skew_LTL(A, overwrite_a=False, calc_L=True, calc_P=True):
    assert A.shape[0] == A.shape[1] > 0
    assert abs((A + A.T).max()) < 1e-14
    n = A.shape[0]
    A = np.asarray(A)
    if not overwrite_a:
        A = A.copy()
    if calc_L:
        L = np.eye(n, dtype=A.dtype)
    if calc_P:
        Pv = np.arange(n)
    for k in range(n - 2):
        #rearrange
        kp = k + 1 + np.abs(A[k + 1 :, k]).argmax()
        #Check if median is required
        if kp != k + 1:
            #swap row
            temp = A[k + 1, k:].copy()
            A[k + 1, k:] = A[kp, k:]
            A[kp, k:] = temp
            #swap column
            temp = A[k:, k + 1].copy()
            A[k:, k + 1] = A[k:, kp]
            A[k:, kp] = temp

            if calc_L:
                #update dimension
                temp = L[k + 1, 1 : k + 1].copy()
                L[k + 1, 1 : k + 1] = L[kp, 1 : k + 1]
                L[kp, 1 : k + 1] = temp
            if calc_P:
                temp = Pv[k + 1]
                Pv[k + 1] = Pv[kp]
                Pv[kp] = temp
        #set Gauss vector
        if A[k + 1, k] != 0.0:
            tau = A[k + 2 :, k].copy()
            tau /= A[k + 1, k]
            #Erase row and column
            A[k + 2 :, k] = 0.0
            A[k, k + 2 :] = 0.0
            #update submatrix
            A[k + 2 :, k + 2 :] += np.outer(tau, A[k + 2 :, k + 1])
            A[k + 2 :, k + 2 :] -= np.outer(A[k + 2 :, k + 1], tau)
            if calc_L:
                L[k + 2 :, k + 1] = tau
    if calc_P:
        #Resynthesize the entire matrix
        P = sp.csr_matrix((np.ones(n), (np.arange(n), Pv)))
    if calc_L:
        if calc_P:
            return (np.asmatrix(A), np.asmatrix(L), P)
        else:
            return (np.asmatrix(A), np.asmatrix(L))
    else:
        if calc_P:
            return (np.asmatrix(A), P)
        else:
            return np.asmatrix(A)

def pfaffian(A, overwrite_a=False, method="P", sign_only=False):
    #square
    assert A.shape[0] == A.shape[1] > 0
    assert abs((A + A.T).max()) < 1e-14, abs((A + A.T).max())
    assert method == "P" or method == "H"
    if method == "H" and sign_only:
        raise Exception("Use `method='P'` when using `sign_only=True`")
    if method == "P":
        return pfaffian_LTL(A, overwrite_a, sign_only)
    else:
        return pfaffian_householder(A, overwrite_a)
def pfaffian_LTL(A, overwrite_a=False, sign_only=False):#Similar to the above, some steps can be repeated
    assert A.shape[0] == A.shape[1] > 0
    assert abs((A + A.T).max()) < 1e-14
    n = A.shape[0]
    A = np.asarray(A)
    if n % 2 == 1:
        return 0
    if not overwrite_a:
        A = A.copy()
    pfaffian_val = 1.0
    for k in range(0, n - 1, 2):
        kp = k + 1 + np.abs(A[k + 1 :, k]).argmax()
        if kp != k + 1:
            temp = A[k + 1, k:].copy()
            A[k + 1, k:] = A[kp, k:]
            A[kp, k:] = temp
            temp = A[k:, k + 1].copy()
            A[k:, k + 1] = A[k:, kp]
            A[k:, kp] = temp
            pfaffian_val *= -1
        if A[k + 1, k] != 0.0:
            tau = A[k, k + 2 :].copy()
            tau /= A[k, k + 1]
            if sign_only:
                pfaffian_val *= np.sign(A[k, k + 1])
            else:
                pfaffian_val *= A[k, k + 1]
            if k + 2 < n:
                A[k + 2 :, k + 2 :] += np.outer(tau, A[k + 2 :, k + 1])
                A[k + 2 :, k + 2 :] -= np.outer(A[k + 2 :, k + 1], tau)
        else:
            return 0.0

    return pfaffian_val


def pfaffian_householder(A, overwrite_a=False):
    assert A.shape[0] == A.shape[1] > 0
    assert abs((A + A.T).max()) < 1e-14
    n = A.shape[0]
    if n % 2 == 1:
        return 0
    A = np.asarray(A)
    if not overwrite_a:
        A = A.copy()
    pfaffian_val = 1.0
    for i in range(A.shape[0] - 2):
        v, tau, alpha = householder(A[i + 1 :, i])
        A[i + 1, i] = alpha
        A[i, i + 1] = -alpha
        A[i + 2 :, i] = 0
        A[i, i + 2 :] = 0
        w = tau * A[i + 1 :, i + 1 :] @ v.conj()
        A[i + 1 :, i + 1 :] += np.outer(v, w) - np.outer(w, v)
        if tau != 0:
            pfaffian_val *= 1 - tau
        if i % 2 == 0:
            pfaffian_val *= -alpha
    pfaffian_val *= A[n - 2, n - 1]
    return pfaffian_val
def pfaffian_schur(A, overwrite_a=False):
    assert np.issubdtype(A.dtype, np.number) and not np.issubdtype(
        A.dtype, np.complexfloating
    )
    assert A.shape[0] == A.shape[1] > 0
    assert abs(A + A.T).max() < 1e-14
    if A.shape[0] % 2 == 1:
        return 0
    (t, z) = la.schur(A, output="real", overwrite_a=overwrite_a)
    l = np.diag(t, 1)
    return np.prod(l[::2]) * la.det(z)
def pfaffian_sign(A, overwrite_a=False):
    assert A.shape[0] == A.shape[1] > 0
    assert abs((A + A.T).max()) < 1e-14, abs((A + A.T).max())
    return pfaffian_LTL_sign(A, overwrite_a)
def pfaffian_LTL_sign(A, overwrite_a=False):
    assert A.shape[0] == A.shape[1] > 0
    assert abs((A + A.T).max()) < 1e-14
    n = A.shape[0]
    A = np.asarray(A)
    if n % 2 == 1:
        return 0
    if not overwrite_a:
        A = A.copy()
    pfaffian_val = 1.0
    for k in range(0, n - 1, 2):
        kp = k + 1 + np.abs(A[k + 1 :, k]).argmax()
        if kp != k + 1:
            temp = A[k + 1, k:].copy()
            A[k + 1, k:] = A[kp, k:]
            A[kp, k:] = temp
            temp = A[k:, k + 1].copy()
            A[k:, k + 1] = A[k:, kp]
            A[k:, kp] = temp
            pfaffian_val *= -1
        if A[k + 1, k] != 0.0:
            tau = A[k, k + 2 :].copy()
            tau /= A[k, k + 1]
            pfaffian_val *= A[k, k + 1]
            if k + 2 < n:
                A[k + 2 :, k + 2 :] += np.outer(tau, A[k + 2 :, k + 1])
                A[k + 2 :, k + 2 :] -= np.outer(A[k + 2 :, k + 1], tau)
        else:
            return 0.0
    return pfaffian_val

lead_pars = dict(
    a=10, r1=50, r2=70, coverage_angle=135, angle=45, with_shell=True, which_lead=""
)

params = dict(
    alpha=20,
    B_x=0,
    B_y=0,
    B_z=0,
    Delta=110,
    g=50,
    orbital=True,
    mu_sc=100,
    c_tunnel=3 / 4,
    V_r=-50,
    mu_="lambda x0, sigma, mu_lead, mu_wire: mu_lead",
    V_="lambda z, V_0, V_r, V_l, x0, sigma, r1: 0",
    V_0=None,
    V_l=None,
    mu_lead=10,
    mu_wire=None,
    r1=None,
    sigma=None,
    x0=None,
    **phase_diagram.constants.__dict__
)

constants = SimpleNamespace(
    m_eff=0.015 * scipy.constants.m_e,
    hbar=scipy.constants.hbar,
    m_e=scipy.constants.m_e,
    eV=scipy.constants.eV,
    e=scipy.constants.e,
    c=1e18 / (scipy.constants.eV * 1e-3),
    mu_B=scipy.constants.physical_constants["Bohr magneton in eV/T"][0] * 1e3,
)

constants.t = (constants.hbar ** 2 / (2 * constants.m_eff)) * constants.c


def get_names(sig):
    names = [
        (name, value)
        for name, value in sig.parameters.items()
        if value.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    return OrderedDict(names)


def filter_kwargs(sig, names, kwargs):
    names_in_kwargs = [(name, value) for name, value in kwargs.items() if name in names]
    return OrderedDict(names_in_kwargs)


def skip_pars(names1, names2, num_skipped):
    skipped_pars1 = list(names1.keys())[:num_skipped]
    skipped_pars2 = list(names2.keys())[:num_skipped]
    if skipped_pars1 == skipped_pars2:
        pars1 = list(names1.values())[num_skipped:]
        pars2 = list(names2.values())[num_skipped:]
    else:
        raise Exception("First {} arguments " "have to be the same".format(num_skipped))
    return pars1, pars2


def combine(f, g, operator, num_skipped=0):
    if not callable(f) or not callable(g):
        raise Exception("One of the functions is not a function")

    sig1 = inspect.signature(f)
    sig2 = inspect.signature(g)

    names1 = get_names(sig1)
    names2 = get_names(sig2)

    pars1, pars2 = skip_pars(names1, names2, num_skipped)
    skipped_pars = list(names1.values())[:num_skipped]

    pars1_names = {p.name for p in pars1}
    pars2 = [p for p in pars2 if p.name not in pars1_names]

    parameters = pars1 + pars2
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    parameters = [p.replace(kind=kind) for p in parameters]
    parameters = skipped_pars + parameters

    def wrapped(*args):
        d = {p.name: arg for arg, p in zip(args, parameters)}
        fval = f(*[d[name] for name in names1.keys()])
        gval = g(*[d[name] for name in names2.keys()])
        return operator(fval, gval)

    wrapped.__signature__ = inspect.Signature(parameters=parameters)
    return wrapped


def memoize(obj):
    cache = obj.cache = {}

    @wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]

    return memoizer


def parse_params(params):
    for k, v in params.items():
        if isinstance(v, str):
            try:
                params[k] = eval(v)
            except NameError:
                pass
    return params


@memoize
def discretized_hamiltonian(a, which_lead=None, subst_sm=None):
    ham = (
        "(0.5 * hbar**2 * (k_x**2 + k_y**2 + k_z**2) / m_eff * c - mu + V) * kron(sigma_0, sigma_z) + "
        "alpha * (k_y * kron(sigma_x, sigma_z) - k_x * kron(sigma_y, sigma_z)) + "
        "0.5 * g * mu_B * (B_x * kron(sigma_x, sigma_0) + B_y * kron(sigma_y, sigma_0) + B_z * kron(sigma_z, sigma_0)) + "
        "Delta * kron(sigma_0, sigma_x)"
    )
    if subst_sm is None:
        subst_sm = {"Delta": 0}

    if which_lead is not None:
        subst_sm["V"] = f"V_{which_lead}(z, V_0, V_r, V_l, x0, sigma, r1)"
        subst_sm["mu"] = f"mu_{which_lead}(x0, sigma, mu_lead, mu_wire)"
    else:
        subst_sm["V"] = "V(x, z, V_0, V_r, V_l, x0, sigma, r1)"
        subst_sm["mu"] = "mu(x, x0, sigma, mu_lead, mu_wire)"

    subst_sc = {"g": 0, "alpha": 0, "mu": "mu_sc", "V": 0}
    subst_interface = {"c": "c * c_tunnel", "alpha": 0, "V": 0}

    templ_sm = discretize(ham, locals=subst_sm, grid_spacing=a)
    templ_sc = discretize(ham, locals=subst_sc, grid_spacing=a)
    templ_interface = discretize(ham, locals=subst_interface, grid_spacing=a)

    return templ_sm, templ_sc, templ_interface


def cylinder_sector(r_out, r_in=0, L=1, L0=0, coverage_angle=360, angle=0, a=10):
    coverage_angle *= np.pi / 360
    angle *= np.pi / 180
    r_out_sq, r_in_sq = r_out ** 2, r_in ** 2

    def shape(site):
        try:
            x, y, z = site.pos
        except AttributeError:
            x, y, z = site
        n = (y + 1j * z) * np.exp(1j * angle)
        y, z = n.real, n.imag
        rsq = y ** 2 + z ** 2
        shape_yz = r_in_sq <= rsq < r_out_sq and z >= np.cos(coverage_angle) * np.sqrt(
            rsq
        )
        return (shape_yz and L0 <= x < L) if L > 0 else shape_yz

    r_mid = (r_out + r_in) / 2
    start_coords = np.array([L - a, r_mid * np.sin(angle), r_mid * np.cos(angle)])

    return shape, start_coords


def is_antisymmetric(H):
    return np.allclose(-H, H.T)


def cell_mats(lead, params, bias=0):
    h = lead.cell_hamiltonian(params=params)
    h -= bias * np.identity(len(h))
    t = lead.inter_cell_hopping(params=params)
    return h, t


def get_h_k(lead, params):
    h, t = cell_mats(lead, params)

    def h_k(k):
        return h + t * np.exp(1j * k) + t.T.conj() * np.exp(-1j * k)

    return h_k


def make_skew_symmetric(ham):
    W = ham.shape[0] // 4
    I = np.eye(2, dtype=complex)
    sigma_y = np.array([[0, 1j], [-1j, 0]], dtype=complex)
    U_1 = np.bmat([[I, I], [1j * I, -1j * I]])
    U_2 = np.bmat([[I, 0 * I], [0 * I, sigma_y]])
    U = U_1 @ U_2
    U = np.kron(np.eye(W, dtype=complex), U)
    skew_ham = U @ ham @ U.H

    assert is_antisymmetric(skew_ham)

    return skew_ham


def calculate_pfaffian(lead, params):
    h_k = get_h_k(lead, params)
    skew_h0 = make_skew_symmetric(h_k(0))
    skew_h_pi = make_skew_symmetric(h_k(np.pi))
    pf_0 = np.sign(pf.pfaffian(1j * skew_h0, sign_only=True).real)
    pf_pi = np.sign(pf.pfaffian(1j * skew_h_pi, sign_only=True).real)
    pfaf = pf_0 * pf_pi
    return pfaf

def change_hopping_at_interface(syst, template, shape1, shape2):
    for (site1, site2), hop in syst.hopping_value_pairs():
        if at_interface(site1, site2, shape1, shape2):
            syst[site1, site2] = template[site1, site2]
    return syst

def apply_peierls_to_template(template, xyz_offset=(0, 0, 0)):
    template = deepcopy(template)  # Needed because kwant.Builder is mutable
    x0, y0, z0 = xyz_offset
    lat = template.lattice
    a = np.max(lat.prim_vecs)  # lattice contant

    def phase(site1, site2, B_x, B_y, B_z, orbital, e, hbar):
        if orbital:
            x, y, z = site1.tag
            direction = site1.tag - site2.tag
            A = [B_y * (z - z0) - B_z * (y - y0), 0, B_x * (y - y0)]
            A = np.dot(A, direction) * a ** 2 * 1e-18 * e / hbar
            phase = np.exp(-1j * A)
            if lat.norbs == 2:  # No PH degrees of freedom
                return phase
            elif lat.norbs == 4:
                return np.array(
                    [phase, phase.conj(), phase, phase.conj()], dtype="complex128"
                )
        else:  # No orbital phase
            return 1

    for (site1, site2), hop in template.hopping_value_pairs():
        template[site1, site2] = combine(hop, phase, operator.mul, 2)
    return template

learner = learners[0]
learner.plot(n=100)
