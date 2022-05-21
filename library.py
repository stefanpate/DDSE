'''
Library of useful things I made doing the DDSE problem sets.
'''

import numpy as np
from scipy.linalg import svd, eig
from collections import Counter
from math import factorial
from itertools import permutations, combinations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def rand_svd(x, r):
    '''
    Randomized Singular Value Decomposition.
    '''

    m, n = x.shape
    p = np.random.randn(n, r) # Random vectors
    z = x @ p # Random projections into col space of x
    q, r = np.linalg.qr(z) # Get orthonormal basis of col space of x
    y = q.T @ x # Project x down to reduced space
    uy, s, vt = svd(y)
    u = q @ uy # Project back up to high dim space

    return u, s, vt


# Dynamic mode decomposition

class dmd:
    def __init__(self):
        self.is_fit = False
        self.lam = None # DMD eigenvalues
        self.phi = None # DMD modes
        self.ur = None # Truncated left singular vectors
        self.wa_inv = None # Matrix used to get init condition (7.29f in DDSE)

    def fit(self, x_full, r, mode='normal'):
        '''
        Get DMD evals and evecs.

        Args:
            - x_full: Data matrix (space x time) 
            - r: Desired rank / last singular value selected
            - mode: 'rand' for randomized svd, 'normal' for deterministic
        '''

        # Time shift data
        x = x_full[:,:-1]
        x_prime = x_full[:,1:]

        d, t = x_prime.shape # Get dimensions
        
        if mode == 'normal':
            u, s, vt = svd(x)
        elif mode == 'rand':
            u, s, vt = rand_svd(x, r)

        tmp = np.zeros((d,t))
        np.fill_diagonal(tmp, s)
        s = tmp # Make s a (d x t) matrix
        v = vt.T

        # Truncate matrices to rank r
        self.ur = u[:,:r]
        sr = s[:r,:r] # Singular values
        vr = v[:,:r]

        sr_inv = np.linalg.inv(sr)
        a_squig = self.ur.T @ x_prime @ vr @ sr_inv # Low dimensional dynamics matrix
        self.lam, w = eig(a_squig)
        self.lam = np.diag(self.lam)

        self.phi = x_prime @ vr @ sr_inv @ w
        self.wa_inv = np.linalg.inv(w @ self.lam)

        self.is_fit = True

    def predict(self, x0, k):
        '''
        Predict system dynamics k steps with DMD model.

        Args:
            - x0: Initial condition (space, 1)
            - k: Number of steps to forecast
        Returns:
            - x_hat: Prediction
        '''

        if not self.is_fit:
            raise("Model fit model before predicting.")

        x0_squig = self.ur.T @ x0
        b = self.wa_inv @ x0_squig

        x_hat = [x0] + [self.phi @ (self.lam**i) @ b for i in range(1,k+1)]

        return np.real(np.hstack(x_hat))


# Sparse identification of nonlinear dynamics

def make_lib(x, d, n, extras=None):
    '''
    Generate library of augmented data up to 
    polynomial order, n, for system of dimension, d,
    from input data, x (space, time). May pass an iterable
    of extra functions to tack onto the end. To add constant
    term pas fcn = lambda x : np.ones(shape=x.shape)

    Returns:
        - theta: Matrix of library fcns
        - lib_fcn_ids: List of tuples (form, term)
        starting at the first non-constant lib_fcn
    '''
    vars = np.arange(d)
    t = x.shape[1]
    theta = []
    lib_fcn_ids = []

    for n in np.arange(1, n + 1):
        for max_exp in range(1, n + 1):
            forms = get_forms(n, max_exp)
            for form in forms:
                terms = get_terms(form, vars)
                form = np.array(form).reshape(-1,1)

                for term in terms:
                    term = np.array(term).reshape(1,-1)
                    lib_fcn = np.prod(np.power(x[term,:].reshape(-1,t), form), axis=0)
                    theta.append(lib_fcn.reshape(-1,1))
                    lib_fcn_ids.append((form.reshape(-1,), term.reshape(-1,)))

    if extras is not None:
        for fcn in extras:
            theta.append(fcn(x).T)

    theta = np.hstack(theta)
    return theta, lib_fcn_ids

def stls(theta, dxdt, lam, d):
    '''
    Sequentially thresholded least squares
    to do sparse regression.

    Args:
        - theta: Library of augmented data (time, # of lib fcns)
        - dxdt: LHS derivatives (time, dimension of system)
        - lam: Sparsity knob
        - d: Dimension of system
    Returns:
        - xi: Coefficients matrix (# of lib fcns, d)
    '''

    xi = np.linalg.pinv(theta) @ dxdt # First guess, least squares

    # Iteratively zero-out smallest coefs and re-regress to make sparse
    for i in range(10):
        small_idxs = abs(xi) < lam
        xi[small_idxs] = 0

        for j in range(d):
            big_idxs = ~small_idxs[:,j]
            xi[big_idxs, j] = np.linalg.pinv(theta[:,big_idxs]) @ dxdt[:,j]

    return xi

def get_forms(n, max_exp):
    '''
    Returns forms of polynomial terms of
    order n involving any exponents from [1, max_exp].
    Returns as a list of lists. Each sub-list contains
    the exponents of the term involved. For example:
    [3, 2, 1] => (a^3)*(b^2)*(c) for arbitrary a, b, c.
    '''
    forms = []
    rem = n - max_exp

    if rem >= 0:
        for exp in range(max_exp, 1, -1):
            mults = rem // exp

            for mult in range(mults, 0, -1):
                more_ones = rem - (mult * exp)
                forms.append([max_exp] + ([exp] * mult) + [1] * more_ones)

        forms.append([max_exp] + ([1] * rem))

    return forms


def get_count(form, d):
    '''
    Counts the number of polynomial terms of specified form
    constructable from d distinct variables.

    Args:
        - form: List of exponents defining a term. If there are
        repeated exponents the variable under these exponents must
        be different, otherwise you could have e.g., (x^1)*(x^1)=(x^2) 
        which is a different form.
        Example form: [3,2,1,1] means (w^3)(x^2)(y)(z)
    '''
    if d < len(form): # Cannot construct form with fewer variables than sub-terms
        return 0
    else:
        repeat_dict = dict(Counter(form))
        ith_repeat_count = []
        total_non_repeats = 0
        for _, v in repeat_dict.items():
            if v > 1:
                ith_repeat_count.append(v)
            else:
                total_non_repeats += 1

        count_denominator = factorial(d - total_non_repeats - sum(ith_repeat_count))
        for elt in ith_repeat_count:
            count_denominator *= factorial(elt)

        return factorial(d) / count_denominator

def append_combos(perm, i, shrinking_vars, ith_repeat_count):
    '''
    Recursive function appends combinations of variables of 
    repeated exponent degree.
    '''
    this_shrinking_vars = shrinking_vars
    combos = list(combinations(this_shrinking_vars, ith_repeat_count[i]))
    
    if i == len(ith_repeat_count) - 1:
        return [perm + list(combo) for combo in combos]

    else:
        terms = []
        for combo in combos:
            
            reduce_mask = np.ones(shape=this_shrinking_vars.shape, dtype=bool)
            for var in combo:
                reduce_mask *= (this_shrinking_vars != var)
            
            shrinking_vars = this_shrinking_vars[reduce_mask]
            
            addendum = append_combos(perm + list(combo), i+1, shrinking_vars, ith_repeat_count)
            terms = terms + addendum
        
        return terms

def get_terms(form, vars):
    '''
    Generates all possible terms of specified form
    from array of variables, vars.
    '''
    repeat_dict = dict(Counter(form))
    ith_repeat_count = []
    total_non_repeats = 0
    for _, v in repeat_dict.items():
        if v > 1:
            ith_repeat_count.append(v)
        else:
            total_non_repeats += 1

    terms = []
    for perm in permutations(vars, total_non_repeats):
        perm = list(perm)
        if total_non_repeats > 0:
            nr_mask = np.ones(shape=vars.shape, dtype=bool)
            for j in range(total_non_repeats):
                nr_mask *= (vars != perm[j])

            shrinking_vars = vars[nr_mask]
        else:
            shrinking_vars = vars

        if len(ith_repeat_count) > 0:
            terms = terms + append_combos(perm, 0, shrinking_vars, ith_repeat_count)
        else:
            terms.append(perm)

    return terms

# Make animated heatmap plot

def make_movie(data, k, loc, cmap='viridis', interval=20, a=5):
    '''
    Makes a gif movie and saves to location 'loc'.
    Uses k timesteps from (y, x, t) data tensor.
    Optional args: cmap, interval, a (scales figure)
    '''
    data = data[:,:,:k]

    # Set up figure
    y, x, t = data.shape
    fig = plt.figure(figsize=(a * (x/y), a * (x/x)))
    ax = plt.axes(xlim=(0, x), ylim=(0, y))

    cax = ax.pcolormesh(np.flipud(data[:-1, :-1, 0]), cmap=cmap)
    fig.colorbar(cax)

    def animate(i):
        cax.set_array(np.flipud(data[:-1, :-1, i]).flatten())

    anim = FuncAnimation(fig, animate, interval=interval, frames=t-1)

    # Save
    anim.save(loc + '.gif', writer='imagemagick')

    plt.close()