import numpy as np
from scipy.linalg import toeplitz, cholesky


def simulate_gaussian_process_new(r_func, T, dt, *args, **kwargs):
    """
    Simulates a zero-mean stationary Gaussian process {X(t)} of duration T,
    sampled at step dt, with correlation function r_func(tau, *args, **kwargs).
    Uses an FFT-based approach (circulant embedding) with correct scaling.
    
    Parameters
    ----------
    r_func : callable
        The correlation function r(tau) of the process. Must accept
        (tau, *args, **kwargs).
    T : float
        Total simulation time.
    dt : float
        Time step.
    *args :
        Additional positional arguments passed to r_func.
    **kwargs :
        Additional keyword arguments passed to r_func.
    
    Returns
    -------
    t : ndarray of shape (N,)
        Array of time points of length N = int(T/dt).
    x : ndarray of shape (N,)
        Simulated Gaussian process values at times in t.
    """
    # 1) Basic discretization
    N = int(round(T / dt)) 
    t = np.arange(N) * dt  # times 0..(N-1)*dt
    
    # 2) Build discrete correlation array: r(0..N-1).
    #    r[n] = r_func(n*dt). For an even function r(-tau)=r(tau),
    #    the "circular embedding" method typically uses 2N-length.
    #    But let's do a minimal version with length= N or 2N.
    
    r_vals = np.array([r_func(n * dt, *args, **kwargs) for n in range(N)])
    
    # We'll embed in length 2N to reduce wrap-around correlation
    # and to ensure positive definiteness more reliably.
    # The first half is r(0..N-1) and the second half is r(N..2N-1) which we
    # can set to small or just mirror if we want an even extension:
    
    r_full = np.zeros(2*N, dtype=float)
    # Put r(0..N-1) in the front:
    r_full[:N] = r_vals
    # Option: we can place the mirrored tail in r_full[N:2N], i.e. for k=1..N-1
    # r_full[2N-k] = r_vals[k], so that it's an even extension.
    # This is a common approach in "circulant embedding."
    for k in range(1, N):
        r_full[2*N - k] = r_vals[k]
    
    # 3) FFT to get eigenvalues of the (2N x 2N) circulant covariance matrix
    R_f = np.fft.fft(r_full)  # length=2N
    
    # 4) Create random complex amplitudes with correct magnitude.
    #    We want to do:  X_f[k] = sqrt(2N * R_f[k]) * (Gaussian complex).
    #    Because ifft in numpy has a 1/(2N) factor, we need *sqrt(2N).
    
    # Safety: ensure nonnegative real part
    lam = np.maximum(R_f.real, 0.0)
    
    # random complex N(0,1)+iN(0,1):
    rng = np.random.default_rng()
    z = rng.normal(size=2*N) + 1j * rng.normal(size=2*N)
    
    # amplitude
    amp = np.sqrt(lam * (2*N))  # the crucial scaling factor!
    
    # frequency components
    X_f = amp * z
    
    # 5) Enforce Hermitian symmetry for a real process
    #    X_f[k] = conj(X_f[2N-k]) for k=1..2N-1
    #    DC (k=0) and Nyquist (k=N) should be purely real.
    
    # Fix the DC component and the Nyquist if needed:
    X_f[0] = amp[0] * rng.normal()  # real only
    # if length=2N, then X_f[N] is also purely real
    X_f[N] = amp[N] * rng.normal()  # real only
    
    # Now mirror the rest:
    for k in range(1, N):
        X_f[2*N - k] = np.conjugate(X_f[k])
    
    # 6) iFFT to get time-domain. np.fft.ifft has a factor 1/(2N) internally
    x_full = np.fft.ifft(X_f).real  # length=2N
    
    # We'll take the first N points as our "realization" of length N
    x = x_full[:N]
    
    # Mean-zero adjustment (optional)
    x -= np.mean(x)
    
    # 7) Check the variance
    var_theoretical = r_func(0.0, *args, **kwargs)
    var_empirical = np.var(x)
    print(f"Theoretical variance = {var_theoretical:.6g}")
    print(f"Empirical variance   = {var_empirical:.6g}")
    
    return t, x


def simulate_gaussian_process(r, T, dt, *args, max_m=None, **kwargs):
    """
    Simulate a Gaussian stationary process with a correlation function r(t, *args, **kwargs)
    using an extended circulant embedding (Davies–Harte) method.

    Parameters
    ----------
    r : function
        Correlation function with signature r(t, *args, **kwargs). It must be defined for t >= 0.
    T : float
        Total simulation time.
    dt : float
        Time discretization step.
    *args : tuple
        Additional positional arguments for r.
    max_m : int, optional
        Maximum allowed circulant embedding size. Default is 2**15.
    **kwargs : dict
        Additional keyword arguments for r.

    Returns
    -------
    t : numpy.ndarray
        Array of time points from 0 to T.
    x : numpy.ndarray
        Simulated process values at the time points in t.
    """
    # Number of time points.
    N = int(np.floor(T / dt)) + 1
    t = np.linspace(0, (N - 1) * dt, N)
    m = 2 * (N - 1)  # minimal embedding size
    if max_m is None:
        max_m = 2**15  # an arbitrary maximum embedding size

    tol = 1e-10
    while True:
        # Build the circulant vector c of length m.
        # For indices 0 <= j < N, use: c[j] = r(j*dt, *args, **kwargs)
        # For indices N <= j < m, use symmetry: c[j] = r((m - j)*dt, *args, **kwargs)
        c = np.empty(m)
        for j in range(N):
            c[j] = r(j * dt, *args, **kwargs)
        for j in range(N, m):
            c[j] = r((m - j) * dt, *args, **kwargs)
        # Compute eigenvalues of the circulant matrix via FFT.
        lam = np.fft.fft(c).real
        if np.all(lam >= -tol):
            lam[lam < 0] = 0.0  # correct tiny numerical negatives
            break
        else:
            if m >= max_m:
                raise ValueError("Could not find a valid circulant embedding with nonnegative eigenvalues "
                                 "up to max_m = {}. Check your correlation function.".format(max_m))
            m *= 2  # increase embedding size and try again

    # Generate the random Fourier coefficients.
    W = np.zeros(m, dtype=complex)
    W[0] = np.sqrt(lam[0] / m) * np.random.randn()  # k=0 is real

    if m % 2 == 0:
        W[m // 2] = np.sqrt(lam[m // 2] / m) * np.random.randn()  # Nyquist frequency must be real
        k_max = m // 2
    else:
        k_max = (m + 1) // 2

    for k in range(1, k_max):
        real_part = np.random.randn()
        imag_part = np.random.randn()
        W[k] = np.sqrt(lam[k] / (2 * m)) * (real_part + 1j * imag_part)
        W[m - k] = np.conjugate(W[k])  # enforce Hermitian symmetry

    # Inverse FFT (note: np.fft.ifft divides by m) and rescale.
    X = np.fft.ifft(W) * np.sqrt(m)
    x = np.real(X[:N])
    return t, x

def simulate_gaussian_process_cov(r, T, dt, jitter=1e-10, max_attempts=5, *args, **kwargs):
    """
    Simulate a Gaussian stationary process by constructing the full covariance matrix
    and using a Cholesky factorization to generate a sample path.

    This is the “standard” method that directly uses independent Gaussian random
    variables. For a process with N time points, the covariance matrix is Toeplitz
    with first row [r(0), r(dt), ..., r((N-1)*dt)].

    Parameters
    ----------
    r : function
        Correlation function with signature r(t, *args, **kwargs). It must be defined for t >= 0.
    T : float
        Total simulation time.
    dt : float
        Time discretization step.
    *args : tuple
        Additional positional arguments to be passed to r.
    **kwargs : dict
        Additional keyword arguments to be passed to r.

    Returns
    -------
    t : numpy.ndarray
        Array of time points from 0 to T.
    x : numpy.ndarray
        Simulated process values at the time points in t.
    
    Notes
    -----
    The function attempts to vectorize the evaluation of r on the required time grid.
    If r is not vectorized, it falls back to a Python loop (list comprehension).
    """
    # Determine number of time points.
    N = int(np.floor(T / dt)) + 1
    t = np.linspace(0, (N - 1) * dt, N)
    
    # Try to compute the covariance vector in a vectorized way.
    try:
        t_vec = np.arange(N) * dt
        first_row = np.asarray(r(t_vec, *args, **kwargs))
        if first_row.ndim != 1 or first_row.shape[0] != N:
            raise ValueError("Correlation function r did not return the expected shape.")
    except Exception:
        first_row = np.array([r(i * dt, *args, **kwargs) for i in range(N)])
    
    # Construct the Toeplitz covariance matrix.
    C = toeplitz(first_row) # the matrix is symmetric

    # Try a Cholesky decomposition; if it fails, add jitter to the diagonal.
    attempt = 0
    while attempt < max_attempts:
        try:
            L = cholesky(C, lower=True)
            break  # success
        except np.linalg.LinAlgError:
            jitter_value = jitter * (10 ** attempt)
            C = C + jitter_value * np.eye(N)
            attempt += 1
    else:
        raise np.linalg.LinAlgError("Cholesky decomposition failed even after adding jitter.")
    
    # Generate a sample path: if z ~ N(0,I) then x = L*z has covariance C.
    z = np.random.randn(N)
    x = L @ z
    return t, x


