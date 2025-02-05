import numpy as np
from scipy.linalg import toeplitz, cholesky

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


