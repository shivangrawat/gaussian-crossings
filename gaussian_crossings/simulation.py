import numpy as np
import scipy
import scipy.linalg
from scipy.linalg import toeplitz, cholesky


def simulate_gaussian_process_fft(r_func, T, dt, *args, **kwargs):
    """
    Simulates a zero-mean stationary Gaussian process {X(t)} of duration T,
    sampled at step dt, with correlation function r_func(tau, *args, **kwargs).
    Uses an FFT-based approach (circulant embedding) with correct scaling.
    Based on the algorithm stated at "https://urbain.vaes.uk/static/teaching/lectures/build/lectures-w4.pdf"
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
    n = int(round(T / dt)) 
    N = 2 * n - 2
    t = np.arange(n) * dt  # times 0..(N-1)*dt
    
    # 2) Build discrete correlation array: r(0..N-1).
    #    r[n] = r_func(n*dt). For an even function r(-tau)=r(tau),
    #    the "circular embedding" method typically uses 2N-length.
    #    But let's do a minimal version with length= N or 2N.
    
    r_vals = np.array([r_func(i * dt, *args, **kwargs) for i in range(n)])
    
    r_full = np.concatenate((r_vals, np.flip(r_vals)[1:-1]))
    
    # 3) FFT to get eigenvalues of the (2N x 2N) circulant covariance matrix
    lam = scipy.fft.fft(r_full)  # length=2N-2
    
    # 4) Create random complex amplitudes with correct magnitude.
    Z = np.random.randn(N) + 1j * np.random.randn(N)

    # 5) Multiply by square root of eigenvalues and inverse FFT
    Y = np.sqrt(lam / N) * Z

    # 6) Calculate the FFt again
    V = scipy.fft.fft(Y)

    # 7) Extract the real part of the inverse FFT
    x = np.real(V[:n])
    
    return t, x

def simulate_gaussian_process_cholesky(r, T, dt, jitter=1e-10, max_attempts=5, *args, **kwargs):
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


