import torch
from scipy.linalg import toeplitz, cholesky


def simulate_gaussian_process_fft(r_func, T, dt, *args, **kwargs):
    """
    Simulates a zero-mean stationary Gaussian process {X(t)} of duration T,
    sampled at step dt, with correlation function r_func(tau, *args, **kwargs).
    Uses an FFT-based approach (circulant embedding) with correct scaling.
    Based on the algorithm stated at:
    "https://urbain.vaes.uk/static/teaching/lectures/build/lectures-w4.pdf"

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
    t : torch.Tensor of shape (n,)
        Tensor of time points of length n = int(T/dt).
    x : torch.Tensor of shape (n,)
        Simulated Gaussian process values at times in t.
    """
    # 1) Basic discretization
    n = int(round(T / dt))
    N = 2 * n - 2
    # Time points
    t = torch.arange(n, dtype=torch.float64) * dt

    # 2) Build discrete correlation array: r(0..N-1).
    try:
        r_vals = r_func(t, *args, **kwargs)
        if not isinstance(r_vals, torch.Tensor):
            r_vals = torch.tensor(r_vals, dtype=torch.float64)
        if r_vals.ndim != 1 or r_vals.shape[0] != n:
            raise ValueError("Correlation function r did not return the expected shape.")
    except Exception:
        r_vals = torch.tensor([r_func(i * dt, *args, **kwargs) for i in range(N)], dtype=torch.float64)

    # r_full is [r(0), r(dt), ..., r((n-1)*dt), r((n-2)*dt), ..., r(dt)]
    r_full = torch.cat((r_vals, torch.flip(r_vals, dims=[0])[1:-1]))

    # 3) FFT to get "eigenvalues" of the corresponding circulant covariance
    lam = torch.fft.fft(r_full)  # shape (N, )

    # 4) Create random complex amplitudes with correct magnitude
    Z_real = torch.randn(N, dtype=torch.float64)
    Z_imag = torch.randn(N, dtype=torch.float64)
    Z = Z_real + 1j * Z_imag

    # 5) Multiply by sqrt of lam / N
    #    (We do not check positivity of lam here, assuming r_full is valid.)
    Y = torch.sqrt(lam / N) * Z

    # 6) "Inverse" transform to get the time-domain signal.
    #    The original code uses fft again, but typically you'd do an ifft:
    V = torch.fft.fft(Y)  
    # If you actually want the standard approach, do:
    # V = torch.fft.ifft(Y)

    # 7) Extract the real part for the first n points
    x = torch.real(V[:n])

    return t, x

def simulate_gaussian_process_cholesky(r, T, dt, jitter=1e-10, max_attempts=5, *args, **kwargs):
    """
    Simulate a Gaussian stationary process by constructing the full covariance matrix
    and using a Cholesky factorization to generate a sample path.

    For a process with N time points, the covariance matrix is Toeplitz
    with first row [r(0), r(dt), ..., r((N-1)*dt)].

    Parameters
    ----------
    r : function
        Correlation function with signature r(t, *args, **kwargs). It must be defined for t >= 0.
    T : float
        Total simulation time.
    dt : float
        Time discretization step.
    jitter : float, optional
        Small diagonal increment added if the matrix is nearly singular.
    max_attempts : int, optional
        Maximum number of times to attempt adding jitter before giving up.
    *args : tuple
        Additional positional arguments to be passed to r.
    **kwargs : dict
        Additional keyword arguments to be passed to r.

    Returns
    -------
    t : torch.Tensor
        Tensor of time points from 0 to T.
    x : torch.Tensor
        Simulated process values at the time points in t.
    """
    # Determine number of time points
    N = int(round(T / dt)) 
    t = torch.arange(N, dtype=torch.float64) * dt

    # Attempt vectorized evaluation of r
    try:
        first_row_vals = r(t, *args, **kwargs)
        if not isinstance(first_row_vals, torch.Tensor):
            first_row_vals = torch.tensor(first_row_vals, dtype=torch.float64)
        if first_row_vals.ndim != 1 or first_row_vals.shape[0] != N:
            raise ValueError("Correlation function r did not return the expected shape.")
    except Exception:
        first_row_vals = torch.tensor([r(i * dt, *args, **kwargs) for i in range(N)], dtype=torch.float64)

    # Construct the Toeplitz covariance matrix
    C = torch.from_numpy(toeplitz(first_row_vals.numpy()))

    # Attempt Cholesky with jitter if necessary
    attempt = 0
    while attempt < max_attempts:
        try:
            L = torch.linalg.cholesky(C)
            break
        except RuntimeError:
            # Add jitter to the diagonal
            jitter_value = jitter * (10 ** attempt)
            C = C + jitter_value * torch.eye(N, dtype=C.dtype)
            attempt += 1
    else:
        raise RuntimeError("Cholesky decomposition failed even after adding jitter.")

    # Generate the sample path: x = L @ z, where z ~ N(0, I)
    z = torch.randn(N, dtype=torch.float64)
    x = L @ z

    return t, x
