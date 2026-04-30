import numpy as np


def DLanczos(A, b, rtol=1e-8, atol=1e-12, m=128):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = b.shape[0]

    if m is None:
        m = n

    b_norm = np.linalg.norm(b)

    x_prev = np.zeros_like(b, dtype=float)
    r_prev = b - A @ x_prev
    beta = np.linalg.norm(r_prev)

    if beta < atol:
        return x_prev.copy(), np.array([beta])

    v_prev = np.zeros(n, dtype=float)
    v_curr = r_prev / beta

    p_prev = np.zeros(n, dtype=float)

    beta_curr = 0.0
    eta_prev = 0.0
    ksi_prev = beta

    residuals = np.empty(m, dtype=float)

    for j in range(m):
        w = A @ v_curr - beta_curr * v_prev
        alpha_curr = np.dot(w, v_curr)

        if j == 0:
            l_curr = 0.0
            ksi_curr = beta
        else:
            l_curr = beta_curr / eta_prev
            ksi_curr = -l_curr * ksi_prev

        eta_curr = alpha_curr - l_curr * beta_curr

        if abs(eta_curr) < atol:
            raise RuntimeError("D-Lanczos breakdown: eta is too small")

        p_curr = (v_curr - beta_curr * p_prev) / eta_curr
        x_curr = x_prev + ksi_curr * p_curr

        residual = np.linalg.norm(b - A @ x_curr)
        residuals[j] = residual

        if residual < atol + rtol * b_norm:
            return x_curr, residuals[: j + 1]

        w = w - alpha_curr * v_curr
        beta_next = np.linalg.norm(w)

        if beta_next < atol:
            return x_curr, residuals[: j + 1]

        v_next = w / beta_next

        v_prev = v_curr
        v_curr = v_next
        beta_curr = beta_next
        eta_prev = eta_curr
        ksi_prev = ksi_curr
        p_prev = p_curr
        x_prev = x_curr

    raise RuntimeError("D-Lanczos did not converge")
