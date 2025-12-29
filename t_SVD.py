# 计算张量的奇异值分解，TSVD
# 导入需要的包
import math
import torch


def t_svd(A):
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)
    if A.is_complex():
        raise ValueError("Input must be real-valued.")

    device = A.device
    dtype = A.dtype
    n3, n1, n2 = A.shape

    # FFT along slice dim (dim=0)
    A_bar = torch.fft.fft(A.to(torch.complex64), dim=0)

    # Pre-allocate full-size tensors
    U_bar = torch.zeros(n3, n1, n1, dtype=torch.complex64, device=device)
    S_bar = torch.zeros(n3, n1, n2, dtype=torch.complex64, device=device)
    Vh_bar = torch.zeros(n3, n2, n2, dtype=torch.complex64, device=device)

    t = math.ceil((n3 + 1) / 2) - 1

    for i in range(t + 1):
        # ✅ CRITICAL: use some=False for full SVD
        U_i, s_i, Vh_i = torch.svd(A_bar[i], some=False)

        # Build full diagonal matrix
        S_i = torch.zeros(n1, n2, dtype=torch.complex64, device=device)
        k = min(n1, n2)
        S_i[:k, :k] = torch.diag(s_i[:k])

        U_bar[i] = U_i  # now (n1, n1) ← OK
        S_bar[i] = S_i  # (n1, n2) ← OK
        Vh_bar[i] = Vh_i  # (n2, n2) ← OK

    # Hermitian symmetry for remaining slices
    for i in range(t + 1, n3):
        j = n3 - i
        U_bar[i] = torch.conj(U_bar[j])
        S_bar[i] = S_bar[j]
        Vh_bar[i] = torch.conj(Vh_bar[j])

    # IFFT + real
    U = torch.fft.ifft(U_bar, dim=0).real.to(dtype)
    S = torch.fft.ifft(S_bar, dim=0).real.to(dtype)
    Vh = torch.fft.ifft(Vh_bar, dim=0).real.to(dtype)

    return U, S, Vh