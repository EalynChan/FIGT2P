# 计算两个张量之间的乘积，t-product
# 导入需要的包
import math
import numpy as np
import scipy.linalg as la
import copy as cp
import torch
import scipy.fftpack as fft

def fft_3d(A):
    return np.fft.fft(A, axis=0)  # np.fft.fft(A, axis=0)
def ifft_3d(A):
    return np.fft.ifftn(A, axes=(0,))  # np.fft.ifftn(A, axes=(0,))

def I_tensor(n, n3):  # 表示单位张量
    I = np.zeros(shape=(n3, n, n))
    I[0] = np.eye(n)
    return I
def block_diag_matrix(matrices):
    # 定义一个函数block_diag_matrix，它接受一个列表作为参数，返回一个分块对角矩阵
    # 初始化一个空矩阵
    result = matrices[0]
    # 遍历列表中的每个矩阵
    for i in range(1, len(matrices)):
        # 使用block_diag函数将当前矩阵和结果矩阵拼接起来
        result = la.block_diag(result, matrices[i])
    # 返回结果矩阵
    return result


import torch
import math

def t_product(A, B):
    """
    Compute the t-product of two real-valued tensors A and B.

    Input layout: (n3, n1, n2) — tube fibers along dim=0
        A: (n3, n1, n2)
        B: (n3, n2, l)   ← must have A.shape[2] == B.shape[1]

    Output:
        C: (n3, n1, l)

    Supports CPU, CUDA, and MPS devices.
    """
    # Ensure inputs are real torch.Tensor
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)
    if not isinstance(B, torch.Tensor):
        B = torch.as_tensor(B)

    if A.is_complex() or B.is_complex():
        raise ValueError("t_product only supports real-valued inputs.")

    device = A.device
    dtype = A.dtype
    n3, n1, n2 = A.shape
    _, n2_B, l = B.shape

    if n2 != n2_B:
        raise ValueError(f"t-product requires A.shape[2] == B.shape[1], got {n2} vs {n2_B}")

    # Step 1: FFT along tube dimension (dim=0)
    A_bar = torch.fft.fft(A.to(torch.complex64), dim=0)  # (n3, n1, n2)
    B_bar = torch.fft.fft(B.to(torch.complex64), dim=0)  # (n3, n2, l)

    # Pre-allocate output in frequency domain
    C_bar = torch.zeros(n3, n1, l, dtype=torch.complex64, device=device)

    # Hermitian symmetry: only compute first t+1 slices
    t = math.ceil((n3 + 1) / 2) - 1  # index of Nyquist frequency

    # Compute first half (including Nyquist)
    for i in range(t + 1):
        C_bar[i] = torch.matmul(A_bar[i], B_bar[i])  # (n1, n2) × (n2, l) → (n1, l)

    # Fill second half using conjugate symmetry
    for i in range(t + 1, n3):
        j = n3 - i
        C_bar[i] = torch.conj(C_bar[j])

    # Step 2: Inverse FFT to time domain
    C = torch.fft.ifft(C_bar, dim=0).real.to(dtype)

    return C

def t_transpose(V):
    """
    Compute t-transpose assuming V has shape (n3, n1, n2),
    where n3 is the number of frontal slices.

    Output: V_t of shape (n3, n2, n1), where:
        V_t[0] = V[0].T
        V_t[j] = V[n3 - j].T  for j = 1, 2, ..., n3-1
    """
    V = torch.as_tensor(V)  # supports numpy input
    if V.is_complex():
        raise ValueError("Input must be real-valued.")

    n3, n1, n2 = V.shape

    if n3 == 1:
        return V.transpose(1, 2).contiguous()  # (1, n1, n2) -> (1, n2, n1)

    # Step 1: Transpose all slices: (n3, n1, n2) -> (n3, n2, n1)
    V_t_all = V.transpose(1, 2)  # vectorized transpose

    # Step 2: Create index mapping: [0, n3-1, n3-2, ..., 1]
    indices = torch.cat([
        torch.tensor([0], device=V.device),
        torch.arange(n3 - 1, 0, -1, device=V.device)  # [n3-1, n3-2, ..., 1]
    ])

    # Step 3: Reorder slices using advanced indexing
    V_t = V_t_all[indices]

    return V_t.contiguous()

def t_conjTranspose(V):
    # 计算张量V的转置
    n3 = V.shape[0]
    V_t = []
    V_temp = cp.deepcopy(V)
    for i in range(0, n3):
        V_temp[i] = np.conjugate(V_temp[i])

    for j in range(0, n3):
        if j == 0:
            V_t_i = np.transpose(V_temp[j])
        else:
            V_t_i = np.transpose(V_temp[n3 - j])
        V_t.append(V_t_i)
    V_t = np.array(V_t)
    return V_t

def t_inverse(A):
    # 计算张量A的逆, 要求A的维度为n x n x n3
    n3, n, n = A.shape
    B_bar = []
    I = np.zeros((n3, n, n))
    I[0] = np.eye(n)
    A_bar, I_bar = fft_3d(A), fft_3d(I)
    for i in range(0, n3):
        A_bar_i_inv = np.linalg.inv(A_bar[i])
        B_bar_i = np.dot(np.array(A_bar_i_inv), np.array(I_bar[i]))
        B_bar.append(B_bar_i)
    B_bar = np.array(B_bar)
    B = ifft_3d(B_bar)
    return B

def t_plus(S, tao):
    # 输入：张量S, 参数tao
    # 输出：S-tao的正数部分, (S-tao)_+=max(S-tao, 0)
    S = S - tao
    S[S<0] = 0
    return S

def t_svt(Y, tao):
    # 输入: 张量Y，参数tao>0
    # 输出：张量奇异值阈值D_tao(Y)
    # step 1: 计算张量Y的离散傅立叶变换
    Y_bar = fft_3d(Y)
    # step 2: 对张量Y_bar的每一个正面切片作矩阵SVT
    n3, n1, n2 = Y.shape
    t = math.ceil((n3 + 1) / 2) - 1
    W_bar = []
    for i in range(0, t+1):
        U, S, Vh = np.linalg.svd(Y_bar[i])
        if n1 == n2:
            S = np.diag(S)
        elif n1 < n2:
            S = np.diag(S)
            S = np.pad(S, ((0, 0), (0, n2 - S.shape[1])), 'constant')
        else:
            S = np.diag(S)
            S = np.pad(S, ((0, n1 - S.shape[0]), (0, 0)), 'constant')
        S_tao = t_plus(S, tao)
        W_bar_i = U @ S_tao @ Vh
        W_bar.append(W_bar_i)
    for j in range(t+1, n3):
        W_bar_j = np.conj(W_bar[n3 - j])
        W_bar.append(W_bar_j)
    # 第 3 步：计算张量Y的svt，通过对W_bar进行逆傅立叶变换
    Y_svt = ifft_3d(W_bar)
    return Y_svt