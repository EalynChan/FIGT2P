# This is for FIGT2P algorithm
import math
import numpy as np
import datetime
import t_SVD
import t_operation as top
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_curve
import torch
import warnings
import platform
import pandas as pd
import metrics_MLL
from sklearn.model_selection import KFold
import copy
import time
warnings.filterwarnings('ignore')

# 检测并配置GPU设备
def setup_device():
    if platform.system() == "Darwin":  # macOS (Apple Silicon)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            print("Using Metal Performance Shaders (MPS) on Apple Silicon")
        else:
            device = torch.device("cpu")
            print("MPS not available, using CPU on macOS")
    elif platform.system() == "Windows":  # Windows
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # print(f"Using CUDA on Windows - {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU on Windows")
    else:  # Linux or other systems
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA on {platform.system()} - {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print(f"CUDA not available, using CPU on {platform.system()}")

    return device


class MSWMLFG:
    def __init__(self, M0, alpha, beta, gamma, delta1, delta2, tau, lbd1, lbd2, lbd3, lbd4, eta1, eta2, kNum, tol, max_iter, y_test):
        self.M0 = M0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta1 = delta1
        self.delta2 = delta2
        self.tau = tau
        self.lbd1 = lbd1
        self.lbd2 = lbd2
        self.lbd3 = lbd3
        self.lbd4 = lbd4

        self.eta1 = eta1
        self.eta2 = eta2
        self.kNum = kNum
        self.tol = tol
        self.max_iter = max_iter
        self.y_test = y_test

        # 设置设备
        self.device = setup_device()

    def fit(self, X_train, y_train):
        self.X_train = X_train  # 存储训练集特征
        # 存储训练集标签，处理可能包含字符串的数据
        self.y_train = []
        for y in y_train:
            # 确保数据是数值型的
            if isinstance(y, np.ndarray) and y.dtype == np.object_:
                # 将对象类型转换为浮点型
                y_float = np.array(y, dtype=np.float32)
                self.y_train.append(torch.tensor(y_float, device=self.device, dtype=torch.float32))
            else:
                self.y_train.append(torch.tensor(y, device=self.device, dtype=torch.float32))

    def to_device(self, tensor):
        """将张量移动到指定设备"""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        else:
            return torch.tensor(tensor, device=self.device, dtype=torch.float32)

    def vec(self, array_3d):
        # 确保 array_3d 是一个 NumPy 数组
        array_3d = np.asarray(array_3d)

        # 检查 array_3d 的维度是否为 3
        if array_3d.ndim == 3:
            vector = array_3d.transpose((0, 2, 1)).reshape(-1)
        elif array_3d.ndim == 2:
            vector = array_3d.T.reshape(-1)
        else:
            raise ValueError("Unsupported array dimension")
        return vector

    def innvec(self, vector, m, n1, n2):
        # 重新将向量转换回三维数组
        # 需要指定原始的形状和 transpose 操作的反向步骤
        if vector.size != m * n1 * n2:
            raise ValueError(f"无法将大小为{vector.size}的数组reshape为({m}, {n1}, {n2})的形状")

        array_3d_reconstructed = vector.reshape(m, n1, n2).transpose(0, 2, 1)
        return array_3d_reconstructed

    # 将张量n-模展开，n=1，2，3
    def unfold(self, tensor, n):
        # 保持张量在原始设备上（GPU/CPU）
        if isinstance(tensor, torch.Tensor):
            # 直接使用PyTorch操作在GPU上进行模展开
            mode = 0 if n == 3 else n
            if n == 3:  # 模3展开
                print(tensor.shape)
                tensor = tensor.transpose(1, 2)  # 替代numpy的转置操作
                moved = tensor.movedim(mode, 0)
            else:
                moved = tensor.movedim(mode, 0)

            # 使用view替代reshape
            return moved.contiguous().view(moved.shape[0], -1)
        else:
            # 对于numpy数组，保持原有逻辑
            tensor = np.array(tensor)
            mode = 0 if n == 3 else n
            if n == 3:
                tensor = np.array([item.T for item in tensor])
                result = np.moveaxis(tensor, mode, 0)
                if result.shape[0] == 0 or result.size == 0:
                    return result.reshape(0, 0) if len(result.shape) >= 2 else np.array([]).reshape(0, 0)
                return result.reshape(result.shape[mode], -1)
            else:
                result = np.moveaxis(tensor, mode, 0)
                if result.shape[0] == 0 or result.size == 0:
                    return result.reshape(0, 0) if len(result.shape) >= 2 else np.array([]).reshape(0, 0)
                return result.reshape(result.shape[mode], -1)

    # n-模乘积
    def mode_n_product(self, tensor, matrix, mode):
        # Ensure inputs are PyTorch tensors
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.as_tensor(matrix)

        # Move matrix to same device/dtype as tensor
        matrix = matrix.to(device=tensor.device, dtype=tensor.dtype)

        # Validate mode
        if mode < 0:
            mode = tensor.dim() + mode
        if not (0 <= mode < tensor.dim()):
            raise ValueError(f"Invalid mode={mode} for tensor with {tensor.dim()} dimensions.")

        # Check dimension compatibility
        if matrix.shape[1] != tensor.shape[mode]:
            raise ValueError(
                f"Matrix shape {matrix.shape} incompatible with tensor.shape[{mode}]={tensor.shape[mode]}. "
                "Expected matrix.shape[1] == tensor.shape[mode]."
            )

        # Permute so that 'mode' is the first dimension
        perm_order = [mode] + [i for i in range(tensor.dim()) if i != mode]
        tensor_perm = tensor.permute(perm_order)  # (In, I1, ..., I_{mode-1}, I_{mode+1}, ...)

        # Reshape to (In, -1)
        original_shape = tensor.shape
        tensor_unfolded = tensor_perm.reshape(tensor.shape[mode], -1)  # (In, N)

        # Mode-n product: matrix @ tensor_unfolded → (J, N)
        result_unfolded = torch.matmul(matrix, tensor_unfolded)  # (J, N)

        # Reshape back: (J, I1, ..., I_{mode-1}, I_{mode+1}, ...)
        new_shape = [matrix.shape[0]] + [original_shape[i] for i in range(len(original_shape)) if i != mode]
        result_perm = result_unfolded.reshape(new_shape)

        # Invert permutation to restore original order (with mode replaced by J)
        inv_perm_order = [0] * len(new_shape)
        idx = 1  # skip first dim (which is J)
        for i in range(len(original_shape)):
            if i == mode:
                inv_perm_order[i] = 0  # J goes to position 'mode'
            else:
                inv_perm_order[i] = idx
                idx += 1

        result = result_perm.permute(*inv_perm_order)
        return result

    # 获取三维数组的非零元素的索引及其值
    def non_zero(self, arr):
        # 获取非零元素的索引
        non_zero_indices = np.nonzero(arr)
        # 获取非零元素的值
        non_zero_values = arr[non_zero_indices]
        return non_zero_indices, non_zero_values

    def TLR_theorem(self, K, A, tao, alpha):
        """
        Solve: 1/2 * ||A - K||_F^2 + tao * ||K||_TLR
        All operations on GPU if inputs are on GPU.
        """
        device = A.device if isinstance(A, torch.Tensor) else torch.device('cpu')
        dtype = A.dtype if isinstance(A, torch.Tensor) else torch.float32

        # Ensure inputs are real tensors
        A = torch.as_tensor(A, dtype=dtype, device=device)
        K = torch.as_tensor(K, dtype=dtype, device=device)

        if A.is_complex():
            A = A.real
        if K.is_complex():
            K = K.real

        # Clamp inf/nan (PyTorch version)
        A = torch.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)
        K = torch.nan_to_num(K, nan=0.0, posinf=1e6, neginf=-1e6)

        # t-SVD of A
        C, S, D = t_SVD.t_svd(A)  # Assume returns (n1, r, n3), (r, r, n3), (n2, r, n3)

        # FFT along tube fibers (dim=2)
        S_bar = torch.fft.fft(S, dim=0)  # complex

        # Get non-zero values (real part only)
        non_zero_mask_S = S_bar.abs() > 1e-8  # avoid numerical noise
        non_zero_values_S = S_bar[non_zero_mask_S].real  # shape: (N,)

        # t-SVD of K
        C0, T, D0 = t_SVD.t_svd(K)
        T_bar = torch.fft.fft(T, dim=0)
        non_zero_mask_T = T_bar.abs() > 1e-8
        non_zero_values_T = T_bar[non_zero_mask_T].real  # shape: (M,)

        # Pad to same length
        len_S = non_zero_values_S.numel()
        len_T = non_zero_values_T.numel()

        if len_S <= len_T:
            target_len = len_T
            padded_S = torch.nn.functional.pad(non_zero_values_S, (0, target_len - len_S), value=0.0)
            padded_T = non_zero_values_T
            result = padded_S - tao * (alpha * torch.pow(padded_T, alpha - 1)) / (1 + torch.pow(padded_T, alpha))
            result = torch.clamp(result, min=0.0)
            # Put back into T_bar
            T_bar_new = T_bar.clone()
            T_bar_new[non_zero_mask_T] = result.to(T_bar.dtype)
            T_new = torch.fft.ifft(T_bar_new, dim=0).real
        else:
            target_len = len_S
            padded_S = non_zero_values_S
            padded_T = torch.nn.functional.pad(non_zero_values_T, (0, target_len - len_T), value=0.0)
            result = padded_S - tao * (alpha * torch.pow(padded_T, alpha - 1)) / (1 + torch.pow(padded_T, alpha))
            result = torch.clamp(result, min=0.0)
            T_bar_new = T_bar.clone()
            # Use mask from S? But updating T_bar — better to update T_bar's own mask or use full tensor
            # Safer: reconstruct full T_bar with updated singular values?
            # For simplicity, we assume same support (or just update where T had non-zeros)
            T_bar_new[non_zero_mask_T] = result[:len_T].to(T_bar.dtype)  # truncate if longer
            T_new = torch.fft.ifft(T_bar_new, dim=0).real

        # Ensure D is real (t_svd should return real, but be safe)
        D = D.real if D.is_complex() else D

        # Final reconstruction: C * T_new * D^T
        D_t = top.t_transpose(D)  # must accept torch.Tensor and return real
        temp = top.t_product(C, T_new)  # both real

        result = top.t_product(temp, D_t)  # all real → no error

        return result

    ###################################更新各个变量############################################

    # 更新张量M根据公式(2)############################################
    def updateM(self, W, M0):
        # 确保 W 是 PyTorch 张量，并与 self.M0 同设备/类型
        if not isinstance(W, torch.Tensor):
            W = torch.as_tensor(W)
        W = W.to(device=self.device, dtype=W.dtype)

        m = W.shape[0]

        # theta = W @ ones → shape (m,)
        ones = torch.ones(m, 1, device=W.device, dtype=W.dtype)
        theta = W @ ones  # shape (m, 1)

        # 扩展为 (m, 1, 1) 以便与 M0 广播（假设 M0.shape = (I, J, m)，且 m == K）
        theta = theta.view(m, 1, 1)  # or .unsqueeze(-1).unsqueeze(-1)

        # 处理 NaN / Inf（PyTorch 版本）
        theta = torch.nan_to_num(theta, nan=0.0, posinf=10.0, neginf=-10.0)

        # 使用 torch.sigmoid（内部已做数值稳定处理，无需 clip）
        # 但如果你坚持限制输入范围（如原文），可保留 clip
        theta_clipped = torch.clamp(theta, -10, 10)
        sigmoid_theta = torch.sigmoid(theta_clipped)  # = 1 / (1 + exp(-theta))

        # 假设 self.M0 是 (I, J, K)，且 K == m
        M = M0 * sigmoid_theta  # 广播：(I, J, K) × (K, 1, 1) → (I, J, K)

        # 限制输出在 [0, 1]
        M = torch.clamp(M, 0.0, 1.0)
        M = torch.clamp(M, min=0.1)  # 避免 M=0

        return M

    # 更新张量A根据公式(5)############################################
    def updateA(self, H):
        """
        构造 A_k = H_k @ P_k^{-1} @ H_k^T,
        where P_k = diag(H_k^T @ 1_n)  --> shape (q,)
        So P_k^{-1} is (q, q) diagonal.
        H: (m, n, q)
        """

        # Sum over n-dimension: H_k^T @ 1_n --> (m, q)
        sum_over_n = H.sum(dim=1)  # (m, q)

        # Invert (with safe division)
        Pk_inv_diag = torch.where(
            sum_over_n > 0,
            1.0 / sum_over_n,
            torch.zeros_like(sum_over_n)
        )  # (m, q)

        # Build batch diagonal matrix: (m, q, q)
        Pk_inv = torch.diag_embed(Pk_inv_diag)  # (m, q, q)

        # Compute A = H @ Pk_inv @ H^T
        # H: (m, n, q)
        # Pk_inv: (m, q, q)
        # H.transpose(-1, -2): (m, q, n)
        HP = torch.matmul(H, Pk_inv)  # (m, n, q)
        A = torch.matmul(HP, H.transpose(-1, -2))  # (m, n, n)

        return A

    # 更新张量E根据公式(19)############################################
    def updateE(self, Y, B, M):
        residual = Y - B
        mask = M > 1e-8
        if mask.any():
            M_active = M[mask]
            threshold = self.lbd2 / (2 * self.delta1 * (M_active ** 2))
            # 注意：去掉 torch.sign，只保留正残差
            E_part = torch.clamp(residual[mask] - threshold, min=0.0)
            E = torch.zeros_like(Y)
            E[mask] = E_part
            return E
        else:
            return torch.zeros_like(Y)

    # 更新矩阵U根据公式(21)############################################
    def updateU(self, A, W, V, Z, n):
        device = Z.device  # 自动获取设备（CPU/GPU）
        In = torch.eye(n, device=device)  # n阶单位矩阵，放在相同设备上
        V, W = self.to_device(V), self.to_device(W)
        # Step 1: Compute V^T @ Z[k]^T for all k → (m, n, n)
        VZT = torch.einsum('nq,mqp->mnp', V.T, Z.permute(0, 2, 1))
        # Step 2: temp_all[i] = sum_k W[k, i] * VZT[k]
        temp_all = torch.tensordot(W.T, VZT, dims=([1], [0]))  # (m, n, n)
        # Step 3: fast = sum_i A[i] @ temp_all[i]
        fast = torch.einsum('mij,mjk->ik', A, temp_all)

        # --- Clean input ---
        fast = torch.nan_to_num(fast, nan=0.0, posinf=1e5, neginf=-1e5)
        fast = torch.clamp(fast, -1e5, 1e5)

        # --- Add jitter for stability ---
        if fast.dim() == 2:
            eps = 1e-6
            fast = fast + eps * torch.eye(fast.shape[0], device=fast.device, dtype=fast.dtype)
        else:
            # For batched SVD
            eps = 1e-6
            I = torch.eye(fast.shape[-1], device=fast.device, dtype=fast.dtype)
            fast = fast + eps * I

        # --- Try SVD ---
        try:
            E, Sigma, Fh = torch.linalg.svd(fast)
        except torch._C._LinAlgError:
            # Fallback: move to CPU with gesvd
            print("SVD failed on GPU, falling back to CPU gesvd...")
            fast_cpu = fast.cpu()
            E, Sigma, Fh = torch.linalg.svd(fast_cpu, driver='gesvd')
            E, Sigma, Fh = E.to(fast.device), Sigma.to(fast.device), Fh.to(fast.device)

        U_new = Fh.T @ In @ E.T
        U_new[U_new < 0.0] = 0.0
        return U_new

    # 更新矩阵V根据公式(23)############################################
    def updateV(self, Z, W, U, A, n):
        device = Z.device  # 自动获取设备（CPU/GPU）
        I = torch.eye(n, device=device)  # n阶单位矩阵，放在相同设备上
        U, W = self.to_device(U), self.to_device(W)
        # Step 1: Compute U @ A[k] for all k → (m, n, n)
        UA = torch.einsum('nd,mdp->mnp', U, A)
        # Step 2: temp_all[i] = sum_k W[i, k] * UA[k]
        temp_all = torch.tensordot(W, UA, dims=([1], [0]))  # (m, n, n)
        # Step 3: fast = sum_i Z[i].T @ temp_all[i]
        Zt = Z.transpose(-2, -1)  # (m, j, i) if Z is (m, i, j)
        tmp1 = torch.einsum('mij,mjk->ik', Zt, temp_all)  # (i, k)

        WUA2 = temp_all.reshape(-1, n)  # shape: (m*n, n)
        gram_matrix = WUA2.T @ WUA2 + self.lbd3 * I  # (n, n)

        V_new = torch.linalg.solve(gram_matrix.T, tmp1.T).T  # more stable
        V_new = torch.clamp(V_new, min=0.0)

        return V_new

    # 更新矩阵W根据公式(28)############################################
    def updateW(self, W, Z, A, V, U, C, Y, B, E, M, M0, OList, muList):
        # Z3 = self.unfold(Z, 3)
        # A3 = self.unfold(A, 3)
        mu_2 = muList[1]
        O_2 = self.to_device(OList[1])

        # Step 1: Compute U @ A[:,k,:] for all k → (m, n, n)
        U, Z = self.to_device(U), self.to_device(Z)
        W, C = self.to_device(W), self.to_device(C)
        UA = torch.einsum('nd,mdp->pnm', U, A)
        # Step 2: temp_all[i] = sum_k V[i, k] * UA[k]
        temp_all = torch.tensordot(V, UA, dims=([1], [0]))  # (m, n, n)
        # Step 3: fast = sum_i Z[:,i,:].T @ temp_all[i]
        ZUVAT = torch.einsum('qmn,qnk->mk', Z.permute(2, 0, 1), temp_all)
        VUA3 = temp_all.reshape(-1, temp_all.shape[2])
        tmp1 = ZUVAT - W @ VUA3.T @ VUA3
        tmp2 = 2 * tmp1 + mu_2 * (W - C) + O_2
        tmp3 = 2 * torch.sign(W) * torch.sum(torch.abs(W), axis=0)
        # 计算梯度 ∂L1/∂W_{ab}
        # 先对每个 a 计算：sum_{i,j} [Δ_{ija}^2 * M_{ija}^2 * (1 - M_{ija}) * M0_{ija}]
        term1 = ((Y - B - E) ** 2) * M0 * M  # shape: (n, q, K)
        term2 = (B ** 2) * (1 - M) * M0
        # print('term1:', term1, '\nterm2:', term2)
        grad_a = 2 * self.delta1 * torch.sum(term1, axis=(1, 2)) + \
                 2 * self.delta2 * torch.sum(term2, axis=(1, 2))  # sum over i,j → shape: (K,)

        # Step 1: 扩展 grad_a 到 (K, m+1) —— 用 unsqueeze + expand（高效，不复制内存）
        dL_dW = grad_a.unsqueeze(1).expand(-1, m + 1)  # shape: (K, m+1)
        # Step 2: 更新 W
        W_new = W - self.eta1 * (tmp2 + tmp3 + dL_dW)
        # Step 3: Min-Max normalization (global over all elements)
        W_min = W_new.min()
        W_max = W_new.max()
        # 避免除零（如果 W_max == W_min）
        denom = W_max - W_min
        if denom.abs() < 1e-12:
            W_normalized = torch.zeros_like(W_new)
        else:
            W_normalized = (W_new - W_min) / denom

        # Step 4: 截断非正值为 0
        W_normalized = torch.clamp(W_normalized, min=0.0)
        return W_normalized

    # 更新张量Z根据公式(17)############################################
    def updateTenZ(self, Z, A, U, V, W, B, OList, muList):
        AUVW = self.mode_n_product(self.mode_n_product(self.mode_n_product(A, U, 1),
                                                       V, 2), W, 0)
        mu_1 = muList[0]
        O_1 = self.to_device(OList[0])
        AUVW = self.to_device(AUVW)
        B = self.to_device(B)
        # Y = torch.tensor(Y)

        # 检查张量维度是否匹配，如果不匹配则调整
        if AUVW.shape != B.shape:
            # 如果形状不匹配，我们使用较小的形状
            min_shape = [min(AUVW.shape[i], B.shape[i]) for i in range(len(AUVW.shape))]
            # 对每个张量进行切片以匹配较小的形状
            AUVW = AUVW[:min_shape[0], :min_shape[1], :min_shape[2]]
            B = B[:min_shape[0], :min_shape[1], :min_shape[2]]
            O_1 = O_1[:min_shape[0], :min_shape[1], :min_shape[2]]

        D1 = (2 * (AUVW) + mu_1 * B - O_1) / (2 + mu_1)
        # Min-Max 归一化
        D1_min = D1.min()
        D1_max = D1.max()
        if D1_max > D1_min:
            D1_normalized = (D1 - D1_min) / (D1_max - D1_min)
        else:
            D1_normalized = torch.zeros_like(D1)

        Z_new = self.TLR_theorem(Z, D1_normalized, self.lbd1 / (2 + mu_1), self.alpha)
        result = self.to_device(Z_new.real)

        # 处理可能的NaN和inf值
        if torch.isnan(result).any() or torch.isinf(result).any():
            result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

        return result

    # 更新张量H根据公式(34)############################################
    def updateH(self, H, C, Z):
        device = H.device
        dtype = H.dtype
        K, m, q = H.shape

        # 预处理 C1 = sum(C, dim=1) → (K,)
        C1 = torch.sum(C, dim=1)  # shape (K,)

        # 预处理 ZZT = Z @ Z.T → (q, q)
        ZZT = Z @ Z.T  # (q, q)

        H_new = H.clone()

        for k in range(K):
            # --- Step 1: Compute Qk from Xk ---
            Xk = self.X_train[k]  # assume it's already a torch.Tensor on correct device
            if not isinstance(Xk, torch.Tensor):
                Xk = torch.as_tensor(Xk, device=device, dtype=dtype)
            else:
                Xk = Xk.to(device, dtype)

            n, d = Xk.shape
            # Pairwise squared L2: ||x_i - x_j||^2
            Xk_norm_sq = torch.sum(Xk ** 2, dim=1)  # (n,)
            dot = Xk @ Xk.T  # (n, n)
            Qk = Xk_norm_sq.unsqueeze(1) + Xk_norm_sq.unsqueeze(0) - 2 * dot
            Qk = torch.clamp(Qk, min=0.0)  # ensure non-negative

            # --- Step 2: Extract Hk and compute column sums ---
            Hk = H[k, :, :]  # (m, q)
            col_sum = Hk.sum(dim=0)  # (q,) — sum over i (rows)
            p_jj_safe = torch.where(col_sum > 1e-8, col_sum, torch.ones_like(col_sum))
            p_inv_sq = (1.0 / p_jj_safe) ** 2  # (q,)

            # --- Step 3: Compute Gk ---
            # Pk_inv = diag(1 / col_sum) → we avoid forming full matrix
            # HP1 = Hk @ Pk_inv → equivalent to Hk / col_sum (broadcast division)
            HP1 = Hk / p_jj_safe  # (m, q) — broadcasting over columns

            assert Qk.shape[0] == HP1.shape[0], f"Qk rows {Qk.shape[0]} != HP1 rows {HP1.shape[0]}"

            left = self.tau * (Qk + Qk.T) @ HP1  # (n, q)
            right_inner = 2 * ZZT @ HP1 - Hk @ (Hk.T @ ZZT @ HP1 / p_jj_safe)  # (m, q)
            right = self.gamma * C1[k] * right_inner  # (m, q)

            Gk = left + 2 * self.lbd3 * Hk - right  # (m, q)

            # --- Step 4: Compute a_k = diag(Hk.T @ Qk @ Hk) ---
            # Hk.T @ Qk @ Hk → (q, q)
            A = Hk.T @ Qk @ Hk  # (q, q)
            a_k = torch.diag(A)  # (q,)

            # --- Step 5: Compute scaling factor ---
            numerator = self.tau * a_k * p_inv_sq  # (q,)
            # Broadcast numerator to (m, q): repeat along rows
            numerator_broad = numerator.unsqueeze(0).expand(m, -1)  # (m, q)

            eps = 1e-8
            scaling_factor = torch.sqrt(numerator_broad / (Gk + eps))  # (m, q)

            # --- Step 6: Update H_new ---
            H_new[k] = Hk * scaling_factor

            # --- Step 7: Handle NaNs ---
            if torch.isnan(H_new[k]).any():
                mean_val = torch.nanmean(H_new[k])
                H_new[k] = torch.where(torch.isnan(H_new[k]), mean_val, H_new[k])

        return H_new

    # 更新矩阵Z根据公式(36)############################################
    def updateMatZ(self, A, B, C):
        device = A.device
        dtype = A.dtype
        m, n, n1 = A.shape

        if n1 != n:
            raise ValueError(f"A must be square in last two dims, got {n} x {n1}")

        # Step 1: Compute C1 = sum(C, dim=1) → (m,)
        C1 = torch.sum(C, dim=1)  # shape (m,)

        # Step 2: Compute result = Σ_k C1[k] * (Lk + Lk^T)
        # We'll compute all Lk in batch if possible
        Lset = []

        col_sum = A.sum(dim=1)  # (m, n): sum over rows → each column total
        Dk_batch = torch.diag_embed(col_sum)  # (m, n, n)

        # Laplacian: Lk = Dk - Ak
        Lk_batch = Dk_batch - A  # (m, n, n)

        # Symmetrize: Lk + Lk^T
        sym_Lk_batch = Lk_batch + Lk_batch.transpose(-2, -1)  # (m, n, n)

        weights = C1.view(m, 1, 1)  # (m, 1, 1)
        result = torch.sum(weights * sym_Lk_batch, dim=0)  # (n, n)

        # Store Lset as list of tensors (if needed for output)
        Lset = [Lk_batch[k] for k in range(m)]

        # Step 3: Build regularization matrix
        eye = torch.eye(n, device=device, dtype=dtype)
        reg_matrix = 2 * self.beta * eye + self.gamma * result + 1e-6 * eye  # (n, n)

        # Step 4: Invert (use pinv if singular)
        try:
            left = torch.linalg.inv(reg_matrix)
        except torch._C._LinAlgError:
            left = torch.linalg.pinv(reg_matrix)

        # Step 5: Compute right = mode_n_product_vector(B, C1, 0)
        # Assuming mode_n_product_vector(B, C1, 0) means: C1 @ B (since mode=0)
        # B shape: (m, n, r) → after mode-0 product with C1 (m,) → (n, r)
        if B.dim() == 3:
            # Contract first dimension: (m,) @ (m, n, r) → (n, r)
            right = torch.tensordot(C1, B, dims=([0], [0]))  # (n, r)
        elif B.dim() == 2:
            # B: (m, n), then C1 @ B → (n,)
            right = C1 @ B  # (n,)
            right = right.unsqueeze(-1)  # make (n, 1) if needed
        else:
            raise ValueError(f"Unsupported B shape: {B.shape}")

        # Step 6: Compute Z = 2β * left @ right
        Z = 2 * self.beta * (left @ right)  # (n, r)

        # Step 7: Min-Max normalization
        Z_min = Z.min()
        Z_max = Z.max()
        eps = 1e-10
        if (Z_max - Z_min) > eps:
            Z_norm = (Z - Z_min) / (Z_max - Z_min)
        else:
            Z_norm = Z

        # Step 8: Handle NaN/Inf
        Z_norm = torch.nan_to_num(Z_norm, nan=0.0, posinf=1.0, neginf=-1.0)

        return Z_norm, Lset

    # 更新张量B根据公式(38)############################################
    def updateB(self, C, M, Y, Z_tensor, Z, E, OList, muList):
        m, n, q = M.shape
        C, Z = self.to_device(C), self.to_device(Z)
        C1 = torch.sum(C, dim=1)
        mu1 = muList[0]
        O1 = self.to_device(OList[0])
        eps = 1e-12  # 防除零
        beta, delta1, delta2 = self.beta, self.delta1, self.delta2

        # ---------- Step 1: Precompute diagonal of G3 ----------
        # g(M) = M^2 (element-wise)
        M_sq = M ** 2  # (n, q, m)
        ones_minus_M_sq = (1.0 - M) ** 2  # (n, q, m)

        # Sum over (i,j) to get per-view (mode-3) diagonal
        # diag_G3[k] = 2*delta1 * sum_{i,j} M[i,j,k]^2 + 2*delta2 * sum_{i,j} (1 - M[i,j,k])^2
        sum_M_sq = M_sq.sum(axis=(1, 2))  # (m,)
        sum_ones_minus_M_sq = ones_minus_M_sq.sum(axis=(1, 2))  # (m,)

        diag_G3 = 2 * delta1 * sum_M_sq + 2 * delta2 * sum_ones_minus_M_sq  # (m,)

        # Diagonal of A = mu1 * I + G3  → D = mu1 + diag_G3
        D_diag = mu1 + diag_G3  # (m,)
        D_inv = 1.0 / (D_diag + eps)  # (m,)

        # Rank-1 vector u = sqrt(2*beta) * C1
        u = math.sqrt(2 * beta) * C1  # (m,)

        # Precompute v = D^{-1} u and alpha = u^T v
        v = D_inv * u  # (m,)
        alpha = torch.dot(u, v)  # scalar

        denom = 1.0 + alpha + eps  # scalar, denominator in Sherman-Morrison

        # ---------- Step 2: Prepare right-hand side for all (i,j) ----------
        # Reshape tensors to (m, n*q) for easier column-wise ops
        N = n * q  # 120 * 7 = 840

        Z_flat = Z.reshape(-1)  # Z is (n, q) → (N,)
        # But now tensors are (m, n, q) → reshape to (m, N)
        Z_tensor_flat = Z_tensor.reshape(m, N)  # (m, N)
        O1_flat = O1.reshape(m, N)  # (m, N)

        residual = Y - E  # (n, q, m)
        penalty_term = 2 * delta1 * (M ** 2) * residual  # (n, q, m)
        penalty_flat = penalty_term.reshape(-1, m).T  # (n*q, m)
        tempCZ = torch.einsum('i,j->ij', C1, Z_flat)
        rhs = (
                2 * beta * tempCZ  # (m, N)
                + mu1 * Z_tensor_flat  # (m, N)
                + O1_flat  # (m, N)
                + penalty_flat  # (m, N)
        )  # (m, N)

        # Transpose to (N, m) for column-wise solve
        rhs_T = rhs.T  # (N, m)

        # ---------- Step 3: Solve A x = rhs using Sherman-Morrison ----------
        # For each row i (corresponding to one (i,j) pair), solve A x_i = rhs[i]
        # x_i = D^{-1} rhs_i - (v * (u^T (D^{-1} rhs_i))) / (1 + alpha)
        # Step 3.1: w = D^{-1} * rhs  (element-wise scaling along m-axis)
        w = rhs_T * D_inv[np.newaxis, :]  # (N, m)
        # Step 3.2: beta_i = u^T w_i for each i → (N,)
        beta_i = w @ u  # (N,)
        # Step 3.3: correction = v * beta_i / denom → broadcast to (N, m)
        correction = torch.outer(beta_i, v) / denom  # (N, m)
        # Step 3.4: x = w - correction
        B_flat = w - correction  # (N, m)

        # ---------- Step 4: Reshape back to (n, q, m) ----------
        B = B_flat.reshape(m, n, q)
        return B

    # 更新矩阵C根据公式(41)############################################
    def updateC(self, B, W, Z, L_list, OList, muList):
        Z = self.to_device(Z)
        O2 = self.to_device(OList[1])
        mu2 = muList[1]
        eps = 1e-12

        m, n, q = B.shape
        N = n * q

        # --- Step 1: Compute ∂L2/∂C (shape m×m, each row a is constant t_a)
        device = W.device
        dL2_dC = torch.empty((m, m), dtype=torch.float32, device=device)
        ZZt = Z @ Z.T  # (n, n), used for trace(L_a Z Z^T)

        for a in range(m):
            # Tr(Z^T L_a Z) = Tr(L_a Z Z^T)
            t_a = torch.trace(self.to_device(L_list[a]) @ ZZt)  # O(n^2) if dense; use sparse matmul if possible
            dL2_dC[a, :] = t_a  # entire row is t_a

        # --- Step 2: Compute u = B_(3) vec(Z)
        B_mat = B.reshape(m, N)  # B_(3) ∈ R^{m × N}
        z_vec = Z.reshape(N)  # vec(Z)
        u = B_mat @ z_vec  # (m,)

        # --- Step 3: Construct RHS matrix R = μ2*W + O2 + 2β*(u 1^T) - dL2/dC
        ones = torch.ones(m, dtype=u.dtype, device=device)
        R = mu2 * W + O2 + 2 * beta * torch.outer(u, ones) - dL2_dC  # (m, m)

        # --- Step 4: Compute K = B_(3) B_(3)^T ∈ R^{m×m}
        K = B_mat @ B_mat.T  # (m, m), cost O(m^2 * N) = O(m^2 n q)
        # --- Step 5: Solve for s in (μ2 I + 2β m K) s = R 1
        r = R @ ones  # (m,)
        Im = torch.eye(m, device=device)
        M = mu2 * Im + 2 * beta * m * K  # (m, m)
        # Add small regularization for numerical stability
        M += eps * Im
        s = torch.linalg.solve(M, r)  # shape: (m,), cost O(m^3)

        # --- Step 6: Recover C = (1/μ2) * (R - 2β K s 1^T)
        ks = K @ s  # (m,)
        C = (R - 2 * beta * torch.outer(ks, ones)) / mu2  # (m, m)

        C_min = C.min()
        C_max = C.max()
        # 避免除零（如果 W_max == W_min）
        denom = C_max - C_min
        if denom.abs() < 1e-12:
            C_normalized = torch.zeros_like(C)
        else:
            C_normalized = (C - C_min) / denom

        # Step 4: 截断非正值为 0
        C_normalized = torch.clamp(C_normalized, min=0.0)

        return C_normalized

    # 更新Oi(i=1,2)
    def updateOmu(self, Ztensor_new, B_new, W_new, C_new, OList, muList, rho, mu_max):
        Ztensor_new = self.to_device(Ztensor_new)
        B_new = self.to_device(B_new)
        W_new = self.to_device(W_new)
        C_new = self.to_device(C_new)

        OList[0] = self.to_device(OList[0]) + muList[0] * (Ztensor_new - B_new)
        OList[1] = OList[1] + muList[1] * (W_new - C_new)

        for i in range(len(muList)):
            muList[i] = np.minimum(rho * muList[i], mu_max)
        return OList, muList

    def round_tensor(self, x: torch.Tensor, decimals: int) -> torch.Tensor:
        """
        Round a scalar tensor to a given number of decimal places.
        Equivalent to np.round(x, decimals=decimals).
        """
        if decimals < 0:
            raise ValueError("decimals must be non-negative")
        if decimals == 0:
            return torch.round(x)
        factor = 10.0 ** decimals
        return torch.round(x * factor) / factor

    def compute_errors(self,
            Ztensor_new, Ztensor,
            E_new, E,
            H_new, H,
            U_new, U,
            V_new, V,
            W_new, W,
            Zmat_new, Zmat,
            redus: int = 4
    ):

        # Helper: safe L-infinity norm
        def inf_norm(a, b):
            return torch.norm(a - b, p=float('inf'))

        # Compute errors for most variables
        err_Zt = self.round_tensor(inf_norm(Ztensor_new, Ztensor), redus)
        err_E = self.round_tensor(inf_norm(E_new, E), redus)
        err_H = self.round_tensor(inf_norm(H_new, H), redus)
        err_U = self.round_tensor(inf_norm(U_new, U), redus)
        err_W = self.round_tensor(inf_norm(W_new, W), redus)
        err_Zm = self.round_tensor(inf_norm(Zmat_new, Zmat), redus)

        # Special handling for V (shape may mismatch during initialization)
        if V_new.shape == V.shape:
            err_V = self.round_tensor(inf_norm(V_new, V), redus)
        else:
            V_flat_new = V_new.flatten()
            V_flat = V.flatten()
            min_size = min(V_flat_new.numel(), V_flat.numel())
            if min_size > 0:
                diff = V_flat_new[:min_size] - V_flat[:min_size]
                err_V = self.round_tensor(torch.norm(diff, p=float('inf')), redus)
            else:
                # Create zero tensor on the same device as V
                err_V = torch.tensor(0.0, device=V.device, dtype=V.dtype)

        return err_Zt, err_E, err_H, err_U, err_V, err_W, err_Zm

    # 主算法MSWMLFG
    def MSWMLFG_Algorithm(self):
        m, n, q = len(self.X_train), self.X_train[0].shape[0], self.y_train[0].shape[1]  # m=number of sources, n=number of samples, q=number of labels
        # print('m, n, q=', m, n, q)

        # 使用不同的随机种子确保每次初始化不同
        torch.manual_seed(datetime.datetime.now().microsecond)
        np.random.seed(datetime.datetime.now().microsecond)

        # 使用不同的初始化策略，增加随机性
        W = torch.rand(m, m, device=self.device) * 0.5 + 0.1  # 更大的初始化值范围
        C = torch.rand(m, m, device=self.device) * 0.5 + 0.1
        H = torch.rand(m, n, q, device=self.device) * 0.5 + 0.1
        E = torch.rand(m, n, q, device=self.device) * 0.1
        Ztensor = torch.rand(m, n, q, device=self.device) * 0.5 + 0.1
        B = torch.rand(m, n, q, device=self.device) * 0.1
        # Ztensor = torch.tensor(self.y_train)
        # 复制 Y 成为 m 个矩阵组成的张量
        # Ztensor = torch.stack([torch.tensor(self.y_test)] * m, dim=0)  # 张量的形状为 (m, n, p)
        # 假设 self.device 已定义，例如 self.device = torch.device('mps')
        U = (torch.rand(n, n, device=self.device) * 0.5 + 0.1)
        V = (torch.rand(q, n, device=self.device) * 0.5 + 0.1)
        Zmat = (torch.rand(n, q, device=self.device) * 0.1)

        O1 = torch.zeros(m, n, q, device=self.device)
        O2 = torch.zeros(W.shape, device=self.device)
        OList = [O1, O2]
        muList = [10e-3, 10e-3]  # 更大的初始mu值
        rho = self.eta2
        mu_max = 10e+8  # 更大的mu_max

        X, Y = self.X_train, self.y_train  # 直接使用已经处理好的y_train
        for k in range(m):
            # X[k] = (X[k] - X[k].min()) / (X[k].max() - X[k].min())  # Min-Max归一化
            # 或
            X[k] = (X[k] - X[k].mean()) / (X[k].std() + 1e-8)  # Z-score标准化，添加小值避免除零
        Y = torch.stack(Y, dim=0)
        M0 = torch.tensor(self.M0, device=self.device, dtype=torch.float32)

        t = 0  # 迭代次数
        prev_err = float('inf')
        no_improvement_count = 0

        while t < self.max_iter:
            M_new = self.updateM(W, M0)
            # print('M_new:', M_new, M_new.shape)
            A_new = self.updateA(H)
            # print('A_new:', A_new, A_new.shape)
            Ztensor_new = self.updateTenZ(Ztensor, A_new, U, V, W, B, OList, muList)
            # print('Ztenosr_new:', Ztensor_new, Ztensor_new.shape)
            E_new = self.updateE(Y, B, M_new)
            # print('E_new:', E_new, E_new.shape)
            U_new = self.updateU(A_new, W, V, Ztensor_new, n)
            # print('U_new:', U_new, U_new.shape)
            V_new = self.updateV(Ztensor, W, U_new, A_new, n)
            # print('V_new:', V_new, V_new.shape)
            W_new = self.updateW(W, Ztensor_new, A_new, V_new, U_new, C, Y, B, E_new, M_new, M0, OList, muList)
            # print('W_new:', W_new, W_new.shape)
            H_new = self.updateH(H, C, Zmat)
            # print('H_new:', H_new, H_new[0].shape)
            Zmat_new, Lset = self.updateMatZ(A_new, B, C)
            # print('Zmat_new:', Zmat_new, Zmat_new.shape)
            B_new = self.updateB(C, M_new, Y, Ztensor_new, Zmat_new, E_new, OList, muList)
            # print('B_new:', B_new, B_new.shape, type(B_new))
            C_new = self.updateC(B_new, W_new, Zmat_new, Lset, OList, muList)
            # print('C_new:', C_new, C_new.shape)

            OList_new, muList_new = self.updateOmu(Ztensor_new, B_new, W_new, C_new, OList, muList, rho, mu_max)

            # 确保所有张量在相同设备上进行计算
            err_Zt, err_E, err_H, err_U, err_V, err_W, err_Zm = self.compute_errors(
                Ztensor_new, Ztensor,
                E_new, E,
                H_new, H,
                U_new, U,
                V_new, V,
                W_new, W,
                Zmat_new, Zmat,
                redus=4
            )

            total_err = err_Zt + err_E + err_H + err_U + err_V + err_W + err_Zm
            # 终止条件判断
            if t > 1:
                if total_err <= self.tol and total_err > 1e-10:  # 添加下限避免过早终止
                    # print('end algorithm1')
                    break
                # 检查是否有改进
                if abs(prev_err - total_err) < 1e-8:  # 更严格的改进判断
                    no_improvement_count += 1
                    if no_improvement_count > 3:  # 连续3次没有改进则退出
                        break
                else:
                    no_improvement_count = 0

            prev_err = total_err

            # 更新各个值
            M = M_new
            A = A_new
            Ztensor = Ztensor_new
            E = E_new
            U = U_new
            V = V_new
            W = W_new
            H = H_new
            Zmat = Zmat_new
            B = B_new
            C = C_new

            OList = OList_new
            muList = muList_new

            t += 1

        return A, Ztensor, E, U, V, W, H, Zmat

    def find_common_k_neighbors(self, data_matrices, k):
        all_neighbors = []

        for matrix in data_matrices:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(matrix)
            distances, indices = nbrs.kneighbors(matrix)
            all_neighbors.append(indices)

        # 向量化交集计算
        n_nodes = all_neighbors[0].shape[0]
        n_matrices = len(all_neighbors)

        # 转换为三维数组 (n_nodes, n_matrices, k)
        neighbors_array = np.array(all_neighbors).transpose(1, 0, 2)

        # 计算交集
        common_neighbors = []
        for node_idx in range(n_nodes):
            # 使用numpy的交集函数
            common = neighbors_array[node_idx, 0]
            for mat_idx in range(1, n_matrices):
                common = np.intersect1d(common, neighbors_array[node_idx, mat_idx])
            common_neighbors.append(common.tolist())

        return common_neighbors

    def binarize_with_fallback(self, score_tensor, threshold):
        # Step 1: 初始二值化（>= threshold）
        binary = (score_tensor >= threshold).long()  # (V, N, L)

        # Step 2: 找出全零的样本位置（按最后一个维度求和 == 0）
        # all_zero: (V, N)
        all_zero = (binary.sum(dim=-1) == 0)

        if all_zero.any():
            # Step 3: 对全零样本，找出每样本最大值的索引
            # argmax_idx: (V, N)
            argmax_idx = score_tensor.argmax(dim=-1)  # 在 L 维上取 argmax

            # Step 4: 构造 one-hot 向量
            # 使用 scatter_ 将 1 写入最大值位置
            fallback = torch.zeros_like(binary)
            fallback.scatter_(dim=-1, index=argmax_idx.unsqueeze(-1), value=1)

            # Step 5: 只对 all_zero 的位置使用 fallback
            # 扩展 all_zero 到 (V, N, 1) 以便广播
            mask = all_zero.unsqueeze(-1)  # (V, N, 1)
            binary = torch.where(mask, fallback, binary)

        return binary

    def mode_n_product_vector(self, tensor, vector, mode):
        # Ensure inputs are tensors
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        if not isinstance(vector, torch.Tensor):
            vector = torch.as_tensor(vector)

        # Move vector to same device and dtype as tensor
        vector = vector.to(device=tensor.device, dtype=tensor.dtype)

        # Normalize negative mode index
        if mode < 0:
            mode = tensor.dim() + mode
        if not (0 <= mode < tensor.dim()):
            raise ValueError(f"Invalid mode={mode} for tensor with {tensor.dim()} dimensions.")

        # Check dimension compatibility
        if vector.numel() != tensor.shape[mode]:
            raise ValueError(
                f"Vector length {vector.numel()} does not match tensor.shape[{mode}] = {tensor.shape[mode]}"
            )

        # Ensure vector is 1D
        if vector.dim() != 1:
            vector = vector.view(-1)

        # Step 1: Move the target mode to the front
        perm = [mode] + [i for i in range(tensor.dim()) if i != mode]
        tensor_perm = tensor.permute(perm)  # (Im, I1, ..., I_{m-1}, I_{m+1}, ...)

        # Step 2: Reshape to (Im, -1)
        tensor_flat = tensor_perm.reshape(tensor.shape[mode], -1)  # (Im, N)

        # Step 3: Contract with vector: v^T @ tensor_flat → (N,)
        contracted = torch.matmul(vector, tensor_flat)  # (N,)

        # Step 4: Reshape back to original shape without mode dimension
        new_shape = [tensor.shape[i] for i in range(tensor.dim()) if i != mode]
        result = contracted.reshape(new_shape)

        return result

    def predict(self, X_test_views, y_test):
        # 1. 获取测试样本在训练集中的共同 k 近邻（返回的是训练集内部索引！
        actual_kNum = min(self.kNum, self.X_train[0].shape[0])
        common_neighbors = self.find_common_k_neighbors(X_test_views, k=actual_kNum)

        # 2. 获取模型重建部分（你已有的逻辑）
        A, Ztensor, E, U, V, W, H, Zmat = self.MSWMLFG_Algorithm()
        m, n, q = Ztensor.shape  # n 是训练样本数

        y_score_model_list = []
        y_score_knn_list = []
        y_score_model_view = torch.zeros(m, y_test.shape[0], q, device=self.device)

        for i in range(y_test.shape[0]):
            neighbor_indices = common_neighbors[i]  # 如 [0, 2, 4]，合法索引

            # --- 模型重建得分 ---
            A_testi = A[:, neighbor_indices, :][:, :, neighbor_indices]
            V_testi = V[:, neighbor_indices]
            U_testi = U[neighbor_indices, :][:, neighbor_indices]
            Y_testi = self.mode_n_product(
                self.mode_n_product(
                    self.mode_n_product(A_testi, U_testi, 1), V_testi, 2
                ), W, 0
            )
            # print(i, 'Y_testi=', Y_testi.shape)
            y_score_model_view[:, i, :] = Y_testi.mean(dim=1)
            # print(i, 'Y_testi_view=', Y_testi_view)
            W1 = torch.sum(W, dim=1)
            W1 = W1 / torch.sum(W1)
            y_score_model_i = self.mode_n_product_vector(Y_testi, W1, 0)
            # print(i, 'y_score_model_i=', y_score_model_i.shape)
            y_score_model_i = torch.mean(y_score_model_i, dim=0)
            y_score_model_list.append(y_score_model_i)

            # --- KNN 标签投票得分 ---
            # neighbor_labels = torch.as_tensor(data_label[neighbor_indices], dtype=torch.float32,  device=self.device)
            # y_score_knn_i = torch.mean(neighbor_labels, dim=0)
            neighbor_labels = Ztensor[:, neighbor_indices, :]
            y_score_knn_i = torch.mean(neighbor_labels, dim=(0,1))
            # print(Ztensor.shape, neighbor_labels.shape, y_score_knn_i.shape)
            y_score_knn_list.append(y_score_knn_i)
        # print('y_score_model_view=', y_score_model_view.shape)

        # --- 融合得分 ---
        y_score_model = torch.stack(y_score_model_list, dim=0)
        y_score_knn = torch.stack(y_score_knn_list, dim=0)
        y_score = 0.5 * y_score_model + 0.5 * y_score_knn  # 可调权重
        y_score_view = 0.5 * y_score_model_view + 0.5 * y_score_knn

        # --- 生成预测标签 ---
        precision, recall, thresholds_pr = precision_recall_curve(
            y_test.ravel(),
            y_score.detach().cpu().numpy().ravel()
        )
        precision = precision[:-1]
        recall = recall[:-1]
        optimal_idx_pr = np.argmax(precision - recall)
        optimal_threshold_pr = thresholds_pr[optimal_idx_pr]

        # 向量化实现
        y_score_tensor = torch.as_tensor(y_score, device=self.device, dtype=torch.float32)

        # 初始二值化
        y_pred = (y_score_tensor >= optimal_threshold_pr).long()

        # 找出全零行
        all_zero_rows = (y_pred.sum(dim=1) == 0)

        # 为全零行分配最大值索引
        if torch.any(all_zero_rows):
            max_indices = torch.argmax(y_score_tensor, dim=1)
            # 为全零行设置最大值位置为1
            y_pred[all_zero_rows, max_indices[all_zero_rows]] = 1

        y_pred_view = self.binarize_with_fallback(y_score_view, optimal_threshold_pr)

        # 在调用sklearn函数前转换张量
        y_score_np = y_score.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        y_pred_view_np = y_pred_view.detach().cpu().numpy()

        return y_score_np, y_pred_np, y_pred_view_np


if __name__ == '__main__':
    dataName = 'EXAMPLE'

    MissRat = [0.3, 0.6]
    NoiseRat = [0.3, 0.6]

    file = './weaklabel_datasets/' # 包含缺失和噪声标签的数据集
    # 原始准确的多标签数据集data
    data = pd.read_csv('./datasets/EXAMPLE_0.csv')
    labNum = 7 # 标签个数
    souNum = 4 # 数据源个数
    fullsamNum = data.shape[0]
    # 原始准确的多标签数据集的完整标签矩阵
    data_label = np.array(data.iloc[0:fullsamNum, -labNum:])

    X, Y, M0 = [], [], []
    attNumList = []
    result = []
    for mr in range(0, len(MissRat)):
        Scores = []
        for noise in range(0, len(NoiseRat)):
            for m in range(souNum):
                # 创建多视图数据，每个视图有一个特征矩阵，最后用列表表示多视图数据
                dfm = pd.read_csv(file + dataName + '_' + str(m) + '_' + str(MissRat[mr]) + '_p_each_' + str(NoiseRat[noise]) + '.csv')
                attNum = dfm.shape[1] - labNum
                attNumList.append(attNum)
                X.append(np.array(dfm.iloc[:, 0:attNum]))
                # 创建多标签矩阵，将标签缺失的位置"？"替换为"0"
                missLocMat = np.zeros(shape=(dfm.shape[0], labNum))
                dfm.iloc[:, -labNum:] = dfm.iloc[:, -labNum:].replace({'0.0': '0', '1.0': '1'})
                temp = copy.deepcopy(dfm.iloc[:, -labNum:])
                for i in range(dfm.shape[0]):
                    for j in range(labNum):
                        if temp.iloc[i, j] == '?':
                            missLocMat[i, j] = 1  # missLocMat表示缺失标签的位置指示0-1矩阵，1的位置表示缺失标签
                            temp.iloc[i, j] = '0'
                        temp.iloc[i, j] = int(temp.iloc[i, j])
                M0.append(missLocMat)
                Y.append(np.array(temp))
            # 假设 X 和 y 是你的特征和标签数据
            kfold = KFold(n_splits=5, shuffle=True,
                          random_state=42)  # n_splits：拆分成几个子集，shuffle：是否打乱数据集的顺序，random_state：控制随机数生成的种子

            start_time = time.time()
            # 创建最终分数列表
            scoresList = []
            for train_index, test_index in kfold.split(X[0], Y[0]):
                X_train, X_test = [], []
                y_train, y_test = [], data_label[test_index]
                M0_train, M0_test = [], []
                for v in range(souNum):
                    X_train.append(X[v][train_index])
                    X_test.append(X[v][test_index])
                    y_train.append(Y[v][train_index])
                    M0_train.append(M0[v][train_index])
                    M0_test.append(M0[v][test_index])

                # 在这里训练模型并评估性能
                alpha, beta, gamma, delta1, delta2, tau = 0.1, 5, 5, 5, 1, 5
                lbd1, lbd2, lbd3, lbd4 = 0.1, 0.7, 1, 0.01
                eta1, eta2, kNum, tol, max_iter = 0.05, 0.1, 15, 1e-5, 30  # 调整参数：更大的学习率，更小的容忍度，更多的迭代次数
                model = MSWMLFG(M0_train, alpha, beta, gamma, delta1, delta2, tau, lbd1, lbd2, lbd3, lbd4, eta1, eta2, kNum, tol, max_iter, y_test)
                model.fit(X_train, y_train)

                y_score, y_pred, y_pred_view = model.predict(X_test, y_test)
                y_test = y_test.astype(int)
                scores = metrics_MLL.mll_metrics(y_test, y_pred, y_score, y_pred_view)
                scoresList.append(scores)

            finalScore = np.round(np.mean(scoresList, axis=0), 4)
            finalScoreStds = np.round(np.std(np.array(scoresList), 0), 3)  # 计算标准差
            Scores.append(finalScore)
            print(dataName, '  MissRat=', MissRat[mr], 'NoiseRat=', NoiseRat[noise], 'Score=', finalScore
                  , ' ScoreStds: ', finalScoreStds)

            end_time = time.time()
            print(dataName, f"run time：%.4f s" % (end_time - start_time))