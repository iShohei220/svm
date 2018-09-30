import numpy as np
import time

class SVM():
    def __init__(self, max_iter=10000, kernel='rbf', C=1.0, gamma=0.001):
        self.max_iter = max_iter
        if kernel == 'linear':
            self.kernel = self.linear_kernel
        elif kernel == 'sigmoid':
            self.kernel = self.sigmoid_kernel
        else:
            self.kernel = self.rbf_kernel
        self.kernel = self.sigmoid_kernel if kernel=='sigmoid' else self.rbf_kernel
        self.C = C
        self.gamma = gamma
    def fit(self, X, y):
        t1 = time.time()
        # 初期化
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros((n))
        kernel = self.kernel
        count = 0
        while count < self.max_iter:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, n):
                i = j
                while i == j:
                    i = np.random.randint(0, n)                
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                k_ij = kernel(x_i, x_i, self.gamma) + kernel(x_j, x_j, self.gamma) - 2 * kernel(x_i, x_j, self.gamma)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # パラメータの更新
                self.w = np.dot(alpha * y, X)
                self.b = np.mean(y - np.dot(self.w.T, X.T))

                # 予測誤差
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # alphaの更新
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)
                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            # 収束判定
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < 0.001:
                break
                
        # パラメータの決定
        self.b = np.mean(y - np.dot(self.w.T, X.T))
        if self.kernel == self.linear_kernel:
            self.w = np.dot(alpha * y, X)
        # 処理時間
        t2 = time.time()
        self.elapsed_time = t2 - t1
    def predict(self, X):
        return self.h(X, self.w, self.b)
    # 予測
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)
    # 予測誤差
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k
    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))
        
    # カーネルを定義
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2.T)
    def sigmoid_kernel(self, x1, x2, gamma):
        return 1 / (1 + np.exp(-gamma * np.dot(x1, x2.T)))
    def rbf_kernel(self, x1, x2, gamma):
        return (np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2))
    
    # 正解率
    def score(self, y, y_pred):
        true_idx = np.where(y_pred == 1)
        TP = np.sum(y_pred[true_idx] == y[true_idx])
        false_idx = np.where(y_pred == -1)
        TN = np.sum(y_pred[false_idx] == y[false_idx])
        return float(TP + TN) / len(y)
