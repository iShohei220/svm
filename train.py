import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SVM import SVM
np.random.seed(0)

def linear_data(N):
    X_1_pos = np.random.rand(N)
    X_2_pos = X_1_pos + np.random.normal(0, 0.3, N) + 0.5
    X_pos = np.array([[x_1, x_2, 1] for x_1, x_2 in zip(X_1_pos, X_2_pos)])
    X_1_neg = np.random.rand(N)
    X_2_neg = X_1_neg + np.random.normal(0, 0.3, N) - 0.5
    X_neg = np.array([[x_1, x_2, -1] for x_1, x_2 in zip(X_1_neg, X_2_neg)])
    X = np.vstack([X_pos, X_neg])
    np.random.shuffle(X)
    y = X[:, 2]
    X = X[:, :2]
    return X, y

def sin_data(N):
    X_1_pos = np.random.rand(N) * 4 * np.pi
    X_2_pos = np.sin(X_1_pos) + np.random.normal(0, 0.4, N)
    X_pos = np.array([[x_1, x_2, 1] for x_1, x_2 in zip(X_1_pos, X_2_pos)])
    X_1_neg = np.random.rand(N) * 4 * np.pi
    X_2_neg = np.sin(X_1_neg) + np.random.normal(0, 0.4, N) - 1.5
    X_neg = np.array([[x_1, x_2, -1] for x_1, x_2 in zip(X_1_neg, X_2_neg)])
    X = np.vstack([X_pos, X_neg])
    np.random.shuffle(X)
    y = X[:, 2]
    X = X[:, :2]
    return X, y

def train_test_split(X, y):
    X_train, X_test = np.split(X, [int(len(X) * 0.8)])
    y_train, y_test = np.split(y, [int(len(y) * 0.8)])
    return X_train, X_test, y_train, y_test

def show_data(X, y):
    X_pos = X[y==1]
    X_neg = X[y==-1]
    plt.plot(X_pos[:, 0], X_pos[:, 1], 'o', c='b')
    plt.plot(X_neg[:, 0], X_neg[:, 1], 'o', c='r')
    
def show_boader(model, X):
    X_border = np.linspace(min(X[:, 0]), max(X[:, 0]))
    y_border = -(model.w[0] * X_border + model.b) / model.w[1]
    plt.plot(X_border, y_border, c='y')
    plt.show()

# 正解率
def score(y, y_pred):
    true_idx = np.where(y_pred == 1)
    TP = np.sum(y_pred[true_idx] == y[true_idx])
    false_idx = np.where(y_pred == -1)
    TN = np.sum(y_pred[false_idx] == y[false_idx])
    return float(TP + TN) / len(y)
# 学習
def train(model, X_train, X_test, y_train, y_test):
    # 学習
    model.fit(X_train, y_train)
    # 予測値
    y_pred = model.predict(X_test)
    # 正解率
    acc = model.score(y_test, y_pred)
    print('正解率: %.3f' % acc)
    print('学習時間： %.3f' % model.elapsed_time)
    return acc

def main():
    print('線形データ')
    X, y = linear_data(500)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # 初期化
    model = SVM(kernel='rbf')
    # 学習
    acc = train(model, X_train, X_test, y_train, y_test)
    show_data(X_test, y_test)
    show_boader(model, X_train)
    
    print('実験2: 非線形データ')
    Ns = [50, 100, 500, 1000]
    print('実験2-1: 異なるカーネルでの実験')
    models = [SVM(kernel='rbf'), SVM(kernel='sigmoid'), SVM(kernel='linear')]
    kernels = ['RBF', 'Sigmoid', 'Linear']
    df_score = pd.DataFrame(index=Ns, columns=kernels)
    df_time = pd.DataFrame(index=Ns, columns=kernels)
    for N in Ns:
        print('データ数： %d' % N)
        X, y = sin_data(N)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        show_data(X, y)
        plt.show()
        for model, kernel in zip(models, kernels):
            print(kernel)
            acc = train(model, X_train, X_test, y_train, y_test)
            df_score.loc[N, kernel] = acc
            df_time.loc[N, kernel] = model.elapsed_time
    print(df_score)
    print(df_time)
    df_score.to_csv('カーネルごとの正解率')
    df_time.to_csv('カーネルごとの学習時間')
   
    print('実験2-2: 異なるパラメータでの実験')
    X, y = sin_data(500)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    show_data(X, y)
    plt.show()
    Cs = [2**a for a in range(-2, 3)]
    gammas = [2**a for a in range(-4, 2)]
    df_score = pd.DataFrame(index=Cs, columns=gammas)
    df_time = pd.DataFrame(index=Cs, columns=gammas)
    for C in Cs:
        for gamma in gammas:
            print('C: %.2f, gamma: %.4f' % (C, gamma))
            model = SVM(C=C, gamma=gamma)
            # 学習
            acc = train(model, X_train, X_test, y_train, y_test)
            df_score.loc[C, gamma] = acc
            df_time.loc[C, gamma] = model.elapsed_time
    print(df_score)
    print(df_time)
    df_score.to_csv('パラメータごとの正解率.csv')
    df_time.to_csv('パラメータごとの学習時間.csv')
            

if __name__ == '__main__':
        main()
