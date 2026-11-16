import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """从零实现一个二分类神经网络（手写版）"""

    def __init__(self, layers, learning_rate=0.05, iterations=2000):
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.params = {}
        self.loss_history = []

    def init_weights(self):
        """随机初始化参数"""
        rng = np.random.default_rng(1)
        self.params["W1"] = rng.normal(0, 1, size=(self.layers[0], self.layers[1]))
        self.params["b1"] = np.zeros((1, self.layers[1]))
        self.params["W2"] = rng.normal(0, 1, size=(self.layers[1], self.layers[2]))
        self.params["b2"] = np.zeros((1, self.layers[2]))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_grad(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y_true, y_pred):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def forward_propagation(self, X, y):
        """前向传播：输入 -> 隐藏层 -> 输出层 -> 计算损失"""
        Z1 = X @ self.params["W1"] + self.params["b1"]
        A1 = self.relu(Z1)
        Z2 = A1 @ self.params["W2"] + self.params["b2"]
        y_hat = self.sigmoid(Z2)
        loss = self.binary_cross_entropy(y, y_hat)

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "y_hat": y_hat,
        }
        return y_hat, loss, cache

    def backward_propagation(self, y, y_hat, cache):
        """反向传播：计算梯度，并更新参数"""
        X = cache["X"]
        A1 = cache["A1"]
        Z1 = cache["Z1"]

        m = X.shape[0]

        # 二分类交叉熵 + sigmoid 的导数可以简化为：dZ2 = (y_hat - y) / m
        dZ2 = (y_hat - y) / m
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.params["W2"].T
        dZ1 = dA1 * self.relu_grad(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # 更新参数（梯度下降）
        self.params["W1"] -= self.learning_rate * dW1
        self.params["b1"] -= self.learning_rate * db1
        self.params["W2"] -= self.learning_rate * dW2
        self.params["b2"] -= self.learning_rate * db2

    def fit(self, X, y):
        """训练模型"""
        self.init_weights()
        for _ in range(self.iterations):
            y_hat, loss, cache = self.forward_propagation(X, y)
            self.backward_propagation(y, y_hat, cache)
            self.loss_history.append(loss)

    def predict_proba(self, X):
        Z1 = X @ self.params["W1"] + self.params["b1"]
        A1 = self.relu(Z1)
        Z2 = A1 @ self.params["W2"] + self.params["b2"]
        return self.sigmoid(Z2)

    def predict(self, X, threshold=0.5):
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Loss Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()


# ------------------------------------------------------------------
# 生成一个简单的二分类数据集
# ------------------------------------------------------------------

def make_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))

    # 通过一个简单的规则来构造标签
    # 如果 x1 + 0.5*x2 > 0 表示类别 1，否则类别 0
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float).reshape(-1, 1)

    # 归一化特征，帮助训练更稳定
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


# ------------------------------------------------------------------
# 主程序
# ------------------------------------------------------------------

def main():
    print("开始训练手写神经网络...")

    X, y = make_data()
    model = NeuralNetwork(layers=[2, 8, 1], learning_rate=0.05, iterations=2000)
    model.fit(X, y)

    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    print("训练准确率:", accuracy)
    print("最后一轮损失:", model.loss_history[-1])

    # 可视化训练损失变化
    model.plot_loss()

    # 预测示例
    test_points = np.array([[1.0, 1.0], [-1.0, -1.0], [2.0, -1.0]])
    probs = model.predict_proba(test_points)
    print("测试样本预测概率:")
    print(probs)
    print("测试样本类别:")
    print(model.predict(test_points))


if __name__ == "__main__":
    main()


# 说明：
# 1. 这段代码是文章“从零实现神经网络”的一个简化版。
# 2. 它包含了：输入层、隐层、输出层、权重、偏置、ReLU、Sigmoid、交叉熵损失、
#    前向传播、反向传播、梯度下降。
# 3. 这就是神经网络学习的核心思路：先猜测，再计算误差，再修正参数，重复训练。
# 4. 如果你要做 “真实数据集版本”，就需要把数据预处理、训练/测试集划分、
#    以及损失函数/分类阈值调成符合实际任务。