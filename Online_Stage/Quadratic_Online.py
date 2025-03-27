# -*- coding: utf-8 -*-
"""
Created on : ------

@author: Ariel_Kantorovich
"""
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import os
import torch


class Quadratic_NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(Quadratic_NN, self).__init__()

        # Define layers
        self.fc0 = nn.Linear(input_size, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_size)

        # Initialize weights
        self.init_weights()

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)



class Quadratic_Nash_Filter:
    def __init__(self, L, N, Q, B):
        self.L = L
        self.N = N
        self.Q = Q
        self.B = B

    def Load_NN_Model(self, path_weights="Quadratic_Net(N=5).pth", input_size=202, output_size=1):
        """
        Load Pytorch NN model
        :param path_weights:
        :param input_size:
        :param output_size:
        :return:
        """
        file_path_weights = os.path.join("Weights", path_weights)
        self.model = Quadratic_NN(input_size, output_size)  # Initialize the model with the same architecture
        self.model.load_state_dict(torch.load(file_path_weights, map_location='cpu', weights_only=True))
        self.model.eval()  # Set the model to evaluation mode

    def forward_pass(self, X):
        """
        :param X:
        :return: ak, bk prediction
        """
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(X)
            predicted_values = outputs.numpy()
        return predicted_values.reshape(self.L, self.N, 1)

    def Exploration_process(self, T_exploration=100, path_N="N=5_quadratic_game"):
        """
        :param T_exploration:
        :return:
        """
        X_train = []
        # Generate Random x
        x = np.random.uniform(low=0.1, high=2.0, size=(L, N, 1))
        Q_diag = np.diagonal(self.Q, axis1=1, axis2=2)
        Q_diag = np.expand_dims(Q_diag, axis=-1)
        for t in range(T_exploration):
            x = x + 1 / N * t
            cost = (0.5 * (2 * np.matmul(self.Q, x) - Q_diag * (x ** 2)) + self.B * x)
            X_train.append(x)
            X_train.append(cost)
        X_train.append(self.B)
        X_train = np.concatenate(X_train, axis=-1)
        X_train = np.append(X_train, self.N * np.ones((L, N, 1)), axis=-1)
        X_train = X_train.reshape(self.L * self.N, -1)
        # Normalize z-score
        X_mean = np.load(f"hyperparameters/{path_N}/X_mean.npy")
        X_std = np.load(f"hyperparameters/{path_N}/X_std.npy")
        epsilon = 0.01
        X_train = (X_train - X_mean) / (X_std + epsilon)
        return X_train

    def __Extract_innerProduct(self, Cn, xn, qnn):
        """
        protected method
        This method try to estimate inner product between <qn, x>
        :param Cn: cost of player n
        :param xn: action of player n
        :param bn: bias of player n
        :param qnn: diagonal that we estimate from NN
        :return: <qn, x> to each player
        """
        inner = (2 * (Cn - self.B * xn) + qnn * (xn ** 2)) / (2 * xn)
        return inner

    def main_loop(self, diagonals_est, T, x, lr):
        """
        main loop of our technique
        :return:
        """
        cost_record = np.zeros((L, T, N))
        diagonals= np.diagonal(self.Q, axis1=1, axis2=2)
        diagonals = np.expand_dims(diagonals, axis=-1)
        for t in range(T):
            cost = (0.5 * (2 * x * np.matmul(self.Q, x) - diagonals * (x ** 2)) + self.B * x)
            cost_record[:, t, :] = cost.squeeze()
            quadratic_product = self.__Extract_innerProduct(cost, x, diagonals)
            nash_gradient = quadratic_product + self.B
            resuidal_gradient = quadratic_product - x * diagonals_est
            total_grad = nash_gradient + resuidal_gradient
            x = x - lr[t] * total_grad
            x = np.clip(x, a_min=gamma_lower, a_max=gamma_upper)
        # Calculate mean cost (L, T, N)
        sum_cost = np.sum(cost_record, axis=2)
        mean_cost = np.mean(sum_cost, axis=0)
        # mean_cost = np.sum(mean_cost, axis=1)
        return mean_cost, sum_cost


def generate_Q_B(N, L, alpha=1, beta=1, low=1, high=2):
    """
    :param N: Number of players
    :param L: Trials to experiment
    :param alpha: scalar
    :param beta: scalar
    :return: Q, B
    """
    # Step 1: Generate L different sets of N random points
    points = np.random.rand(L, N, 2)

    # Step 2: Build distance maps D for all trials
    points_expanded = points[:, np.newaxis, :, :]
    differences = points_expanded - points[:, :, np.newaxis, :]
    distances = np.linalg.norm(differences, axis=-1)

    # Step 3: Generate Q = exp(-alpha*D) for all trials
    Q = np.exp(-alpha * distances)

    # Step 4: Randomly sample diagonal values and update Q diagonals
    random_diagonals = np.random.uniform(low, high, size=(L, N))  # Generate random values for each diagonal
    for i in range(L):  # Update each trial's diagonal individually
        np.fill_diagonal(Q[i], random_diagonals[i])

    # Step 5: Generate B
    B = np.random.uniform(low=-beta, high=beta, size=(L, N, 1))
    return Q, B

def main_loop(L, N, Q, B, T, x, lr, is_global = False):
    """
    :param L: (int) Number of games
    :param N: (int) Number of players
    :param Q: (np.array LxNxN)
    :param B: (np.array LxNx1)
    :param T: (int) number of iteration
    :param x: (np.array LxNx1)
    :param lr: (float/np.array) learning rate
    :param is_global: (bool) consider residual gradient if True
    :return:
    """
    # Initialize parameters
    cost_record = np.zeros((L, T, N))
    grad_record = np.zeros((L, T, N))
    diagonals = np.diagonal(Q, axis1=1, axis2=2)
    diagonals = np.expand_dims(diagonals, axis=-1)
    residual_gradient = 0
    for t in range(T):
        cost_record[:, t, :] = (0.5 * (2 * x * np.matmul(Q, x) - diagonals * (x ** 2)) + B * x).squeeze()
        local_gradient = np.matmul(Q, x) + B
        if is_global:
            residual_gradient = np.matmul(Q, x) - diagonals * x
        total_grad = local_gradient + residual_gradient
        grad_record[:, t, :] = total_grad.squeeze()
        x = x - lr[t] * total_grad
        x = np.clip(x, a_min=gamma_lower, a_max=gamma_upper)
    # Calculate mean cost
    sum_cost = np.sum(cost_record, axis=2)
    mean_cost = np.mean(sum_cost, axis=0)
    mean_grad = np.mean(grad_record, axis=0)
    return mean_cost, mean_grad, x, sum_cost




if __name__ == '__main__':
    # Define Global Var
    L = 800 # Trials each point
    N = 5 # Number og agents
    alpha = 5.0
    beta = 1
    gamma_lower = -20.0
    gamma_upper = 20.0
    T = 1100
    learning_rate = 0.006 * np.ones((T, ))
    learning_rate_2 = 0.09 * np.ones((T, )) # 0.06
    # learning_rate = 0.09 * np.reciprocal(np.power(range(1, T + 1), 0.6))
    DEBUG_FLAG = False

    # Calculate Q and B
    Q_game, B_game = generate_Q_B(N, L, alpha, beta, low=1.2, high=2.2)

    # Generate Random x
    x_init = np.random.uniform(0.1, 1.1, size=(L, N, 1))
    x_ne = x_init.copy()
    x_global = x_init.copy()
    x_ours = x_init.copy()

    # Quadratic Nash filter (Our method)
    nash_filter = Quadratic_Nash_Filter(L, N, Q_game, B_game)
    nash_filter.Load_NN_Model("Quadratic_Net(N=5).pth")
    X_train = nash_filter.Exploration_process(100, "N=5_quadratic_game")
    diag_est = nash_filter.forward_pass(X_train)
    ml_cost, sum_cost_ml = nash_filter.main_loop(diag_est, T, x_ours, learning_rate_2)

    # Run Main Loop
    ne_cost, grad_ne, x_ne, sum_cost_ne = main_loop(L, N, Q_game, B_game, T, x_ne, learning_rate_2, False)
    global_cost, grad_global, x_global, sum_cost_global = main_loop(L, N, Q_game, B_game, T, x_global, learning_rate_2, True)

    # sum cost is in size (L, T)
    sum_cost_global = sum_cost_global[:, -1].reshape(-1, 1)
    ne_ratio = np.abs(sum_cost_ne - sum_cost_global) / np.abs(sum_cost_global)
    std_ne = np.std(ne_ratio, axis=0)
    mean_ne_ratio = np.mean(ne_ratio, axis=0)
    ml_ratio = np.abs(sum_cost_ml - sum_cost_global) / np.abs(sum_cost_global)
    std_ml = np.std(ml_ratio, axis=0)
    mean_ml_ratio = np.mean(ml_ratio, axis=0)

    # Plot results
    t = np.arange(T)
    plt.figure(1)
    plt.title(f"N={N}")
    plt.plot(t, ne_cost, color='b',label='Nash'), plt.legend(), plt.xlabel("# Iteration")
    plt.plot(t, ml_cost, color='r',label='Ours'), plt.legend(), plt.xlabel("# Iteration")
    plt.plot(t, global_cost, "--k", label='Global'), plt.legend(), plt.xlabel("Iteration"), plt.ylabel('Cost')

    plt.figure(2)
    plt.title(f"N={N}")
    plt.plot(t, mean_ne_ratio, color='b', label='Nash'), plt.legend(), plt.xlabel("# Iteration")
    plt.fill_between(t, mean_ne_ratio - std_ne, mean_ne_ratio + std_ne, color='b', alpha=0.2)
    plt.plot(t, mean_ml_ratio, color='r', label='Ours'), plt.legend(), plt.xlabel("# Iteration")
    plt.fill_between(t, mean_ml_ratio - std_ml, mean_ml_ratio + std_ml, color='r', alpha=0.2)
    plt.ylim(-2, 1.2)
    plt.xlim(0, T)
    if DEBUG_FLAG:
        Q_factor = Q_game.copy()
        Q_factor[:, range(N), range(N)] *= 0.5
        Q_inv = np.linalg.inv(Q_factor)
        B_trans = B_game.transpose(0, 2, 1)  # Shape: (L, 1, N)
        x_opt = -0.5 * np.matmul(Q_inv, B_game)
        x_opt = np.clip(x_opt, a_min=gamma_lower, a_max=gamma_upper)
        C_debug = np.matmul(x_opt.transpose(0, 2, 1), np.matmul(Q_factor, x_opt)) + np.matmul(B_trans, x_opt)
        C_debug = np.mean(C_debug, axis=0)
        plt.plot(t, C_debug[0] * np.ones((T, )), "--b", label='Debug'), plt.legend(), plt.xlabel("# Iteration")

    # plt.figure(2)
    # plt.subplot(1,2,1)
    # for n in range(N):
    #     plt.plot(t, grad_ne[:, n], label=f"n={n}"), plt.legend(), plt.xlabel("# Iteration"), plt.ylabel('Grad Nash')
    # plt.subplot(1,2,2)
    # for n in range(N):
    #     plt.plot(t, grad_global[:, n], label=f'n={n}'), plt.legend(), plt.xlabel("# Iteration"), plt.ylabel("Grad Global")
    plt.show()



    print("Finsh .... ")

