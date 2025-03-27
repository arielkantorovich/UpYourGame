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

class Wireless_naive(nn.Module):
    def __init__(self, input_size, output_size):
        super(Wireless_naive, self).__init__()

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


class Wireless_poly(nn.Module):
    def __init__(self, input_size, output_size):
        """
        :param input_size: In general (2 * T_exploration) because [(Pn, In)_t=0, ...., (Pn, In)_t=T_explor]
        :param output_size: (2, ) because beta_n and alpha_n
        """
        super(Wireless_poly, self).__init__()
        self.fc0 = nn.Linear(input_size, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)

        # Initialize weights
        self.init_weights()

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
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


class WirelessKalman:
    def __init__(self, L, N, T_explor):
        self.L = L
        self.N = N
        self.T_explor = T_explor

    def Load_NN_model(self, folder_weight="N=5(Rlink=0.1)", path_weights="Wireless_Net_poly.pth", output_size=2):
        input_size = 2 * self.T_explor
        file_path_weights = os.path.join("Weights", folder_weight, path_weights)
        self.model = Wireless_poly(input_size, output_size)  # Initialize the model with the same architecture
        self.model.load_state_dict(torch.load(file_path_weights, map_location='cpu', weights_only=True))
        self.model.eval()  # Set the model to evaluation mode

    def forward_propg(self, X, output_size=2):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(X)
            predicted_values = outputs.numpy()
        return predicted_values.reshape(self.L, self.N, output_size)


def generate_symmetric_gain(N, L, alpha=1, R_link=0.1):
    """
    :param N: Number of players
    :param L: Trials to experiment
    :param alpha: scalar
    :param beta: scalar
    :return: Q, B
    """
    # Generate random transceiver locations in a radius of 1
    Transreceivers = np.random.rand(L, N, 2) * 2 - 1  # Scale to (-1, 1) and then to (-1, 1) radius 1
    # Generate random receiver locations in a radius of 0.1 around each transceiver
    Rlink = R_link #0.1/ 0.3
    ReceiverOffsets = Rlink * (np.random.rand(L, N, 2) * 2 - 1)  # Scale to (-1, 1) and then to (-0.1, 0.1)
    Receivers = Transreceivers + ReceiverOffsets
    # Calculate distances between transceivers and receivers
    distances = np.linalg.norm(Transreceivers[:, :, np.newaxis, :] - Receivers[:, np.newaxis, :, :], axis=3)
    g = alpha / (distances ** 2)
    return g

def calc_resudial_grad(grad_local, SNR, g_diag, g_zero):
    """
    This method calculate the prior gradient
    :param grad_local:
    :param SNR:
    :param g_diag:
    :param g_zero:
    :return:
    """
    L, N, N = g_zero.shape
    temp = grad_local * SNR * (1/g_diag)
    diag_mask = np.eye(N, dtype=bool)  # This creates a (N, N) mask with True on the diagonal
    g_zero_copy = g_zero.copy()
    g_zero_copy[:, diag_mask] = 0
    results = -1.0 * np.matmul(g_zero_copy, temp)
    return results

def Adding_Noise(g, sigma, C, K):
    """
    This method adding noise to gain matrix
    :param g: (L, N, N)
    :param sigma: scalar - standard deviation of the gaussian noise
    :param C:  scalar - bounded uniform noise in column
    :param K: scalar - number of bandit feedback
    :return: g_noise (L, N, K, N, N) noise tensor where the colum is uniform noise
    """
    L, N, _ = g.shape
    # Step 1: Expand dimensions to prepare for repetition Shape becomes (L, 1, 1, N, N)
    g_expanded = g[:, np.newaxis, np.newaxis, :, :]
    # Step 2: Repeat along the player (N) and repetition (K) axes Shape becomes (L, N, 1, N, N) then (L, N, K, N, N)
    g_repeated = np.repeat(g_expanded, N, axis=1)
    g_repeated = np.repeat(g_repeated, K, axis=2)
    # Generate Noise
    Gamma_G = sigma * np.random.randn(L, N, K, N, N)
    # each row noise is zero
    for n in range(N):
        Gamma_G[:, n, :, :, n] = np.random.uniform(low=-C, high=C, size=(N, )) # colum noise bounded in C
        Gamma_G[:, n, :, n, :] = 0  # row noise zero
    G_Noise = g_repeated + Gamma_G
    return G_Noise


def main_loop(P, N, L, T, g, lr, alpha_nn, beta_nn, gamma_nn, isGlobal=True):
    # Initialize Parameters
    Border_floor = 0
    Border_ceil = 1
    N0 = 0.001

    # Initialize record variables
    P_record = np.zeros((T, L, N, 1))
    global_objective = np.zeros((T, L))
    gradient_resuidal = np.zeros((L, N, 1))
    grad_record = np.zeros((T, L, N, 1))
    grad_record_local = np.zeros((T, L, N, 1))
    # Prepare g to calculation
    g_diag = np.diagonal(g, axis1=1, axis2=2).copy()
    g_diag = g_diag.reshape(L, N, 1)
    g_colum = np.transpose(g, axes=(0, 2, 1))
    for t in range(T):
        # calculate instance
        In = np.matmul(g_colum, P) - g_diag * P
        # calculate gradients
        numerator = (g_diag / (In + N0))
        SNR = numerator * P
        gradients_local = (numerator / (1 + SNR))

        if isGlobal:
            gradient_resuidal = calc_resudial_grad(gradients_local, SNR, g_diag, g)

        grad_record[t] = gradient_resuidal
        grad_record_local[t] = gradients_local
        # Adding ours prior
        grad_NN = alpha_nn * P + beta_nn * In + gamma_nn

        # Update action vector
        P = P + lr[t] * (gradients_local + gradient_resuidal + grad_NN)

        # Project the action to [Border_floor, Border_ceil] (Normalization)
        P = np.minimum(np.maximum(P, Border_floor), Border_ceil)

        # Save results in record
        P_record[t] = P

        # Calculate global objective
        temp = np.log(1 + numerator * P)
        temp = temp.squeeze()
        global_objective[t] = np.sum(temp, axis=1)
    # Finally Let's mean for all L trials
    P_record = P_record.squeeze()
    grad_record = grad_record.squeeze()
    grad_record_local = grad_record_local.squeeze()
    mean_P_record = np.mean(P_record, axis=1)
    mean_grad_record = np.mean(grad_record, axis=1)
    mean_global_objective = np.mean(global_objective, axis=1)
    mean_grad_record_local = np.mean(grad_record_local, axis=1)
    return  mean_P_record, mean_global_objective, mean_grad_record, mean_grad_record_local


def build_Data_Randomize(N, L, T_exploration, g):
    """This method build data for NN randomly each turn pick P and play prior"""
    # Initialize Noise Parameters
    N0 = 0.001

    # Prepare g to calculation
    g_diag = np.diagonal(g, axis1=1, axis2=2).copy()
    g_diag = g_diag.reshape(L, N, 1)
    g_colum = np.transpose(g, axes=(0, 2, 1))

    # Initialize Train list
    X_train = []
    Y_train = []
    for t in range(T_exploration):
        P = np.random.rand(L, N, 1)  # Generate Power from uniform distributed
        # calculate instance
        In = np.matmul(g_colum, P) - g_diag * P

        # calculate gradients
        numerator = (g_diag / (In + N0))
        SNR = numerator * P
        gradients_local = (numerator / (1 + SNR))

        gradient_resuidal = calc_resudial_grad(gradients_local, SNR, g_diag, g)

        # Build Train
        X_train.append(np.concatenate([P, In], axis=-1))
        Y_train.append(gradient_resuidal)

    X_train = np.concatenate(X_train, axis=-1) # size (L, N, 2 * T_exploration) where 2 is number of features In and Pn
    X_train = X_train.reshape(L * N, 2*T_exploration)
    Y_train = np.concatenate(Y_train, axis=-1) # size (L, N, T_exploration)
    Y_train = Y_train.reshape(L * N, T_exploration)
    return X_train, Y_train

def build_DataToTrain(P, N, L, T_exploration, g, lr, isGlobal=False):
    """This function build data for poly prior using
    play gradient for T_exploration = 50 in our case"""
    # Initialize Parameters
    Border_floor = 0
    Border_ceil = 1
    N0 = 0.001

    # Initialize record variables
    P_record = np.zeros((T, L, N, 1))

    # Prepare g to calculation
    g_diag = np.diagonal(g, axis1=1, axis2=2).copy()
    g_diag = g_diag.reshape(L, N, 1)
    g_colum = np.transpose(g, axes=(0, 2, 1))

    # Initialize Train list
    X_train = []
    Y_train = []
    for t in range(T_exploration):
        # calculate instance
        In = np.matmul(g_colum, P) - g_diag * P

        # calculate gradients
        numerator = (g_diag / (In + N0))
        SNR = numerator * P
        gradients_local = (numerator / (1 + SNR))

        gradient_resuidal = calc_resudial_grad(gradients_local, SNR, g_diag, g)

        # Build Train
        X_train.append(np.concatenate([P, In], axis=-1))
        Y_train.append(gradient_resuidal)

        if isGlobal:
            P = P + lr[t] * (gradients_local + gradient_resuidal)
        else:
            # Update action vector
            P = P + lr[t] * gradients_local

        # Project the action to [Border_floor, Border_ceil] (Normalization)
        P = np.minimum(np.maximum(P, Border_floor), Border_ceil)

        # Save results in record
        P_record[t] = P

    X_train = np.concatenate(X_train, axis=-1) # size (L, N, 2 * T_exploration) where 2 is number of features In and Pn
    X_train = X_train.reshape(L * N, 2*T_exploration)
    Y_train = np.concatenate(Y_train, axis=-1) # size (L, N, T_exploration)
    Y_train = Y_train.reshape(L * N, T_exploration)
    return X_train, Y_train



if __name__ == '__main__':
    # Define Scalar Parameters
    N = 80
    alpha = 10e-3 # 10e-3 for N=5.
    T = 2000
    T_exp = 100 #50 / 100
    R_link = 0.1
    isValid = False
    isTest = True

    # learning_rate = 0.06 * np.reciprocal(np.power(range(1, T + 1), 0.9))
    learning_rate = 0.0098/4 * np.ones((T, ))
    # learning_rate[int(T/2):] = 0.0098/4
    learning_rate_2 = 0.009 * np.ones((T,))

    if isValid:
        L = 2000
        file_x_pre = 'Numpy_array_save/N=5_wireless_poly(Rlink=0.3)/x_valid_pre.npy'
        file_x = 'Numpy_array_save/N=5_wireless_poly(Rlink=0.3)/x_valid.npy'
        file_y = 'Numpy_array_save/N=5_wireless_poly(Rlink=0.3)/y_valid.npy'

    else:
        L = 15000
        file_x_pre = 'Numpy_array_save/N=5_wireless_poly(Rlink=0.3)/x_train_pre.npy'
        file_x = 'Numpy_array_save/N=5_wireless_poly(Rlink=0.3)/x_train.npy'
        file_y = 'Numpy_array_save/N=5_wireless_poly(Rlink=0.3)/y_train.npy'

    if isTest:
        L = 200

    # Define gain matrices
    gain = generate_symmetric_gain(N, L, alpha, R_link)

    P_init = np.random.rand(L, N, 1)
    # Build Data for training alpha and beta
    X_data_pre, _ = build_DataToTrain(P_init, N, L, T_exp, gain, learning_rate_2)

    if not isTest:
        X_data, Y_data = build_DataToTrain(P_init, N, L, T_exp, gain, learning_rate_2, True)
        np.save(file_x_pre, X_data_pre)
        np.save(file_x, X_data)
        np.save(file_y, Y_data)
        print("Finsh Generate Wireless training Data, isValid = ", isValid)
        exit(-1)

    # Use Training Network
    nash_filter = WirelessKalman(L, N, T_exp)
    nash_filter.Load_NN_model("N=80(Rlink=0.1)", 'Wireless_Net_poly.pth', output_size=2)
    output = nash_filter.forward_propg(X_data_pre, 2)
    alpha_ne, beta_ne = output[:, :, 0].reshape(L, N, 1), output[:, :, 1].reshape(L, N, 1)

    # Use Training Network(projection)
    nash_filter = WirelessKalman(L, N, T_exp)
    nash_filter.Load_NN_model("N=80(Rlink=0.1)", 'Wireless_Net_poly(alpha).pth', output_size=1)
    output = nash_filter.forward_propg(X_data_pre, output_size=1)
    alpha_p = output[:, :, 0].reshape(L, N, 1)

    # nash_filter = WirelessKalman(L, N, T_exp)
    # nash_filter.Load_NN_model('Wireless_Net_poly(gamma).pth', output_size=3)
    # output = nash_filter.forward_propg(X_data, 3)
    # alpha_gam, beta_gam, gamma_gam = output[:, :, 0].reshape(L, N, 1), output[:, :, 1].reshape(L, N, 1), output[:, :, 1].reshape(L, N, 1)


    # Get optimal and estimate results
    P_opt, global_opt, grad_opt, grad_loc = main_loop(P_init, N, L, T, gain, learning_rate_2, 0, 0, 0,True)
    P_ne, global_ne, _, grad_ne = main_loop(P_init, N, L, T, gain, learning_rate_2, 0, 0, 0,False)
    P_ours, global_ours, _, _ = main_loop(P_init, N, L, T, gain, learning_rate, alpha_ne, beta_ne, 0, False)
    P_ours_p, global_ours_p, _, _ = main_loop(P_init, N, L, T, gain, learning_rate, alpha_p, 0, 0, False)
    # P_ours_gam, global_ours_gam, _, _ = main_loop(P_init, N, L, T, gain, learning_rate, alpha_gam, beta_gam, gamma_gam, False)

    # Finally plot results
    t = np.arange(0, T, 1)

    # plt.figure(1)
    # plt.subplot(121)
    # for n in range(N):
    #     plt.plot(t, P_opt[:, n], label=f"P{n} opt"), plt.xlabel("# Iteration"),
    #     plt.legend()
    # plt.subplot(122)
    # for n in range(N):
    #     plt.plot(t, P_ne[:, n], label=f"P{n} NE"), plt.xlabel("# Iteration"),
    #     plt.legend()
    # plt.subplot(223)
    # for n in range(N):
    #     plt.plot(t, P_ours[:, n], label=f"P{n} alpha + beta"), plt.xlabel("# Iteration"),
    #     plt.legend()
    # plt.subplot(224)
    # for n in range(N):
    #     plt.plot(t, P_ours_p[:, n], label=f"P{n} alpha"), plt.xlabel("# Iteration"),
    #     plt.legend()


    plt.figure(2)
    plt.title(f'N={N}, Rlink={R_link}')
    plt.plot(t, global_opt, label="global"), plt.xlabel("# Iteration")
    plt.plot(t, global_ne, label="NE"), plt.xlabel("# Iteration")
    plt.plot(t, global_ours, label=r"$(\alpha_{n}, \beta_{n})$")
    plt.plot(t, global_ours_p, label=r"$\alpha_{n}$")
    # plt.plot(t, global_ours_gam, label="alpha+beta+gamma"), plt.xlabel("# Iteration")
    plt.legend()

    # plt.figure(3)
    # plt.subplot(1, 2, 1)
    # for n in range(N):
    #     plt.plot(t, grad_opt[:, n], label=f"Grad-residual n={n}"), plt.xlabel("# Iteration"), plt.legend()
    # plt.subplot(1, 2, 2)
    # for n in range(N):
    #     plt.plot(t, grad_loc[:, n], label=f"Grad-local n={n}"), plt.xlabel("# Iteration"), plt.legend()
    #
    # plt.figure(4)
    # for n in range(N):
    #     plt.plot(t, grad_ne[:, n], label=f"Grad-Ne n={n}"), plt.xlabel("# Iteration"), plt.legend()

    plt.show()
