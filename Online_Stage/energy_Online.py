import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import os
import torch
import argparse

class Energy_naive(nn.Module):
    def __init__(self, input_size, output_size):
        super(Energy_naive, self).__init__()

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

class Nash_Filter:
    def __init__(self, L, N, K, A_k, B_k, const_V):
        self.L = L
        self.N = N
        self.K = K
        self.A_k = A_k
        self.B_k = B_k
        self.const_V = const_V

    def Estimate_ak_bk(self, P_list, X_list):
        """
        This method try estimate ak and bk og game
        by using data from exploration here we use the least square approach
        :param P_list: (T_exploration, L, K)
        :param X_list: (T_exploration, L, K)
        :return:
        """
        T_exploration, L, K = X_list.shape

        # Reshape X_list and P_list for batch processing
        X_flat = X_list.reshape(T_exploration, -1)  # Shape: (T_exploration, L * K)
        P_flat = P_list.reshape(T_exploration, -1)  # Shape: (T_exploration, L * K)

        # Build the design matrix A
        A = np.stack([(self.N ** 2) * (X_flat ** 3), self.N * (X_flat ** 2)], axis=2)  # Shape: (T_exploration, L*K, 2)

        # Transpose A for least squares, resulting in shape (L*K, T_exploration, 2)
        A = A.transpose(1, 0, 2)

        # Solve least squares for each (L*K) pair
        coefficients = np.array([
            np.linalg.lstsq(A[i], P_flat[:, i], rcond=None)[0]
            for i in range(A.shape[0])
        ])  # Shape: (L*K, 2)

        # Extract ak and bk, and reshape them to (L, K)
        ak = coefficients[:, 0].reshape(L, K)
        bk = coefficients[:, 1].reshape(L, K)

        return ak, bk


    def Exploration_process(self, K=24, explor_factor=2, gamma_n_k=7.35):
        """
        :param K: default 24 hours
        :param explor_factor: multiply by 2 need to be at least 2
        :param gamma_n_k: (int) limit the produce energy the player can put on resource k
        :return:
        """
        if explor_factor < 2: raise ValueError("Exploration factor must be greater or equal 2")
        domain_sample = np.arange(0.2, gamma_n_k, 0.1)
        if len(domain_sample) < 2*K: raise ValueError("more unknowns from domain sample please check delta (default 0.1)")
        T_explor = K * explor_factor
        x = np.random.choice(domain_sample, size=T_explor, replace=False)
        x = np.repeat(x[:, np.newaxis, np.newaxis], self.L, axis=1)
        x = np.repeat(x, K, axis=2)
        Sk = self.N * x
        P = self.A_k * x * (Sk ** 2) + self.B_k * Sk * x
        return P, x

    def Get_sk(self, Ak, Bk, Xnk, Pnk):
        # Form polynomial coefficients
        coeffs_a = Ak * Xnk  # Coefficient for x^2
        coeffs_b = Bk * Xnk  # Coefficient for x^1
        coeffs_c = -Pnk  # Coefficient for x^0
        # Solve poly problem
        delta = np.sqrt(coeffs_b ** 2 -4 * coeffs_a * coeffs_c)
        S1 = (-coeffs_b + delta) / (2 * coeffs_a + 1e-6)
        S2 = (-coeffs_b - delta) / (2 * coeffs_a + 1e-6)
        # Select the positive root
        S = np.where(S1 > 0, S1, S2)  # Choose positive value element-wise
        # Ensure shape is (L, N, K)
        return S


class Nash_Filter_NN(Nash_Filter):
    def __init__(self, L, N, K, A_k, B_k, const_V):
        super().__init__(L, N, K, A_k, B_k, const_V)

    def Load_NN_Model(self, path_weights, input_size=23, output_size=2):
        """
        Load Pytorch NN model
        :param path_weights:
        :param input_size:
        :param output_size:
        :return:
        """
        # file_path_weights = os.path.join("trains_record", "energy_Net", path_weights)
        file_path_weights = os.path.join("Weights", path_weights)
        self.model = Energy_naive(input_size, output_size)  # Initialize the model with the same architecture
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
        return predicted_values


    def Exploration_process(self, T_exploration=5, pathN="N=5_energey_game"):
        """
        The method build dataset for inference time
        :param T_exploration:
        :return: X_train (L, N, K, input_size)
        """
        low_bound = ((3 * T_exploration) / self.N) + 0.1
        Xn_k = np.random.uniform(low=low_bound, high=7.0, size=(L, N, K))
        X_train = []
        for t in range(-T_exploration, T_exploration+1, 1):
            # Calculate sum of action and duplicate for vectorization operation
            Sk = np.sum(Xn_k, axis=1)
            Sk = np.repeat(Sk[:, np.newaxis, :], N, axis=1)
            # Calculate reward
            P = A_k * Xn_k * (Sk ** 2) + B_k * Xn_k * Sk
            X_train.append(P)
            X_train.append(Xn_k)
            Xn_k = Xn_k + (1 / N) * t
        # Concatenate the list into a single array with shape (L, N, K, input_size - 1)
        X_train = np.stack(X_train, axis=-1)
        X_train = np.append(X_train, N * np.ones((self.L, self.N, self.K, 1)), axis=-1)
        # Normalize z-score
        X_mean = np.load(f"hyperparameters/{pathN}/X_mean.npy")
        X_std = np.load(f"hyperparameters/{pathN}/X_std.npy")
        epsilon = 0.01
        X_train = (X_train - X_mean) / (X_std + epsilon)

        return X_train


    def Estimate_ak_bk(self, X_train):
        """
        The method estimate ak and bk (L, K)
        :param X_train: (L, N, K, input_size)
        :return:
        """
        # Get the dimensions of the input
        L, N, K, input_size = X_train.shape

        # Reshape X_train to prepare it for batch processing by the neural network
        # New shape: (L * N * K, input_size)
        X_train_flat = X_train.reshape(-1, input_size)

        # Perform forward propagation through the neural network
        # NN output shape will be (L * N * K, 2), where 2 corresponds to [ak, bk]
        predictions = self.forward_pass(X_train_flat)

        # Split the predictions into ak and bk
        ak_flat, bk_flat = predictions[:, 0], predictions[:, 1]

        # Reshape ak and bk back to (L, N, K)
        ak = ak_flat.reshape(L, N, K)
        bk = bk_flat.reshape(L, N, K)

        return ak, bk





def main_loop_Nash_Filter(const_V, N, T, gamma_n_k, gamma_n, save_grad_debug, learning_rate,
              Xn_k, A_k, B_k,
              reward_list, grad_list, is_global=False, model_path="Energy_NetPath(N=5).pth", hyper_param_path=""):
    nash_filter = Nash_Filter_NN(L, N, K, A_k[:, 0, :], B_k[:, 0, :], const_V)
    nash_filter.Load_NN_Model(model_path, input_size=23, output_size=2)
    x_train = nash_filter.Exploration_process(5, pathN=hyper_param_path)
    Ak_est, Bk_est = nash_filter.Estimate_ak_bk(x_train)
    print(f"MSE: Ak_error = {np.sum((Ak_est - A_k) ** 2) / (L * N * K)}    Bk_error = {np.sum((Bk_est - B_k) ** 2) / (L * N * K)}")
    # Game loop
    for t in range(T):
        # Calculate P (pretend you can find Sk)
        Sk = np.sum(Xn_k, axis=1)
        Sk = np.repeat(Sk[:, np.newaxis, :], N, axis=1)
        # Calculate reward
        V = const_V * np.log(1 + Xn_k)
        P = A_k * Xn_k * (Sk ** 2) + B_k * Xn_k * Sk
        r_n = np.sum(V - P, axis=2)
        reward_list[t] = np.mean(r_n, axis=0)
        # Estimate Sk
        Sk_est = nash_filter.Get_sk(Ak_est, Bk_est, Xn_k, P)
        # Gradient ascent
        grad_Xnk_NE = calculate_NE_gradient(const_V, Xn_k, Ak_est, Bk_est, Sk_est)
        grad_Xnk_resudial = calculate_residual_gradient(Xn_k, Ak_est, Bk_est, Sk_est, N)
        total_grad = grad_Xnk_NE + grad_Xnk_resudial
        Xn_k = Xn_k + learning_rate[t] * total_grad
        # Project to action to the set:
        Xn_k = np.clip(Xn_k, a_min=0, a_max=gamma_n_k)
        Xn_k = project_onto_simplex(Xn_k, z=gamma_n)
    return Xn_k, reward_list



def project_onto_simplex(V, z=1):
    """
    Project matrix V onto the probability simplex if the sum exceeds z.
    :param V: numpy array, matrix of size (LxNxK)
    :param z: float, norm radius of the projected vector
    :return: numpy array, matrix of the same size as V, with vectors conditionally projected onto the probability simplex
    """
    L, N, K = V.shape
    # Sum along the last axis for each (L, N) slice
    sum_V = np.sum(V, axis=-1, keepdims=True)

    # Identify vectors that need projection
    mask = sum_V > z

    # Only sort and perform projection where sum exceeds z
    U = np.sort(V, axis=-1)[:, :, ::-1]
    cssv = np.cumsum(U, axis=-1) - z
    ind = np.arange(1, V.shape[-1] + 1)
    cond = (U - cssv / ind) > 0
    rho = np.count_nonzero(cond, axis=-1, keepdims=True)

    # Compute theta for projection where needed
    theta = np.take_along_axis(cssv, rho - 1, axis=-1) / rho
    projection = np.maximum(V - theta, 0)

    # Use mask to select between original V and projected values
    result = np.where(mask, projection, V)
    return result

def calculate_NE_gradient(const_V, Xn_k, A_k, B_k, Sk):
    """
    The function return vectorization gradient of the player's grad_rnk
    :param Xn_k: np array size (L, N, K)
    :param A_k: np array size (L, N, K)
    :param B_k: np array size (L, N, K)
    :param Sk: np array size (L, N, K)
    :return: grad_r_nk: np array size (L, N)
    """
    grad_Pk = A_k * (Sk ** 2 + 2 * Sk * Xn_k) + B_k * (Sk + Xn_k)
    grad_vk = const_V / (1 + Xn_k)
    grad_r_nk = grad_vk - grad_Pk
    return grad_r_nk

def calculate_residual_gradient(Xn_k, A_k, B_k, Sk, N=5):
    """
    This function calculate the residual gradient for global reward
    :param Xn_k: (L, N, K) np.array
    :param A_k: (L, N, K) np.array
    :param B_k: (L, N, K) np.array
    :param Sk: (L, N, K) np.array
    :return: residual gradient (L, N, K) np.array
    """
    temp = - (2 * A_k * Xn_k * Sk + B_k * Xn_k)
    grad_sum = np.sum(temp, axis=1)
    grad_sum = np.repeat(grad_sum[:, np.newaxis, :], N, axis=1)
    results = grad_sum - temp
    return results

def main_loop(const_V, N, T, gamma_n_k, gamma_n, save_grad_debug, learning_rate,
              Xn_k, A_k, B_k,
              reward_list, grad_list, is_global=False, ML_grad=False):
    """
    :param const_V: (int) constant parameter to control V reward
    :param N: (int) number of players
    :param T: (int) number of iteration
    :param gamma_n_k: (int)
    :param gamma_n: (int)
    :param save_grad_debug: (bool)
    :param learning_rate: (np.array) size (T, )
    :param Xn_k: (np.array) size (L, N, K)
    :param A_k: (np.array) size (L, N, K)
    :param B_k: (np.array) size (L, N, K)
    :param is_global: (bool)
    :param ML_grad: (np.array) (L, N, K)
    :return:
    """
    std_list = np.zeros_like(reward_list)
    grad_Xnk_global = np.zeros((L, N, K))
    for t in range(T):
        # Calculate sum of action and duplicate for vectorization operation
        Sk = np.sum(Xn_k, axis=1)
        Sk = np.repeat(Sk[:, np.newaxis, :], N, axis=1)
        # Calculate reward
        V = const_V * np.log(1 + Xn_k)
        P = A_k * Xn_k * (Sk ** 2) + B_k * Xn_k * Sk
        r_n = np.sum(V - P, axis=2)
        reward_list[t] = np.mean(r_n, axis=0)
        std_list[t] = np.std(r_n, axis=0)
        # Gradient ascent
        grad_Xnk_NE = calculate_NE_gradient(const_V, Xn_k, A_k, B_k, Sk)
        if is_global:
            grad_Xnk_global = calculate_residual_gradient(Xn_k, A_k, B_k, Sk, N)
        total_grad = grad_Xnk_NE + grad_Xnk_global
        Xn_k = Xn_k + learning_rate[t] * total_grad
        if save_grad_debug:
            temp = np.sum(grad_Xnk_global, axis=2)
            grad_list[t] = np.mean(temp, axis=0)
        # Project to action to the set:
        Xn_k = np.clip(Xn_k, a_min=0, a_max=gamma_n_k)
        Xn_k = project_onto_simplex(Xn_k, z=gamma_n)

    return Xn_k, reward_list, grad_list, std_list



def Parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Build Data set for Energy Game.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the NN model *.pth torch weights')
    parser.add_argument('--hyper_path', type=str, required=True, help='Path to the hyperparameter std and mean')
    parser.add_argument('--outputDir', type=str, default="output", help='output directory path')
    parser.add_argument('--N', type=int, default=5, help='batch size')
    parser.add_argument('--L', type=int, default=100, help='Number of games')
    parser.add_argument('--K', type=int, default=24, help='Sts the Intersection Over Union threshold')
    parser.add_argument('--dist', type=int, default=0, help='Sample distribution 0-uniform, 1 - exponential')
    parser.add_argument('--isValid', type=int, default=1, help='data for validation - 1 or training - 0')
    parser.add_argument('--T_exp', type=int, default=5, help='T_exploration, exploration process time take from -T to T so in general is twice size')
    parser.add_argument('--T_loss', type=int, default=200, help='T_loss, path loss samples time')
    return parser.parse_args()

if __name__ == '__main__':
    # Initialize constant parameters
    args = Parse_args()
    const_V = 25
    L = args.L
    K = args.K
    N = args.N
    T = 300

    # Game Parameters
    gamma_n_k = 7.35
    gamma_n = 15.55
    alpha = 1.5
    beta = 0.97
    dist = args.dist
    model_path = args.model_path

    if dist == 0:
        # Uniform Distribution
        dist = "uniform"

        # Sample ak and bk
        A_k = alpha * np.random.uniform(low=0.1, high=1.8, size=(L, K))
        B_k = beta * np.random.uniform(low=0.0, high=5.0, size=(L, K))

        # Duplicate ak and bk that will be same for each player
        A_k = np.repeat(A_k[:, np.newaxis, :], N, axis=1)
        B_k = np.repeat(B_k[:, np.newaxis, :], N, axis=1)

    elif dist == 1:
        # Exponential Distribution
        dist = "exponential"
        mean_target_a = (0.1 + 1.8) / 2  # ≈ 0.95
        lambda_a = 1.0 / mean_target_a
        mean_target_b = 2.5  # Roughly mid of [0, 5]
        lambda_b = 1.0 / mean_target_b

        A_k_raw = np.random.exponential(scale=1.0 / lambda_a, size=(L, K))
        A_k_clipped = np.clip(A_k_raw, 0.1, 1.8)
        A_k = alpha * A_k_clipped

        B_k_raw = np.random.exponential(scale=1.0 / lambda_b, size=(L, K))
        B_k_clipped = np.clip(B_k_raw, 0.0, 5.0)
        B_k = beta * B_k_clipped
        # Duplicate ak and bk that will be same for each player
        A_k = np.repeat(A_k[:, np.newaxis, :], N, axis=1)
        B_k = np.repeat(B_k[:, np.newaxis, :], N, axis=1)


    # learning_rate = 0.005 * np.reciprocal(np.power(range(1, T + 1), 0.6))
    learning_rate = 0.0005 * np.ones((T,))
    Xn_k = np.random.uniform(low=0.0, high=1.0, size=(L, N, K))

    # define X nash and x global
    X_NE = Xn_k.copy()
    X_global = Xn_k.copy()
    X_ml = Xn_k.copy()


    # Save results
    save_grad_debug = False
    reward_list_NE = np.zeros((T, N))
    grad_list_NE = np.zeros((T, N))
    reward_list_global = np.zeros((T, N))
    grad_list_global = np.zeros((T, N))
    reward_list_ml = np.zeros((T, N))
    grad_list_ml = np.zeros((T, N))

    # Read to main loop
    X_ml, reward_list_ml = main_loop_Nash_Filter(const_V, N, T, gamma_n_k, gamma_n, save_grad_debug, learning_rate,
                          X_ml, A_k, B_k,
                          reward_list_global, grad_list_global, is_global=False,
                                                 model_path=model_path, hyper_param_path=args.hyper_path)





    X_NE, reward_list_NE, grad_list_NE, std_ne = main_loop(const_V, N, T, gamma_n_k,
                                                   gamma_n, save_grad_debug, learning_rate,
                                                   X_NE, A_k, B_k,
                                                   reward_list_NE, grad_list_NE, is_global=False,
                                                   ML_grad=False)


    X_global, reward_list_global, grad_list_global, _ = main_loop(const_V, N, T, gamma_n_k,
                                                               gamma_n, save_grad_debug, learning_rate,
                                                               X_global, A_k, B_k,
                                                               reward_list_global, grad_list_global, is_global=True,
                                                             ML_grad=False)


    # Plot Section
    t = np.arange(T)

    total_NE_reward = np.sum(reward_list_NE, axis=1)
    total_NE_std = np.mean(std_ne, axis=1)

    total_global_reward = np.sum(reward_list_global, axis=1)

    total_ml_reward = np.sum(reward_list_ml, axis=1)

    plt.figure(1)
    plt.title(f'N={N}')
    plt.plot(t, total_NE_reward, color='b', label="$Nash$"), plt.xlabel("# Iteration"), plt.legend()
    # plt.fill_between(t, total_NE_reward - total_NE_std, total_NE_reward + total_NE_std, color='b', alpha=0.2)

    plt.plot(t, total_ml_reward, color='lightcoral', linestyle='-', label="$Ours$"), plt.xlabel("# Iteration"), plt.legend()
    plt.plot(t, total_global_reward, '--k', label="$Global$"), plt.xlabel("# Iteration"), plt.legend()
    plt.ylim(np.max(total_NE_reward)-200, np.max(total_global_reward)+100)

    plt.show()

    print("Finsh ..... ")