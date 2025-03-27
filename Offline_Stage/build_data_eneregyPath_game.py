import numpy as np


def BuildData_for_kalman(x_init, T_exploration, N=5, file_x="N=5_energey_game", file_y_sanity="N=5_energey_game"):
    Xn_k = x_init.copy()
    A_k = alpha * np.random.uniform(low=0.1, high=1.8, size=(L, K))
    B_k = beta * np.random.uniform(low=0.0, high=5.0, size=(L, K))

    # Duplicate ak and bk that will be same for each player
    A_k = np.repeat(A_k[:, np.newaxis, :], N, axis=1)
    B_k = np.repeat(B_k[:, np.newaxis, :], N, axis=1)
    X_train = []
    Y_train = [A_k[:, 0, 0].reshape(L, 1), B_k[:, 0, 0].reshape(L, 1)]
    x_debug = []
    for t in range(-T_exploration, T_exploration+1, 1):
        # Calculate sum of action and duplicate for vectorization operation
        Sk = np.sum(Xn_k, axis=1)
        Sk = np.repeat(Sk[:, np.newaxis, :], N, axis=1)
        # Calculate reward
        P = A_k * Xn_k * (Sk ** 2) + B_k * Xn_k * Sk
        X_train.append(P[:, 0, 0].reshape(L, 1))
        X_train.append(Xn_k[:, 0, 0].reshape(L, 1))
        Xn_k = Xn_k + (1 / N )* t

    X_train = np.concatenate(X_train, axis=-1)
    X_train = np.append(X_train, N * np.ones((L, 1)), axis=1)
    Y_train = np.concatenate(Y_train, axis=-1)
    np.save(file_x, X_train)
    np.save(file_y_sanity, Y_train)
    print("Succeed save first data to training.")
    return A_k, B_k

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

if __name__ == "__main__":
    # Initialize constant parameters
    const_V = 25
    K = 24
    N = 5
    T = 200
    alpha = 1.5
    beta = 0.97
    gamma_n_k = 7.35
    gamma_n = 15.55
    learning_rate = 0.0005 * np.ones((T,))
    isValid = True
    # Define Paths to save arrays
    if isValid:
        L = 3000
        state = "valid"
    else:
        L = 30000
        state = "train"

    pathN = "N=5_energey_game_check"
    file_x = f"Numpy_array_save/{pathN}/X_{state}.npy"
    file_y_sanity = f"Numpy_array_save/{pathN}/Y_{state}_sanity.npy"
    file_z = f"Numpy_array_save/{pathN}/Z_{state}.npy"
    file_y = f"Numpy_array_save/{pathN}/Y_{state}.npy"

    x = np.random.uniform(low=3.1, high=7.0, size=(L, N, K))
    A_k, B_k = BuildData_for_kalman(x_init=x, T_exploration=5, N=N, file_x=file_x, file_y_sanity=file_y_sanity)
    # Now collect for T data gradient points
    Z_train = np.zeros((L, 2*T))
    Y_train = np.zeros((L, T))
    Xn_k = x.copy()
    for t in range(T):
        # Calculate sum of action and duplicate for vectorization operation
        Sk = np.sum(Xn_k, axis=1)
        Sk = np.repeat(Sk[:, np.newaxis, :], N, axis=1)
        # Calculate reward
        V = const_V * np.log(1 + Xn_k)
        P = A_k * Xn_k * (Sk ** 2) + B_k * Xn_k * Sk
        # Calculate gradients
        grad_Xnk_NE = calculate_NE_gradient(const_V, Xn_k, A_k, B_k, Sk)
        grad_Xnk_global = calculate_residual_gradient(Xn_k, A_k, B_k, Sk, N)
        total_grad = grad_Xnk_NE + grad_Xnk_global
        # Save datapoints results in
        Z_train[:, 2 * t] = Xn_k[:, 0, 0]
        Z_train[:, 2 * t + 1] = Sk[:, 0, 0]
        Y_train[:, t] = grad_Xnk_global[:, 0, 0]
        # Gradients ascent
        Xn_k = Xn_k + learning_rate[t] * total_grad
        # Project to action to the set:
        Xn_k = np.clip(Xn_k, a_min=0, a_max=gamma_n_k)
        Xn_k = project_onto_simplex(Xn_k, z=gamma_n)

    np.save(file_z, Z_train)
    np.save(file_y, Y_train)
    print("Succeed save second data to training.")