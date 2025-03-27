import numpy as np

def generate_Q_B(N, L, alpha=1, beta=1, low=1, high=2):
    """
    The following method Generate Q PDS and B - bias for quadratic games
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
    # Expand dimensions to Q diagonal
    random_diagonals = np.expand_dims(random_diagonals, axis=-1)
    return Q, B, random_diagonals


def Generate_Train_data(T, x_init_2, is_global=False, StepJump=1):
    # Initialize Parameters
    lr = 0.09 * np.ones((T,))  # 0.06
    x = x_init_2.copy()
    X_train = np.zeros((L, N, 2* (T//StepJump) + 2))
    Y_train = np.zeros((L, N, T//StepJump))
    residual_gradient = np.zeros((L, N, 1))

    # Iteration Step:
    for t in range(T):
        cost = (0.5 * (2 * np.matmul(Q_game, x) * x - Q_diag * (x ** 2)) + B_game * x)
        local_gradient = np.matmul(Q_game, x) + B_game
        if is_global:
            residual_gradient = np.matmul(Q_game, x) - Q_diag * x
        total_grad = local_gradient + residual_gradient
        if t % StepJump == 0:
            t_jump = t // StepJump
            # Store x and cost in X_train and Y _train
            X_train[:, :, t_jump * 2] = x[:, :, 0]  # Store x
            X_train[:, :, t_jump * 2 + 1] = cost[:, :, 0]  # Store cost
            Y_train[:, :, t_jump] = residual_gradient[:, :, 0]
        x = x - lr[t] * total_grad
        x = np.clip(x, a_min=gamma_lower, a_max=gamma_upper)

    # Add B and N as the final two features
    X_train[:, :, -2] = B_game[:, :, 0]  # Add B_game
    X_train[:, :, -1] = N  # Add N (constant value for all entries)

    return X_train, Y_train




if __name__ == '__main__':
    # Define Global Var
    N = 5 # Number og agents
    alpha = 5.0
    beta = 1
    gamma_lower = -20.0
    gamma_upper = 20.0
    isValid = False

    T_exploration = 100
    T_stat = 600

    if isValid:
        L = 2500
        file_x = f"Numpy_array_save/N={N}_quadraticPath/X_valid.npy"
        file_z = f"Numpy_array_save/N={N}_quadraticPath/Z_valid.npy"
        file_y = f"Numpy_array_save/N={N}_quadraticPath/Y_valid.npy"
        file_y_sanity = f"Numpy_array_save/N={N}_quadraticPath/Y_sanity_valid.npy"
    else:
        L = 25000
        file_x = f"Numpy_array_save/N={N}_quadraticPath/X_train.npy"
        file_z = f"Numpy_array_save/N={N}_quadraticPath/Z_train.npy"
        file_y = f"Numpy_array_save/N={N}_quadraticPath/Y_train.npy"
        file_y_sanity = f"Numpy_array_save/N={N}_quadraticPath/Y_sanity_train.npy"

    # Calculate Q and B
    Q_game, B_game, Q_diag = generate_Q_B(N, L, alpha, beta, low=1.2, high=2.2)
    # Generate Data
    # x_init = np.random.uniform(1.1, 12.9, size=(L, N, 1))
    # Get Data to training
    # x_data, _ = Generate_Train_data(T_exploration, x_init, is_global=False, StepJump=1)
    # z_data, y_data = Generate_Train_data(T_stat, x_init, is_global=True, StepJump=10)
    y_sanity = Q_diag.copy()

    # Manual  Option (old)
    x_init = 1 * np.ones((L, N, 1))
    x_data = np.zeros((L, N, 2 * T_exploration + 2))  # Preallocate
    x = x_init.copy()
    for t in range(T_exploration):
        x = x + 10 / N
        cost = (0.5 * (2 * np.matmul(Q_game, x) * x - Q_diag * (x ** 2)) + B_game * x)
        help = cost / x
        # Store x and cost in X_train
        x_data[:, :, t * 2] = x[:, :, 0]  # Store x
        # x_data[:, :, t * 3 + 1] = cost[:, :, 0]  # Store cost
        x_data[:, :, t * 2 + 1] = help[:, :, 0]  # Store cost
    # Add B_game and N as the final two features
    x_data[:, :, -2] = B_game[:, :, 0]  # Add B_game
    x_data[:, :, -1] = N  # Add N (constant value for all entries)

    # Save results
    np.save(file_x, x_data[:, 0, :])
    # np.save(file_z, z_data[:, 0, :])
    # np.save(file_y, y_data[:, 0, :])
    np.save(file_y_sanity, y_sanity[:, 0, :])

    print("Finsh .....")