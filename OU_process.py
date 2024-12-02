import numpy as np
import matplotlib.pyplot as plt


def simulate_ou_process(theta, mu, sigma, x0, T, dt, n_paths):
    """
    Symuluje trajektorie procesu Ornsteina-Uhlenbecka.

    Parameters:
        theta (float): Szybkość powrotu do średniej.
        mu (float): Średnia stacjonarna.
        sigma (float): Intensywność fluktuacji.
        x0 (float): Wartość początkowa.
        T (float): Całkowity czas symulacji.
        dt (float): Długość kroku czasowego.
        n_paths (int): Liczba trajektorii do zasymulowania.

    Returns:
        t (numpy array): Tablica czasów.
        paths (numpy array): Tablica trajektorii (n_paths x len(t)).
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0

    for i in range(n_steps):
        dw = np.sqrt(dt) * np.random.randn(n_paths)  # Przyrost procesu Wienera / Browna
        paths[:, i + 1] = paths[:, i] + theta * (mu - paths[:, i]) * dt + sigma * dw

    return t, paths


# Parametry symulacji
theta = 0.7 # Szybkość powrotu
mu = 0.0  # Średnia stacjonarna
sigma = 0.3  # Intensywność fluktuacji
x0 = 1.0  # Wartość początkowa
T = 20.0  # Całkowity czas symulacji
dt = 0.01  # Długość kroku czasowego
n_paths = 2 # Liczba trajektorii

# Symulacja
t, paths = simulate_ou_process(theta, mu, sigma, x0, T, dt, n_paths)

# Wizualizacja
plt.figure(figsize=(10, 6))
for i in range(n_paths):
    plt.plot(t, paths[i, :], label=f'Trajektoria {i + 1}')
plt.axhline(mu, color='red', linestyle='--', label='Mean')
plt.title("Trajektorie procesu Ornsteina-Uhlenbecka")
plt.xlabel("Czas")
plt.ylabel("X(t)")
plt.legend()
plt.grid(True)
plt.show()
