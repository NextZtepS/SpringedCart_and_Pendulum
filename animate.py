from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


# Define the model
def model(Y, τ, M, m, g, k, l, L, T):
    x, dx_dτ, θ, dθ_dτ = Y

    # Equations of motion
    dx_dτ = dx_dτ
    # d2x_dτ2 = (1 / (M + m) * L) * (- m * l * np.cos(θ) * d2θ_dτ2 + m * l * np.sin(θ) * dθ_dτ**2 - k * L * T**2 * x)
    d2x_dτ2 = (1 / (L * ((M + m) - m * np.cos(θ)**2))) * (m * g * T**2 * np.cos(θ) * np.sin(θ) + m * l * np.sin(θ) * dθ_dτ**2 - k * L * T**2 * x)
    dθ_dτ = dθ_dτ
    d2θ_dτ2 = (1 / l) * (- L * np.cos(θ) * d2x_dτ2 - g * T**2 * np.sin(θ))

    return [dx_dτ, d2x_dτ2, dθ_dτ, d2θ_dτ2]

def solve_and_extract(model, Y0, τ, constants):
    Y = odeint(model, Y0, τ, args=constants)
    x = Y[:, 0]
    dx_dτ = Y[:, 1]
    θ = Y[:, 2]
    θ = np.mod(θ - np.pi, 2 * np.pi) - np.pi  # make θ between -π and π only
    dθ_dτ = Y[:, 3]
    return x, dx_dτ, θ, dθ_dτ

# Constants
M = 0.5  # Mass of the cart
m = 0.5  # Mass of the pendulum
g = 1  # Gravity
k = 1  # Spring constant
l = 1  # Length of the pendulum
L = 1  # Unit length
T = 1  # Unit time

# Initial conditions
x0 = 0.5
dx_dτ0 = 2
θ0 = np.pi / 2
dθ_dτ0 = -2
Y0 = [x0, dx_dτ0, θ0, dθ_dτ0]

# Solve the Differential Equations
τ_max = 30
num_point = 1_000
τ = np.linspace(0, τ_max, num_point)
x, dx_dτ, θ, dθ_dτ = solve_and_extract(model, Y0, τ, (M, m, g, k, l, L, T))


if __name__ == "__main__":
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Cart-Pendulum System")
    ax.set_xlim((min(x) - l) * 1.1, (max(x) + l) * 1.1)
    ax.set_ylim(-l * 1.1, l * 1.1)
    ax.set_aspect("equal")
    ax.grid()

    # Pendulum position
    pendulum, = ax.plot([], [], "o-", markersize=10, label="Pendulum")
    # Cart position
    cart, = ax.plot([], [], "s", markersize=15, label="Cart")

    # Initialize the animation
    def init():
        pendulum.set_data([], [])
        cart.set_data([], [])
        return pendulum, cart

    # Update the animation
    def update(frame):
        # Update the cart and pendulum position
        pendulum.set_data(
            [x[frame], x[frame] + l * np.sin(θ[frame])],
            [0, -l * np.cos(θ[frame])],
        )
        cart.set_data([x[frame]], [0])
        return pendulum, cart

    # Animate the system
    ani = FuncAnimation(fig, update, frames=len(τ), init_func=init, blit=True, interval=(τ_max * T * 1000) / num_point)

    # The following line may take a long time to render the animation
    # ani.save("animation.gif", writer=PillowWriter(fps=num_point / (τ_max * T)))
    
    plt.show()
    