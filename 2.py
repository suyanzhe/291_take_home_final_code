import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Line search on f(x)=x^4

def steepest_descent_x4(alpha, x):
    derivative = 4 * np.power(x, 3)
    return x - alpha * derivative

def compute_optimal_step_size(x):
    derivative = 4 * np.power(x, 3)
    return x / derivative if derivative else 0

def newton_x4(step_size, x):
    derivative = 4 * np.power(x, 3)
    second_derivative = 12 * np.power(x, 2)
    return x - step_size * derivative / second_derivative

def iterate(x, function, num_of_iterations, alpha):
    values = []
    for i in range(num_of_iterations):
        values.append(np.power(x, 4))
        x = function(alpha, x)
    values.append(np.power(x, 4))
    return values

def iterate_with_optimal_step_size(x, num_of_iterations):
    values = []
    for i in range(num_of_iterations):
        values.append(np.power(x, 4))
        alpha = compute_optimal_step_size(x)
        x = steepest_descent_x4(alpha, x)
    values.append(np.power(x, 4))
    return values

def draw_curves(data_group, num_of_iterations = 100, fig_name = None, colors = None):
    plt.rcParams["font.family"] = "STIX"
    plt.rcParams["mathtext.fontset"] = "stix"
    iters = range(num_of_iterations + 1)
    figure = plt.figure(figsize=(8, 2.5))
    if colors is not None:
        for data, c in zip(data_group, colors):
            plt.plot(iters, data[1], label = data[0], color = c)
    else:
        for data in data_group:
            plt.plot(iters, data[1], label = data[0])
    plt.legend()
    plt.title(r"How the function values of $f\left(x\right)=x^4$ change in the first " + str(num_of_iterations) + " iterations")
    plt.xlabel("After iteration #")
    plt.ylabel(r"Value of $x^4$")
    plt.subplots_adjust(0.1, 0.175, 0.975, 0.9)
    if fig_name is not None:
        plt.savefig(fig_name + ".pdf", dpi = 3584)
    plt.show()

def main(num_of_iterations):
    c = ["#F44336", "#FF9800", "#000000", "#2196F3", "#00BCD4", "#4CAF50"]
    x_0 = 1
    data_group = [("Steepest descent, learning rate = " + str(alpha), iterate(x_0, steepest_descent_x4, num_of_iterations, alpha)) for alpha in [0.1, 0.01]]
    draw_curves(data_group, num_of_iterations, fig_name = "steepestdescent", colors = c[:len(data_group)])
    data_group.append(("Steepest descent with the optimal step size", iterate_with_optimal_step_size(x_0, num_of_iterations)))
    draw_curves(data_group, num_of_iterations, fig_name = "optimalstep", colors = c[:len(data_group)])
    data_group.extend([("Newton direction, step size = " + str(step_size), iterate(x_0, newton_x4, num_of_iterations, step_size)) for step_size in [1, 0.3, 0.1]])
    draw_curves(data_group, num_of_iterations, fig_name = "Newton", colors = c[:len(data_group)])

if __name__ == "__main__":
     main(100)