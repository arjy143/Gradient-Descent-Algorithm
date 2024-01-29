import matplotlib.pyplot as plt
import sympy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class gradient_descent_class:
    
    def __init__(self, x_values, y_values, learning_rate, input_gradient, input_intercept):
        self.x_values = x_values
        self.y_values = y_values
        self.learning_rate = learning_rate
        self.input_gradient = input_gradient
        self.input_intercept = input_intercept

    def display_final_graph(self, input_gradient, input_intercept, count):
        plt.scatter(self.x_values, self.y_values)
        plt.xlabel(f'x axis; Gradient: {input_gradient}; Intercept: {input_intercept}; Count(epochs): {count}')
        plt.ylabel('y - axis')
        plt.plot(self.x_values, input_gradient * self.x_values + input_intercept, color="red", label="Line of best fit")
        plt.legend()
        plt.show()

    def create_graph(self, input_gradient, input_intercept):
        plt.scatter(self.x_values, self.y_values)
        plt.plot(self.x_values, input_gradient * self.x_values + input_intercept, color="blue")

    def display_scatter_graph(self):
        plt.scatter(self.x_values, self.y_values)
        plt.show()

    def display_loss_function_graph(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        M = np.linspace(-100, 100, 100)
        C = np.linspace(-100, 100, 100)
        m, c = np.meshgrid(M, C)
        eq_terms = [(self.y_values[i] - (m + c * self.x_values[i]))**2 for i in range(len(self.x_values))]  
        f = sum(eq_terms) 
        ax.plot_surface(c,m, f)
        ax.set_xlabel('m')
        ax.set_ylabel('c')
        ax.set_zlabel('f')
        plt.show()

    def start_gradient_descent(self, step_size_tolerance, max_count):
        m_step_size = 1
        c_step_size = 1
        count = 0
    
        m , c = sympy.symbols('m c')
        eq_terms = [(self.y_values[i] - (c + m * self.x_values[i]))**2 for i in range(len(self.x_values))]  
        f = sum(eq_terms) 
        df_dm = f.diff(m)
        df_dc = f.diff(c)

        while (abs(m_step_size) > step_size_tolerance and abs(c_step_size) > step_size_tolerance) or count < max_count:
            m_derivative_value = df_dm.subs([(m, self.input_gradient), (c, self.input_intercept)])
            c_derivative_value = df_dc.subs([(m, self.input_gradient), (c, self.input_intercept)])

            m_step_size = m_derivative_value * self.learning_rate
            c_step_size = c_derivative_value * self.learning_rate
            self.learning_rate *= 0.9
            self.input_gradient -= m_step_size
            self.input_intercept -= c_step_size 
            count += 1
            if count % 1 == 0:
                self.create_graph(self.input_gradient, self.input_intercept)

        return self.input_gradient, self.input_intercept, count
    
    