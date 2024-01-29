import numpy as np
import gradient_descent

x = np.array([0.5,-5.6,20,1,6])
y = np.array([0.02,27,3,1,0])

learning_rate = 0.01
input_gradient = 1
input_intercept = 0
step_size_tolerance = 0.001
max_count = 500

program = gradient_descent.gradient_descent_class(x, y, learning_rate, input_gradient, input_intercept)
program.display_scatter_graph()
program.display_loss_function_graph()
output_gradient, output_intercept, count= program.start_gradient_descent(step_size_tolerance, max_count)

print(output_gradient)
print(output_intercept)
print(count)

program.display_final_graph(output_gradient,output_intercept, count)
