import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # Your code here
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    input_matrix = np.pad(input_matrix, padding, 'empty')

    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    output_matrix = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            start_i, start_j = i * stride, j * stride
            output_matrix[i][j] = np.sum(input_matrix[start_i:start_i+kernel_height, start_j:start_j+kernel_width] * kernel)
    return output_matrix
