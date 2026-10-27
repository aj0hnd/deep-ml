def determinant_4x4(matrix: list[list[int|float]]) -> float:
	# Your recursive implementation here
	def laplace_expansion(mat):
        if len(mat) == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        
        i, det = 0, 0
        for j in range(len(mat[0])):
            sub_mat = []
            for m in range(1, len(mat)):
                sub_mat.append([])
                for n in range(len(mat[0])):
                    if n != j:
                        sub_mat[-1].append(mat[m][n])
            if (i + j) % 2 == 0:
                det += mat[i][j] * laplace_expansion(sub_mat)
            else:
                det -= mat[i][j] * laplace_expansion(sub_mat)
        
        return det    
    return laplace_expansion(matrix)