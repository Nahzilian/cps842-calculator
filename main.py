import numpy as np

# Link analysis
 
def matrix_calculator(vec1,vec2):
    """
    @Param
    vec1: List, 1x3 vector
    vec2: List, 3x3 vector
    """
    return np.dot(vec1,vec2)
a = np.array([[0.03, 0.48, 0.48],[0.03, 0.03, 0.93],[0.03, 0.93, 0.93]])
b = np.array([0.03,0.48,0.48])
print(matrix_calculator(b,a))

def page_rank_hit(arr,num,alpha):
    x_0 = np.array([1,0,0])
    arr = arr * (1 - alpha)
    arr = arr + (alpha/arr.shape[0])
    init_state = np.round(x_0.dot(arr),2)
    for _ in range(num):
        init_state = np.dot(init_state,arr)
    return init_state

test = np.array([[0, 0.5, 0.5],[0, 0, 1],[0, 1, 0]])
print(page_rank_hit(test,2,0.1))
