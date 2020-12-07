import numpy as np
import math
# Link analysis
 
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


# Collaborative filtering (CF)
def user_based_sim(arr):
    """
    Accept only 2 users per array
    """
    avg = [sum(arr[i])/arr.shape[1] for i in range(arr.shape[0])]
    numarator = 0
    a_set = []
    b_set = []
    for i in range(arr[0].shape[0]):
        numarator += (arr[0][i] - avg[0]) * (arr[1][i] - avg[1])
        a_set.append((arr[0][i] - avg[0]) ** 2)
        b_set.append((arr[1][i] - avg[1]) ** 2)
    return numarator/ math.sqrt(sum(a_set) * sum(b_set))
print(user_based_sim(np.array([[2,3,4],[3,4,5]])))

# np.array([
# [2,3,4,5], book1
# [3,4,4,2], book2
# [5,0,2,2], book3
# [4,5,0,4], book4
# [0,5,3,5]]) book5

def sim_calc(a,b, avg_a,avg_b):
    numarator = 0
    a_set = []
    b_set = []
    for i in range(a.shape[0]):
        numarator += (a[i] - avg_a) * (b[i] - avg_b)
        a_set.append((a[i] - avg_a) ** 2)
        b_set.append((b[i] - avg_b) ** 2)
    result = numarator/ math.sqrt(sum(a_set) * sum(b_set))
    if not math.isnan(result):
        return result
    else:
        return 0
# This function should accept all, however, it will not give right result
# due to the different in arrays (i.e If there's ranking for 1 item but not the other)

def item_based(arr, r_item):
    avg = [sum(arr[i])/arr.shape[1] for i in range(arr.shape[0])]
    sim = []
    for i in range(arr.shape[0]):
        avg_a = avg[i]
        avg_b = sum(r_item)/len(r_item)
        sim.append( sim_calc(arr[i],r_item,avg_a,avg_b))
    print(sim)

item_based(np.array([[3,5]]),[2,2])


# Clustering

# HAC
# K-mean

def clustering_sim_1(init_centroids,arr):
    for i in init_centroids:
        sim = [x.dot(i) for x in arr]
        print(sim)

def clustering_sim_2(arr):
    for i in arr:
        sim = [ np.round(x.dot(i),2) if not np.array_equal(x,i) else 1 for x in arr]
        print(sim)

print("-------------------")
a = np.array([
    [0,0.9,0.4],
    [0.8,0.3,0.5],
    [1,0,0],
    [0,1,0],
    [0.7,0.4,0.6]])
clustering_sim_2(a)


    


