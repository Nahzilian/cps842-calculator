import numpy as np
import math

# Link analysis

# page rank
# HITS
def page_rank(arr,num,alpha):
    """
    Accept numpy array of 0 and 1
    the number alpha
    the amount of iterarion num
    """
    for i in range(arr.shape[0]):
        count_non = np.count_nonzero(arr[i])
        if count_non == 0:
            arr[i] += 1/(np.count(arr[i]))
        else:
            arr[i] = arr[i] * (1/count_non)
    x_0 = np.array([1] + [0 for _ in range(arr.shape[1] - 1)])
    arr = arr * (1 - alpha)
    arr = arr + (alpha/arr.shape[0])
    arr = np.round(arr,2)
    init_state = np.round(x_0.dot(arr),2)
    for _ in range(num):
        init_state = np.round(init_state,2).dot(arr)
    return init_state

# test = np.array([[0, 1, 1],[0, 0, 1],[0, 1, 0]],np.double)
# print(page_rank(test,2,0.1))

# Collaborative filtering (CF)
def user_based_sim(arr,user_r):
    """
    Accept separated arrays
    """
    result = []
    for i in range(arr.shape[0]):
        index = []
        temp = []
        for j,value in enumerate(arr[i]):
            if not value == 0:
                temp.append(value)
            else:
                index.append(j)
        user_temp = [user_r[x] for x in range(len(user_r)) if not x in index]
        avg_user = round(sum(user_temp)/len(user_temp),2)
        avg = round(sum(temp)/len(temp),2)
        numarator = 0
        a_set = []
        b_set = []
        for j,value in enumerate(user_temp):
            numarator += (temp[j] - avg) * (value - avg_user)
            a_set.append((temp[j] - avg) ** 2)
            b_set.append((value - avg_user) ** 2)
        den = math.sqrt(sum(a_set) * sum(b_set))
        if den > 0:
            result.append(np.round(numarator/den,2))
        else:
            result.append(0)
    return result

def user_score_calc(arr,others):
    return  sum([arr[x] * others[x] for x in range(len(arr))]) / sum(arr)

book = np.array([
[5,3,0,3],
[3,0,4,3]])

# user = np.array([4,2,5,4])
# print(user_based_sim(book,user))
# t = [x for x in user_based_sim(book,user) if x > 0]
# print(t)
# print(user_score_calc(t,[4,4]))


# np.array([
# [2,3,4,5], book1
# [3,4,4,2], book2
# [5,0,2,2], book3
# [4,5,0,4], book4
# [0,5,3,5]]) book5

# This function should accept all, however, it will not give right result
# due to the different in arrays (i.e If there's ranking for 1 item but not the other)

def item_based_sim(arr,item_r):
    """
    Accept only 2 users per array
    """
    result = []
    for i in range(arr.shape[0]):
        index = []
        temp = []
        for j,value in enumerate(arr[i]):
            if not value == 0:
                temp.append(value)
            else:
                index.append(j)
        item_temp = [item_r[x] for x in range(len(item_r)) if not x in index]
        avg_item = round(sum(item_temp)/len(item_temp),2)
        avg = round(sum(temp)/len(temp),2)
        numarator = 0
        a_set = []
        b_set = []
        for j,value in enumerate(item_temp):
            numarator += (temp[j] - avg) * (value - avg_item)
            a_set.append((temp[j] - avg) ** 2)
            b_set.append((value - avg_item) ** 2)
        den = math.sqrt(sum(a_set) * sum(b_set))
        if den > 0:
            result.append(np.round(numarator/den,2))
        else:
            result.append(0)
    return result

def item_score_calc(arr,others):
    return  sum([arr[x] * others[x] for x in range(len(arr))]) / sum(arr)

items = np.array([
[5,3],
[3,0],
[0,4],
[3,3]])

item = np.array([4,4])
# print(item_based_sim(items,item))
# t = [x for x in item_based_sim(items,item) if x > 0]
# print(t)
# print(item_score_calc(t,[4,4]))


# Clustering

# HAC
# K-mean

def k_mean_clustering(arr,centroids,num,org_array,org_num):
    if num == 0:
        print(centroids)
        return centroids
    elif num < org_num:
        sim_results = []
        new_centroids = []
        for element in org_array:
            sim_results.append([np.round(element.dot(centroid),2) for centroid in centroids])
        for i,centroid in enumerate(centroids):
            temp = []
            for j,sim_val in enumerate(sim_results):
                max_n = max(sim_val)
                if sim_val.index(max_n) == i:
                    temp.append(org_array[j])
            new_centroids.append(temp)
        print(new_centroids)
        re_cacl_centroids = [np.round(np.round(sum_row(x),2) / len(x),2) for x in new_centroids]
        k_mean_clustering(arr,re_cacl_centroids,num - 1,org_array,org_num)
    else:
        sim_results = []
        new_centroids = []
        for element in arr:
            sim_results.append([np.round(element.dot(centroid),2) for centroid in centroids])
        for i,centroid in enumerate(centroids):
            temp = [centroids[i]]
            for j,sim_val in enumerate(sim_results):
                max_n = max(sim_val)
                if sim_val.index(max_n) == i:
                    temp.append(arr[j])
            new_centroids.append(temp)
        re_cacl_centroids = [np.round(np.round(sum_row(x),2) / len(x),2) for x in new_centroids]
        k_mean_clustering(arr,re_cacl_centroids,num - 1,org_array,org_num)

def sum_row(arr):
    result = np.zeros((1,len(arr[0])))
    for i in arr:
        result += i
    return np.round(result[0],2)


test = np.array([
    [1,0,0],
    [0,1,0],
    [0.7,0.4,0.6]
])

centroids = np.array([
    [0,0.9,0.4],
    [0.8,0.3,0.5]
])

org_arr = np.array([
    [0,0.9,0.4],
    [0.8,0.3,0.5],
    [1,0,0],
    [0,1,0],
    [0.7,0.4,0.6]
])

print(k_mean_clustering(test,centroids,100,org_arr,100))

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
# clustering_sim_2(a)


def eval_clustering():
    print("hello world")

# a = np.array([[2,3,4,5],
# [3,4,4,2],
# [5,0,2,2]])
# b = np.array([1,2,3,4])
# print(a @ b)