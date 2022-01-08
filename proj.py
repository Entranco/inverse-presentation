import numpy as np
import numpy.linalg as lin
import random
import matplotlib.pyplot as plt
import time
import scipy.optimize as opt

def test_mapping(mapping, R, n):
    total = 0
    for i in range(n):
        for j in range(i):
            total = total + (R[i][j] * ((mapping[i] - mapping[j])**2))
    return total

def test_mapping_matrix(mapping, R, n):
    returnMatrix = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(i):
            returnMatrix[i][j] = (R[i][j] * ((mapping[i] - mapping[j])**2))
            returnMatrix[j][i] = (R[i][j] * ((mapping[i] - mapping[j])**2))
    return returnMatrix

def gen_mapping(n, order):
    mapping = [(i, order[i]) for i in range(n)]
    mapping.sort(key=lambda x : x[1])
    #return [mapping[i][0] for i in range(n)]
    returnList = [0 for i in range(n)]
    for i in range(n):
        returnList[mapping[i][0]] = i
    return returnList


def generate_data(n, steps, mapping):
    steps = 200
    birth = lambda k : 0.1*(0.98**(k**2))
    death = lambda k : 0.2
    graphs = list()
    graph = [[0 for i in range(n)] for j in range(n)]
    graphs.append(graph)
    

    for i in range(steps):
        new_graph = [[0 for i in range(n)] for j in range(n)]
        for j in range(n):
            for k in range(j):
                result = random.randint(1, 10000)
                if graph[j][k] == 0:
                    if birth(mapping[j]-mapping[k]) * 10000 > result:
                        new_graph[j][k] = 1
                    else:
                        new_graph[j][k] = 0
                else:
                    if death(mapping[j]-mapping[k]) * 10000 > result:
                        new_graph[j][k] = 0
                    else:
                        new_graph[j][k] = 1
        graph = new_graph 
        graphs.append(graph)
    return graphs

def generate_data_social(n, start, steps, mapping, birth, death):
    graphs = list()
    graphs.append(start)
    
    for i in range(steps):
        new_graph = [set() for j in range(n)]
        for j in range(n):
            for k in range(j):
                result = random.randint(1, 10000)
                if k not in graphs[i][j]:
                    if birth(mapping[j]-mapping[k]) * 10000 > result:
                        new_graph[j].add(k)
                        new_graph[k].add(j)
                else:
                    if death(mapping[j]-mapping[k]) * 10000 < result:
                        new_graph[j].add(k)
                        new_graph[k].add(j)
        graphs.append(new_graph)
    return graphs

def generate_social(n, steps):
    initial_birth = 0.1
    initial_list = [set() for j in range(n)]
    graphs = list()
    for i in range(n):
        for j in range(i):
            result = random.randint(1, 10000)
            if result > 10000 - (10000 * initial_birth):
                initial_list[i].add(j)
                initial_list[j].add(i)
    
    graphs.append(initial_list)
    for k in range(1, steps):
        prob_matrix = [[0.02 for i in range(n)] for j in range(n)]
        for i in range(n):
            for j in range(i):
                if j in graphs[k-1][i]:
                    for el in graphs[k-1][i]:
                        prob_matrix[i][j] = prob_matrix[i][j] + 0.025
                    for el in graphs[k-1][j]:
                        prob_matrix[i][j] = prob_matrix[i][j] + 0.025
        
        new_list = [set() for j in range(n)]
        for i in range(n):
            for j in range(i):
                if j in graphs[k-1][i]:
                    result = random.randint(1, 10000)
                    if result < 10000 - (0.03 * 10000):
                        new_list[i].add(j)
                        new_list[j].add(i)
                else:
                    result = random.randint(1, 10000)
                    if result > 10000 - (prob_matrix[i][j] * 10000):
                        new_list[i].add(j)
                        new_list[j].add(i)
        graphs.append(new_list)
        
    return graphs
                        



def analyze_graph(graphs, n):
    R = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(i):
            freq = 0
            for k in range(len(graphs)):
                if graphs[k][i][j] != 0:
                    freq = freq + 1
            R[i][j] = freq
            R[j][i] = freq

    D = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        D[i][i] = sum(R[i]) + sum(R[:][i])

    deltaR = list()
    for i in range(n):
        deltaR.append([D[i][j] - R[i][j] for j in range(n)])
    return R, D, deltaR

def analyze_list(graphs, n, step_cap):
    R = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(i):
            freq = 0
            for k in range(step_cap):
                if j in graphs[k][i]:
                    freq = freq + 1
            R[i][j] = freq
            R[j][i] = freq

    D = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        D[i][i] = sum(R[i]) + sum(R[:][i])

    deltaR = list()
    for i in range(n):
        deltaR.append([D[i][j] - R[i][j] for j in range(n)])
    return R, D, deltaR

def run_eigenanalysis(mapping, graphs, R, D, deltaR, n):
    [w, v] = lin.eig(deltaR)

    index = 0
    for i in range(len(w)):
        if w[i] > w[index]:
            index = i

    max_index = index

    for i in range(len(w)):
        if w[i] < w[index] and w[i] > 0:
            index = i
    min_index = index

    max_pairs = [(i, v[max_index][i]) for i in range(n)]
    min_pairs = [(i, v[min_index][i]) for i in range(n)]
    max_pairs.sort(key=lambda x : x[1])
    min_pairs.sort(key=lambda x : x[1])

    max_results = gen_mapping(n, v[max_index])
    min_results = gen_mapping(n, v[min_index])

    #print(mapping)
    #print(max_results)
    #print(min_results)

    #print(test_mapping(mapping, R, n))
    #print(test_mapping(max_results, R, n))
    #print(test_mapping(min_results, R, n))
    #random_mapping = [i for i in range(n)]
    #random.shuffle(random_mapping)
    #print(test_mapping(random_mapping, R, n))

def run_sgd(n, R, deltaR):
    q = [i for i in range(n)]
    random.shuffle(q)
    iterations = 0
    lamb = 0.00001
    err_arr = list()
    while iterations < 10000:
        #mapping = gen_mapping(n, q)
        #err = test_mapping(mapping, R, n)
        err = obj(q, deltaR)
        err_arr.append(err)
        #print(err)
        #if iterations > 300 and err >= err_arr[iterations - 300]:
            #break
        derivative = deriv(q, deltaR)#np.matmul(np.transpose(q), deltaR + np.transpose(deltaR))
        nor = lin.norm(derivative)
        #print(nor)
        if nor == 0.0:
            break
        q = q - (lamb * derivative * (iterations / 100))
        iterations = iterations + 1
        #mapping = gen_mapping(n, q)
        #print(test_mapping(mapping, R, n))
    
    mapping = gen_mapping(n, q)
    return mapping

def deriv(q, deltaR):
    return np.matmul(np.transpose(q), deltaR + np.transpose(deltaR))
    
def obj(q, deltaR):
    return np.matmul(np.matmul(np.transpose(q), deltaR), q)

#lambda x: test_mapping(gen_mapping(n,x), R, n)
def run_scipy(n, R, deltaR):
    guess = [i for i in range(n)]
    #random.shuffle(guess)
    res = opt.minimize(lambda x : obj(x, deltaR), guess, method='Newton-CG', jac=lambda x : deriv(x, deltaR))
    return gen_mapping(n, res.x)

def list_to_matrix(adj_list):
    matrix = [[0 for i in range(len(adj_list))] for j in range(len(adj_list))]
    for i in range(len(adj_list)):
        for j in range(i):
            if j in adj_list[i]:
                matrix[i][j] = 1
    return matrix

def run(n):
    steps = 200
    mapping = [el for el in range(n)]
    random.shuffle(mapping)
    graphs = generate_data(n, steps, mapping)
    R, D, deltaR = analyze_graph(graphs, n)
    #t0 = time.time()
    sgd_mapping = run_sgd(n, R, deltaR)
    #t1 = time.time()
    #return t1 - t0
    #t0 = time.time()
    scip_mapping = run_scipy(n, R, deltaR)
    #t1 = time.time()
    #return t1 - t0
    #print(scip_mapping)
    

    print("Actual Solution: " + str(test_mapping(mapping, R, n)))
    print("SGD Solution: " + str(test_mapping(mapping, R, n)))
    print("Scip Solution: " + str(test_mapping(scip_mapping, R, n)))
    scip_graphs = generate_data(n, steps, scip_mapping)
    sgd_graphs = generate_data(n, steps, sgd_mapping)
    
    plt.imshow(graphs[steps-1], cmap='hot', interpolation='nearest')
    plt.title = 'Original Matrix'
    plt.show()

    plt.imshow(sgd_graphs[steps-1], cmap='hot', interpolation='nearest')
    plt.title = 'Matrix produced by Gradient Descent'
    plt.show()

    plt.imshow(scip_graphs[steps-1], cmap='hot', interpolation='nearest')
    plt.title = 'Matrix produced by COBYLA'
    plt.show()

def run_social(n):
    steps = 100
    step_cap = 50
    graphs = generate_social(n, steps)
    R, D, deltaR = analyze_list(graphs, n, step_cap)
    
    #t0 = time.time()
    sgd_mapping = run_sgd(n, R, deltaR)
    #t1 = time.time()
    #return t1 - t0
    #t0 = time.time()
    scip_mapping = run_scipy(n, R, deltaR)
    #t1 = time.time()
    #return t1 - t0
    #print(scip_mapping)

    sgd_graphs = generate_data_social(n, graphs[step_cap-1], steps - step_cap, sgd_mapping, lambda k : 0.2 * (0.8**(k)), lambda k : 0.03)
    scip_graphs = generate_data_social(n, graphs[step_cap-1], steps - step_cap, sgd_mapping, lambda k : 0.2 * (0.8**(k)), lambda k : 0.03)

    plt.imshow(list_to_matrix(graphs[steps-1]), cmap='hot', interpolation='nearest')
    plt.title = 'Original Matrix'
    plt.show()

    plt.imshow(list_to_matrix(sgd_graphs[steps-step_cap-1]), cmap='hot', interpolation='nearest')
    plt.title = 'Matrix produced by Gradient Descent'
    plt.show()

    plt.imshow(list_to_matrix(scip_graphs[steps-step_cap-1]), cmap='hot', interpolation='nearest')
    plt.title = 'Matrix produced by COBYLA'
    plt.show()



'''
n = 100
steps = 200
mapping = [el for el in range(n)]
#random.shuffle(mapping)
graphs = generate_data(n, steps, mapping)
R, D, deltaR = analyze_graph(mapping, graphs, n)
run_eigenanalysis(mapping, graphs, R, D, deltaR)
sgd_mapping = run_sgd(n, R, deltaR)
print("Actual Solution: " + str(test_mapping(mapping, R, n)))
#print(mapping)
#print(sgd_mapping)
#originalHeat = test_mapping_matrix(mapping, R, n)
#sgdHeat = test_mapping(sgd_mapping, R, n)
sgd_graphs = generate_data(n, steps, sgd_mapping)

plt.imshow(graphs[steps-1], cmap='hot', interpolation='nearest')
plt.show()

plt.imshow(sgd_graphs[steps-1], cmap='hot', interpolation='nearest')
plt.show()
'''

def timing_func():
    times = list()
    x = list()
    for i in range(25, 301, 25):
        x.append(i)
        times.append(run(i))

    plt.plot(x, times)
    plt.title('Runtime of Newton-CG')
    plt.xlabel('Number of Vertices')
    plt.ylabel('Runtime (seconds)')
    plt.show()

#run(100)
run_social(100)