from heapq import heappush, heappop
import random
import matplotlib.pyplot as plt
from timeit import timeit
import math
import copy
import itertools

grader = None  # will be set by grader.py

def SP_naive (graph, s):
    '''
    Shortest path algorithm with naive implementation.
    graph: adjacency list of the graph
    s: source vertex
    return: a dictionary of shortest distances from s to all other vertices
    '''
    frontier = [s]
    dist = {s:0}

    while len(frontier) > 0:

        x = min(frontier, key = lambda k: dist[k])
        frontier.remove(x)

        for y, dxy in graph[x].items():
            dy = dist[x] + dxy

            if y not in dist:
                frontier.append(y)
                dist[y] = dy

            elif dist[y] > dy:
                dist[y] = dy

    return dist

# Shortest path algorithm with binary heap

def SP_heap (graph, s):
    '''
    Shortest path algorithm with binary heap.
    graph: adjacency list of the graph
    s: source vertex
    return: a dictionary of shortest distances from s to all other vertices
    '''
    frontier = []
    heappush(frontier, (0, s))
    done = set()
    dist = {s: 0}

    while len(frontier) > 0:

        dx, x = heappop(frontier)
        if x in done:
            continue

        done.add(x)

        for y, dxy in graph[x].items():
            dy = dx + dxy

            if y not in dist or dist[y] > dy:
                heappush(frontier,(dy, y))
                dist[y] = dy

    return dist

def random_sparse_graph (n, step):
    '''
    Generate a random sparse graph with n vertices.
    n: number of vertices
    step: maximum distance between two adjacent vertices
    return: adjacency list of the graph
    '''
    graph = {f'{i}_{j}': {} for i in range(1, n+1) for j in range(1, n+1)}

    for i in range(1, n+1):
        for j in range(1, n):
            d = random.randint(step+1, 2*step)
            graph[f'{i}_{j}'][f'{i}_{j+1}'] = d
            graph[f'{i}_{j+1}'][f'{i}_{j}'] = d

    for i in range(1, n):
        for j in range(1, n+1):
            d = random.randint(step+1, 2*step)
            graph[f'{i}_{j}'][f'{i+1}_{j}'] = d
            graph[f'{i+1}_{j}'][f'{i}_{j}'] = d

    return graph

def random_dense_graph (n, d_max):
    '''
    Generate a random dense graph with n vertices.
    n: number of vertices
    d_max: maximum distance between two adjacent vertices
    return: adjacency list of the graph
    '''
    graph = {f'{i}':{} for i in range(n)}

    for n1 in graph:
        for n2 in graph:
            if n2!= n1 and n2 not in graph[n1]:
                d = random.randint(1, d_max)
                graph[n1][n2] = d
                graph[n2][n1] = d

    return graph

####################### Benchmark b1 #####################

def benchmark():
    # ----- mesures pour graphes PEU DENSES (grille N×N), N = 10..100 -----
    Time_heap_sparse, Time_naive_sparse, n_list_sparse = [], [], []
    n_starts = 10  # nombre de départs aléatoires par graphe

    for N in range(10, 101):  # jusqu'à 100x100
        n = N * N
        n_list_sparse.append(n)
        print(f'Benchmark sparse N={N} (n={n})...')
        graph_sparse = random_sparse_graph(n=N, step=100)

        # moyenne sur n_starts lancements avec départs aléatoires
        Time_naive_sparse.append(
            timeit(lambda: SP_naive(graph_sparse, random.choice(list(graph_sparse))), number=n_starts) / n_starts
        )
        Time_heap_sparse.append(
            timeit(lambda: SP_heap(graph_sparse, random.choice(list(graph_sparse))), number=n_starts) / n_starts
        )

    # ----- mesures pour graphes DENSES, N = 10..30 -----
    Time_heap_dense, Time_naive_dense, n_list_dense = [], [], []
    for N in range(10, 30):  # limité à 30x30
        n = N * N
        n_list_dense.append(n)
        print(f'Benchmark dense N={N} (n={n})...')
        graph_dense = random_dense_graph(n=n, d_max=10_000)

        Time_naive_dense.append(
            timeit(lambda: SP_naive(graph_dense, random.choice(list(graph_dense))), number=n_starts) / n_starts
        )
        Time_heap_dense.append(
            timeit(lambda: SP_heap(graph_dense, random.choice(list(graph_dense))), number=n_starts) / n_starts
        )
    print('Benchmark terminé.')
    # ----- tracé : 2 sous-graphes séparés -----
    fig, (ax_dense, ax_sparse) = plt.subplots(2, 1, figsize=(9, 9), sharex=False)

    # Sous-graphe DENSE
    ax_dense.set_title("Graphes denses (|E| ≈ |V|²) et peu denses (|E| ≈ |V|) — N jusqu’à 30×30")
    ax_dense.set_xlabel("|V| (= N×N)")
    ax_dense.set_ylabel("Temps moyen (s)")
    ax_dense.plot(n_list_dense, Time_naive_dense, 'r*-', label="naive (dense)")
    ax_dense.plot(n_list_dense, Time_heap_dense,  'b*-', label="heap (dense)")
    # Plot 
    ax_dense.plot(n_list_sparse[:len(n_list_dense)], Time_naive_sparse[:len(n_list_dense)], 'r^-', label="naive (sparse)")
    ax_dense.plot(n_list_sparse[:len(n_list_dense)], Time_heap_sparse[:len(n_list_dense)],  'b^-', label="heap (sparse)")

    # Échelle log utile si les courbes se séparent beaucoup :
    # ax_dense.set_yscale('log')
    ax_dense.grid(True, alpha=0.25)
    ax_dense.legend(loc="best")

    # Sous-graphe SPARSE
    ax_sparse.set_title("Zoom sur graphes peu denses (|E| ≈ |V|) — N jusqu’à 100×100")
    ax_sparse.set_xlabel("|V| (= N×N)")
    ax_sparse.set_ylabel("Temps moyen (s)")
    ax_sparse.plot(n_list_sparse, Time_naive_sparse, 'r^-', label="naive (sparse)")
    ax_sparse.plot(n_list_sparse, Time_heap_sparse,  'b^-', label="heap (sparse)")
    ax_sparse.grid(True, alpha=0.25)
    # Fenêtre verticale resserrée autour des temps sparse
    ymin = min(Time_naive_sparse + Time_heap_sparse)
    ymax = max(Time_naive_sparse + Time_heap_sparse)
    margin = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ax_sparse.set_ylim(ymin - margin, ymax + margin)
    ax_sparse.legend(loc="best")

    plt.tight_layout()
    plt.show()
####################### Question q1 #####################

def add_source(graph, src):
    '''
    Add a source vertex to the graph.
    graph: adjacency list of the graph
    src: source vertex
    return: adjacency list of the graph with the source vertex
    '''
    if src in graph:
        return "error: the source vertex is already in the graph"

    graph_src=graph.copy()

    graph_src[src]={}
    for v in graph.keys():
        graph_src[src][v]=0

    return graph_src

def test_add_source():
    sim_graph = {"A":{"B":4,"C":2,"D":3},
                "B":{"A":6,"C":-5},
                "C":{"D":1},
                "D":{}}

    src="source"
    answer = {"A":{"B":4,"C":2,"D":3},
                "B":{"A":6,"C":-5},
                "C":{"D":1},
                "D":{},
                "source":{"A":0, "B":0, "C":0, "D":0}}
    grader.addMessage(f'Testing add_source on: {grader.printer.pformat(sim_graph)}, {src}')
    grader.addMessage(f'Expecting: {grader.printer.pformat(answer)}')

    sim_graph_src=add_source(sim_graph, src)
    
    if grader.requireIsEqual(answer, sim_graph_src):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {grader.printer.pformat(sim_graph_src)}')
    return "Test passed"


####################### Question q2 #####################

def bellman_ford(graph, src):
    '''
    Bellman-Ford algorithm to find the shortest path from a source vertex to all other vertices.
    graph: adjacency list of the graph
    src: source vertex
    return: a dictionary of shortest distances from the source vertex to all other vertices
    '''
    n=len(graph)

    dist={}

    #initialize dist
    ############TODO : complete code#############
    for i in graph:
        dist[i]=math.inf
    dist[src]=0

    # calculate optimal distance
    ############TODO : complete code#############

    # detect negative cycle: return None if there is any
    ############TODO : complete code#############

    return dist

def test_bellman_ford():
    sim_graph = {"A":{"B":4,"C":2,"D":3},
                "B":{"A":6,"C":-5},
                "C":{"D":1},
                "D":{}}

    src="source"
    answer = {"A":0, "B":0, "C":-5, "D":-4, "source":0}
    sim_graph_src=add_source(sim_graph, src)
    grader.addMessage(f'Testing bellman_ford on: {grader.printer.pformat(sim_graph_src)}, {src}')
    grader.addMessage(f'Expecting: {answer}')
    dist=bellman_ford(sim_graph_src, src)

    if grader.requireIsEqual(answer, dist):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {dist}')

    neg_cycle_graph = {"A":{"B":-5,"C":2,"D":3},
             "B":{"A":3,"C":4},
             "C":{"D":1},
             "D":{}}
    grader.addMessage(f'Testing bellman_ford on: {grader.printer.pformat(neg_cycle_graph)}, {src}')
    grader.addMessage( 'Expecting: None')
    neg_dist = bellman_ford(neg_cycle_graph, "A")
    if grader.requireIsEqual(None, neg_dist):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {neg_dist}')


####################### Question q3 #####################

def rewrite_weights(graph, dist):
    '''
    Rewrite the weights of the graph to make them nonnegative.
    graph: adjacency list of the graph
    dist: a dictionary of shortest distances from the source vertex to all other vertices
    return: adjacency list of the graph with nonnegative weights
    '''
    # use deepcopy
    altered_graph = copy.deepcopy(graph)

    # Recalculate the new nonnegative weights
    ############### TODO : complete code ##################

    return altered_graph

def test_rewrite_weights():
    sim_graph = {"A":{"B":4,"C":2,"D":3},
                "B":{"A":6,"C":-5},
                "C":{"D":1},
                "D":{}}

    opt_distance={'A': 0, 'B': 0, 'C': -5, 'D': -4, 'source': 0}
    answer = {'A': {'B': 4, 'C': 7, 'D': 7}, 'B': {'A': 6, 'C': 0}, 'C': {'D': 0}, 'D': {}}
    grader.addMessage(f'Testing rewrite_weights on: {sim_graph}, {opt_distance}')
    grader.addMessage(f'Expecting: {grader.printer.pformat(answer)}')
    nonneg_graph=rewrite_weights(sim_graph, opt_distance)
    if grader.requireIsEqual(answer, nonneg_graph):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {nonneg_graph}')

####################### Question q4 #####################

def all_distances(graph):
    '''
    Compute all shortest distances between all pairs of vertices in the graph.
    graph: adjacency list of the graph
    return: a dictionary of all shortest distances between all pairs of vertices
    '''
    d = {(u,v):None for u in graph for v in graph}

    ############### TODO : complete code ##################

    return d

def test_all_distances():
    nonneg_graph={'A': {'B': 4, 'C': 7, 'D': 7}, 'B': {'A': 6, 'C': 0}, 'C': {'D': 0}, 'D': {}}
    answer = {('A', 'A'): 0, ('A', 'B'): 4, ('A', 'C'): 4, ('A', 'D'): 4, 
                                     ('B', 'A'): 6, ('B', 'B'): 0, ('B', 'C'): 0, ('B', 'D'): 0, 
                                     ('C', 'A'): None, ('C', 'B'): None, ('C', 'C'): 0, ('C', 'D'): 0, 
                                     ('D', 'A'): None, ('D', 'B'): None, ('D', 'C'): None, ('D', 'D'): 0}
    grader.addMessage(f'Testing all_distances on: {nonneg_graph}')
    grader.addMessage(f'Expecting: {grader.printer.pformat(answer)}')
    dist=all_distances(nonneg_graph)

    if grader.requireIsEqual(answer, dist):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {dist}')

####################### Question 5 #####################

def BF_SP_all_pairs(graph, src="source"):

    ############### TODO : complete code ##################
    d = {(u,v):None for u in graph for v in graph}

    # Return a dictionary of distances for all pairs of nodes

    return d

def test_BF_SP_all_pairs():
    sim_graph = {"A":{"B":4,"C":2,"D":3},
             "B":{"A":6,"C":-5},
             "C":{"D":1},
             "D":{}}
    
    answer = {('A', 'A'): 0, ('A', 'B'): 4, ('A', 'C'): -1, ('A', 'D'): 0, 
              ('B', 'A'): 6, ('B', 'B'): 0, ('B', 'C'): -5, ('B', 'D'): -4, 
              ('C', 'A'): None, ('C', 'B'): None, ('C', 'C'): 0, ('C', 'D'): 1, 
              ('D', 'A'): None, ('D', 'B'): None, ('D', 'C'): None, ('D', 'D'): 0}

    grader.addMessage(f'Testing BF_SP_all_pairs on: {grader.printer.pformat(sim_graph)}')
    grader.addMessage(f'Expecting: {grader.printer.pformat(answer)}')
    dist=BF_SP_all_pairs(sim_graph)

    if grader.requireIsEqual(answer, dist):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {dist}')

####################### Question q6 #####################

def closest_oven(house, oven_houses, distance_dict):
    '''
    Find the closest oven house to a given house.
    house: a house
    oven_houses: a list of oven houses
    distance_dict: a dictionary of all shortest distances between all pairs of vertices
    return: a tuple of the distance to the closest oven house and the closest oven house
    '''
    ############### TODO : complete code ##################
    return (0, None)
def test_closest_oven():
    toy_village = {'A': {'B': -3, 'E': 20, 'F': 30},
           'B': {'A': 6, 'C': 9, 'F': 39},
           'C': {'B': 9, 'D': 8},
           'D': {'C': 8, 'F': 50},
           'E': {'A': -10, 'F': 6},
           'F': {'A': -20, 'B': -25, 'D': -15, 'E': 6} }

    grader.addMessage(f'Testing closest_oven on: {grader.printer.pformat(toy_village)}')
    toy_distance_dict = BF_SP_all_pairs(toy_village)
    for node, node_list, answer in [('A', ['B','E','D'], (-3, "B")), 
                                    ('B', ['B','E','D'], (0, "B")),
                                    ('C', ['B','E','D'], (8, "D")),
                                    ('D', ['B','E','D'], (0, "D")),
                                    ('E', ['B','E','D'], (-19, "B")),
                                    ('F', ['B','E','D'], (-25, "B"))]:
        grader.addMessage(f'Expecting: closest oven from {node} among {node_list} is {answer}')
        if grader.requireIsEqual(answer, closest_oven(node, node_list, toy_distance_dict)):
            grader.addMessage('Got it!')
        else:
            grader.addMessage(f'Got: {closest_oven(node, node_list, toy_distance_dict)}')

####################### Question q7 #####################

def kcentre_value(village, oven_houses, distance_dict):
    '''
    Compute the maximum distance between a house and the closest oven house.
    village: adjacency list of the village
    oven_houses: a list of oven houses
    distance_dict: a dictionary of all shortest distances between all pairs of vertices
    return: the maximum distance between a house and the closest oven house
    '''

    ############### TODO : complete code ##################
    return 0

def test_kcentre_value():
    toy_village = {'A': {'B': -3, 'E': 20, 'F': 30},
            'B': {'A': 6, 'C': 9, 'F': 39},
            'C': {'B': 9, 'D': 8},
            'D': {'C': 8, 'F': 50},
            'E': {'A': -10, 'F': 6},
            'F': {'A': -20, 'B': -25, 'D': -15, 'E': 6} }
    grader.addMessage(f'Testing kcentre_value on: {grader.printer.pformat(toy_village)} and [B, E, D]')
    grader.addMessage(f'Expecting: 8')
    toy_distance_dict = BF_SP_all_pairs(toy_village)
    result = kcentre_value(toy_village, ['B','E','D'], toy_distance_dict)
    if grader.requireIsTrue(result == 8):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {result}')

####################### Benchmark b2 #####################

def read_map(filename):
    '''
    Read a map from a file.
    filename: name of the file
    return: adjacency list of the map
    '''
    f = open(file=filename, mode='r', encoding='utf-8')

    map = {}
    while True:  # reading list of cities from file
        ligne = f.readline().rstrip()
        if (ligne == '--'):
            break
        info = ligne.split(':')
        map[info[0]] = {}

    while True:  # reading list of distances from file
        ligne = f.readline().rstrip()
        if (ligne == ''):
            break
        info = ligne.split(':')
        map[info[0]][info[1]] = int(info[2])

    return map

def brute_force(map, candidates, k, distance_dict) :
    best_combi = []
    best_dist = math.inf
    for combi in itertools.combinations(candidates, k):
        dist= kcentre_value(map, list(combi), distance_dict)
        if  dist<best_dist:
            best_combi= list(combi)
            best_dist= dist
    return  best_dist, set(best_combi)

def BF_benchmark():

    village = read_map('village.map')

    village_distance_dict = BF_SP_all_pairs(village)

    # assert brute_force(village, list(village), 3, village_distance_dict) == (0, {'C', 'D', 'B'})
    # assert brute_force(village, list(village), 2, village_distance_dict) == (8, {'C', 'A'})

    Time_brute_force = []

    k_list = []

    for k in range(1, 20):
        print(f'k={k} is being processed...')
        Time_brute_force.append(timeit(lambda: brute_force(village, list(village), k, village_distance_dict), number=1))
        k_list.append(k)

    #print N_list, time_list
    plt.xlabel('k')
    plt.ylabel('T')
    plt.plot(k_list, Time_brute_force, 'r^')
    plt.xticks(k_list)
    plt.show()

####################### Question q8 #####################

def greedy_algorithm(map, candidates, k, distance_dict):
    '''
    Greedy algorithm to find the k-centre of a village.
    map: adjacency list of the village
    candidates: list of houses in the village
    k: number of oven houses
    distance_dict: a dictionary of all shortest distances between all pairs of vertices
    return: a tuple of the maximum distance between a house and the closest oven house and the set of oven houses
    '''
    ############### TODO : complete code ##################
    return 0, set()
def test_greedy_algorithm():
    village = read_map('village.map')

    village_distance_dict = BF_SP_all_pairs(village)

    grader.addMessage(f'Testing greedy_algorithm on: {grader.printer.pformat(village)}')
    force_d, force_h = brute_force(village, list(village), 5, village_distance_dict)
    grader.addMessage(f'Got: {force_d} with houses {force_h} with brute_force algorithm')
    greed_d, greed_h = greedy_algorithm(village, list(village), 5, village_distance_dict) # >>> 502!!
    grader.addMessage(f'Got: {greed_d} with houses {greed_h} with greedy_algorithm algorithm')
    if grader.requireIsTrue(force_d > 0):
        grader.addMessage(f'{force_d} > 0 as expected')
        if grader.requireIsTrue(greed_d >= force_d):
            grader.addMessage(f'{greed_d} >= {force_d} as expected')
        else:
            grader.addMessage(f'{greed_d} < {force_d} not as expected')
    else:
        grader.addMessage(f'{force_d} <= 0 not as expected')
    if grader.requireIsTrue(greedy_algorithm(village, list(village), 1, village_distance_dict) == (14, {'CL5'})):
        grader.addMessage('greedy with 1 house is correct')
    else:
        grader.addMessage('greedy with 1 house is incorrect')
    if grader.requireIsTrue(greedy_algorithm(village, list(village), 5, village_distance_dict) == (4, {'CL2', 'CL7', 'AH1', 'CL5', 'AH2'})):
        grader.addMessage('greedy with 5 houses is correct')
    else:
        grader.addMessage('greedy with 5 houses is incorrect')


####################### Question q9 #####################

def random_algorithm(map, candidates, k, distance_dict, trials=100) :
    '''
    Random algorithm to find the k-centre of a village.
    map: adjacency list of the village
    candidates: list of houses in the village
    k: number of oven houses
    distance_dict: a dictionary of all shortest distances between all pairs of vertices
    trials: number of trials
    return: a tuple of the maximum distance between a house and the closest oven house and the set of oven houses
    '''
    ############### TODO : complete code ##################
    return 0, set()

def test_random_algorithm():
    village = read_map('village.map')
    dist = BF_SP_all_pairs(village)
    candidates = list(village)
    
    # --- ORACLE : optimum par bruteforce pour quelques k
    ks = [2, 3, 4]  # adaptez à votre carte
    opt = {}
    opt_sets = {}
    for k in ks:
        d_star, S_star = brute_force(village, candidates, k, dist)
        opt[k] = d_star
        # on recense toutes les solutions optimales (utile si on veut tester l’appartenance)
        sols = set()
        for combi in itertools.combinations(candidates, k):
            if kcentre_value(village, list(combi), dist) == d_star:
                sols.add(frozenset(combi))
        opt_sets[k] = sols
    grader.addMessage(f'Oracle optimum values: {opt}')
    # --- (1) Déterminisme intra-implémentation (quelle que soit la méthode interne)
    seed = 123456
    k = 4
    random.seed(seed)
    d1, h1 = random_algorithm(village, candidates, k, dist, trials=200)
    random.seed(seed)
    d2, h2 = random_algorithm(village, candidates, k, dist, trials=200)

    grader.addMessage(f'[determinism] seed={seed}, k={k} → (d1={d1},h1={set(h1)}) vs (d2={d2},h2={set(h2)})')
    grader.requireIsEqual(d1, d2)
    grader.requireIsEqual(set(h1), set(h2))

    # --- (2) Cohérence interne et optimalité sur petite instance
    for k_ref in ks:
        # graine fixée mais indépendante de l'implémentation
        random.seed(424242 + k_ref)
        d_got, houses_got = random_algorithm(village, candidates, k_ref, dist, trials=500)

        # cohérence : la valeur renvoyée correspond bien aux maisons renvoyées
        d_check = kcentre_value(village, list(houses_got), dist)
        grader.addMessage(f'[consistency] k={k_ref} → got d={d_got}, check={d_check}')
        grader.requireIsEqual(d_got, d_check)

        # optimalité (sur petite carte) : égal au bruteforce
        grader.addMessage(f'[optimality] k={k_ref} → d_got={d_got}, d*={opt[k_ref]}')
        grader.requireIsEqual(d_got, opt[k_ref])

        # si vous voulez être stricts sur l’ensemble, acceptez TOUT optimum :
        grader.requireIsTrue(frozenset(houses_got) in opt_sets[k_ref])

    # --- (3) Monotonicité trials (même graine) : plus d’essais ne doit pas dégrader
    k = 5
    random.seed(999)
    d_small, _ = random_algorithm(village, candidates, k, dist, trials=20)
    random.seed(999)
    d_big, _ = random_algorithm(village, candidates, k, dist, trials=2000)
    grader.addMessage(f'[trials monotonicity] k={k}: 20→{d_small} vs 2000→{d_big}')
    # d_big ≤ d_small attendu (on minimise d)
    grader.requireIsGreaterThanOrEqual(d_big, d_small)

    # --- (4) Sanity checks structurels
    k = 4
    random.seed(321)
    d_any, h_any = random_algorithm(village, candidates, k, dist, trials=100)
    grader.requireIsEqual(len(h_any), k)
    grader.requireIsTrue(set(h_any).issubset(set(candidates)))
    grader.requireIsEqual(d_any, kcentre_value(village, list(h_any), dist))

    grader.addMessage('random_algorithm: property-based tests passed')

####################### Benchmark b3 #####################

def BF_G_R_benchmark(max_k=12, random_trials=100):
    village = read_map('village.map')
    village_distance_dict = BF_SP_all_pairs(village)

    time_bf, time_gr, time_rd = [], [], []
    d_bf, d_gr, d_rd = [], [], []
    k_list = []

    for k in range(2, max_k+1):
        print(f'k={k} is being processed...')
        k_list.append(k)

        time_bf.append(timeit(lambda: d_bf.append(
            brute_force(village, list(village), k, village_distance_dict)[0]), number=1))

        time_gr.append(timeit(lambda: d_gr.append(
            greedy_algorithm(village, list(village), k, village_distance_dict)[0]), number=1))

        time_rd.append(timeit(lambda: d_rd.append(
            random_algorithm(village, list(village), k, village_distance_dict, random_trials)[0]), number=1))

    # --- Styles homogènes (couleurs/markers/linestyles)
    PAL = {
        "bf":   {"color": "#d62728", "marker": "s", "ls": "-",  "label": "brute force"},
        "gr":   {"color": "#1f77b4", "marker": "^", "ls": "-.", "label": "glouton"},
        "rand": {"color": "#ff7f0e", "marker": "o", "ls": "--", "label": "aléatoire"}
    }
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axT, axD = axes

    # --- Sous-graphe 1 : temps (échelle log)
    axT.set_yscale('log')
    axT.plot(k_list, time_bf,  PAL["bf"]["ls"],  color=PAL["bf"]["color"],
             marker=PAL["bf"]["marker"],  ms=6, lw=1.6)
    axT.plot(k_list, time_gr,  PAL["gr"]["ls"],  color=PAL["gr"]["color"],
             marker=PAL["gr"]["marker"],  ms=6, lw=1.6)
    axT.plot(k_list, time_rd,  PAL["rand"]["ls"], color=PAL["rand"]["color"],
             marker=PAL["rand"]["marker"], ms=6, lw=1.6)
    axT.set_ylabel("Temps T (log)")
    axT.grid(True, alpha=0.25)

    # --- Sous-graphe 2 : qualité d
    axD.plot(k_list, d_bf,  PAL["bf"]["ls"],  color=PAL["bf"]["color"],
             marker=PAL["bf"]["marker"],  ms=6, lw=1.6)
    axD.plot(k_list, d_gr,  PAL["gr"]["ls"],  color=PAL["gr"]["color"],
             marker=PAL["gr"]["marker"],  ms=6, lw=1.6)
    axD.plot(k_list, d_rd,  PAL["rand"]["ls"], color=PAL["rand"]["color"],
             marker=PAL["rand"]["marker"], ms=6, lw=1.6)
    axD.set_xlabel("k")
    axD.set_ylabel("Distance max d")
    axD.set_xticks(k_list)
    axD.grid(True, alpha=0.25)

    # --- Légende commune (une seule, centrée au-dessus des deux sous-graphes)
    handles = [
        Line2D([0], [0], color=PAL["bf"]["color"],  ls=PAL["bf"]["ls"],  marker=PAL["bf"]["marker"],  lw=1.6, ms=6, label=PAL["bf"]["label"]),
        Line2D([0], [0], color=PAL["gr"]["color"],  ls=PAL["gr"]["ls"],  marker=PAL["gr"]["marker"],  lw=1.6, ms=6, label=PAL["gr"]["label"]),
        Line2D([0], [0], color=PAL["rand"]["color"],ls=PAL["rand"]["ls"], marker=PAL["rand"]["marker"], lw=1.6, ms=6, label=PAL["rand"]["label"]),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))

    fig.tight_layout(rect=[0, 0, 1, 0.94])  # laisse de la place pour la légende
    plt.show()
    
