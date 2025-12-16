from SP import SP_naive
from SP import SP_heap

import random

def random_sparse_graph (n, step):
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
    graph = {f'{i}':{} for i in range(n)}

    for n1 in graph:
        for n2 in graph:
            if n2!= n1 and n2 not in graph[n1]:
                d = random.randint(1, d_max)
                graph[n1][n2] = d
                graph[n2][n1] = d

    return graph

import matplotlib.pyplot as plt
from timeit import timeit

def benchmark():
    Time_heap_sparse = []
    Time_naive_sparse = []
    Time_heap_dense = []
    Time_naive_dense = []
    n_list = []

    for N in range(10, 30):
        n = N * N
        n_list.append(n)

        # Graphes peu denses (grille N×N)
        graph_sparse = random_sparse_graph(n=N, step=100)
        Time_naive_sparse.append(
            timeit(lambda: SP_naive(graph_sparse, random.choice(list(graph_sparse))), number=N) / N
        )
        Time_heap_sparse.append(
            timeit(lambda: SP_heap(graph_sparse, random.choice(list(graph_sparse))), number=N) / N
        )

        # Graphes denses (complets)
        graph_dense = random_dense_graph(n=N * N, d_max=10000)
        Time_naive_dense.append(
            timeit(lambda: SP_naive(graph_dense, random.choice(list(graph_dense))), number=N) / N
        )
        Time_heap_dense.append(
            timeit(lambda: SP_heap(graph_dense, random.choice(list(graph_dense))), number=N) / N
        )

    # ----------- tracé : 2 sous-graphes -----------
    fig, (ax_all, ax_sparse) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Subplot 1 : vue d’ensemble
    ax_all.set_title("Comparaison UCS — liste vs tas (dense et sparse)")
    ax_all.set_ylabel("Temps moyen (s)")
    ax_all.plot(n_list, Time_naive_sparse, 'r^', label="naive sparse")
    ax_all.plot(n_list, Time_heap_sparse,  'b^', label="heap sparse")
    ax_all.plot(n_list, Time_naive_dense, 'r*', label="naive dense")
    ax_all.plot(n_list, Time_heap_dense,  'b*', label="heap dense")
    ax_all.legend(loc="best")
    ax_all.grid(True, alpha=0.25)

    # Subplot 2 : focus sur les courbes sparse
    ax_sparse.set_title("Focus : graphes peu denses (sparse)")
    ax_sparse.set_xlabel("|V| (= N×N)")
    ax_sparse.set_ylabel("Temps moyen (s)")
    ax_sparse.plot(n_list, Time_naive_sparse, 'r^-', label="naive sparse")
    ax_sparse.plot(n_list, Time_heap_sparse,  'b^-', label="heap sparse")
    ax_sparse.grid(True, alpha=0.25)

    # Fenêtre verticale resserrée autour des temps sparse
    ymin = min(Time_naive_sparse + Time_heap_sparse)
    ymax = max(Time_naive_sparse + Time_heap_sparse)
    margin = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ax_sparse.set_ylim(ymin - margin, ymax + margin)

    plt.tight_layout()
    plt.show()