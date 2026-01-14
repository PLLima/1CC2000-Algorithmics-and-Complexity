import random, math, statistics
from matplotlib import pyplot as plt
from timeit import timeit
import graderUtil

grader = graderUtil.Grader()
submission = grader.load('tdp_3')

assert submission is not None, "Failed to load submission"
# Set grader variable in submission for messages etc.
submission.grader = grader  # give access to grader inside submission


# Utility fuctions
compute_graph = submission.compute_graph
generate_random_coords = submission.generate_random_coords
compute_tour_length = submission.compute_tour_length

# student code
greedy = submission.greedy_tsp
bb = submission.branch_and_bound_tsp
prim = submission.prim_mst
dfs = submission.dfs
approx = submission.approximate_tsp
odd_degree = submission.find_odd_degree_vertices
sbg = submission.find_sbg
eulerian_tour = submission.build_eulerian_tour
hamiltonian_tour = submission.convert_to_hamiltonian_tour
christofides = submission.christofides_tsp

# test data
graph1 = {0:{1:1,2:3,3:6},
          1:{0:1,2:2,3:3},
          2:{0:3,1:2,3:1},
          3:{0:6,1:3,2:1}}

graph2 = {0: {1: 42, 2: 39, 3: 17, 4: 30, 5: 18, 6: 30, 7: 62, 8: 57, 9: 46},
          1: {0: 42, 2: 6, 3: 54, 4: 21, 5: 25, 6: 52, 7: 56, 8: 67, 9: 13},
          2: {0: 39, 1: 6, 3: 52, 4: 22, 5: 23, 6: 52, 7: 60, 8: 69, 9: 19},
          3: {0: 17, 1: 54, 2: 52, 4: 37, 5: 29, 6: 19, 7: 56, 8: 46, 9: 54},
          4: {0: 30, 1: 21, 2: 22, 3: 37, 5: 13, 6: 32, 7: 40, 8: 47, 9: 17},
          5: {0: 18, 1: 25, 2: 23, 3: 29, 4: 13, 6: 31, 7: 50, 8: 52, 9: 28},
          6: {0: 30, 1: 52, 2: 52, 3: 19, 4: 32, 5: 31, 7: 38, 8: 28, 9: 47},
          7: {0: 62, 1: 56, 2: 60, 3: 56, 4: 40, 5: 50, 6: 38, 8: 22, 9: 44},
          8: {0: 57, 1: 67, 2: 69, 3: 46, 4: 47, 5: 52, 6: 28, 7: 22, 9: 57},
          9: {0: 46, 1: 13, 2: 19, 3: 54, 4: 17, 5: 28, 6: 47, 7: 44, 8: 57}}


edges1 = [(0, 1), (1, 2), (2, 3)]

edges2 = [(0, 3), (0, 5), (5, 4), (4, 9), (9, 1), (1, 2), (3, 6), (6, 8), (8, 7)]

simple_coord = [(0,0),
                (1,0),
                (1,1),
                (0,1)]

eult1 = [1,2,3,1,2,4]
eult2 = [5,2,3,1,2,4,7,7,7,3,2,5,5,4,1,4,1]




# Benchmark functions
def benchmark_time(algos,nb_nodes_begin=4, nb_nodes_end=20, node_inc=1, nb_tries=10):

    n_list = []
    algo_time = {}
    algo_sol = {}
    t={}
    s={}
    for algo in algos:
        algo_name = algo.__name__
        algo_time[algo_name] = []
        algo_sol[algo_name] = []

    for nb_nodes in range(nb_nodes_begin,nb_nodes_end,node_inc):
        print(str(nb_nodes) + "/" + str(nb_nodes_end))
        n_list.append(nb_nodes)

        # IMPORTANT: accumulate across tries (moved OUT of the inner loop)
        for algo in algos:
            algo_name = algo.__name__
            t[algo_name] = []
            s[algo_name] = []

        for i in range(nb_tries):
            coords = generate_random_coords(nb_nodes)
            graph = compute_graph(coords)
            for algo in algos:
                algo_name = algo.__name__

                elapsed = timeit(lambda: s[algo_name].append(compute_tour_length(algo(graph),graph)), number=1)
                t[algo_name].append(elapsed)


        for algo in algos:
            algo_name = algo.__name__
            # Average over tries
            algo_time[algo_name].append(statistics.mean(t[algo_name]))
            algo_sol[algo_name].append(statistics.mean(s[algo_name]))


    # Time curves
    for algo in algos:
        algo_name = algo.__name__
        plt.plot(n_list, algo_time[algo_name], label=algo_name)
    plt.xlabel('nb_nodes')
    plt.ylabel('time (s)')
    plt.legend()

    plt.show()
    return

def benchmark_time_greedy():
    benchmark_time([greedy],nb_nodes_end=100,node_inc = 5)

def benchmark_time_bb():
    benchmark_time([bb],nb_nodes_end=12)

def benchmark_time_approx():
    benchmark_time([approx],nb_nodes_end=100,node_inc = 5)

def benchmark_time_christofides():
    benchmark_time([christofides],nb_nodes_end=100,node_inc = 5)

def benchmark_time_all():
    benchmark_time([greedy,approx,christofides],nb_nodes_end=100,node_inc = 5)




def benchmark_perf(algos,nb_nodes=10,nb_tries=100):
    algo_ratio = {}
    algo_ratio_l = {}
    algo_ratio_w = {}
    for algo in algos:
        algo_name = algo.__name__
        algo_ratio_l[algo_name] = []
        algo_ratio_w[algo_name] = 1

    for i in range(nb_tries):
        print(str(i) + "/" + str(nb_tries))
        coords = generate_random_coords(nb_nodes)
        graph = compute_graph(coords)
        val_opt = compute_tour_length(bb(graph),graph)
        for algo in algos:
            algo_name = algo.__name__
            val_algo = compute_tour_length(algo(graph),graph)
            ratio = val_algo/val_opt
            algo_ratio_l[algo_name].append(ratio)
            if ratio>algo_ratio_w[algo_name]:
                algo_ratio_w[algo_name] = ratio



    for algo in algos:
        algo_name = algo.__name__
        # Average over tries
        algo_ratio[algo_name] = statistics.mean(algo_ratio_l[algo_name])
        grader.addMessage(f"### {algo_name} ### average ratio: {algo_ratio[algo_name]:.2f}  worst ratio: {algo_ratio_w[algo_name]:.2f}")


def benchmark_perf_greedy():
    benchmark_perf([greedy])

def benchmark_perf_approx():
    benchmark_perf([approx])

def benchmark_perf_christofides():
    benchmark_perf([christofides])

def benchmark_perf_all():
    benchmark_perf([greedy,approx,christofides])

grader.addUtilityPart('b1', benchmark_perf_greedy, maxSeconds=180, description='benchmark perf of greedy')
grader.addUtilityPart('b2', benchmark_perf_approx, maxSeconds=180, description='benchmark perf of approx')
grader.addUtilityPart('b3', benchmark_perf_christofides, maxSeconds=180, description='benchmark perf of christofides')
grader.addUtilityPart('b4', benchmark_perf_all, maxSeconds=180, description='benchmark perf of all')

grader.addUtilityPart('t1', benchmark_time_greedy, maxSeconds=180, description='benchmark time of greedy')
grader.addUtilityPart('t2', benchmark_time_bb, maxSeconds=180, description='benchmark time of branch and bound')
grader.addUtilityPart('t3', benchmark_time_approx, maxSeconds=180, description='benchmark time of approx')
grader.addUtilityPart('t4', benchmark_time_christofides, maxSeconds=180, description='benchmark time of christofides')
grader.addUtilityPart('t5', benchmark_time_all, maxSeconds=180, description='benchmark time of all approximate algorithms')

# Helper functions for tests

# -------------------------
# Robust check helpers
# -------------------------

from collections import Counter, deque

def _undirected_edge(u, v):
    return (u, v) if u <= v else (v, u)

def _tour_is_permutation(tour, n):
    return isinstance(tour, list) and len(tour) == n and set(tour) == set(range(n))

def _canonical_cycle(tour):
    """
    Canonical form of a Hamiltonian cycle represented as a list of vertices visited once.
    We accept rotations and reverse direction as equivalent.

    Returns a canonical tuple.
    """
    n = len(tour)
    if n == 0:
        return tuple()

    # rotate so that smallest vertex is first
    m = min(tour)
    idxs = [i for i, v in enumerate(tour) if v == m]
    # choose best rotation among all occurrences of min (rare but safe)
    best = None
    for i0 in idxs:
        rot = tour[i0:] + tour[:i0]
        rot_rev = [rot[0]] + list(reversed(rot[1:]))  # keep first fixed, reverse direction
        cand = min(tuple(rot), tuple(rot_rev))
        best = cand if best is None else min(best, cand)
    return best

def require_valid_tour(grader, tour, graph, name="tour"):
    n = len(graph)
    if not _tour_is_permutation(tour, n):
        grader.fail(f"{name} must be a permutation of 0..{n-1}, got: {tour}")
    # also ensure all edges exist (complete graph assumed, but still)
    for i in range(n):
        u = tour[i]
        v = tour[(i+1) % n]
        if v not in graph.get(u, {}):
            grader.fail(f"{name} uses missing edge ({u},{v}) in graph")
    return True

def require_tour_cost_equal(grader, tour, graph, expected, name="tour"):
    val = compute_tour_length(tour, graph)
    grader.requireIsEqual(expected, val)
    return val

def require_tour_cost_leq(grader, tour, graph, bound, name="tour"):
    val = compute_tour_length(tour, graph)
    grader.requireIsLessThanOrEqual(bound, val)
    return val

def require_cycle_equivalent_if_wanted(grader, tour, expected_tour, name="tour"):
    """
    Optional: if you still want to accept multiple equivalent outputs,
    compare canonical forms (rotation + reverse).
    """
    if _canonical_cycle(tour) != _canonical_cycle(expected_tour):
        grader.fail(f"{name} is not equivalent to expected cycle up to rotation/reversal.\n"
                    f"Expected (canonical): {_canonical_cycle(expected_tour)}\n"
                    f"Got (canonical):      {_canonical_cycle(tour)}")

# ---- MST checks ----

class _DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True

def _mst_cost_kruskal(graph):
    n = len(graph)
    edges = []
    for u, nbrs in graph.items():
        for v, w in nbrs.items():
            if u < v:
                edges.append((w, u, v))
    edges.sort()
    dsu = _DSU(n)
    cost = 0
    cnt = 0
    for w, u, v in edges:
        if dsu.union(u, v):
            cost += w
            cnt += 1
            if cnt == n-1:
                break
    if cnt != n-1:
        raise ValueError("Graph is not connected; MST undefined.")
    return cost

def require_valid_mst(grader, mst_edges, graph, name="mst"):
    n = len(graph)
    if not isinstance(mst_edges, list):
        grader.fail(f"{name} must be a list of edges, got: {type(mst_edges)}")
    if len(mst_edges) != n - 1:
        grader.fail(f"{name} must have exactly n-1 edges ({n-1}), got {len(mst_edges)}")

    # Check edges exist + acyclic + connectivity
    dsu = _DSU(n)
    used = set()
    cost = 0
    for e in mst_edges:
        if not (isinstance(e, tuple) or isinstance(e, list)) or len(e) != 2:
            grader.fail(f"{name} edges must be pairs (u,v), got {e}")
        u, v = int(e[0]), int(e[1])
        uu, vv = _undirected_edge(u, v)
        if (uu, vv) in used:
            grader.fail(f"{name} contains duplicate edge {(uu, vv)}")
        used.add((uu, vv))
        if v not in graph.get(u, {}):
            grader.fail(f"{name} contains non-edge ({u},{v})")
        if not dsu.union(u, v):
            grader.fail(f"{name} contains a cycle (edge {u}-{v})")
        cost += graph[u][v]

    # connectivity: all nodes must be in same component
    root0 = dsu.find(0)
    for i in range(n):
        if dsu.find(i) != root0:
            grader.fail(f"{name} is not spanning/connected")
    
    # Verify the cost is minimal (actual MST check)
    expected_cost = _mst_cost_kruskal(graph)
    if cost != expected_cost:
        grader.fail(f"{name} has cost {cost} but minimum spanning tree has cost {expected_cost}")
    
    return cost

# ---- DFS order check (accepts any valid DFS preorder, not exact list) ----

def _edges_to_adjacency(edges):
    """Convert edge list to adjacency list (undirected)"""
    adj = {}
    for u, v in edges:
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        adj[u].append(v)
        adj[v].append(u)
    return adj

def require_valid_dfs_order(grader, order, edges, start=0, name="dfs"):
    adj = _edges_to_adjacency(edges)

    # IMPORTANT: len(adj) can be wrong if some vertices have degree 0.
    # Prefer n inferred from order (since you require a permutation anyway).
    if not isinstance(order, list) or len(order) == 0:
        grader.fail(f"{name} must be a non-empty list, got: {order}")
    n = len(order)

    if len(order) != n or set(order) != set(range(n)):
        grader.fail(f"{name} must be a permutation of 0..{n-1}, got: {order}")
    if order[0] != start:
        grader.fail(f"{name} must start at {start}, got {order[0]}")

    visited = {start}
    stack = [start]

    for i in range(1, n):
        next_v = order[i]
        if next_v in visited:
            grader.fail(f"{name} invalid: vertex {next_v} repeated at position {i}")

        # 1) Backtrack as DFS would: pop finished nodes
        while stack:
            top = stack[-1]
            if any(nei not in visited for nei in adj.get(top, [])):
                break
            stack.pop()

        if not stack:
            grader.fail(f"{name} invalid: traversal ended early before visiting {next_v}")

        # 2) Now next_v must be a neighbor of the CURRENT top
        top = stack[-1]
        if next_v not in adj.get(top, []):
            grader.fail(
                f"{name} invalid: expected next vertex at position {i} "
                f"to be an unvisited neighbor of {top}, got {next_v}"
            )

        visited.add(next_v)
        stack.append(next_v)

    return True

# ---- Graph (dict-of-dict) normalization for sbg etc ----

def _edge_set_from_adjdict(g):
    """
    Convert {u:{v:w}} to set of undirected weighted edges {(min,max,w)}.
    """
    out = set()
    for u, nbrs in g.items():
        for v, w in nbrs.items():
            if u < v:
                out.add((u, v, w))
    return out

def require_same_weighted_undirected_graph(grader, got, expected, name="graph"):
    if _edge_set_from_adjdict(got) != _edge_set_from_adjdict(expected):
        grader.fail(f"{name} differs.\nExpected edges: {_edge_set_from_adjdict(expected)}\n"
                    f"Got edges:      {_edge_set_from_adjdict(got)}")
    return True
# -------------------------
# ---- Edge trail check for build_eulerian_tour ----

def require_valid_edge_trail(grader, trail, edges, name="trail", must_be_closed=False):
    """
    trail: list of vertices
    edges: list of undirected edges (u,v) possibly with duplicates
    We check the trail uses each edge exactly once (as a multiset).
    For a non-eulerian input, this is still a valid test for "use all edges exactly once".
    """
    if not isinstance(trail, list) or len(trail) != len(edges) + 1:
        grader.fail(f"{name} must have length m+1 (m edges). Expected {len(edges)+1}, got {len(trail)}")

    mult = Counter((_undirected_edge(u, v) for (u, v) in edges))
    for i in range(len(trail) - 1):
        u, v = trail[i], trail[i+1]
        e = _undirected_edge(u, v)
        if mult[e] <= 0:
            grader.fail(f"{name} uses an edge {e} not available (or too many times)")
        mult[e] -= 1

    if any(mult.values()):
        grader.fail(f"{name} does not use all edges exactly once. Remaining: {mult}")

    if must_be_closed and trail[0] != trail[-1]:
        grader.fail(f"{name} must be a closed tour (start=end), got {trail[0]} != {trail[-1]}")
    
    return True

# ---- Hamiltonian conversion check ----

def require_hamiltonian_is_first_occurrences(grader, ham, euler, name="hamiltonian"):
    expected = list(dict.fromkeys(euler))
    if ham != expected:
        grader.fail(f"{name} should be the list of first occurrences in Eulerian walk.\n"
                    f"Expected: {expected}\nGot:      {ham}")
    return True
# -------------------------


# Question 1 : greedy implementation

def test_greedy():
    grader.addMessage('--- TEST GREEDY ---')

    sol1 = greedy(graph1)
    grader.addMessage('--- simple case : 4 nodes --- greedy solution {} of value {}'.format(sol1, compute_tour_length(sol1, graph1)))
    require_valid_tour(grader, sol1, graph1, "greedy(graph1)")
    require_tour_cost_equal(grader, sol1, graph1, 10, "greedy(graph1)")


    sol2 = greedy(graph2)
    grader.addMessage('--- more complex case : 10 nodes --- greedy solution {} of value {}'.format(sol2, compute_tour_length(sol2, graph2)))
    require_valid_tour(grader, sol2, graph2, "greedy(graph2)")
    require_tour_cost_equal(grader, sol2, graph2, 227, "greedy(graph2)")

    return grader

grader.addBasicPart('q1', test_greedy, 2, description='Test greedy implementation')


# Question 2 : branch and bound implementation

def test_bb():
    grader.addMessage('--- TEST BRANCH AND BOUND ---')

    sol1 = bb(graph1)
    grader.addMessage('--- simple case : 4 nodes --- bb solution {} of value {}'.format(sol1, compute_tour_length(sol1, graph1)))
    require_valid_tour(grader, sol1, graph1, "bb(graph1)")
    require_tour_cost_equal(grader, sol1, graph1, 8, "bb(graph1)")

    sol2 = bb(graph2)
    grader.addMessage('--- more complex case : 10 nodes --- bb solution {} of value {}'.format(sol2, compute_tour_length(sol2, graph2)))
    require_valid_tour(grader, sol2, graph2, "bb(graph2)")
    require_tour_cost_equal(grader, sol2, graph2, 202, "bb(graph2)")

    return grader


grader.addBasicPart('q2', test_bb, 2, description='Test branch and bound implementation')



# Question 3 : Prim implementation

def test_prim():
    grader.addMessage('--- TEST MST (Prim/Kruskal) ---')

    mst1 = prim(graph1)
    grader.addMessage(f'--- simple case : 4 nodes --- prim mst {mst1}')
    cost1 = require_valid_mst(grader, mst1, graph1, "mst(graph1)")
    grader.requireIsEqual(_mst_cost_kruskal(graph1), cost1)

    mst2 = prim(graph2)
    grader.addMessage(f'--- more complex case : 10 nodes --- prim mst {mst2}')
    cost2 = require_valid_mst(grader, mst2, graph2, "mst(graph2)")
    grader.requireIsEqual(_mst_cost_kruskal(graph2), cost2)

    return grader

grader.addBasicPart('q3', test_prim, 2, description='Test Prim implementation')


# Question 4 : DFS implementation

def test_dfs():
    grader.addMessage('--- TEST DFS ---')

    sol1 = dfs(edges1)
    grader.addMessage(f'--- simple case : 3 edges --- dfs solution {sol1}')
    grader.requireIsTrue(require_valid_dfs_order(grader, sol1, edges1, start=0, name="dfs(edges1)"))
    
    sol2 = dfs(edges2)
    grader.addMessage(f'--- more complex case : 9 edges --- dfs solution {sol2}')
    grader.requireIsTrue(require_valid_dfs_order(grader, sol2, edges2, start=0, name="dfs(edges2)"))

    return grader

grader.addBasicPart('q4', test_dfs, 2, description='Test DFS implementation')


# Question 5 : 2 Approx implementation

def test_approx():
    grader.addMessage('--- TEST 2 APPROX ---')


    approxSol1 = approx(graph1)
    val_approxSol1 = compute_tour_length(approxSol1,graph1)
    grader.addMessage('--- simple case : 4 nodes --- approx solution {} of value {}'.format(approxSol1, val_approxSol1))
    require_valid_tour(grader, approxSol1, graph1, "approx(graph1)")
    require_tour_cost_equal(grader, approxSol1, graph1, 10, "approx(graph1)")


    approxSol2 = approx(graph2)
    val_approxSol2 = compute_tour_length(approxSol2,graph2)
    grader.addMessage('--- more complex case : 10 nodes --- approx solution {} of value {}'.format(approxSol2, val_approxSol2))
    require_valid_tour(grader, approxSol2, graph2, "approx(graph2)")
    sol2 = bb(graph2)
    opt = compute_tour_length(sol2,graph2)
    # opt = 202  # si tu veux réutiliser la valeur du test BB
    require_tour_cost_leq(grader, approxSol2, graph2, 2 * opt, "approx(graph2)")
    return grader  


grader.addBasicPart('q5', test_approx, 2, description='Test approx implementation')


# Question 6 : find odd degree vertices

def test_odd_degree():
    grader.addMessage('--- TEST ODD DEGREE VERTICES ---')

    odd_degreeSol1 = odd_degree(edges1)
    grader.addMessage('--- simple case : 3 edges --- odd_degree solution {}'.format(odd_degreeSol1))
    grader.requireIsEqual(set([0, 3]), set(odd_degreeSol1))

    odd_degreeSol2 = odd_degree(edges2)
    grader.addMessage('--- simple case : 9 edges --- odd_degree solution {}'.format(odd_degreeSol2))
    grader.requireIsEqual(set([2, 7]), set(odd_degreeSol2))

    return grader  


grader.addBasicPart('q6', test_odd_degree, 2, description='Test odd_degree implementation')


# Question 7 : find sub graph
def test_sbg():
    grader.addMessage('--- TEST FIND SUB GRAPH ---')

    odd1 = [0, 3]
    got1 = sbg(odd1, graph1)
    exp1 = {0: {3: 6}, 3: {0: 6}}
    grader.addMessage(f'--- simple case : 4 nodes --- sbg solution {got1}')
    grader.requireIsTrue(require_same_weighted_undirected_graph(grader, got1, exp1, "sbg(graph1)"))

    odd2 = [2, 7]
    got2 = sbg(odd2, graph2)
    exp2 = {2: {7: 60}, 7: {2: 60}}
    grader.addMessage(f'--- more complex case : 10 nodes --- sbg solution {got2}')
    grader.requireIsTrue(require_same_weighted_undirected_graph(grader, got2, exp2, "sbg(graph2, odd=[2,7])"))

    odd3 = [2, 7, 3, 4]
    got3 = sbg(odd3, graph2)
    exp3 = {
        2: {7: 60, 3: 52, 4: 22},
        7: {2: 60, 3: 56, 4: 40},
        3: {2: 52, 7: 56, 4: 37},
        4: {2: 22, 7: 40, 3: 37},
    }
    grader.addMessage(f'--- more complex case 2 : 10 nodes --- sbg solution {got3}')
    grader.requireIsTrue(require_same_weighted_undirected_graph(grader, got3, exp3, "sbg(graph2, odd=[2,7,3,4])"))

    return grader  


grader.addBasicPart('q7', test_sbg, 2, description='Test sbg implementation')


# Question 8 : build eulerian tour

def test_eulerian_tour():
    grader.addMessage('--- TEST EULERIAN TOUR / TRAIL ---')

    trail1 = eulerian_tour(edges1)
    grader.addMessage(f'--- simple case : 3 edges --- eulerian_tour solution {trail1}')
    grader.requireIsTrue(require_valid_edge_trail(grader, trail1, edges1, name="build_eulerian_tour(edges1)", must_be_closed=False))

    trail2 = eulerian_tour(edges2)
    grader.addMessage(f'--- more complex case : 9 edges --- eulerian_tour solution {trail2}')
    grader.requireIsTrue(require_valid_edge_trail(grader, trail2, edges2, name="build_eulerian_tour(edges2)", must_be_closed=False))
    return grader


grader.addBasicPart('q8', test_eulerian_tour, 2, description='Test eulerian_tour implementation')


# Question 9 : build hamiltonian tour

def test_hamiltonian_tour():
    grader.addMessage('--- TEST HAMILTONIAN TOUR ---')

    sol1 = hamiltonian_tour(eult1)
    grader.addMessage(f'--- simple case --- hamiltonian_tour solution {sol1}')
    grader.requireIsTrue(require_hamiltonian_is_first_occurrences(grader, sol1, eult1, name="hamiltonian(eult1)"))

    sol2 = hamiltonian_tour(eult2)
    grader.addMessage(f'--- more complex case --- hamiltonian_tour solution {sol2}')
    grader.requireIsTrue(require_hamiltonian_is_first_occurrences(grader, sol2, eult2, name="hamiltonian(eult2)"))
    return grader

grader.addBasicPart('q9', test_hamiltonian_tour, 2, description='Test hamiltonian_tour implementation')


# Question 10 : Christofides

def test_christofides():
    grader.addMessage('--- TEST CHRISTOFIDES ---')

    sol1 = christofides(graph1)
    grader.addMessage(f'--- simple case : 4 nodes --- christofides solution {sol1} of value {compute_tour_length(sol1, graph1)}')
    require_valid_tour(grader, sol1, graph1, "christofides(graph1)")
    require_tour_cost_equal(grader, sol1, graph1, 10, "christofides(graph1)")

    sol2 = christofides(graph2)
    grader.addMessage(f'--- more complex case : 10 nodes --- christofides solution {sol2} of value {compute_tour_length(sol2, graph2)}')
    require_valid_tour(grader, sol2, graph2, "christofides(graph2)")
    require_tour_cost_equal(grader, sol2, graph2, 213, "christofides(graph2)")

    return grader

grader.addBasicPart('q10', test_christofides, 2, description='Test christofides implementation')

if __name__ == "__main__":
    grader.grade()
