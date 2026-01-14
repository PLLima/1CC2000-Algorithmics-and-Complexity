# TP3 - problème du voyageur de commerce (TSP)

# importations
import networkx as nx
import random
from math import dist

grader = None

# compute the matrix of Euclidean distances between cities
def compute_graph(coords):
    n = len(coords)
    graph = {}
    for i in range(n):
        graph[i] = {}
        for j in range(n):
            if i != j:
                graph[i][j] = dist(coords[i],coords[j])
    return graph


# generate a list of random coordinates
def generate_random_coords(n, x_max=100, y_max=100):
    coords = []
    for _ in range(n):
        coord = (random.randint(0,x_max),random.randint(0,y_max))
        coords.append(coord)
    return coords

# compute the total length of the tour
def compute_tour_length(tour, graph):
    length = 0
    for i in range(len(tour)):
        length += graph[tour[i]][tour[(i + 1) % len(tour)]]
    return length



# greedy heuristic to solve the TSP
def greedy_tsp(graph):
    n = len(graph)
    unvisited = set(range(n))
    current_city = 0
    tour = [current_city]
    unvisited.remove(current_city)

    while unvisited:
        nearest_city = min(unvisited, key=lambda city: graph[current_city][city])
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city

    return tour




# branch and bound algorithm to solve the TSP
def branch_and_bound_tsp(graph):
    n = len(graph)
    best_tour = list(range(n))
    best_length = compute_tour_length(best_tour, graph)

    def bb_rec(current_tour, current_length, visited):
        nonlocal best_tour, best_length

        if current_length >= best_length:
            return

        if len(visited) == n:
            final_length = current_length + graph[current_tour[-1]][current_tour[0]]
            if final_length < best_length:
                best_length = final_length
                best_tour = list(current_tour)
            return

        last_city = current_tour[-1]
        for next_city in graph[last_city]:
            if next_city not in visited:
                bb_rec(current_tour + [next_city], 
                       current_length + graph[last_city][next_city], 
                       visited | {next_city})

    bb_rec([0], 0, {0})
    return best_tour




# build a minimum spanning tree (MST) using Prim's algorithm
def prim_mst(graph):
    n = len(graph)
    edges = []
    visited = {0}
    
    while len(visited) < n:
        min_weight = float('inf')
        min_edge = None
        
        for u in visited:
            for v, weight in graph[u].items():
                if v not in visited:
                    if weight < min_weight:
                        min_weight = weight
                        min_edge = (u, v)
        
        if min_edge:
            u, v = min_edge
            edges.append((u, v))
            visited.add(v)
        else:
            break
            
    return edges



# perform a depth-first search (DFS) to obtain the list of vertices without duplicates
def dfs(edges):
    tour = []

    def dfs_rec(edge_list, city):
        nonlocal tour
        tour.append(city)

        neighbors = []
        for u, v in edge_list:
            if u == city:
                neighbors.append(v)
            elif v == city:
                neighbors.append(u)
        
        neighbors.sort()

        for neighbor in neighbors:
            if neighbor not in tour:
                dfs_rec(edge_list, neighbor)

    dfs_rec(edges,0)
    return tour


#  approximation algorithm to solve the TSP
def approximate_tsp(graph):

    # calculate an MST of the graph
    mst_edges = prim_mst(graph)

    # perform a depth-first search to obtain the list of vertices
    tour = dfs(mst_edges)

    return tour





# compute the odd degree vertices of a graph
def find_odd_degree_vertices(edges):
    degree = {}

    for u, v in edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    
    odd_vertices = [v for v, d in degree.items() if d % 2 != 0]

    return odd_vertices
    
# compute a subgraph from a graph and a list of vertices
def find_sbg(lst_vertices, graph):
    sbg = {}

    for u in lst_vertices:
        sbg[u] = {}
        for v in lst_vertices:
            if u != v and v in graph[u]:
                sbg[u][v] = graph[u][v]

    return sbg

#  find the minimum weight perfect matching of a graph
# returns a list of edges
def find_perfect_matching(graph):
    G = nx.Graph()
    edges = []
    max_val = 0
    for u in graph:
        for v in graph[u]:
            if graph[u][v] > max_val:
                max_val = graph[u][v]
    for u in graph:
        for v in graph[u]:
            edges.append((u,v,max_val - graph[u][v]))
    G.add_weighted_edges_from(edges)
    match = nx.max_weight_matching(G) # the function finds the maximum weight perfect matching, hence the inversion of weights
    return list(match)


# build an Eulerian tour from a list of edges
def build_eulerian_tour(edges):
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)

    # Find starting vertex: if there are odd-degree vertices, start at one of them
    # Otherwise start at any vertex (for Eulerian circuit)
    odd_vertices = find_odd_degree_vertices(edges)
    start_vertex = odd_vertices[0] if odd_vertices else edges[0][0]
    
    stack = [start_vertex]
    tour = []
  
    while stack:
        u = stack[-1]
        
        if u in graph and graph[u]:
            v = graph[u][0]
            
            graph[u].remove(v)
            graph[v].remove(u)
            
            stack.append(v)
        else:
            tour.append(stack.pop())

    return tour[::-1]

# build a Hamiltonian tour from an Eulerian tour
def convert_to_hamiltonian_tour(eulerian_tour):
    visited = set()
    hamiltonian_tour = []

    for city in eulerian_tour:
        if city not in visited:
            hamiltonian_tour.append(city)
            visited.add(city)

    return hamiltonian_tour
   


# Christophides' approximation algorithm to solve the TSP
def christofides_tsp(graph):
    
    # Step 1: Find a minimum spanning tree (MST)
    mst_edges = prim_mst(graph)

    # Step 2: Find the odd degree vertices
    odd_degree_vertices = find_odd_degree_vertices(mst_edges)

    # Step 3: Find the subgraph corresponding to the odd degree vertices
    sbg_odd = find_sbg(odd_degree_vertices, graph)

    # Step 4: Find a perfect matching on the subgraph
    matching = find_perfect_matching(sbg_odd)

    # Step 5: Build the Eulerian tour
    eulerian_tour = build_eulerian_tour(mst_edges+matching)

    # Step 6: Convert the Eulerian tour to a Hamiltonian tour
    hamiltonian_tour = convert_to_hamiltonian_tour(eulerian_tour)

    return hamiltonian_tour




