import random
import numpy as np

grader = None  # will be set by grader

# a small instance of Knapsack with 4 objects 
W4 = 5                                                    # weight limit of the knapsack
O4 = { 'o1': {'w': 1, 'v': 10}, 'o2': {'w': 2, 'v': 20},       # 'w' for weight and 'v' for value
       'o3': {'w': 3, 'v': 15}, 'o4': {'w': 4, 'v': 20} }


# a bigger instance of  Knapsack with 15 objects
W15 = 750                             
O15 = { 'o1':  {'w': 70,  'v': 135}, 'o2':  {'w': 73,  'v': 139}, 'o3':  {'w': 77,  'v': 149}, 
        'o4':  {'w': 80,  'v': 150}, 'o5':  {'w': 82,  'v': 156}, 'o6':  {'w': 87,  'v': 163}, 
        'o7':  {'w': 90,  'v': 173}, 'o8':  {'w': 94,  'v': 184}, 'o9':  {'w': 98,  'v': 192}, 
        'o10': {'w': 106, 'v': 201}, 'o11': {'w': 110, 'v': 210}, 'o12': {'w': 113, 'v': 214}, 
        'o13': {'w': 115, 'v': 221}, 'o14': {'w': 118, 'v': 229}, 'o15': {'w': 120, 'v': 240} }

def generate_random_instance(nb_objects, max_weight, max_value):
    return {
        f'o{i}': {'w': random.randint(1, max_weight), 'v': random.randint(1, max_value)}
        for i in range(1, nb_objects + 1)
    }

# Example
# random_instance = generate_random_instance(30, 120, 240)

# Question 1: Backtracking implementation

def backtracking(O_dict, W) :    
    global bestSol, objs_list, nodes_explored

    # initialization
    bestSol = {'selected': set(), 'weight': 0, 'score': 0}
    objs_list = sorted(O_dict.keys()) # lexicographical order on objects
    nodes_explored = 0

    # computes the children of the current partial solution
    def children(curParSol):
        global objs_list

        childrenSols = []
        currentObjs = curParSol['selected']
        start_index = curParSol['index']
        for i in range(start_index, len(objs_list)):
            obj = objs_list[i]
            if obj not in currentObjs:
                newWeight = curParSol['weight'] + O_dict[obj]['w']
                if newWeight <= W:
                    newSelected = currentObjs.union({obj})
                    newScore = curParSol['score'] + O_dict[obj]['v']
                    childSol = {
                        'selected': newSelected,
                        'index': i + 1,
                        'weight': newWeight,
                        'score': newScore
                    }
                    childrenSols.append(childSol)
        return childrenSols

    # a partial solution is terminal when it has no children
    def terminal(curParSol):
        return len(children(curParSol)) == 0

    def backtracking_rec(curParSol) :
        global objs_list, bestSol, nodes_explored

        nodes_explored += 1
        if terminal(curParSol):
            if curParSol['score'] > bestSol['score']: # search for a minimum
                bestSol = curParSol
        else:
            for childParSol in children(curParSol):
                backtracking_rec(childParSol)

    # call backtracking on the root node
    rootSol = {'selected': set() , 'index': 0, 'weight': 0, 'score': 0}
    backtracking_rec(rootSol)

    # return the best found solution (the index entry is no more relevent)
    if 'index' in bestSol: 
        bestSol.pop('index')
    bestSol['nodes'] = nodes_explored
    return bestSol

# Question 2 : Greedy implementation

def greedy(O_dict, W):

    # initialization
    sol = {'selected': set() , 'weight': 0, 'score': 0}

    sorted_objs = sorted(O_dict.keys(), 
                         key=lambda x: O_dict[x]['v'] / O_dict[x]['w'], 
                         reverse=True)

    for obj in sorted_objs:
        obj_weight = O_dict[obj]['w']
        obj_value = O_dict[obj]['v']
        
        if sol['weight'] + obj_weight <= W:
            sol['selected'].add(obj)
            sol['weight'] += obj_weight
            sol['score'] += obj_value

    return sol


# Question 3: Trap instance

def trap_instance(k):
    O, W = {}, 0

    W = 2 * k
    O = {
        'o1': {'w': 1, 'v': 2},
        'o2': {'w': W, 'v': W}
    }

    return O, W

# Question 4 : Dynamic Programming implementation

def dynprog(O_dict, W):

    n = len(O_dict)
    objs_list = sorted(O_dict.keys())

    # FILLING THE TABLE ITERATIVELY
    V = np.zeros((n+1, W+1), dtype=int)

    for i in range(1, n+1):
        obj_name = objs_list[i-1]
        w_i = O_dict[obj_name]['w']
        v_i = O_dict[obj_name]['v']
        for j in range(W+1):
            if(i == 0 or j == 0):
                V[i][j] = 0
            elif(w_i > j):
                V[i][j] = V[i-1][j]
            elif(w_i <= j):
                V[i][j] = max(V[i-1][j], V[i-1][j - w_i] + v_i)

    # RETRIEVE THE SOLUTION
    selected = set()
    w = W
    for i in range(n, 0, -1):
        if V[i, w] != V[i-1, w]:
            obj_name = objs_list[i-1]
            selected.add(obj_name)
            w -= O_dict[obj_name]['w']

    return {'selected': selected, 'score': V[n,W]}

# Question 5 : Branch-and-Bound implementation with fractional relaxation

def knapsack_bb_fractional(O_dict, W):
    """
    Branch and Bound with fractional relaxation using priority queue.
    State is a dict:
        {'selected': set[str], 'index': int, 'weight': int, 'score': int}

    Returns {'selected': set(...), 'score': int, 'nodes': int}
    
    This is a MAXIMIZATION problem (we want to maximize value).
    We use negative values to work with a min-heap as if it were a max-heap.
    """
    import heapq
    from dataclasses import dataclass, field
    from typing import Any, Optional, TypedDict, Set, Tuple
    from math import inf
    
    # Sort objects by value/weight ratio (descending)
    objs_list = sorted(O_dict.keys(),
                       key=lambda o: O_dict[o]['v'] / O_dict[o]['w'],
                       reverse=True)
    n = len(objs_list)
    
    # State representation: (index, weight, value, selected_set)
    # where index is the next item to consider
    class State(TypedDict):
        selected: Set[str]
        index: int
        weight: int
        score: int        

    @dataclass(order=True)
    class Node:
        # For maximization: store negative upper bound so heapq pops the best bound first
        priority: float
        cand: Any = field(compare=False)   # the state dict

    nodes_explored = [0]    # Counter for statistics

    def fractional_upper_bound(state: State) -> float:
        """
        Compute upper bound using fractional knapsack relaxation.
        Returns the maximum value we could possibly achieve from this state.
        """
        index = state['index']
        current_weight = state['weight']
        current_value = state['score']

        if current_weight > W:
            return -inf  # Invalid state

        remaining_capacity = W - current_weight
        bound = float(current_value)

        for i in range(index, n):
            obj = objs_list[i]
            w_i = O_dict[obj]['w']
            v_i = O_dict[obj]['v']

            if w_i <= remaining_capacity:
                bound += v_i
                remaining_capacity -= w_i
            else:
                bound += v_i * (remaining_capacity / w_i)
                break

        return bound

    def is_leaf(state: State) -> bool:
        """Check if state represents a complete solution."""
        return state['index'] >= n

    def partition(state: State):
        """
        Generate children states: include or exclude the next object.
        **Yields** valid child states.
        """
        index = state['index']
        if index >= n:
            return  # No children for leaf nodes

        obj = objs_list[index]
        obj_weight = O_dict[obj]['w']
        obj_value = O_dict[obj]['v']

    ############### TODO : complete code ####################        
        
    ###############################################################


    def branch_and_bound(C0: State, f, is_leaf, partition) -> Optional[Tuple[State, float]]:
        """
        Generic branch and bound algorithm.
        For MAXIMIZATION: f returns upper bound, we use negative values in heap.
        Returns (solution_state, actual_value) if found, else None.
        """
        OPEN = []
        heapq.heappush(OPEN, Node(-f(C0), C0))  # Negative for max-heap behavior

        best_cost = -inf  # For maximization: best value found so far
        best_sol: Optional[State] = None

        while OPEN:
            # If the best upper bound is worse than our current best, stop
            if OPEN[0].priority >= -best_cost:  # priority is negative upper bound
                break

    ############### TODO : complete code ####################        
            
    ###############################################################

        return (best_sol, best_cost) if best_sol is not None else None

    # Initial state: no items processed, no weight, no value, empty selection
    C0 = {'selected': set(), 'index': 0, 'weight': 0, 'score': 0}

    result = branch_and_bound(C0, fractional_upper_bound, is_leaf, partition)

    if result is None:
        return {'selected': set(), 'score': 0, 'nodes': nodes_explored[0]}

    best_state, best_value = result
    return {'selected': set(best_state['selected']),
            'score': int(best_value),
            'nodes': nodes_explored[0]}

# Question 6 : Meet-in-the-Middle implementation
       
def knapsack_meet_in_the_middle(O_dict, W):
    """Retourne {'selected': set(...), 'score': int}"""

    from itertools import combinations
    import bisect

    objs = sorted(O_dict)
    n = len(objs)
    half = n // 2
    A = objs[:half]
    B = objs[half:]

    def subsets(objs):
        """
        Generate all subsets of the given list of objects.
        
        :param objs: List of object keys
        :return: Generator of (weight, value, selected_set) tuples
        """
        for r in range(len(objs) + 1):
            for combo in combinations(objs, r):
                w = sum(O_dict[o]['w'] for o in combo)
                v = sum(O_dict[o]['v'] for o in combo)
                yield (w, v, set(combo))

    A_sub = list(subsets(A))
    B_sub = list(subsets(B))

    # Tri par poids
    B_sub.sort(key=lambda x: x[0])

    # --- IMPORTANT: remove dominated pairs ---
    # cleaned_B holds the (w, v, S) from B_sub that are not dominated
    cleaned_B = []
    max_v = -1

    ############### TODO : complete code ####################        

    ###############################################################

    # Weights only for binary search
    B_weights = [w for w, v, S in cleaned_B]

    best_value = 0
    best_comb = set()

    # Optimal search
    # Use bisect.bisect_right: this function returns the index where to insert an element using binary search
    ############### TODO : complete code ####################        

    ###############################################################
    return {'selected': best_comb, 'score': best_value}
