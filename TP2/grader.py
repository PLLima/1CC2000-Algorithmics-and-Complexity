import random, math, statistics
from matplotlib import pyplot as plt
from timeit import timeit
import graderUtil

grader = graderUtil.Grader()
submission = grader.load('tdp_2')

assert submission is not None, "Failed to load submission"
# Set grader variable in submission for messages etc.
submission.grader = grader  # give access to grader inside submission

# Access the variables
O4 = submission.O4
W4 = submission.W4
O15 = submission.O15
W15 = submission.W15

greedy = submission.greedy
backtracking = submission.backtracking
trap_instance = submission.trap_instance
dynprog = submission.dynprog
knapsack_bb_fractional = submission.knapsack_bb_fractional
knapsack_meet_in_the_middle = submission.knapsack_meet_in_the_middle
generate_random_instance = submission.generate_random_instance

# Question 1 : backtracking implementation

def test_backtracking():
    grader.addMessage('--- TEST BACKTRACKING ---')

    bestSol4 = backtracking(O4, W4)
    grader.addMessage(f"--- simple case : 4 objects --- best found solution {bestSol4['selected']} "
                 f"of value {bestSol4['score']} ({bestSol4['nodes']} nodes explored)")
    grader.requireIsEqual(35, bestSol4['score'])
    grader.requireIsEqual({'o2', 'o3'}, bestSol4['selected'])
    grader.requireIsLessThanOrEqual(23, bestSol4['nodes'])

    bestSol15 = backtracking(O15, W15)
    grader.addMessage(f"--- bigger case : 15 objects --- best found solution {bestSol15['selected']} "
                 f"of value {bestSol15['score']} ({bestSol15['nodes']} nodes explored)")
    grader.requireIsEqual(1458, bestSol15['score'])
    grader.requireIsEqual({'o1', 'o3', 'o5', 'o7', 'o8', 'o9', 'o14', 'o15'}, bestSol15['selected'])
    grader.requireIsLessThanOrEqual(43051, bestSol15['nodes'])

    return grader  # so callers can inspect messages/points if desired


grader.addBasicPart('q1', test_backtracking, 2, description='Test backtracking implementation')

# Question 2 : greedy implementation

def test_greedy():
    grader.addMessage('--- TEST GREEDY ---')

    bestGreedySol4 = greedy(O4, W4)
    grader.addMessage('--- simple case : 4 objects --- best greedy solution {} of value {}'.format(bestGreedySol4['selected'], bestGreedySol4['score']))
    grader.requireIsEqual(30, bestGreedySol4['score'])
    grader.requireIsEqual({'o2', 'o1'}, bestGreedySol4['selected'])

    bestGreedySol15 = greedy(O15, W15)
    grader.addMessage('--- bigger case : 15 objects --- best greedy solution {} of value {}'.format(bestGreedySol15['selected'], bestGreedySol15['score']))
    grader.requireIsEqual(1441, bestGreedySol15['score'])
    grader.requireIsEqual({'o1', 'o2', 'o3', 'o7', 'o8', 'o9', 'o14', 'o15'}, bestGreedySol15['selected'])

    return grader

grader.addBasicPart('q2', test_greedy, 2, description='Test greedy implementation')

# Benchmark

def benchmark(algo1=backtracking, algo2=greedy, start=10, end=21, num_instances=10):
    print('--- BENCHMARK ---')
    algo1_time, algo2_time = [], []
    algo1_sol, algo2_sol = [], []
    n_list = []
    
    W = 1000
    max_value = 100
    max_weight = 250

    for n in range(start, end):
        n_list.append(n)

        nb_tries = num_instances  # reduce variance

        # IMPORTANT: accumulate across tries (moved OUT of the inner loop)
        t1, t2, s1, s2 = [], [], [], []

        for i in range(nb_tries):
            print(f'n = {n}, try = {i+1}')
            O_dict = generate_random_instance(n, max_weight, max_value)
            # Measure algo1
            elapsed1 = timeit(lambda: s1.append(int(algo1(O_dict, W)['score'])), number=1)
            t1.append(elapsed1)

            # Measure algo2
            elapsed2 = timeit(lambda: s2.append(int(algo2(O_dict, W)['score'])), number=1)
            t2.append(elapsed2)


        # print(f'  {algo1.__name__}:\n\tinstances={instances}\n\ts1={s1}\n\ts2={s2}')
        # Average over tries
        algo1_time.append(statistics.mean(t1))
        algo2_time.append(statistics.mean(t2))
        # Use mean for readability (ratio is unchanged either way)
        algo1_sol.append(statistics.mean(s1))
        algo2_sol.append(statistics.mean(s2))

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Value ratio (algo2 vs algo1)
    axs[0].plot(n_list, [v2 / v1 for v1, v2 in zip(algo1_sol, algo2_sol)], 'g-')
    axs[0].set_title(f'{algo2.__name__} value / {algo1.__name__} value')
    axs[0].set_xlabel('n')
    axs[0].set_ylabel('value ratio')

    # Time curves
    axs[1].plot(n_list, algo1_time, 'r^', label=algo1.__name__)
    axs[1].plot(n_list, algo2_time, 'b^', label=algo2.__name__)
    axs[1].set_xlabel('n')
    axs[1].set_ylabel('time (s)')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Benchmark 1

grader.addUtilityPart('b1', benchmark, maxSeconds=180, description='Benchmark backtracking/greedy implementations')

# Question 3: Trap instance

def test_trap():
    grader.addMessage('--- TEST TRAP ---')
    O, W = trap_instance(10)
    grader.requireIsGreaterThan(0, W)
    grader.requireIsTrue(len(O) > 0)
    grader.requireIsTrue(greedy(O, W)['score'] * 10 <= backtracking(O, W)['score'])

grader.addBasicPart('q3', test_trap, 2, description='Test trap instance performance')

# Question 4: Dynamic Programming implementation

def test_dynprog():
    grader.addMessage('--- DYNAMIC PROGRAMMING ---')

    # 4-object instance
    bestSol4 = dynprog(O4, W4)
    grader.addMessage(f"4 objects → selected={bestSol4['selected']} value={bestSol4['score']}")
    grader.requireIsEqual(35, bestSol4['score'])
    grader.requireIsEqual({'o2', 'o3'}, bestSol4['selected'])

    # 15-object instance
    bestSol15 = dynprog(O15, W15)
    grader.addMessage(f"15 objects → selected={bestSol15['selected']} value={bestSol15['score']}")
    grader.requireIsEqual(1458, bestSol15['score'])
    grader.requireIsEqual({'o1', 'o3', 'o5', 'o7', 'o8', 'o9', 'o14', 'o15'}, bestSol15['selected'])

    # Random consistency check vs backtracking
    grader.addMessage('Random 20-object instance → compare dynprog vs backtracking values')
    O = generate_random_instance(20, 100, 250)
    W = 1000
    bt = backtracking(O, W)
    dp = dynprog(O, W)
    grader.addMessage(f"backtracking={bt['score']} dynprog={dp['score']}")
    grader.requireIsEqual(bt['score'], dp['score'])

grader.addBasicPart('q4', test_dynprog, 5, description='Test dynamic programming implementation')

def benchmark_dynprog():
    """
    Benchmark backtracking vs dynamic programming on random instances.
    """
    benchmark(algo1=backtracking, algo2=dynprog)

grader.addUtilityPart('b2', benchmark_dynprog, maxSeconds=360, description='Benchmark backtracking/dynprog implementations')

# Question 5

def test_bb_fractional():
    grader.addMessage('--- TEST BRANCH AND BOUND WITH FRACTIONAL RELAXATION ---')

    result4 = knapsack_bb_fractional(O4, W4)
    grader.addMessage(f"--- simple case : 4 objects --- Best solution: {result4['selected']} with value {result4['score']} explored {result4['nodes']} nodes")
    grader.requireIsEqual(result4['score'], 35)
    grader.requireIsEqual(result4['selected'], {'o2', 'o3'})
    grader.requireIsLessThanOrEqual(9, result4['nodes'])

    result15 = knapsack_bb_fractional(O15, W15)
    grader.addMessage(f"--- bigger case : 15 objects --- Best solution: {result15['selected']} with value {result15['score']} explored {result15['nodes']} nodes")
    grader.requireIsEqual(result15['score'], 1458)
    grader.requireIsEqual(result15['selected'], {'o1', 'o3', 'o5', 'o7', 'o8', 'o9', 'o14', 'o15'})
    grader.requireIsLessThanOrEqual(result15['nodes'], 104)

grader.addBasicPart('q5', test_bb_fractional, 5, description='Test branch-and-bound implementation')

# Benchmark 3

def benchmark_bb_fractional():
    """
    Benchmark backtracking vs branch-and-bound (fractional relaxation) on random instances.
    Compares both implementations on:
      - execution time (seconds)
      - number of explored nodes
    Averages are computed ONLY over successful runs (instances without timeout).
    Plots two curves (time and nodes) and returns a results dict.
    """
    from timeit import timeit

    def mean_valid(xs):
            xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
            return (sum(xs) / len(xs)) if xs else float('nan')
    random_seed_base = 42

    W = 1000
    max_value = 100
    max_weight = 250

    # Results structure -----------------------------------------------
    results = {
        'n': [],
        'backtracking': {'time': [], 'nodes': [], 'score': []},
        'bb_fractional': {'time': [], 'nodes': [], 'score': []},
        'counts': {'bt_runs': [], 'bb_runs': []}  # how many successful runs per n
    }

    num_instances = 10

    # Main loop --------------------------------------------------------
    for n in range(10, 21):
        print(f'\n=== n={n} (averaging over {num_instances} instances) ===')
        results['n'].append(n)

        bt_times, bt_nodes, bt_scores = [], [], []
        bb_times, bb_nodes, bb_scores = [], [], []

        for inst in range(num_instances):
            # Deterministic seed per (n, inst) for reproducibility
            random.seed(random_seed_base * n + inst)

            O_dict = generate_random_instance(n, max_weight, max_value)

            # Backtracking
            bt_sol = [None]  # Use list to capture result in lambda
            bt_t = timeit(lambda: bt_sol.__setitem__(0, backtracking(O_dict, W)), number=1)
            bt_sol = bt_sol[0]
            if bt_sol is not None:
                bt_times.append(bt_t)
                bt_nodes.append(bt_sol.get('nodes', float('nan')))
                bt_scores.append(bt_sol.get('score', float('nan')))
            else:
                print(f'  [BT timeout] n={n} inst={inst+1}/{num_instances}')

            # B&B (fractional relaxation)
            bb_sol = [None]  # Use list to capture result in lambda
            bb_t = timeit(lambda: bb_sol.__setitem__(0, knapsack_bb_fractional(O_dict, W)), number=1)
            bb_sol = bb_sol[0]
            if bb_sol is not None:
                bb_times.append(bb_t)
                bb_nodes.append(bb_sol.get('nodes', float('nan')))
                bb_scores.append(bb_sol.get('score', float('nan')))
            else:
                print(f'  [BB timeout] n={n} inst={inst+1}/{num_instances}')

            # Optionnel : cohérence si les deux ont réussi
            if bt_sol is not None and bb_sol is not None:
                if bt_sol.get('score') != bb_sol.get('score'):
                    print(f'  [WARN] score mismatch on n={n} inst={inst+1}: '
                          f'BT={bt_sol.get("score")} vs BB={bb_sol.get("score")}')

        # Agrégation (uniquement sur les runs réussis)
        results['backtracking']['time'].append(mean_valid(bt_times))
        results['backtracking']['nodes'].append(mean_valid(bt_nodes))
        results['backtracking']['score'].append(mean_valid(bt_scores))
        results['bb_fractional']['time'].append(mean_valid(bb_times))
        results['bb_fractional']['nodes'].append(mean_valid(bb_nodes))
        results['bb_fractional']['score'].append(mean_valid(bb_scores))
        results['counts']['bt_runs'].append(len(bt_times))
        results['counts']['bb_runs'].append(len(bb_times))

        print(f"  BT: avg_time={results['backtracking']['time'][-1]:.4f}s "
              f"avg_nodes={results['backtracking']['nodes'][-1]:.0f} "
              f"(runs={len(bt_times)}/{num_instances})")
        print(f"  BB: avg_time={results['bb_fractional']['time'][-1]:.4f}s "
              f"avg_nodes={results['bb_fractional']['nodes'][-1]:.0f} "
              f"(runs={len(bb_times)}/{num_instances})")

    # Plots ------------------------------------------------------------
    n_vals = results['n']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (A) Time vs n
    ax = axes[0]
    ax.plot(n_vals, results['backtracking']['time'], marker='o', label='backtracking')
    ax.plot(n_vals, results['bb_fractional']['time'], marker='o', label='branch-and-bound (fractional)')
    ax.set_xlabel('n (number of objects)')
    ax.set_ylabel('average time (s)')
    ax.set_title('Execution time vs n')
    ax.set_yscale('log')
    ax.grid(True, which='both', ls=':')
    ax.legend()

    # (B) Nodes vs n
    ax = axes[1]
    ax.plot(n_vals, results['backtracking']['nodes'], marker='o', label='backtracking')
    ax.plot(n_vals, results['bb_fractional']['nodes'], marker='o', label='branch-and-bound (fractional)')
    ax.set_xlabel('n (number of objects)')
    ax.set_ylabel('average explored nodes')
    ax.set_title('Nodes explored vs n')
    ax.set_yscale('log')
    ax.grid(True, which='both', ls=':')
    ax.legend()

    plt.tight_layout()
    plt.show()

    return results

grader.addUtilityPart('b3', benchmark_bb_fractional, maxSeconds=600, description='Benchmark backtracking/B&B fractional implementations')

# Question 6

def test_meet_in_the_middle():
    grader.addMessage('--- TEST MEET-IN-THE-MIDDLE ---')

    result4 = knapsack_meet_in_the_middle(O4, W4)
    grader.addMessage(f"--- simple case : 4 objects --- Best solution: {result4['selected']} with value {result4['score']}")
    grader.requireIsEqual(result4['score'], 35)
    grader.requireIsEqual(result4['selected'], {'o2', 'o3'})

    result15 = knapsack_meet_in_the_middle(O15, W15)
    grader.addMessage(f"--- bigger case : 15 objects --- Best solution: {result15['selected']} with value {result15['score']}")
    grader.requireIsEqual(result15['score'], 1458)
    grader.requireIsEqual(result15['selected'], {'o1', 'o3', 'o5', 'o7', 'o8', 'o9', 'o14', 'o15'})

grader.addBasicPart('q6', test_meet_in_the_middle, 4, description='Test meet-in-the-middle implementation')

# Benchmark 4

def benchmark_meet_in_the_middle():
    """
    Benchmark backtracking vs meet-in-the-middle on random instances.
    """
    benchmark(algo1=knapsack_bb_fractional, algo2=knapsack_meet_in_the_middle, start=10, end=31)

grader.addUtilityPart('b4', benchmark_meet_in_the_middle, maxSeconds=600, description='Benchmark backtracking/meet_in-the-middle implementations')

# Benchmark 5

def benchmark_all_algorithms(max_n=40, step=5, timeout_seconds=20, num_instances=20):
    """
    Compare all knapsack algorithms on random instances.
    Returns a dict 'results' with per-algo: time[], score[], quality[] and nodes[] when dispo.
    'quality' = moyenne des (score_algo / score_DP) calculés par instance (si DP a réussi).
    """
    from timeit import default_timer as timer
    import statistics
    import math
    import threading

    def run_with_timeout(func, args, timeout):
        """
        Run func(*args) with TimeoutFunction.
        Returns (result, elapsed) on success; (None, None) on timeout.
        """
        start = timer()
        try:
            wrapper = graderUtil.TimeoutFunction(func, maxSeconds=timeout) if timeout and timeout > 0 else None
            result = (wrapper or func)(*args) if wrapper else func(*args)
        except graderUtil.TimeoutFunctionException:
            return (None, None)
        elapsed = timer() - start
        return (result, elapsed)
    
    results = {
        'n': [],
        'backtracking':   {'time': [], 'score': [], 'nodes': [], 'quality': []},
        'greedy':         {'time': [], 'score': [],               'quality': []},
        'dynprog':        {'time': [], 'score': [],               'quality': []},  # quality = 1 si DP ok
        'bb_fractional':  {'time': [], 'score': [], 'nodes': [],  'quality': []},
        'mim':            {'time': [], 'score': [],               'quality': []},
    }

    W = 1000
    max_weight = 250
    max_value = 100

    for n in range(5, max_n + 1, step):
        print(f'\n=== n={n} (averaging over {num_instances} instances) ===')
        results['n'].append(n)

        # Collecte per-instance pour les moyennes
        per = {
            'backtracking':  {'time': [], 'score': [], 'nodes': [], 'quality': []},
            'greedy':        {'time': [], 'score': [],             'quality': []},
            'dynprog':       {'time': [], 'score': [],             'quality': []},
            'bb_fractional': {'time': [], 'score': [], 'nodes': [], 'quality': []},
            'mim':           {'time': [], 'score': [],             'quality': []},
        }

        for inst in range(num_instances):
            O_dict = generate_random_instance(n, max_weight, max_value)

            # Lancer tous les algos sur la même instance
            instance_scores = {}

            # DP (référence pour la qualité)
            dp_sol, dp_t = run_with_timeout(dynprog, (O_dict, W), timeout_seconds)
            if dp_sol is not None:
                per['dynprog']['time'].append(dp_t)
                per['dynprog']['score'].append(dp_sol['score'])
                per['dynprog']['quality'].append(1.0)
                instance_scores['dynprog'] = dp_sol['score']
            else:
                # Pas d'optimum → on ne calculera pas de ratio qualité pour cette instance
                print(f'  DP timeout (> {timeout_seconds}s) on instance {inst+1}')

            # Backtracking
            bt_sol, bt_t = run_with_timeout(backtracking, (O_dict, W), timeout_seconds)
            if bt_sol is not None:
                per['backtracking']['time'].append(bt_t)
                per['backtracking']['score'].append(bt_sol['score'])
                per['backtracking']['nodes'].append(bt_sol['nodes'])
                if 'dynprog' in instance_scores and instance_scores['dynprog'] > 0:
                    per['backtracking']['quality'].append(bt_sol['score'] / instance_scores['dynprog'])

            # Greedy
            gr_sol, gr_t = run_with_timeout(greedy, (O_dict, W), timeout_seconds)
            if gr_sol is not None:
                per['greedy']['time'].append(gr_t)
                per['greedy']['score'].append(gr_sol['score'])
                if 'dynprog' in instance_scores and instance_scores['dynprog'] > 0:
                    per['greedy']['quality'].append(gr_sol['score'] / instance_scores['dynprog'])

            # Branch & Bound + relaxation fractionnaire
            bb_sol, bb_t = run_with_timeout(knapsack_bb_fractional, (O_dict, W), timeout_seconds)
            if bb_sol is not None:
                per['bb_fractional']['time'].append(bb_t)
                per['bb_fractional']['score'].append(bb_sol['score'])
                per['bb_fractional']['nodes'].append(bb_sol['nodes'])
                if 'dynprog' in instance_scores and instance_scores['dynprog'] > 0:
                    per['bb_fractional']['quality'].append(bb_sol['score'] / instance_scores['dynprog'])

            # Meet-in-the-Middle (version corrigée)
            mim_sol, mim_t = run_with_timeout(knapsack_meet_in_the_middle, (O_dict, W), timeout_seconds)
            if mim_sol is not None:
                per['mim']['time'].append(mim_t)
                per['mim']['score'].append(mim_sol['score'])
                if 'dynprog' in instance_scores and instance_scores['dynprog'] > 0:
                    per['mim']['quality'].append(mim_sol['score'] / instance_scores['dynprog'])

        # Moyennes (ignorer les listes vides)
        for algo, dic in per.items():
            def mean_or_nan(xs):
                return (sum(xs) / len(xs)) if xs else float('nan')
            print(f'  {algo}: times={dic["time"]}, scores={dic["score"]}, quality={dic["quality"]}')
            results[algo]['time'].append(mean_or_nan(dic['time']))
            results[algo]['score'].append(mean_or_nan(dic['score']))
            if 'nodes' in dic:
                results[algo]['nodes'].append(mean_or_nan(dic['nodes']))
            results[algo]['quality'].append(mean_or_nan(dic['quality']))

        # Log court
        print(f"  DP avg_score={results['dynprog']['score'][-1]:.1f}, "
              f"BT avg_quality={results['backtracking']['quality'][-1]:.3f}, "
              f"Greedy avg_quality={results['greedy']['quality'][-1]:.3f}, "
              f"BB avg_quality={results['bb_fractional']['quality'][-1]:.3f}, "
              f"MIM avg_quality={results['mim']['quality'][-1]:.3f}")

    return results


def plot_benchmark_results(results):
    """Plot comprehensive comparison of all algorithms."""
    import math
    import matplotlib.pyplot as plt

    n_values = results['n']
    algos = ['backtracking', 'greedy', 'dynprog', 'bb_fractional', 'mim']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Time vs n (log y)
    ax = axes[0, 0]
    for algo in algos:
        times = [t if (t is not None and not math.isnan(t)) else float('nan')
                 for t in results[algo]['time']]
        ax.plot(n_values, times, marker='o', label=algo)
    ax.set_xlabel('Number of objects (n)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution Time')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    # (2) Solution quality (ratio to DP)
    ax = axes[0, 1]
    for algo in algos:
        quals = [q if (q is not None and not math.isnan(q)) else float('nan')
                 for q in results[algo]['quality']]
        ax.plot(n_values, quals, marker='o', label=algo)
    ax.set_xlabel('Number of objects (n)')
    ax.set_ylabel('Score / Optimal Score')
    ax.set_title('Solution Quality (relative to DP)')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    ax.legend()
    ax.grid(True)

    # (3) Nodes explored (si dispo)
    ax = axes[1, 0]
    for algo in ['backtracking', 'bb_fractional']:
        if 'nodes' in results[algo]:
            nodes = [n if (n is not None and not math.isnan(n)) else float('nan')
                     for n in results[algo]['nodes']]
            ax.plot(n_values, nodes, marker='o', label=algo)
    ax.set_xlabel('Number of objects (n)')
    ax.set_ylabel('Nodes explored')
    ax.set_title('Search Space Exploration')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

    # (4) Time vs Quality (moyennes globales, en ignorant NaN)
    ax = axes[1, 1]
    for algo in algos:
        times = [t for t in results[algo]['time'] if (t is not None and not math.isnan(t))]
        quals = [q for q in results[algo]['quality'] if (q is not None and not math.isnan(q))]
        if times and quals:
            avg_time = sum(times) / len(times)
            avg_quality = sum(quals) / len(quals)
            ax.scatter(avg_time, avg_quality, s=100, label=algo)
            ax.annotate(algo, (avg_time, avg_quality), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)
    ax.set_xlabel('Average Time (seconds)')
    ax.set_ylabel('Average Solution Quality')
    ax.set_title('Time vs Quality')
    ax.set_xscale('log')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def run_comprehensive_benchmark(timeout_seconds=5, num_instances=10, max_n=30, step=5):
    """Run comprehensive benchmark and display results."""
    print("=" * 60)
    print("COMPREHENSIVE BENCHMARK OF KNAPSACK ALGORITHMS")
    print(f"(Averaging over {num_instances} random instances per n)")
    print("=" * 60)

    results = benchmark_all_algorithms(
        max_n=max_n, step=step, timeout_seconds=timeout_seconds, num_instances=num_instances
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Print summary table
    print(f"\n{'n':<5} {'BT Time':<10} {'BT Nodes':<10} {'BB Time':<10} {'BB Nodes':<10} "
          f"{'DP Time':<10} {'Greedy':<10}")
    print("-" * 80)

    for i, n in enumerate(results['n']):
        bt_time = f"{results['backtracking']['time'][i]:.4f}" if results['backtracking']['time'][i] == results['backtracking']['time'][i] else "N/A"
        bt_nodes = f"{int(results['backtracking']['nodes'][i])}" if results['backtracking']['nodes'][i] == results['backtracking']['nodes'][i] else "N/A"
        bb_time = f"{results['bb_fractional']['time'][i]:.4f}" if results['bb_fractional']['time'][i] == results['bb_fractional']['time'][i] else "N/A"
        bb_nodes = f"{int(results['bb_fractional']['nodes'][i])}" if results['bb_fractional']['nodes'][i] == results['bb_fractional']['nodes'][i] else "N/A"
        dp_time = f"{results['dynprog']['time'][i]:.4f}" if results['dynprog']['time'][i] == results['dynprog']['time'][i] else "N/A"
        gr_time = f"{results['greedy']['time'][i]:.4f}" if results['greedy']['time'][i] == results['greedy']['time'][i] else "N/A"

        print(f"{n:<5} {bt_time:<10} {bt_nodes:<10} {bb_time:<10} {bb_nodes:<10} {dp_time:<10} {gr_time:<10}")

    plot_benchmark_results(results)
    return results

grader.addUtilityPart('b5', run_comprehensive_benchmark, maxSeconds=600, description='Full benchmark of all implementations')

if __name__ == "__main__":
    grader.grade()
