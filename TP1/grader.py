import random
import graderUtil

grader = graderUtil.Grader()

submission = grader.load('tdp_1')

assert submission is not None, "Failed to load submission"

submission.grader = grader  # give access to grader inside submission

grader.addUtilityPart('b1', grader.submission.benchmark, description='Benchmark shortest path implementations')

grader.addBasicPart('q1', grader.submission.test_add_source, 2, description='Test add_source implementation')

grader.addBasicPart('q2', grader.submission.test_bellman_ford, 2, description='Test bellman_ford implementation')

grader.addBasicPart('q3', grader.submission.test_rewrite_weights, 2, description='Test rewrite_weights implementation')

grader.addBasicPart('q4', grader.submission.test_all_distances, 2, description='Test all_distances implementation')

grader.addBasicPart('q5', grader.submission.test_BF_SP_all_pairs, 2, description='Test BF_SP_all_pairs implementation')

grader.addBasicPart('q6', grader.submission.test_closest_oven, 2, description='Test closest_oven implementation')

grader.addBasicPart('q7', grader.submission.test_kcentre_value, 2, description='Test kcentre_value implementation')

grader.addUtilityPart('b2', grader.submission.BF_benchmark, description='Benchmark Brute Force all-pairs shortest paths implementation')

grader.addBasicPart('q8', grader.submission.test_greedy_algorithm, 2, description='Test greedy_algorithm implementation')

grader.addBasicPart('q9', grader.submission.test_random_algorithm, 2, description='Test random_algorithm implementation')

grader.addUtilityPart('b3', grader.submission.BF_G_R_benchmark, description='Benchmark Brute Force all-pairs shortest paths implementation')

if __name__ == "__main__":
    grader.grade()
