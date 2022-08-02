from os import listdir

from aalpy.SULs import MealySUL
from aalpy.learning_algs import run_Lstar, run_RPNI
from aalpy.utils import load_automaton_from_file

from data_classes import RPNIExperiment, CachedLStarExperiment, LStarExperiment
from data_generation import DataSet, l_star_with_populated_cache
from model_comparison import compare_learned_models


def l_star_experiment(model, test_cases_coverage, test_cases_random, alphabet, eq_oracle):
    # L*
    sul = MealySUL(model)
    l_star_model, data = run_Lstar(alphabet, sul, eq_oracle, 'mealy', print_level=0, return_data=True)

    # L* info
    output_queries = data['queries_learning']
    steps_output_queries = data['steps_learning']
    eq_oracle_queries = data['queries_eq_oracle']
    steps_eq_queries = data['steps_eq_oracle']
    learning_rounds = data['learning_rounds']

    coverage_diff = compare_learned_models(model, l_star_model, test_cases_coverage)
    random_diff = compare_learned_models(model, l_star_model, test_cases_random)

    return LStarExperiment(l_star_model.size, output_queries, steps_output_queries, eq_oracle_queries, steps_eq_queries,
                           learning_rounds, 100 - coverage_diff, 100 - random_diff)


def rpni_experiment(data, model, test_cases_coverage, test_cases_random):
    rpni_model = run_RPNI(data.data, automaton_type='mealy', input_completeness='sink_state', print_info=False)

    conformance_coverage = 100 - compare_learned_models(model, rpni_model, test_cases_coverage)
    conformance_random = 100 - compare_learned_models(model, rpni_model, test_cases_random)

    return RPNIExperiment(rpni_model.size, conformance_coverage, conformance_random, data.size, data.average_len(),
                          conformance_coverage == 100)


def l_star_with_initial_cache(cached_data: DataSet, model, eq_oracle, test_cases_coverage):
    learned_model, queries_to_fill_holes, cache_hits, learning_rounds = l_star_with_populated_cache(model,
                                                                                                    cached_data.data,
                                                                                                    eq_oracle)

    conformance_coverage = 100 - compare_learned_models(model, learned_model, test_cases_coverage)

    return CachedLStarExperiment(conformance_coverage, cached_data.size, queries_to_fill_holes, cache_hits,
                                 learning_rounds)
