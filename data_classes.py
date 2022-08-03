from collections import defaultdict
from os import listdir
from statistics import stdev, mean

from aalpy.utils import load_automaton_from_file

from csv_export import LStarExportEntry, RPNIExportEntry, CachedLStarExportEntry


class LStarExperiment:
    def __init__(self, model_size, output_queries, steps_output_queries, eq_oracle_queries, steps_eq_queries,
                 learning_rounds, conformance_coverage, conformance_random) -> None:
        self.model_size = model_size
        self.output_queries = output_queries
        self.steps_output_queries = steps_output_queries
        self.eq_oracle_queries = eq_oracle_queries
        self.steps_eq_queries = steps_eq_queries
        self.conformance_coverage = conformance_coverage
        self.conformance_random = conformance_random
        self.learning_rounds = learning_rounds
        self.sum_queries = output_queries + eq_oracle_queries
        self.sum_steps = steps_output_queries + steps_eq_queries
        self.average_trace_len = self.sum_steps / self.sum_queries


class RPNIExperiment:
    def __init__(self, model_size, conformance_coverage, conformance_random, data_size, average_len,
                 correctly_learned_model) -> None:
        self.model_size = model_size
        self.conformance_coverage = conformance_coverage
        self.conformance_random = conformance_random
        self.data_size = data_size
        self.average_len = average_len
        self.correctly_learned_model = correctly_learned_model


class CachedLStarExperiment:
    def __init__(self, conformance_coverage, random_sample_size, performed_queries, cache_hits,
                 learning_rounds) -> None:
        self.conformance_coverage = conformance_coverage
        self.random_sample_size = random_sample_size
        self.performed_queries = performed_queries
        self.cache_hits = cache_hits
        self.learning_rounds = learning_rounds


def data_stats(field, l_star_data):
    field_data = [getattr(elem, field) for elem in l_star_data]
    field_data_average = mean(field_data)
    field_data_stdev = stdev(field_data)
    return field_data_average, field_data_stdev


def l_star_summary(l_star_experiment_data, verbose):
    number_states = data_stats("model_size", l_star_experiment_data)
    learning_rounds = data_stats("learning_rounds", l_star_experiment_data)
    output_queries = data_stats("output_queries", l_star_experiment_data)
    steps_output_queries = data_stats("steps_output_queries", l_star_experiment_data)
    eq_oracle_queries = data_stats("eq_oracle_queries", l_star_experiment_data)
    steps_eq_queries = data_stats("steps_eq_queries", l_star_experiment_data)
    average_trace_len = data_stats("average_trace_len", l_star_experiment_data)
    conformance_coverage = data_stats("conformance_coverage", l_star_experiment_data)
    conformance_random = data_stats("conformance_random", l_star_experiment_data)
    sum_queries = data_stats("sum_queries", l_star_experiment_data)
    sum_steps = data_stats("sum_steps", l_star_experiment_data)

    if verbose:
        print(f'\n----L* summary----')
        print(f'States: {number_states[0]} ({number_states[1]})')
        print(f'Learning rounds: {learning_rounds[0]} ({learning_rounds[1]})')
        print(f'Output queries: {output_queries[0]} ({output_queries[1]})')
        print(f'Steps Output queries: {steps_output_queries[0]} ({steps_output_queries[1]})')
        print(f'Equivalence oracle queries: {eq_oracle_queries[0]} ({eq_oracle_queries[1]})')
        print(f'Steps equivalence oracle: {steps_eq_queries[0]} ({steps_eq_queries[1]})')
        print(f'Average trace length: {average_trace_len[0]} ({average_trace_len[1]})')
        print(f'Conformance (coverage): {conformance_coverage[0]} ({conformance_coverage[1]})')
        print(f'Conformance (random): {conformance_random[0]} ({conformance_random[1]})')

    return LStarExportEntry(number_states, output_queries, steps_output_queries, eq_oracle_queries, steps_eq_queries,
                            conformance_coverage, learning_rounds, sum_queries, sum_steps, average_trace_len)


def rpni_summary(rpni_experiment_data, rpni_data_names, minimized_l_star, verbose):
    rpni_export_data = defaultdict(RPNIExportEntry)

    minimized_l_star_data = rpni_experiment_data[minimized_l_star][0]

    if verbose:
        print(f'\n----RPNI summary----')
        print(f'\n--Experiment: {minimized_l_star}')
        print(f'States: {minimized_l_star_data.model_size}')
        print(f'Sample size: {minimized_l_star_data.data_size}')
        print(f'Average trace length: {minimized_l_star_data.average_len}')
        print(f'Conformance (coverage): {minimized_l_star_data.conformance_coverage}')
        print(f'Conformance (random): {minimized_l_star_data.conformance_random}')

    rpni_export_data[minimized_l_star] = RPNIExportEntry((minimized_l_star_data.model_size, 0),
                                                         (minimized_l_star_data.conformance_coverage, 0),
                                                         (minimized_l_star_data.conformance_random, 0),
                                                         (minimized_l_star_data.data_size, 0),
                                                         (minimized_l_star_data.average_len, 0), ("", 0))

    for experiment_name in rpni_data_names:
        number_states = data_stats("model_size", rpni_experiment_data[experiment_name])
        data_size = data_stats("data_size", rpni_experiment_data[experiment_name])
        average_len = data_stats("average_len", rpni_experiment_data[experiment_name])
        conformance_coverage = data_stats("conformance_coverage", rpni_experiment_data[experiment_name])
        conformance_random = data_stats("conformance_random", rpni_experiment_data[experiment_name])
        correctly_learned_model = [getattr(elem, "correctly_learned_model") for elem in
                                   rpni_experiment_data[experiment_name] if
                                   getattr(elem, "correctly_learned_model") == True]

        rpni_export_data[experiment_name] = RPNIExportEntry(number_states, conformance_coverage, conformance_random,
                                                            data_size, average_len, (len(correctly_learned_model), 0))

        if verbose:
            print(f'\n--Experiment: {experiment_name}')
            print(f'States: {number_states[0]} ({number_states[1]})')
            print(f'Sample size: {data_size[0]} ({data_size[1]})')
            print(f'Average trace length: {average_len[0]} ({average_len[1]})')
            print(f'Conformance (coverage): {conformance_coverage[0]} ({conformance_coverage[1]})')
            print(f'Conformance (random): {conformance_random[0]} ({conformance_random[1]})')
            print(
                f'Correctly learned models: {len(correctly_learned_model)}/{len(rpni_experiment_data[experiment_name])}')

    return rpni_export_data


def cached_l_star_summary(cached_l_star_experiment_data, verbose):
    conformance_coverage = data_stats("conformance_coverage", cached_l_star_experiment_data)
    random_sample_size = data_stats("random_sample_size", cached_l_star_experiment_data)
    performed_queries = data_stats("performed_queries", cached_l_star_experiment_data)
    cache_hits = data_stats("cache_hits", cached_l_star_experiment_data)
    learning_rounds = data_stats("learning_rounds", cached_l_star_experiment_data)

    if verbose:
        print(f'\n----Cached L* summary----')
        print(f'Conformance (coverage): {conformance_coverage[0]} ({conformance_coverage[1]})')
        print(f'Random sample size: {random_sample_size[0]} ({random_sample_size[1]})')
        print(f'Performed queries: {performed_queries[0]} ({performed_queries[1]})')
        print(f'Cached Queries: {cache_hits[0]} ({cache_hits[1]})')
        print(f'Learning Rounds: {learning_rounds[0]} ({learning_rounds[1]})')

    return CachedLStarExportEntry(conformance_coverage, random_sample_size, performed_queries, cache_hits,
                                  learning_rounds)


def load_dot_files(benchmark):
    loaded_models = []
    for dot_file in listdir(f'./automata/{benchmark}'):
        if dot_file[-4:] == ".dot":
            model_name = dot_file[:-4]
            model = load_automaton_from_file(f'./automata/{benchmark}/{dot_file}', automaton_type='mealy')
            loaded_models.append((model_name, model))
    return loaded_models
