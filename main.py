from calendar import c
from collections import defaultdict
from functools import cache
from os import listdir
from statistics import mean, stdev
import string

from aalpy.learning_algs import run_RPNI
from aalpy.utils import load_automaton_from_file, compare_automata
from numpy import number

from data_generation import *
from model_comparison import *
from csv_export import *

class LStarExperiment:
    def __init__(self, model_size, output_queries, steps_output_queries, eq_oracle_queries, steps_eq_queries, learning_rounds, conformance_coverage, conformance_random) -> None:
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
    def __init__(self, model_size, conformance_coverage, conformance_random, data_size, average_len, correctly_learned_model) -> None:
        self.model_size = model_size
        self.conformance_coverage = conformance_coverage
        self.conformance_random = conformance_random
        self.data_size = data_size
        self.average_len = average_len
        self.correctly_learned_model = correctly_learned_model

class CachedLStarExperiment:
    def __init__(self, conformance_coverage, random_sample_size, performed_queries, cache_hits, learning_rounds) -> None:
        self.conformance_coverage = conformance_coverage
        self.random_sample_size = random_sample_size
        self.performed_queries = performed_queries
        self.cache_hits = cache_hits
        self.learning_rounds = learning_rounds


def data_stats(field, l_star_data):
    field_data = [getattr(elem, field) for elem in l_star_data]
    field_data_average = mean(field_data)
    field_data_stdev = stdev(field_data)
    return (field_data_average, field_data_stdev)


def load_dot_files(benchmark):
    '''
    
    '''
    loaded_models = []
    for dot_file in listdir(f'./automata/{benchmark}'):
        if dot_file[-4:] == ".dot":
            model_name = dot_file[:-4]
            model = load_automaton_from_file(f'./automata/{benchmark}/{dot_file}', automaton_type='mealy')
            loaded_models.append((model_name, model))
    return loaded_models

def l_star_experiment(sul_model, test_cases_coverage, test_cases_random, alphabet, eq_oracle):
    # L*
    sul = MealySUL(model)
    l_star_model, data = run_Lstar(alphabet, sul, eq_oracle, 'mealy', print_level=0, return_data=True)

    # L* info
    output_queries = data['queries_learning']
    steps_output_queries = data['steps_learning']
    eq_oracle_queries = data['queries_eq_oracle']
    steps_eq_queries = data['steps_eq_oracle']
    learning_rounds = data['learning_rounds']

    coverage_diff = compare_learned_models(sul_model, l_star_model, test_cases_coverage)
    random_diff = compare_learned_models(sul_model, l_star_model, test_cases_random)

    return LStarExperiment(l_star_model.size, output_queries, steps_output_queries, eq_oracle_queries, steps_eq_queries, learning_rounds, 100-coverage_diff, 100-random_diff)

def rpni_experiment(data, sul_model, test_cases_coverage, test_cases_random):
    rpni_model = run_RPNI(data.data, automaton_type='mealy', input_completeness='sink_state', print_info=False)

    conformance_coverage = 100 - compare_learned_models(sul_model, rpni_model, test_cases_coverage)
    conformance_random = 100 - compare_learned_models(sul_model, rpni_model, test_cases_random)

    return RPNIExperiment(rpni_model.size, conformance_coverage, conformance_random, data.size, data.average_len(), conformance_coverage == 100)

def l_star_with_initial_cache(cached_data : DataSet, sul_model, eq_oracle, test_cases_coverage):

    learned_model, queries_to_fill_holes, cache_hits, learning_rounds = l_star_with_populated_cache(sul_model, cached_data.data, eq_oracle)

    conformance_coverage = 100 - compare_learned_models(sul_model, learned_model, test_cases_coverage)

    return CachedLStarExperiment(conformance_coverage, cached_data.size, queries_to_fill_holes, cache_hits, learning_rounds)


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
        print(f'\nL* summary:')
        print(f'States: {number_states[0]} ({number_states[1]})')
        print(f'Learning rounds: {learning_rounds[0]} ({learning_rounds[1]})')
        print(f'Output queries: {output_queries[0]} ({output_queries[1]})')
        print(f'Steps Output queries: {steps_output_queries[0]} ({steps_output_queries[1]})')
        print(f'Equivalence oracle queries: {eq_oracle_queries[0]} ({eq_oracle_queries[1]})')
        print(f'Steps equivalence oracle: {steps_eq_queries[0]} ({steps_eq_queries[1]})')
        print(f'Average trace length: {average_trace_len[0]} ({average_trace_len[1]})')
        print(f'Conformance (coverage): {conformance_coverage[0]} ({conformance_coverage[1]})')
        print(f'Conformance (random): {conformance_random[0]} ({conformance_random[1]})')

    return LStarExportEntry(number_states, output_queries, steps_output_queries, eq_oracle_queries, steps_eq_queries, conformance_coverage, learning_rounds, sum_queries, sum_steps, average_trace_len)


def rpni_summary(rpni_experiment_data, rpni_data_names, minimized_l_star, verbose):

    rpni_export_data = defaultdict(RPNIExportEntry)

    minimized_l_star_data = rpni_experiment_data[minimized_l_star][0]

    if verbose:
        print(f'\nRPNI summary:')
        print(f'States: {minimized_l_star_data.model_size}')
        print(f'Sample size: {minimized_l_star_data.data_size}')
        print(f'Average trace length: {minimized_l_star_data.average_len}')
        print(f'Conformance (coverage): {minimized_l_star_data.conformance_coverage}')
        print(f'Conformance (random): {minimized_l_star_data.conformance_random}')
    
    rpni_export_data[minimized_l_star] = RPNIExportEntry((minimized_l_star_data.model_size,0), (minimized_l_star_data.conformance_coverage,0), (minimized_l_star_data.conformance_random, 0), (minimized_l_star_data.data_size, 0) , (minimized_l_star_data.average_len, 0), ("",0))


    for experiment_name in rpni_data_names:
        number_states = data_stats("model_size", rpni_experiment_data[experiment_name])
        data_size = data_stats("data_size", rpni_experiment_data[experiment_name])
        average_len = data_stats("average_len", rpni_experiment_data[experiment_name])
        conformance_coverage = data_stats("conformance_coverage", rpni_experiment_data[experiment_name])
        conformance_random = data_stats("conformance_random", rpni_experiment_data[experiment_name])
        correctly_learned_model = [getattr(elem, "correctly_learned_model") for elem in rpni_experiment_data[experiment_name] if getattr(elem, "correctly_learned_model")==True]

        rpni_export_data[experiment_name] = RPNIExportEntry(number_states, conformance_coverage, conformance_random, data_size, average_len, (len(correctly_learned_model),0))
    
        if verbose:
            print(f'\nExperiment: {experiment_name}')
            print(f'States: {number_states[0]} ({number_states[1]})')
            print(f'Sample size: {data_size[0]} ({data_size[1]})')
            print(f'Average trace length: {average_len[0]} ({average_len[1]})')
            print(f'Conformance (coverage): {conformance_coverage[0]} ({conformance_coverage[1]})')
            print(f'Conformance (random): {conformance_random[0]} ({conformance_random[1]})')
            print(f'Correctly learned models: {len(correctly_learned_model)}/{len(rpni_experiment_data[experiment_name])}')

        return rpni_export_data




def cached_l_star_summary(cached_l_star_experiment_data, verbose):
    conformance_coverage = data_stats("conformance_coverage", cached_l_star_experiment_data)
    random_sample_size = data_stats("random_sample_size", cached_l_star_experiment_data)
    performed_queries = data_stats("performed_queries", cached_l_star_experiment_data)
    cache_hits = data_stats("cache_hits", cached_l_star_experiment_data)
    learning_rounds = data_stats("learning_rounds", cached_l_star_experiment_data)

    if verbose:
        print(f'\nCached L* summary:')
        print(f'Conformance (coverage): {conformance_coverage[0]} ({conformance_coverage[1]})')
        print(f'Random sample size: {random_sample_size[0]} ({random_sample_size[1]})')
        print(f'Performed queries: {performed_queries[0]} ({performed_queries[1]})')
        print(f'Cached Queries: {cache_hits[0]} ({cache_hits[1]})')
        print(f'Learning Rounds: {learning_rounds[0]} ({learning_rounds[1]})')

    return CachedLStarExportEntry(conformance_coverage, random_sample_size, performed_queries, cache_hits, learning_rounds)


if __name__ == "__main__":

    # load all automata from benchmark
    benchmark = 'MQTT' # 'BLE'
    benchmark_models = load_dot_files(benchmark)

    # generate test suite for conformance testing after learning
    num_tests = 10000
    test_cases_coverage = create_test_cases(benchmark_models, num_tests, 'coverage')
    test_cases_random = create_test_cases(benchmark_models, num_tests, 'random')

    # number of repetition of each learning algorithm
    repeats_per_experiment = 2 # 5

    # levels on which output is printed
    # 0: no output is printed
    # 1: data summary is printed
    # 2: data summary + debug output 
    verbose_level = 1

    # export csv files that contain the results of the performed evaluation
    csv = True
    l_star_data_export = DataExporter(LStarExportEntry.pretty_printed_attr())
    rpni_data_export = RPNIDataExporter(RPNIExportEntry.pretty_printed_attr())
    cached_l_star_data_export = DataExporter(CachedLStarExportEntry.pretty_printed_attr())

    #rpni_model_l_star_str = "l* data"
    rpni_model_random_l_star_length_str = "random |l* data|"
    rpni_model_random_large_set_str = "random 2*|l* data|"
    rpni_model_random_long_traces_str = "random long traces"
    rpni_model_minimized_char_set_str = "l* data (minimized)"
    #rpni_model_random_good_enough_str = "random corr"

    rpni_data = {
            #rpni_model_l_star_str: data_l_star,
            rpni_model_random_l_star_length_str: None,
            #rpni_model_random_large_set_str: None,
            #rpni_model_random_long_traces_str: None,
            #rpni_model_minimized_char_set_str: data_minimized_char_set,
            #rpni_model_random_good_enough_str: data_random_good_enough
    }

    rpni_data_names = rpni_data.keys()
    
    for model_name, model in benchmark_models:

        l_star_experiment_data = defaultdict(list)
        rpni_experiment_data = defaultdict(defaultdict)
        cached_l_star_experiment_data = defaultdict(list)

        # parameter for equivalence oracle
        walks_per_state=25
        walk_len=25

        sul = MealySUL(model)
        alphabet = model.get_input_alphabet()

        rpni_data_export.add_model(model_name)
        
        for _ in range(repeats_per_experiment):
            eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=walks_per_state, walk_len=walk_len)
            l_star_res = l_star_experiment(model, test_cases_coverage[model_name], test_cases_random[model_name], alphabet, eq_oracle)
            l_star_experiment_data[model_name].append(l_star_res)
        
        avg_query_steps = data_stats("average_trace_len", l_star_experiment_data[model_name])[0]
        max_sequence_length = round((avg_query_steps - 0.5) * 2)

        learning_queries = round(data_stats("sum_queries", l_star_experiment_data[model_name])[0])

        data_l_star= data_from_computed_e_set(model,include_extended_s_set=True, verbose=verbose_level == 2)

        data_minimized_char_set = minimized_char_set_data(model, include_extended_s_set=True, verbose=verbose_level == 2)

        model_size = model.size

        rpni_experiment_data[model_name] = defaultdict(list)

        for _ in range(repeats_per_experiment):
            # data generation:

            # random |l* data|
            data_random_l_star_length = generate_random_data(model, num_sequences=learning_queries, min_sequence_len=1,max_sequence_len=max_sequence_length, verbose=verbose_level == 2)

            # random 2*|l* data|
            data_random_large_set = generate_random_data(model, num_sequences=(learning_queries * 2), min_sequence_len=1,max_sequence_len=max_sequence_length, verbose=verbose_level == 2)

            # random long traces
            data_random_long_traces = generate_random_data(model, num_sequences=learning_queries, min_sequence_len=model_size,max_sequence_len=(model_size * 2), verbose=verbose_level == 2)

            # good enough data (randomly generated data that learns correctly)
            # data_random_good_enough = generate_random_data(model, num_sequences= learning_queries * 1, min_sequence_len=model_size,max_sequence_len=max(10,model_size)  + model_size, verbose=verbose)

            rpni_data = {
            #rpni_model_l_star_str: data_l_star,
            rpni_model_random_l_star_length_str: data_random_l_star_length,
            #rpni_model_random_large_set_str: data_random_large_set,
            #rpni_model_random_long_traces_str: data_random_long_traces,
            #rpni_model_minimized_char_set_str: data_minimized_char_set,
            #rpni_model_random_good_enough_str: data_random_good_enough
            }

           
            
            for data_name, data in rpni_data.items():
                rpni_experiment_data[model_name][data_name].append(rpni_experiment(data,model,test_cases_coverage[model_name], test_cases_random[model_name]))

            eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=walks_per_state, walk_len=walk_len)
            cached_l_star_experiment = l_star_with_initial_cache(data_random_l_star_length, model, eq_oracle, test_cases_coverage[model_name])
            # L* cached with random samples
            cached_l_star_experiment_data[model_name].append(cached_l_star_experiment)

        

        rpni_experiment_data[model_name][rpni_model_minimized_char_set_str].append(rpni_experiment(data_minimized_char_set,model,test_cases_coverage[model_name], test_cases_random[model_name]))
        

        print(f'------------------{model_name}------------------')
        l_star_data_export.add_entry(model_name, l_star_summary(l_star_experiment_data[model_name], verbose = verbose_level >= 1))

        rpni_data_export.add_entry(model_name, rpni_summary(rpni_experiment_data[model_name],rpni_data_names,rpni_model_minimized_char_set_str, verbose = verbose_level >= 1))

        cached_l_star_data_export.add_entry(model_name, cached_l_star_summary(cached_l_star_experiment_data[model_name], verbose = verbose_level >= 1))
    
    if csv:
        l_star_data_export.export_csv(f'{benchmark}_l_star_data')
        rpni_data_export.export_csv(f'{benchmark}_rpni_data', rpni_data_names)
        cached_l_star_data_export.export_csv(f'{benchmark}_cached_l_star_data')
        
