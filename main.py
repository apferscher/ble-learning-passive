from collections import defaultdict
from os import listdir
from statistics import mean

from aalpy.learning_algs import run_RPNI
from aalpy.utils import load_automaton_from_file, compare_automata

from data_generation import *
from model_comparison import *

class Experiment:
    def __init__(self, model_size, coverage_diff, random_diff, data_size, average_len) -> None:
        self.model_size = model_size
        self.coverage_diff = coverage_diff
        self.random_diff = random_diff
        self.data_size = data_size
        self.average_len = average_len

bluetooth_models = []

#model = load_automaton_from_file(f'./automata/CC2650.dot', automaton_type='mealy')
#model_name = 'CC2650'
#bluetooth_models.append((model_name, model))

for dot_file in listdir('./automata'):
    model_name = dot_file[:-4]
    model = load_automaton_from_file(f'./automata/{dot_file}', automaton_type='mealy')
    bluetooth_models.append((model_name, model))

num_tests = 10000
test_cases_coverage = create_test_cases(bluetooth_models, num_tests, 'coverage')
test_cases_random = create_test_cases(bluetooth_models, num_tests, 'random')

repeats_per_experiment = 5
verbose = False

for model_name, model in bluetooth_models:
    l_star_experiment_data = list()
    rpni_experiment_data = defaultdict(list)

    # L*
    sul = MealySUL(model)
    alphabet = model.get_input_alphabet()
    # eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=100, min_walk_len=4, max_walk_len=8)
    eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=10, walk_len=10)

    l_star_model, data = run_Lstar(alphabet, sul, eq_oracle, 'mealy', print_level=0, return_data=True)
    l_star_model_size = l_star_model.size

    # L* info
    e_set = data['characterization set']
    output_queries = data['queries_learning']
    steps_output_queries = data['steps_learning']
    eq_oracle_queries = data['queries_eq_oracle']
    steps_eq_queries = data['steps_eq_oracle']

    learning_queries = (output_queries + eq_oracle_queries)

    avg_query_steps = (data['steps_learning'] + data['steps_eq_oracle']) / learning_queries

    print('-' * 70)
    print(f'ACTIVE LEARNING DATA:')
    print('-' * 70)

    print(f'------------------{model_name}------------------')
    print(f'Model size: {l_star_model_size}')
    print(f'Output queries: {output_queries}')
    print(f'Output queries steps: {steps_output_queries}')
    print(f'Conformance queries: {eq_oracle_queries}')
    print(f'Output query steps: {steps_eq_queries}')
    print(f'Average query steps: {avg_query_steps}')


    if verbose:
        print(f'L* data size: {learning_queries}')
        print(f'Average length of L* samples: {avg_query_steps}')

    # long sequence  
    max_sequence_length = round((avg_query_steps - 0.5) * 2)

    rpni_model_l_star_str = "rpni_model_l_star"
    rpni_model_random_l_star_length_str = "rpni_model_random_l_star_length"
    rpni_model_random_large_set_str = "rpni_model_random_large_set"
    rpni_model_random_long_traces_str = "rpni_model_random_long_traces"
    rpni_model_minimized_char_set_str = "rpni_model_minimized_char_set"
    rpni_model_random_good_enough_str = "rpni_model_random_good_enough"

    for _ in range(repeats_per_experiment):

        if verbose:
            print('-' * 5 + f' data gen: {rpni_model_l_star_str} ' + '-' * 5)

        data_l_star= data_from_computed_e_set(l_star_model, include_extended_s_set=True, verbose=verbose)

        if verbose:
            print('-' * 5 + f' data gen: {rpni_model_random_l_star_length_str} ' + '-' * 5)
        data_random_l_star_length = generate_random_data(model, num_sequences=learning_queries, min_sequence_len=1,
                                                         max_sequence_len=max_sequence_length, verbose=verbose)

        if verbose:
            print('-' * 5 + f' data gen: {rpni_model_random_large_set_str} ' + '-' * 5)
        data_random_large_set = generate_random_data(model, num_sequences=(learning_queries * 2), min_sequence_len=1,
                                                     max_sequence_len=max_sequence_length, verbose=verbose)

        if verbose:
            print('-' * 5 + f' data gen: {rpni_model_random_long_traces_str} ' + '-' * 5)
        data_random_long_traces = generate_random_data(model, num_sequences=learning_queries,
                                                       min_sequence_len=l_star_model.size,
                                                       max_sequence_len=(l_star_model.size *
                                                                         2), verbose=verbose)

        if verbose:
            print('-' * 5 + f' data gen: {rpni_model_random_good_enough_str} ' + '-' * 5)
  
        
        data_random_good_enough = generate_random_data(model, num_sequences= learning_queries * 25, min_sequence_len=l_star_model_size,max_sequence_len=10 + l_star_model_size, verbose=verbose)


        if verbose:
            print('-' * 5 + f' data gen: {rpni_model_minimized_char_set_str} ' + '-' * 5)
        data_minimized_char_set = minimized_char_set_data(l_star_model, include_extended_s_set=True, verbose=verbose)

        rpni_data = {
            rpni_model_l_star_str: data_l_star,
            rpni_model_random_l_star_length_str: data_random_l_star_length,
            rpni_model_random_large_set_str: data_random_large_set,
            rpni_model_random_long_traces_str: data_random_long_traces,
            rpni_model_minimized_char_set_str: data_minimized_char_set,
            rpni_model_random_good_enough_str: data_random_good_enough
        }

        # L* with caching
        queries_to_fill_holes, cache_hits = l_star_with_populated_cache(model, data_random_l_star_length.data, eq_oracle)
        l_star_experiment_data.append((l_star_model.size, learning_queries, queries_to_fill_holes, cache_hits))

        if verbose:
            print(f'L* with caching initialed with random data of size {learning_queries}: '
                  f'queries {queries_to_fill_holes}, cache hits {cache_hits}')

        if verbose:
            print(f'Experiment: {model_name}')
            print(f'L* learned {l_star_model.size} state model.')
            print(f'Number of queries required by L*  : {learning_queries}')

        for data_name, data in rpni_data.items():
            if verbose:
                print('-' * 5 + f' {data_name} ' + '-' * 5)
            rpni_model = run_RPNI(data.data, automaton_type='mealy', input_completeness='sink_state',
                                  print_info=False)

            if verbose:
                print(f'RPNI Learned {rpni_model.size} state model.')
                # wrong size
                # print(f'Number of samples provided to RPNI: {len(data)}')

            if set(rpni_model.get_input_alphabet()) != set(l_star_model.get_input_alphabet()):
                if verbose:
                    print('Learned models do not have the same input alphabets => RPNI model is not input complete.')
                continue

            cex = compare_automata(rpni_model, l_star_model)

            coverage_diff, random_diff = 0, 0
            if cex:
                coverage_diff = compare_learned_models(l_star_model, rpni_model, test_cases_coverage[model_name])
                random_diff = compare_learned_models(l_star_model, rpni_model, test_cases_random[model_name])
                if verbose:
                    print('Counterexample found between models learned by RPNI and L*.')
                    print(f'Coverage test cases: {round(coverage_diff * 100, 2)}% non-conforming test-cases.')
                    print(f'Random test cases  : {round(random_diff * 100, 2)}% non-conforming test-cases.')
            else:
                if verbose:
                    print('RPNI and L* learned same models.')
                if rpni_model.size != l_star_model.size and verbose:
                    print(f'    Models do have different size.\n    RPNI {rpni_model.size} vs. L* {l_star_model.size}')

            rpni_experiment_data[data_name].append(Experiment(rpni_model.size, coverage_diff, random_diff, data.size, data.average_len()))
    
    print('-' * 70)
    print(f'PASSIVE LEARNING DATA:')
    print('-' * 70)

    print(f'------------------{model_name}------------------')
    if len(set([i[0] for i in l_star_experiment_data])) != 1:
        print(f"L* did not always learn model of the same size: {[i[0] for i in l_star_experiment_data]}")

    #print(f"L* model size: {l_star_experiment_data[0][0]}")

    print(f'L* with caching initialed with random data of size equal to data required for L*.\n'
          f'  # random samples: {[i[1] for i in l_star_experiment_data]}\n'
          f'  # queries       : {[i[2] for i in l_star_experiment_data]}\n'
          f'  # cache hits    : {[i[3] for i in l_star_experiment_data]}')

    for experiment, data in rpni_experiment_data.items():
        print(f'\n{experiment} data summary')
        print(f'RPNI Model sizes {[i.model_size for i in data]}')
        print(f'Coverage testing conformance %: {[round(100 - i.coverage_diff * 100, 2) for i in data]} '
              f'Avg:{round(100 - mean([i.coverage_diff for i in data]) * 100, 2)}%')
        print(f'Random testing conformance %: {[round(100 - i.random_diff * 100, 2) for i in data]} '
              f'Avg:{round(100 - mean([i.random_diff for i in data]) * 100, 2)}%')

        print(f'Data size avg:{round(mean([i.data_size for i in data]), 2)}')
        print(f'Average steps avg:{round(mean([i.average_len for i in data]), 2)}')

    if verbose:
        print('----------------------------------------------------------------')
