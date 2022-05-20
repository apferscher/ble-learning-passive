from math import ceil
from os import listdir

from aalpy.learning_algs import run_RPNI
from aalpy.utils import load_automaton_from_file, compare_automata

from data_generation import *
from model_comparison import *

bluetooth_models = []

for dot_file in listdir('./automata'):
    model_name = dot_file[:-4]
    model = load_automaton_from_file(f'./automata/{dot_file}', automaton_type='mealy')
    bluetooth_models.append((model_name, model))

test_cases = create_test_cases(bluetooth_models, 10000)

for model_name, model in bluetooth_models:
    # L*
    sul = MealySUL(model)
    alphabet = model.get_input_alphabet()
    eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=100, min_walk_len=4, max_walk_len=8)

    l_star_model, data = run_Lstar(alphabet, sul, eq_oracle, 'mealy', print_level=0, return_data=True)

    # L* info
    e_set = data['characterization set']
    output_queries = data['queries_learning']
    eq_oracle_queries = data['queries_eq_oracle']

    learning_queries = output_queries + eq_oracle_queries
    learning_steps = int(
        ceil((output_queries + eq_oracle_queries) / (data['steps_learning'] + data['steps_eq_oracle'])))
    max_sequence_length = learning_steps * 2

    data_l_star = data_from_computed_e_set(l_star_model, include_extended_s_set=True)

    data_random_l_star_length = generate_random_data(model, num_sequences=learning_queries, min_sequence_len=1,
                                                     max_sequence_len=max_sequence_length)

    data_random_large_set = generate_random_data(model, num_sequences=(learning_queries * 2), min_sequence_len=1,
                                                 max_sequence_len=max_sequence_length)

    data_random_long_traces = generate_random_data(model, num_sequences=ceil(learning_queries / 2),
                                                   min_sequence_len=l_star_model.size,
                                                   max_sequence_len=(l_star_model.size * 2))

    data_minimized_char_set = minimized_char_set_data(l_star_model, include_extended_s_set=True)

    # print(learning_queries - len(data))

    rnpi_models = {
        'rpni_model_random_l_star_length': (
            run_RPNI(data_random_l_star_length, automaton_type='mealy', input_completeness='sink_state',
                     print_info=False),
            data_random_l_star_length),
        'rpni_model_andom_large_set': (
            run_RPNI(data_random_large_set, automaton_type='mealy', input_completeness='sink_state', print_info=False),
            data_random_large_set),
        'rpni_model_random_long_traces': (
            run_RPNI(data_random_long_traces, automaton_type='mealy', input_completeness='sink_state',
                     print_info=False),
            data_random_long_traces),
        'rpni_model_l_star': (
            run_RPNI(data_l_star, automaton_type='mealy', input_completeness='sink_state', print_info=False),
            data_l_star),
        'rpni_model_minimized_char_set': (
            run_RPNI(data_minimized_char_set, automaton_type='mealy', input_completeness='sink_state',
                     print_info=False),
            data_minimized_char_set)}

    print(f'Experiment: {model_name}')
    print(f'L* learned {l_star_model.size} state model.')
    print(f'Number of queries required by L*  : {learning_queries}')

    for k in rnpi_models.keys():
        rpni_model = rnpi_models[k][0]
        data = rnpi_models[k][1]
        print('-' * 5 + f' {k} ' + '-' * 5)
        print(f'RPNI Learned {rpni_model.size} state model.')
        print(f'Number of samples provided to RPNI: {len(data)}')

        if set(rpni_model.get_input_alphabet()) != set(l_star_model.get_input_alphabet()):
            print('Learned models do not have the same input alphabets => RPNI model is not input complete.')
            continue

        cex = compare_automata(rpni_model, l_star_model)
        if cex:
            # model_diff = compare_learned_models(l_star_model, rpni_model, num_tests=10000)
            model_diff = compare_learned_models(l_star_model, rpni_model, test_cases[model_name])
            print('Counterexample found between models learned by RPNI and L*.')
            print(f'Models display different bahaviour for {round(model_diff * 100, 2)}% of test cases.')
        else:
            print('RPNI and L* learned same models.')
            if rpni_model.size != l_star_model.size:
                print(f'    Models do have different size.\n    RPNI {rpni_model.size} vs. L* {l_star_model.size}')

    print('----------------------------------------------------------------')
