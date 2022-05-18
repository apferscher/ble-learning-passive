import random
from os import listdir

from aalpy.SULs import MealySUL
from aalpy.learning_algs import run_Lstar, run_RPNI
from aalpy.oracles import RandomWordEqOracle, StatePrefixEqOracle
from aalpy.utils import load_automaton_from_file, compare_automata, generate_test_cases


def data_from_l_star_E_set(hypothesis, e_set, include_extended_s_set=True):
    data = []
    prefixes = [state.prefix for state in hypothesis.states]

    if include_extended_s_set:
        extended_prefixes = []
        for p in prefixes:
            for a in hypothesis.get_input_alphabet():
                extended_prefixes.append(p + tuple([a]))

        prefixes.extend(extended_prefixes)

    for prefix in prefixes:
        for suffix in e_set:
            inputs = prefix + suffix
            output = hypothesis.compute_output_seq(hypothesis.initial_state, inputs)
            data.append(list(zip(inputs, output)))

    return data


def data_from_computed_e_set(hypothesis, include_extended_s_set=True):
    e_set = hypothesis.compute_characterization_set()
    return data_from_l_star_E_set(hypothesis, e_set, include_extended_s_set)


def minimized_char_set_data(hypothesis, e_set, include_extended_s_set=True):
    from aalpy.learning_algs.deterministic_passive.rpni_helper_functions import extract_unique_sequences, createPTA
    data = data_from_l_star_E_set(hypothesis, e_set, include_extended_s_set)
    pruned_data = []
    for seq in extract_unique_sequences(createPTA(data, automaton_type='mealy')):
        pruned_data.append([io[0] for io in seq])
    return pruned_data


def generate_random_data(model, num_sequences, min_sequence_len, max_sequence_len):
    data = []
    input_alphabet = model.get_input_alphabet()
    for _ in range(num_sequences):
        inputs = random.choices(input_alphabet, k=random.randint(min_sequence_len, max_sequence_len))
        output = model.compute_output_seq(model.initial_state, inputs)
        data.append(list(zip(inputs, output)))
    return data


def compare_learned_models(model_1, model_2, num_tests, min_test_len=3, max_test_len=16):
    diff = 0
    inputs = model_1.get_input_alphabet()
    for _ in range(num_tests):
        test_case = random.choices(inputs, k=random.randint(min_test_len, max_test_len))
        o_1 = model_1.compute_output_seq(model_1.initial_state, test_case)
        o_2 = model_2.compute_output_seq(model_2.initial_state, test_case)

        if o_1 != o_2:
            diff += 1

    return diff / num_tests


def compare_learned_models_with_state_coverage(model_1, model_2, num_tests):
    diff = 0
    inputs = model_2.get_input_alphabet()

    walks_per_state = int(num_tests / model_2.size)
    eq_oracle = StatePrefixEqOracle(inputs, sul=None, walks_per_state=walks_per_state, walk_len=10)
    test_cases = [tc[0] for tc in generate_test_cases(model_2, eq_oracle)]

    for test_case in test_cases:
        o_1 = model_1.compute_output_seq(model_1.initial_state, test_case)
        o_2 = model_2.compute_output_seq(model_2.initial_state, test_case)

        if o_1 != o_2:
            diff += 1

    return diff / len(test_cases)


bluetooth_models = []

for dot_file in listdir('./automata'):
    model_name = dot_file[:-4]
    model = load_automaton_from_file(f'./automata/{dot_file}', automaton_type='mealy')
    bluetooth_models.append((model_name, model))

for model_name, model in bluetooth_models:
    sul = MealySUL(model)
    alphabet = model.get_input_alphabet()
    eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=100, min_walk_len=4, max_walk_len=8)

    l_star_model, data = run_Lstar(alphabet, sul, eq_oracle, 'mealy', print_level=0, return_data=True)

    e_set = data['characterization set']
    learning_queries = data['queries_learning']
    eq_oracle_queries = data['queries_eq_oracle']

    # data = data_from_l_star_E_set(l_star_model, e_set, include_extended_s_set=True)
    # data = data_from_computed_e_set(l_star_model, include_extended_s_set=True)
    data = minimized_char_set_data(l_star_model, e_set, include_extended_s_set=True)
    data = generate_random_data(model, num_sequences=learning_queries - 100, min_sequence_len=10, max_sequence_len=20)
    # print(learning_queries - len(data))

    rpni_model = run_RPNI(data, automaton_type='mealy', input_completeness='sink_state', print_info=False)

    print(f'Experiment: {model_name}')
    print(f'L* learned {l_star_model.size} state model.')
    print(f'RPNI Learned {rpni_model.size} state model.')
    print(f'Number of queries required by L*  : {learning_queries}')
    print(f'Number of samples provided to RPNI: {len(data)}')

    if set(rpni_model.get_input_alphabet()) != set(l_star_model.get_input_alphabet()):
        print('Learned models do not have the same input alphabets => RPNI model is not input complete.')
        continue

    cex = compare_automata(rpni_model, l_star_model)
    if cex:
        # model_diff = compare_learned_models(l_star_model, rpni_model, num_tests=10000)
        model_diff = compare_learned_models_with_state_coverage(l_star_model, rpni_model, num_tests=10000)
        print('Counterexample found between models learned by RPNI and L*.')
        print(f'Models display different bahaviour for {round(model_diff * 100, 2)}% of test cases.')
    else:
        print('RPNI and L* learned same models.')
        if rpni_model.size != l_star_model.size:
            print(f'    Models do have different size.\n    RPNI {rpni_model.size} vs. L* {l_star_model.size}')

    print('----------------------------------------------------------------')