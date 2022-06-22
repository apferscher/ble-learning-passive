from math import ceil

from aalpy.SULs import MealySUL
from aalpy.learning_algs import run_Lstar, run_RPNI
from aalpy.oracles import StatePrefixEqOracle, RandomWordEqOracle
from aalpy.utils import generate_test_cases

from data_generation import generate_random_data


def compare_learned_models(model_1, model_2, test_cases):
    diff = 0

    for test_case in test_cases:
        o_1 = model_1.compute_output_seq(model_1.initial_state, test_case)
        o_2 = model_2.compute_output_seq(model_2.initial_state, test_case)

        if o_1 != o_2:
            diff += 1

    return diff / len(test_cases)


def create_test_cases(experiment_list, num_test_cases, method):
    assert method in {'coverage', 'random'}
    test_cases = dict()
    for model_name, model in experiment_list:
        inputs = model.get_input_alphabet()
        walks_per_state = ceil(num_test_cases / model.size)
        if method == 'coverage':
            eq_oracle = StatePrefixEqOracle(inputs, sul=None, walks_per_state=walks_per_state, walk_len=10)
        else:
            # min size: size of smallest model
            # max size: doubled size of largest model
            eq_oracle = RandomWordEqOracle(inputs, sul=None, num_walks=num_test_cases, min_walk_len=3,
                                           max_walk_len=16 * 2)
        test_cases[model_name] = [tc[0] for tc in generate_test_cases(model, eq_oracle)]

    return test_cases


def compare_mealy_and_moore_learning():
    from aalpy.learning_algs import run_RPNI
    from aalpy.utils import load_automaton_from_file

    from data_generation import generate_random_data

    model = load_automaton_from_file('automata/CYBLE-416045-02.dot', automaton_type='mealy')
    data = generate_random_data(model, num_sequences=10000, min_sequence_len=3, max_sequence_len=8)

    model_mealy = run_RPNI(data, automaton_type='mealy', input_completeness='sink_state')

    model_moore = run_RPNI(data, automaton_type='moore', input_completeness='sink_state')

    tc_random = create_test_cases([('ex1', model)], 10000, 'random')['ex1']
    # tc_coverage = create_test_cases([('ex1', model)], 10000, 'coverage')

    res_random1 = compare_learned_models(model, model_mealy, tc_random)
    res_random2 = compare_learned_models(model, model_moore, tc_random)

    print(f'Mealy {round((1 - res_random1) * 100)}, Moore {round((1 - res_random2) * 100)}')


def increasing_parameters_exp(target_model, step_increase=2, query_increase=100, num_increases=10):
    # if you want to examine effect of increasing steps without increasing the number of queries simply put
    # query_increase to 0, and vice versa if you want to have steps fixed
    # Returns: a list of tuples (num_queries, query_len, rpni_model_acc)

    sul = MealySUL(target_model)
    alphabet = target_model.get_input_alphabet()
    eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=100, min_walk_len=4, max_walk_len=8)

    l_star_model, data = run_Lstar(alphabet, sul, eq_oracle, 'mealy', print_level=0, return_data=True)

    learning_queries = data['queries_learning'] + data['queries_eq_oracle']
    avg_query_steps = int((data['steps_learning'] + data['steps_eq_oracle']) / learning_queries)

    tc = create_test_cases([('ex1', target_model)], 10000, 'coverage')['ex1']

    experiment_data = []
    curr_steps = avg_query_steps
    curr_queries = learning_queries
    for _ in range(num_increases):
        curr_steps += step_increase
        curr_queries += query_increase

        random_data = generate_random_data(target_model, num_sequences=learning_queries,
                                           min_sequence_len=curr_steps, max_sequence_len=curr_steps)

        rpni_model = run_RPNI(random_data, 'mealy', print_info=True, input_completeness='sink_state')

        non_conformance = compare_learned_models(l_star_model, rpni_model, tc)
        conformance = round((1 - non_conformance) * 100, 2)

        experiment_data.append((curr_queries, curr_steps, conformance))

    return experiment_data
