import random

from aalpy.oracles import StatePrefixEqOracle
from aalpy.utils import generate_test_cases


def compare_learned_models_random_word(model_1, model_2, num_tests, min_test_len=3, max_test_len=16):
    diff = 0
    inputs = model_1.get_input_alphabet()
    for _ in range(num_tests):
        test_case = random.choices(inputs, k=random.randint(min_test_len, max_test_len))
        o_1 = model_1.compute_output_seq(model_1.initial_state, test_case)
        o_2 = model_2.compute_output_seq(model_2.initial_state, test_case)

        if o_1 != o_2:
            diff += 1

    return diff / num_tests


def compare_learned_models(model_1, model_2, test_cases):
    diff = 0

    for test_case in test_cases:
        o_1 = model_1.compute_output_seq(model_1.initial_state, test_case)
        o_2 = model_2.compute_output_seq(model_2.initial_state, test_case)

        if o_1 != o_2:
            diff += 1

    return diff / len(test_cases)


def create_test_cases(experiment_list, num_test_cases):
    test_cases = dict()
    for model_name, model in experiment_list:
        inputs = model.get_input_alphabet()
        walks_per_state = int(num_test_cases / model.size)
        eq_oracle = StatePrefixEqOracle(inputs, sul=None, walks_per_state=walks_per_state, walk_len=10)
        test_cases[model_name] = [tc[0] for tc in generate_test_cases(model, eq_oracle)]

    return test_cases