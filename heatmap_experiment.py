from math import sqrt

from aalpy.SULs import MealySUL
from aalpy.learning_algs import run_Lstar, run_RPNI
from aalpy.oracles import StatePrefixEqOracle
from aalpy.utils import load_automaton_from_file

from data_generation import generate_random_data
from model_comparison import create_test_cases, compare_learned_models


def increasing_parameters_exp(model):
    sul = MealySUL(model)
    alphabet = model.get_input_alphabet()
    eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=25, walk_len=25)

    l_star_model, data = run_Lstar(alphabet, sul, eq_oracle, 'mealy', print_level=0, return_data=True)

    learning_queries = data['queries_learning'] + data['queries_eq_oracle']
    avg_query_steps = int((data['steps_learning'] + data['steps_eq_oracle']) / learning_queries)

    validation_test_cases = create_test_cases([('ex1', model)], 10000, 'coverage')['ex1']

    experiment_data = []

    num_of_queries_multipliers = list(range(1, 11))
    num_of_steps = list(range(5, 26, 2))

    for query_multiplier in num_of_queries_multipliers:
        for steps in num_of_steps:
            random_data = generate_random_data(model, num_sequences=query_multiplier * learning_queries,
                                               min_sequence_len=steps - 2, max_sequence_len=steps + 2)

            rpni_model = run_RPNI(random_data.data, 'mealy', print_info=True, input_completeness='sink_state')

            non_conformance = compare_learned_models(model, rpni_model, validation_test_cases)
            conformance = round((1 - non_conformance) * 100, 2)

            experiment_data.append((query_multiplier * learning_queries, steps, conformance))

    return experiment_data


def plot_heatmap(experiment_data):
    import matplotlib.pylab as plt
    import seaborn as sns

    num_increases = int(sqrt(len(experiment_data)))

    exp_data_values = {(i[0], i[1]): i[2] for i in experiment_data}

    x, y, z = [i[0] for i in experiment_data], [i[1] for i in experiment_data], [i[2] for i in experiment_data]
    x, y = sorted(list(set(x))), sorted(list(set(y)))

    z_2d_array = []
    for i in x:
        row = []
        for j in y:
            row.append(exp_data_values[(i, j)])
        row.reverse()
        z_2d_array.append(row)

    # transpose
    z_2d_array = list(map(list, zip(*z_2d_array)))

    print(x)
    print(y)
    print(z_2d_array)

    sns.heatmap(z_2d_array, xticklabels=x, yticklabels=sorted(y, reverse=True), annot=True, cmap="Greens")
    plt.show()


if __name__ == '__main__':
    model = load_automaton_from_file('automata/MQTT/mosquitto__two_client_will_retain.dot', automaton_type='mealy')
    experiment_data = increasing_parameters_exp(model)

    print(experiment_data)

    plot_heatmap(experiment_data)

# automata/BLE/CC2640R2-no-feature-req.dot
# [(979, 5, 52.94), (979, 7, 69.98), (979, 9, 72.36), (979, 11, 78.52), (979, 13, 75.97), (979, 15, 87.17), (979, 17, 78.81), (979, 19, 84.64), (979, 21, 92.25), (979, 23, 90.67), (979, 25, 95.02), (1958, 5, 55.94), (1958, 7, 76.97), (1958, 9, 88.47), (1958, 11, 90.76), (1958, 13, 87.76), (1958, 15, 93.41), (1958, 17, 98.21), (1958, 19, 98.62), (1958, 21, 100.0), (1958, 23, 93.98), (1958, 25, 100.0), (2937, 5, 67.97), (2937, 7, 75.67), (2937, 9, 87.88), (2937, 11, 90.72), (2937, 13, 92.69), (2937, 15, 98.62), (2937, 17, 100.0), (2937, 19, 98.35), (2937, 21, 100.0), (2937, 23, 100.0), (2937, 25, 100.0), (3916, 5, 66.33), (3916, 7, 83.22), (3916, 9, 87.55), (3916, 11, 93.87), (3916, 13, 100.0), (3916, 15, 98.41), (3916, 17, 100.0), (3916, 19, 100.0), (3916, 21, 100.0), (3916, 23, 100.0), (3916, 25, 100.0), (4895, 5, 72.25), (4895, 7, 84.23), (4895, 9, 89.89), (4895, 11, 98.32), (4895, 13, 100.0), (4895, 15, 100.0), (4895, 17, 100.0), (4895, 19, 100.0), (4895, 21, 100.0), (4895, 23, 100.0), (4895, 25, 100.0), (5874, 5, 72.78), (5874, 7, 86.53), (5874, 9, 92.21), (5874, 11, 95.63), (5874, 13, 100.0), (5874, 15, 100.0), (5874, 17, 98.52), (5874, 19, 98.68), (5874, 21, 100.0), (5874, 23, 100.0), (5874, 25, 100.0), (6853, 5, 67.72), (6853, 7, 84.52), (6853, 9, 95.75), (6853, 11, 100.0), (6853, 13, 100.0), (6853, 15, 100.0), (6853, 17, 100.0), (6853, 19, 100.0), (6853, 21, 100.0), (6853, 23, 100.0), (6853, 25, 100.0), (7832, 5, 76.23), (7832, 7, 86.27), (7832, 9, 96.65), (7832, 11, 100.0), (7832, 13, 98.28), (7832, 15, 100.0), (7832, 17, 100.0), (7832, 19, 100.0), (7832, 21, 100.0), (7832, 23, 100.0), (7832, 25, 100.0), (8811, 5, 77.28), (8811, 7, 95.56), (8811, 9, 95.46), (8811, 11, 100.0), (8811, 13, 100.0), (8811, 15, 100.0), (8811, 17, 100.0), (8811, 19, 100.0), (8811, 21, 100.0), (8811, 23, 100.0), (8811, 25, 100.0), (9790, 5, 76.55), (9790, 7, 89.38), (9790, 9, 97.0), (9790, 11, 98.19), (9790, 13, 100.0), (9790, 15, 100.0), (9790, 17, 100.0), (9790, 19, 100.0), (9790, 21, 100.0), (9790, 23, 100.0), (9790, 25, 100.0)]

# automata/MQTT/mosquitto__two_client_will_retain.dot
# [(3697, 5, 62.97), (3697, 7, 85.2), (3697, 9, 83.9), (3697, 11, 86.54), (3697, 13, 89.61), (3697, 15, 93.79), (3697, 17, 96.46), (3697, 19, 97.21), (3697, 21, 93.88), (3697, 23, 95.2), (3697, 25, 96.62), (7394, 5, 78.61), (7394, 7, 84.79), (7394, 9, 89.65), (7394, 11, 97.96), (7394, 13, 96.62), (7394, 15, 97.33), (7394, 17, 96.99), (7394, 19, 99.22), (7394, 21, 96.64), (7394, 23, 97.55), (7394, 25, 97.94), (11091, 5, 73.4), (11091, 7, 86.08), (11091, 9, 90.5), (11091, 11, 97.55), (11091, 13, 96.56), (11091, 15, 98.85), (11091, 17, 98.87), (11091, 19, 99.03), (11091, 21, 98.84), (11091, 23, 98.88), (11091, 25, 98.94), (14788, 5, 79.72), (14788, 7, 89.67), (14788, 9, 94.84), (14788, 11, 96.28), (14788, 13, 97.3), (14788, 15, 98.64), (14788, 17, 98.45), (14788, 19, 99.14), (14788, 21, 99.08), (14788, 23, 99.04), (14788, 25, 99.92), (18485, 5, 82.36), (18485, 7, 88.9), (18485, 9, 95.32), (18485, 11, 98.91), (18485, 13, 97.5), (18485, 15, 98.14), (18485, 17, 99.14), (18485, 19, 98.7), (18485, 21, 99.55), (18485, 23, 98.95), (18485, 25, 99.17), (22182, 5, 77.67), (22182, 7, 94.5), (22182, 9, 96.56), (22182, 11, 98.41), (22182, 13, 98.85), (22182, 15, 99.79), (22182, 17, 99.94), (22182, 19, 99.98), (22182, 21, 100.0), (22182, 23, 100.0), (22182, 25, 98.81), (25879, 5, 84.09), (25879, 7, 93.56), (25879, 9, 98.8), (25879, 11, 98.22), (25879, 13, 98.33), (25879, 15, 99.25), (25879, 17, 99.98), (25879, 19, 99.7), (25879, 21, 100.0), (25879, 23, 99.95), (25879, 25, 100.0), (29576, 5, 86.03), (29576, 7, 94.13), (29576, 9, 99.03), (29576, 11, 98.28), (29576, 13, 98.92), (29576, 15, 99.18), (29576, 17, 99.91), (29576, 19, 100.0), (29576, 21, 100.0), (29576, 23, 99.99), (29576, 25, 100.0), (33273, 5, 78.83), (33273, 7, 97.4), (33273, 9, 98.53), (33273, 11, 98.96), (33273, 13, 99.06), (33273, 15, 98.99), (33273, 17, 98.9), (33273, 19, 99.31), (33273, 21, 100.0), (33273, 23, 100.0), (33273, 25, 100.0), (36970, 5, 80.88), (36970, 7, 96.28), (36970, 9, 98.23), (36970, 11, 98.9), (36970, 13, 99.55), (36970, 15, 99.48), (36970, 17, 99.55), (36970, 19, 99.95), (36970, 21, 99.79), (36970, 23, 100.0), (36970, 25, 100.0)]
