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