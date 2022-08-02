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

    # z_2d_array.reverse()
    z_2d_array = list(map(list, zip(*z_2d_array)))

    print(x)
    print(y)
    print(z_2d_array)

    sns.heatmap(z_2d_array, xticklabels=x, yticklabels=sorted(y, reverse=True), annot=True, cmap="Greens")
    plt.show()


if __name__ == '__main__':
    model = load_automaton_from_file('automata/BLE/CC2640R2-no-feature-req.dot', automaton_type='mealy')
    experiment_data = increasing_parameters_exp(model)

    print(experiment_data)

    plot_heatmap(experiment_data)
