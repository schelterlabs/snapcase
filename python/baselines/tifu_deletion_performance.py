import numpy as np
import pandas as pd
import time
import sys

from tifuknn import read_claim2vector_embedding_file_no_vector, partition_the_data, \
    temporal_decay_sum_history, KNN, merge_history, predict_with_elements_in_input

training_chunk = 0
test_chunk = 1

def tifu_predictions(data_chunk, training_key_set, test_key_set, input_size, num_nearest_neighbors,
                     within_decay_rate, group_decay_rate, alpha, group_size):

    topk = 20
    next_k_step = 1

    temporal_decay_sum_history_training = temporal_decay_sum_history(data_chunk[training_chunk],
                                                                 training_key_set, input_size,
                                                                 group_size, within_decay_rate,
                                                                 group_decay_rate)

    temporal_decay_sum_history_test = temporal_decay_sum_history(data_chunk[training_chunk],
                                                             test_key_set, input_size,
                                                             group_size, within_decay_rate,
                                                             group_decay_rate)

    index, distance = KNN(temporal_decay_sum_history_test, temporal_decay_sum_history_training,
                      num_nearest_neighbors)

    sum_history = merge_history(temporal_decay_sum_history_test, test_key_set,
                                temporal_decay_sum_history_training,
                                training_key_set, index, alpha)

    num_ele = topk
    for iter in range(len(test_key_set)):
        input_variable = data_chunk[training_chunk][test_key_set[iter]]
        target_variable = data_chunk[training_chunk][test_key_set[iter]]

        if len(target_variable) < 2 + next_k_step:
            continue

        output_vectors = predict_with_elements_in_input(sum_history, test_key_set[iter])
        top = 400
        hit = 0
        for idx in range(len(output_vectors)):
            # for idx in [2]:

            output = np.zeros(input_size)
            target_topi = output_vectors[idx].argsort()[::-1][:top]
            c = 0
            for i in range(top):
                if c >= num_ele:
                    break
                output[target_topi[i]] = 1
                c += 1

            vectorized_target = np.zeros(input_size)
            for ii in target_variable[1 + idx]:
                vectorized_target[ii] = 1

def run_experiment(dataset, num_queries_to_evaluate, num_baskets_to_delete, baskets_file, num_nearest_neighbors,
                   within_decay_rate, group_decay_rate, alpha, group_size):

    for num_queries in num_queries_to_evaluate:

        print(f'# {dataset} - num_queries={num_queries},num_baskets_to_delete={num_baskets_to_delete}',
              file=sys.stderr)

        data_chunk, input_size, code_freq_at_first_claim = read_claim2vector_embedding_file_no_vector([baskets_file])

        training_key_set = list(data_chunk[0].keys())
        training_key_set = [key for key in training_key_set if len(data_chunk[0][key]) >= 3]
        test_key_set = np.random.choice(training_key_set, size=num_queries)

        deletion_candidates = [key for key in training_key_set if key not in test_key_set]

        keys_with_deletions = np.random.choice(deletion_candidates, size=num_baskets_to_delete)

        for key_with_deletion in keys_with_deletions:

            baskets = data_chunk[0][key_with_deletion]

            random_basket_to_remove = np.random.randint(len(baskets))
            baskets_after_removal = [basket for index, basket in enumerate(baskets)
                                     if index != random_basket_to_remove]

            data_chunk[0][key_with_deletion] = baskets_after_removal



            start_time = time.time()
            tifu_predictions(data_chunk, training_key_set, test_key_set, input_size, num_nearest_neighbors=300,
                                 within_decay_rate=0.9, group_decay_rate=0.7, alpha=0.7, group_size=7)

            duration = time.time() - start_time
            print(f'tifu_python,deletion_performance,{dataset},{num_queries},{duration * 1000}')


num_queries_to_evaluate = [10, 100, 1000]
num_baskets_to_delete = 100

for seed in [42, 767, 999]:

    np.random.seed(seed)

    run_experiment(dataset = 'tafang', num_queries_to_evaluate=num_queries_to_evaluate,
                   num_baskets_to_delete=num_baskets_to_delete,
                   baskets_file = '/Users/ssc/projects/snapcase/datasets/nbr/TaFang_history_NB.csv',
                   num_nearest_neighbors=300, within_decay_rate=0.9, group_decay_rate=0.7, alpha=0.7, group_size=7)

    run_experiment(dataset = 'instacart', num_queries_to_evaluate=num_queries_to_evaluate,
                   num_baskets_to_delete=num_baskets_to_delete,
                   baskets_file = '/Users/ssc/projects/snapcase/datasets/nbr/Instacart_history.csv',
                   num_nearest_neighbors=900, within_decay_rate=0.9, group_decay_rate=0.7, alpha=0.9, group_size=3)

    run_experiment(dataset = 'valuedshopper', num_queries_to_evaluate=num_queries_to_evaluate,
                   num_baskets_to_delete=num_baskets_to_delete,
                   baskets_file = '/Users/ssc/projects/snapcase/datasets/nbr/VS_history_order.csv',
                   num_nearest_neighbors=300, within_decay_rate=1.0, group_decay_rate=6, alpha=0.9, group_size=7)
