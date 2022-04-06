import pandas as pd
import numpy as np
import time
import sys

from vsknn import VSKnn

# Samples a set of random evolving sessions to predict for
def random_queries(evolving_sessions, num_queries):
    evolving_session_ids = evolving_sessions.SessionId.unique()
    sampled_evolving_session_ids = np.random.choice(evolving_session_ids, size=num_queries)

    sampled_evolving_sessions = {id: [] for id in sampled_evolving_session_ids}

    for index, row in evolving_sessions.iterrows():
        if row['SessionId'] in sampled_evolving_sessions:
            sampled_evolving_sessions[row['SessionId']].append(row['ItemId'])

    query_sessions = {}

    for session_id, items in sampled_evolving_sessions.items():
        random_session_length = np.random.randint(len(items) - 1) + 1
        query_sessions[session_id] = items[:random_session_length]

    return query_sessions


def run_experiment(dataset, historical_sessions_file, evolving_sessions_file,
                   num_clicks_to_delete, num_queries_to_evaluate):

    historical_sessions = pd.read_csv(historical_sessions_file, sep='\t')
    evolving_sessions = pd.read_csv(evolving_sessions_file, sep='\t')


    for num_queries in num_queries_to_evaluate:

        if dataset == "ecom60m" and num_queries > 100:
            print(f"#Skipping num_queries={num_queries} for ecom60m")
            continue

        print(f'# {dataset} - num_queries={num_queries},num_clicks_to_delete={num_clicks_to_delete}',
              file=sys.stderr)
        query_sessions = random_queries(evolving_sessions, num_queries)

        historical_clicks = historical_sessions.copy(deep=True)

        for run in range(num_clicks_to_delete):
            row_to_delete = np.random.choice(historical_clicks.index, 1, replace=False)
            historical_clicks = historical_clicks.drop(row_to_delete)

            start_time = time.time()

            vsknn = VSKnn(k=100)
            vsknn.fit(historical_clicks)

            for query_session_id, query_items in query_sessions.items():
                for query_item in query_items:
                    for item_to_add in query_items[:-1]:
                        # We only add this click and skip the predictions
                        vsknn.predict_next(query_session_id, item_to_add, skip=True)

                    item_to_predict_for = query_items[-1]
                    vsknn.predict_next(query_session_id, item_to_predict_for)

            duration = time.time() - start_time
            print(f'vs_python,deletion_performance,{dataset},{num_queries},{duration * 1000}')


num_queries_to_evaluate = [100, 1000, 10000]
# Wish we could do more, but the experiment takes already too long
num_clicks_to_delete = 20

for seed in [42, 767, 999]:
    run_experiment('ecom1m', "../../datasets/session/bolcom-clicks-1m_train.txt",
                   "../../datasets/session/bolcom-clicks-1m_test.txt",
                   num_clicks_to_delete, num_queries_to_evaluate)

    run_experiment('rsc15', "../../datasets/session/rsc15-clicks_train_full.txt",
                   "../../datasets/session/rsc15-clicks_test.txt",
                   num_clicks_to_delete, num_queries_to_evaluate)

    run_experiment('ecom60m', "../../datasets/session/bolcom-clicks-50m_train.txt",
                    "../../datasets/session/bolcom-clicks-50m_test.txt",
                    num_clicks_to_delete, num_queries_to_evaluate)