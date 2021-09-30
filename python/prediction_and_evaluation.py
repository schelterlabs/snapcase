"""
generate the user vectors before run this.
"""
import argparse
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

from dataset import Dataset
from metrics import get_precision_recall_fscore, get_ndcg, has_hit_in_top_k

seed = 1027


def load_user_vectors_from_file(path, vector_len):
    with open(path, "r") as fp:
        all_lines = fp.readlines()
    customer_ids = []
    user_vector_dict = {}
    user_rows = [line for line in all_lines if line.startswith("USER-")]  # filter user vector rows
    for r in user_rows:
        _, user_id, vector_str = r.split("-")
        customer_ids.append(int(user_id))
        # convert user vector to a numpy array
        user_vector = np.zeros(vector_len)
        vector_str = vector_str.strip()
        idx_vals = vector_str.split(";")
        for x in idx_vals:
            idx, val = x.split(":")
            user_vector[int(idx)] = float(val)
        user_vector_dict[int(user_id)] = user_vector
    return customer_ids, user_vector_dict

def load_recommendation_vectors_from_file(path, vector_len):
    with open(path, "r") as fp:
        all_lines = fp.readlines()
    customer_ids = []
    user_vector_dict = {}
    user_rows = [line for line in all_lines if line.startswith("RECO-")]  # filter user vector rows
    for r in user_rows:
        _, user_id, vector_str = r.split("-")
        customer_ids.append(int(user_id))
        # convert user vector to a numpy array
        user_vector = np.zeros(vector_len)
        vector_str = vector_str.strip()
        idx_vals = vector_str.split(";")
        for x in idx_vals:
            idx, val = x.split(":")
            user_vector[int(idx)] = float(val)
        user_vector_dict[int(user_id)] = user_vector
    return customer_ids, user_vector_dict

def predict(user_vector_path, history_path, num_neighbours=200, alpha=0.5, topn=20, test_split=0.1):
    # load_history dataset
    ds_hist = Dataset()
    ds_hist.load_from_file(history_path)
    # load user vectors
    customer_ids, user_vector_dict = load_user_vectors_from_file(path=user_vector_path,
                                                                 vector_len=len(ds_hist.item_ids))

    print("Loading DD recommendations...")
    customer_ids_reco, reco_vector_dict = load_recommendation_vectors_from_file(path=user_vector_path,
                                                                 vector_len=len(ds_hist.item_ids))

    customer_ids.sort()
    random.seed(seed)
    print("random seed:", seed)
    # these customer only have 1 basket, hence no future prefiction label, don't remember why remove 2939...
    # customer_ids_copy.remove(2939)
    # sample a fraction of all customers as test customers
    customer_ids_copy = customer_ids.copy()
    test_customer_ids = random.sample(customer_ids_copy, int(test_split * len(customer_ids_copy)))
    test_customer_ids.sort()
    training_user_mat = np.vstack([user_vector_dict[cid] for cid in customer_ids])

    # create test user vector matrix and an index to customer id mapping
    test_user_vecs = []
    index_to_cid = {}
    for i, cid in enumerate(test_customer_ids):
        test_user_vecs.append(user_vector_dict[cid])
        index_to_cid[i] = cid
    test_user_mat = np.vstack(test_user_vecs)

    print(f"Num of training users: {len(customer_ids)}")
    print(f"Num of test users: {len(test_customer_ids)}")

#     print("Searching for neighbors for test users in training users...")
#     # find neighbors
#     nbrs = NearestNeighbors(n_neighbors=num_neighbours, algorithm='brute').fit(training_user_mat)
#     distances, indices = nbrs.kneighbors(test_user_mat)
#     print(f"Started generating predictions...")
#     final_prediction_vector = dict()
#     # the indices are row number in training_user_mat, note it includes the test user itself!
#     for index_list in indices:
#         # index from 1 to skip the target user herself
#         neighbours = training_user_mat[index_list[1:], :]
#         nbrs_mean = np.mean(neighbours, axis=0)
#         target_cust_idx = index_list[0]
#         target_cust_id = index_to_cid[target_cust_idx]
#         final_prediction_vector[target_cust_id] = alpha * training_user_mat[target_cust_idx] + (1 - alpha) * nbrs_mean
#     # now we just need to sort the prediction vector to arrive at the final top n predictions
#
    final_prediction_vector = reco_vector_dict

    topn_recommendations = dict()
    for cid, pred_v in final_prediction_vector.items():
        topn_items = pred_v.argsort()[::-1][:topn] + 1
        # note that we convert both customer id and items ids to str to later evaluation
        topn_recommendations[str(cid)] = topn_items.astype("str")
    return topn_recommendations


def evaluate(predictions, ground_truth, topn):
    all_precisions = []
    all_recalls = []
    all_fscores = []
    all_ndcgs = []
    all_hits = 0
    for uid, predict_item_ids in predictions.items():
        # note that future basket are stored as list of 1 basket, hence the extra index 0
        target_item_ids = ground_truth[uid]  # these contain item IDs counting from 1!

        precision, recall, fscore, _ = get_precision_recall_fscore(target_item_ids, predict_item_ids, topn)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_fscores.append(fscore)

        ndcg = get_ndcg(target_item_ids, predict_item_ids, topn)
        all_ndcgs.append(ndcg)
        all_hits += has_hit_in_top_k(target_item_ids, predict_item_ids, topn)

    recall = np.mean(all_recalls)
    precision = np.mean(all_precisions)
    fscore = np.mean(all_fscores)
    ndcg = np.mean(all_ndcgs)
    hit_ratio = all_hits / len(predictions)
    return recall, precision, fscore, ndcg, hit_ratio


def main(args):
    print("Started running prediction...")
    predictions = predict(
        user_vector_path=args.user_vector_path,
        num_neighbours=args.num_neighbours,
        history_path=args.history_path,
        alpha=args.alpha,
        topn=args.topn,
        test_split=args.test_split)
    print(f"Started loading the ground truth labels")
    ground_truth_data = Dataset()
    ground_truth_data.load_from_file(args.ground_truth_path)
    ground_truth = {cid: array[0][1] for cid, array in ground_truth_data.customer_baskets.items()}
    # evaluate metrics
    print("Started evaluating...")
    topn = 10
    recall, precision, fscore, ndcg, hit_ratio = evaluate(predictions=predictions,
                                                          ground_truth=ground_truth,
                                                          topn=topn)
    print(f'top n: {topn}')
    print(f'recall@{topn}:{recall}')
    print(f'ndcg@{topn}: {ndcg}')
    topn = 20
    recall, precision, fscore, ndcg, hit_ratio = evaluate(predictions=predictions,
                                                          ground_truth=ground_truth,
                                                          topn=topn)
    print(f'top n: {topn}')
    print(f'recall@{topn}:{recall}')
    print(f'ndcg@{topn}: {ndcg}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_vector_path", default="../tifu-Instacart.txt")
    parser.add_argument("--history_path", default="../datasets/nbr/Instacart_history.csv")
    parser.add_argument("--ground_truth_path", default="../datasets/nbr/Instacart_future.csv")
    parser.add_argument("--num_neighbours", default=300, type=int, help="the number of neighbors")
    parser.add_argument("--alpha", default=0.7, type=float, help="the prediction vector weight")
    parser.add_argument("--topn", default=20, type=int, help="the topn recommendations")
    parser.add_argument("--test_split", default=1.0, type=float, help="the fraction of users we want to evaluate")
    args = parser.parse_args()
    print(args)
    main(args)
