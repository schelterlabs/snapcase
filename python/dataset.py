import json
import os
import time

import pandas as pd

from utils import BasketVectorizer


class Dataset:
    def __init__(self):
        self.customer_baskets = dict()
        # this is to store the sparse matrix
        self.customer_baskets_vectorized = dict()
        self.item_ids = None
        self.customer_ids = None

    def __repr__(self):
        res = f"customers: {len(self.customer_ids)}, unique items: {len(self.item_ids)}"
        return res

    def load_from_file(self, file_path):
        # note that it is important to keep order_number as int, so that it is sorted properly, that order is preserved.
        df = pd.read_csv(file_path, dtype={"CUSTOMER_ID": str, "ORDER_NUMBER": int, "MATERIAL_NUMBER": str})
        print(f"number of records in {file_path}: {len(df)}")
        for customer_id, customer_baskets_df in df.groupby("CUSTOMER_ID"):
            self.customer_baskets[customer_id] = []
            for order_id, basket in customer_baskets_df.groupby("ORDER_NUMBER"):
                items_in_basket = basket["MATERIAL_NUMBER"].values
                self.customer_baskets[customer_id].append((order_id, items_in_basket))
        # update other attributes based on customer_baskets
        self._update_stats()

    def _update_stats(self):
        self.number_of_baskets_per_customer = dict()
        customer_ids = set()
        item_ids = set()
        for _cid, _baskets in self.customer_baskets.items():
            customer_ids.add(_cid)
            self.number_of_baskets_per_customer[_cid] = len(_baskets)
            for _order_id, _b in _baskets:
                item_ids.update(_b)
        # we first sort the item ids, then convert it to string to be used as vocabulary
        # note that this is done in two steps to avoid miss sorting, [1,2,3..11,] into [1,11,2,3]
        self.item_ids = sorted(item_ids, key=lambda x: int(x))  # sort a dict will return a list
        self.customer_ids = sorted(customer_ids, key=lambda x: int(x))  # sort a dict will return a list

    def vectorize(self, vocabulary=None):
        if vocabulary is None:
            vocabulary = self.item_ids
        bc = BasketVectorizer(vocabulary=vocabulary)
        for cid in self.customer_ids:
            self.customer_baskets_vectorized[cid] = []
            for order_id, raw_basket in self.customer_baskets[cid]:
                vec_tuple = (order_id, bc.transform([raw_basket], toarray=False)[0])
                self.customer_baskets_vectorized[cid].append(vec_tuple)

    def prune(self, min_baskets_per_customer=2, min_items_per_basket=2):
        """DONT USE THIS YET. USE the prune function in tifuknn"""
        # removing small baskets
        for _cid, _baskets in self.customer_baskets.items():
            left_baskets = []
            for _basket in _baskets:
                if len(_basket) >= min_items_per_basket:
                    left_baskets.append(_basket)
            self.customer_baskets[_cid] = left_baskets
        # remove customer with less than min_baskets_per_customer
        self.customer_baskets = {_cid: _baskets for _cid, _baskets in self.customer_baskets.items()
                                 if len(_baskets) >= min_baskets_per_customer}
        self._update_stats()

    def to_vocab(self, file_path):
        if ds.item_ids is None:
            raise ValueError("no vocab to save, item_ids missing")
        df = pd.DataFrame({"itemId": ds.item_ids})
        df.to_csv(file_path, index=False)

    def to_json_baskets(self, jsondata_path):
        # For simulating file source streaming
        for customerId, list_of_baskets in self.customer_baskets.items():
            # suppose the list_of_baskets is sorted
            for orderId, basket_array in list_of_baskets:
                d_temp = {"customerId": int(customerId), "orderId": orderId,
                          "basket": list(map(lambda x: int(x), basket_array.tolist())), "isDeletion": False}

                fp = os.path.join(jsondata_path, f"customer_{customerId}_order_{orderId}.csv")
                json.dump(d_temp, open(fp, "w"))
                print(f"Writing {fp} ...")
                time.sleep(0.1)


if __name__ == '__main__':
    ds = Dataset()
    ds.load_from_file("../datasets/TaFang_history_NB.csv")
    print(ds)
