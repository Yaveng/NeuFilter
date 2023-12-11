from __future__ import division
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import scale
import pandas as pd
import random


def load_network(args, time_scaling=True):
    datapath = args.datapath
    user_sequence = []
    item_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []

    f = open(datapath,"r")
    f.readline()
    for cnt, l in enumerate(f):
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        if start_timestamp is None:
            start_timestamp = float(ls[2])
        timestamp_sequence.append(float(ls[2]) - start_timestamp)
        if len(ls) > 3:
            y_true_labels.append(int(ls[3])) # label = 1 at state change, 0 otherwise
            feature_sequence.append(list(map(float,ls[4:])))
        else:
            y_true_labels.append(0)
            feature_sequence.append([0.0,0.0])
    f.close()

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    nodeid = 0
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items)
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    if time_scaling:
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, \
        feature_sequence, \
        y_true_labels]

