#!/usr/bin/env python3
import os
from collections import defaultdict
from tqdm import tqdm
import pickle

MIN_SESSION_NUM = 2

def get_user_session(f_name, session=3600):
    all_session = []
    with open(f_name, 'r') as in_f:
        for i, line in tqdm(enumerate(in_f)):
            user, behaviors = line.strip().split('\t')
            behavior_list = [tuple(items.split('@')) for items in behaviors.split(',')]
            sess_data = []
            start_idx = [0]
            for index, (_item_id, _type, ts) in enumerate(behavior_list):
                sess_data.append((ts, _item_id))
                if index == 0:  continue
                if int(ts) - int(behavior_list[index-1][-1]) >= session:
                    start_idx.append(index)
                
            start_idx.append(len(sess_data))
            for idx in range(len(start_idx)-1):
                sess = sess_data[start_idx[idx]:start_idx[idx+1]]
                if len(sess) <= MIN_SESSION_NUM:
                    continue

                all_session.append(sess)

    print(f'Total session : {len(all_session)}')

    with open('suumo_sess_data.csv', 'w') as out_f:
        print("SessionID,Time,ItemID", file=out_f)
        for idx, session in tqdm(enumerate(all_session)):
            for ts, item_id in session:
                print(f"{idx},{ts},{item_id}", file=out_f)


if __name__ == '__main__':
    tag = 'summo_web'
    dt = '2020-08-01'
    data_f = f"../solr_recall/data/{tag}/{dt}/merged.data"

    get_user_session(data_f)

