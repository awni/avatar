
import json
import os
import random
import tqdm

def split(data, dev_size=5000, test_size=5000):
    random.shuffle(data)
    held_out = dev_size + test_size
    train = data[held_out:]
    dev = data[test_size:held_out]
    test = data[:test_size]
    return train, dev, test

def save_json(path, name, dataset):
    file_name = os.path.join(path, name + ".json")
    with open(file_name, 'w') as fid:
        for conv in tqdm.tqdm(dataset):
            conv = [{"text" : text} for text in conv]
            json.dump(conv, fid)
            fid.write("\n")

def read_convs(data_path):
    raw = os.path.join(data_path, "raw.txt")
    with open(raw, 'r') as fid:
        convs = []
        query = None
        for l in fid:
            reply = l.strip().split()
            if not l:
                query = None
            if query:
                convs.append((query, reply))
            query = reply
    return convs

if __name__ == "__main__":
    data_path = "data/"

    convs = read_convs(data_path)

    # Split conversations into train, val, test
    train, val, test = split(convs)
    print("Writing training set..")
    save_json(data_path, "train", train)
    print("Writing dev set..")
    save_json(data_path, "dev", val)
    print("Writing test set..")
    save_json(data_path, "test", test)

