
import json
import nltk.tokenize as nltok
import os
import random
import tqdm

SEP = " +++$+++ "

# Preprocess into a training, a dev and a test set.
def read_lines(path, file_name):
    fn = os.path.join(path, file_name)
    with open(fn, 'r') as fid:
        lines = [l.strip().split(SEP) for l in fid]
    return lines

REPLACE = { # mix of character encodings in text..
    "\x96" : "-", "\x97" : "-", "\x85" : "...", "\xe9" : "e",
    "\xed" : "'", "\xea" : "e", "\xfc" : "u", "\xe4" : "a",
    "\xe8" : "e", "\xe0" : "a", "\xa3" : "$", "\xf3" : "o",
    "\xf1" : "n", "\x91" : "'", "\x92" : "'", "\x93" : "'",
    "\x94" : "'", "\xb9" : "'", "\xd4" : "'", "\xd5" : "'",
    "\xef" : "\"", "\xe7" : "c", "\xe2" : "a", "\x8c" : "'",
    "\xd2" : "'", "\xd3" : "'", "\x82" : "e", "\xb2" : "'",
    "\xb3" : "'", "\xb4" : "'", "\xc8" : "e", "\xe1" : "a",
    "\xc9" : "E", "\xfb" : "u", "\xf9" : "u", "\xad" : "-",
    "\xab" : "'", "\xdc" : "U", "\xc7" : "C", "\xfa" : "u",
    "\xb7" : "", "\xdf" : "", "\xa5" : "", "\x8a" : ""}

def clean(text):
    # TODO, should probably move this into
    # a shared function in the loader
    line = "".join(REPLACE.get(t, t) for t in text)
    line = line.lower()
    return nltok.word_tokenize(line)

def parse_convs(path):
    """
    Fields of movie_conversations.txt are:
        first charactar id, second character id,
        movie id, list of line ids

    Returns a list of conversations. Each conversation
    is a list of line ids.
    """
    convs = read_lines(path, "movie_conversations.txt")
    convs = [eval(conv[-1]) for conv in convs]
    return convs

def parse_lines(path):
    """
    Fields of movie_lines.txt are:
        line id, character id, movie id,
        character name, text

    Returns a dictionary of line id to a tuple of
    character id and text.
    """
    lines = read_lines(path, "movie_lines.txt")
    lines = {line[0] : (line[1], line[-1])
                for line in lines}
    return lines

def split(data, dev_size=1000, test_size=1000):
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
            conv = [{"text" : clean(text), "char_id" : char}
                    for char, text in conv]
            json.dump(conv, fid)
            fid.write("\n")

if __name__ == "__main__":
    data_path = "data/"
    convs = parse_convs(data_path)
    lines = parse_lines(data_path)

    # Join lines and conversations on line id
    convs = [[lines[c] for c in conv]
                for conv in convs]

    # Split conversations into train, val, test
    train, val, test = split(convs)
    print("Cleaning and writing training set..")
    save_json(data_path, "train", train)
    print("Cleaning and writing dev set..")
    save_json(data_path, "dev", val)
    print("Cleaning and writing test set..")
    save_json(data_path, "test", test)

