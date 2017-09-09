"""
This file mostly from
https://github.com/inikdom/opensubtitles-parser
"""
import xml.etree.ElementTree as ET
import os
import re
import errno
import tqdm

raw_file = "raw.txt"
inc = 0

def main():
    data_dir = "data"

    files = find_files(data_dir)
    fid = open("data/raw.txt", 'w')
    for f in tqdm.tqdm(files):
        try:
            extract(f, fid)
        except KeyboardInterrupt:
            print("Process stopped by user...")
            return 0
        except Exception as e:
            print(e)
            print("Error in " + f)
            pass

def find_files(directory):
    xml_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.xml'):
                xml_files.append(os.path.join(root, f))
    return xml_files

def extract(xml_file, fid):
    '''
    The assumption is made that each <s> node in the
    xml docs represents a token, meaning everything
    has already been tokenized.
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for child in root.findall('s'):
        A = []
        for node in child.getiterator():
            if node.tag == 'w':
                A.append(node.text.encode('ascii', 'ignore').replace('-', ''))
        text = " ".join(A)
        text = clean(text)
        if len(text) == 0:
            continue
        if text[0] != '[' and text[-1] != ':':
            fid.write(text + "\n")
    # Empty line denotes end of movie
    fid.write("\n")

def clean(text):
    '''
    This function removes funky things in text. There
    is probably a much better way to do it, but unless
    the token list is much bigger this shouldn't
    really matter how inefficient it is.
    '''
    t = text.strip('-')
    t = t.lower()
    t = t.strip('\"')
    regex = re.compile('\(.+?\)')
    t = regex.sub('', t)
    t.replace('  ', ' ')
    regex = re.compile('\{.+?\}')
    t = regex.sub('', t)
    t = t.replace('  ', ' ')
    t = t.replace("~", "")
    t = t.strip(' ')
    return t

if __name__ == "__main__":
    main()
