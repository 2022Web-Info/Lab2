import copy
import gzip
import json

import pandas as pd
import csv

movie_tags = pd.read_csv("Movie_tag.csv")
movie = {}
for i in range(len(movie_tags)):
    movie[movie_tags["id"][i]] = movie_tags['tag'][i]

# print(movie.keys())

movie_entity = {}
with open('douban2fb.txt', 'rb') as f:
    for line in f:
        line = line.strip()
        list1 = line.decode().split('\t')
        # print(list1)
        movie_entity[list1[1]] = {"tag": movie[int(list1[0])]}


data = json.dumps(movie_entity, indent=1, ensure_ascii=False)
# print(data)
with open('movie_entity.json', 'w', newline='\n') as f:
    f.write(data)
# print(movie_entity)

triples = {}
triples1 = {}
for key in movie_entity.keys():
    triples[key] = []
# triples1 = copy.deepcopy(triples)
# print(triples)

with gzip.open('freebase_douban.gz', 'rb') as f:
    i = 0
    for line in f:
        i = i + 1
        if i % 10000000 == 0:
            print(i)
        line = line.strip()
        list1 = line.decode().split('\t')
        patten_str = r"<http://rdf.freebase.com/ns/"
        if (patten_str not in list1[0]) or (patten_str not in list1[2]):
            continue
        word = list1[0][len(patten_str):].strip('>')
        word2 = list1[2][len(patten_str):].strip('>')
        if word in triples.keys():
            triples[word].append([list1[1], {word2: []}])
        if word2 in triples.keys():
            if word not in triples1.keys():
                triples1[word] = []
            triples1[word].append([list1[1], {word2: []}])
        # print(list1)
        # if i == 10000000:
        #     break

triples2 = copy.deepcopy(triples)
for key1 in triples2.keys():
    list1 = triples2[key1]
    for i1 in range(len(list1)):
        list2 = list1[i1]
        dict1 = list2[1]
        for key2 in dict1.keys():
            # print(triples2[key1][i1][1])
            if key2 in triples.keys():
                triples2[key1][i1][1][key2] = triples[key2]
            elif key2 in triples1.keys():
                triples2[key1][i1][1][key2] = triples1[key2]
                # print(triples1[key2])

data = json.dumps(triples2, indent=1)
# print(data)
with open('knowledge_graph.json', 'w', newline='\n') as f:
    f.write(data)

    # for item in triples.items():
    #     if len(item[1]) > 0:
    #         json.dump(item, f)
    #         json.dump('\n', f)
