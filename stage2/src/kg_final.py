import json


movies = {}
with open('douban2fb.txt', 'rb') as f:
    for line in f:
        line = line.strip()
        list1 = line.decode().split('\t')
        # print(list1)
        movies[list1[1]] = int(list1[0])

movies_2 = {}
with open('movie_id_map.txt', 'rb') as f:
    for line in f:
        line = line.strip()
        list1 = line.decode().split('\t')
        # print(list1)
        movies_2[int(list1[0])] = int(list1[1])

for key in movies.keys():
    movies[key] = movies_2[movies[key]]
print(movies)

with open("knowledge_graph.json", 'r') as f:
    kg_dict = json.load(f)

kg_list = []
relation_dict = {}
for key1 in kg_dict.keys():
    list1 = kg_dict[key1]
    for list2 in list1:
        relation1 = list2[0]
        dict2 = list2[1]
        if relation1 not in relation_dict.keys():
            relation_dict[relation1] = len(relation_dict.keys())
        for key2 in dict2.keys():
            if key2 not in movies.keys():
                movies[key2] = len(movies.keys())
                print(len(movies.keys()))
            kg_list.append([movies[key1], relation_dict[relation1], movies[key2]])

            list3 = dict2[key2]
            for list4 in list3:
                relation2 = list4[0]
                dict3 = list4[1]
                if relation2 not in relation_dict.keys():
                    relation_dict[relation2] = len(relation_dict.keys())
                for key3 in dict3.keys():
                    if key3 not in movies.keys():
                        movies[key3] = len(movies.keys())
                        print(len(movies.keys()))
                    kg_list.append([movies[key2], relation_dict[relation2], movies[key3]])

set1 = set(kg_list)
with open('kg_final.txt', 'w') as f:
    for list1 in set1:
        f.write(str(list1[0]))
        f.write(" ")
        f.write(str(list1[1]))
        f.write(" ")
        f.write(str(list1[2]))
        f.write("\n")
