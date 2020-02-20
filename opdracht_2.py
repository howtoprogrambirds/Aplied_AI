from datetime import datetime
import operator
import random
import math
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gen_csv_from_txt_to_ndarray(file: str, dtype: List, delimeter: str = None, converters: dict = None) -> np.array:
    '''
        Deze functie is gemaakt om van een csv text een structured numpy (door de
        gegeven dtype) te maken. Hierbij kunnen delimeters en converters toegevoegd
        worden.

        Gebruik bij deze opdracht:
        - Er wordt een ";" als delimeter meegegeven omdat de text verdeelt wordt bij ";"
        - Er worden de standaard converters meegegeven die de -1 veranderen in een 0
        - Ik heb gekozen om een dtype mee te geven zodat men een structured ndarray kan
            maken dit verzorgt ervoor dat men "headers" heeft en zo makkelijke de collom
            kan pakken bij naam in plaats van index.
        - Ik heb er ook voor gekozen om de seizoenen en data in één array te stoppen
            hierdoor heb je een makkelijk data structuur en is het randomizen van de
            data nauwkeuriger. Omdat alle rijen gelijk blijven.
        - Dus eerst wordt alle data uit de text gehaald en in een numpy array gezet die
            niet de seizoenen bevat. Daarna wordt een numpy structured array(met dezelfde
            hoeveelheid rijen) aangemaakt met null waardes. Deze worden gevult
            met de data uit de text.
    '''
    data = np.genfromtxt(file,
                         delimiter=delimeter,
                         converters=converters,
                         dtype=dtype[:-1],
                         )
    result = np.empty(len(data), dtype=dtype)
    for name in result.dtype.names:
        if name != 'season':
            result[name] = data[name]
    return result

def set_meteorological_seasons(dataset):
    '''
        Dit is een functie die in de structured numpy array het kolom seizoen("season") vult.
        Het is de oude implementatie alleen is het verandert zodat de datastructure van een
        structured array gebruikt kan worden.
    '''
    for row in range(0, len(dataset)):
        if dataset["date"][row] < 20000301:
            dataset["season"][row] = 'winter'
        elif 20000301 <= dataset["date"][row] < 20000601:
            dataset["season"][row] = 'lente'
        elif 20000601 <= dataset["date"][row] < 20000901:
            dataset["season"][row] = 'zomer'
        elif 20000901 <= dataset["date"][row] < 20001201:
            dataset["season"][row] = 'herfst'
        else:  # from 01-12 to end of year
            dataset["season"][row] = 'winter'

def euclidean_distance(instance1, instance2, length):
    '''
    Als men de afstand van twee punten wilt weten kan dat op twee manieren gemeten worden.
    Men kan de Manhattan distance meten en/of euclidean_distance.

    Om het verschil uit te leggen gebruiken we een land kaart:
    - Manhattan distance zijn alle wegen en routes die men moet nemen om van punt a naar
        b te gaan.
    - Euclidean distance is alsof bij punt a een vliegtuig pakt en een rechte lijn trekt
        naar punt b.

    Omdat Manhattan distance hierbij niet van toepassing is gebruiken we de Euclidean
    distance. Deze afstand wordt gemeten door het kwadraat van het verschil te nemen en
    daarvan de wortel. Als er meerdere dimensies (gegeven door length) zijn worden
    eerst alle kwadraten bij elkaar opgeteld en daarvan de wortel genomen.
    '''
    distance = 0
    for x in range(length):
        if instance1[x] == -1 or instance2[x] == -1:
            pass
        else:
            distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def assignment(df, centroids):
    nearest_distances = {}
    for i in centroids.keys():
        row = math.inf
        nearest_distance = math.inf
        nearest_distances[i] = [row, nearest_distance]
        furthest_distance = 0
        distances = []
        for item, rows in df.iterrows():
            dist = euclidean_distance(np.array(rows), np.array(centroids[i])[0], 8)
            distances.append(dist)
            if dist < nearest_distances[i][1]:
                nearest_distances[i][1] = dist
                nearest_distances[i][0] = rows["season"]
            if dist > furthest_distance:
                furthest_distance = dist
        centroids[i]["season"] = nearest_distances[i][0]
        df['distance_from_{}'.format(i)] = distances
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    return df

def update(centroids, df):
    for i in centroids.keys():
        for name in centroids[i].columns.values:
            if name != "season":
                centroids[i][name] = np.mean(df[df['closest'] == i][name])
    return centroids

def get_season_by_centroid(df, centroids):
    print("the default seasons", np.unique(df["season"].values))
    for index, rows in df.iterrows():
        df.loc[index, "season"] = centroids[rows["closest"]]["season"].values[0]
        # rows["season"] = centroids[rows["closest"]]["season"]
    print("the seasons chosen by centroid(s)", np.unique(df["season"].values))
    return df

def get_furthest_distance_intra_cluster(df, centroids):
    furthest_distances = []
    for key in centroids.keys():
        try:
            furthest_distances.append(max(df.loc[df["closest"] == key][f"distance_from_{key}"].values))
        except Exception:
            continue
    return max(furthest_distances)

dtype_float = np.dtype(float)
dtype_int = np.dtype(int)
dtype = [('date', dtype_int),
         ('avg_WS', dtype_float),
         ('avg_TEMP', dtype_float),
         ('min_TEMP', dtype_float),
         ('max_TEMP', dtype_float),
         ('amount_S', dtype_float),
         ("time_precipitation", dtype_float),
         ("sum_precipitation", dtype_float),
         ("season", np.dtype("U8")),
         ]
converters = {5: lambda s: 0 if s == b"-1" else float(s),
              7: lambda s: 0 if s == b"-1" else float(s)}

complete_dataset = gen_csv_from_txt_to_ndarray(
    file='dataset1.csv', delimeter=';', dtype=dtype, converters=converters
)
set_meteorological_seasons(complete_dataset)

df_without_answer = pd.DataFrame(complete_dataset)

furthest_distances = []
k_list = []

for k in range(1, 10):
    df = pd.DataFrame(complete_dataset)
    intra_cluster_furthest_distances = []
    centroids = {
        i+1: df.sample() for i in range(k)
    }

    df = assignment(df, centroids)

    print("relocate centriods")
    update(centroids, df)
    while True:
        closest_centroids = df['closest'].copy(deep=True)
        centroids = update(centroids, df)
        df = assignment(df, centroids)
        if closest_centroids.equals(df['closest']):
            break
        else:
            print("relocate centriods")

    df = get_season_by_centroid(df, centroids)

    good = 0
    wrong = 0
    for i in range(len(df["season"].values)):
        if df["season"].values[i] == df_without_answer["season"].values[i]:
            good += 1
        else:
            wrong += 1

    k_list.append(k)
    print(f"K = {k} with {good} good and {wrong} wrong ({int((good / (good + wrong)) * 100.0)}%) ")
    furthest_distances.append(get_furthest_distance_intra_cluster(df, centroids))
plt.plot(k_list, furthest_distances)
plt.xlabel("k")
plt.title("k-Means plot")
plt.ylabel("Max intra cluster distance")
plt.show()




