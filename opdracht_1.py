import operator
import random
import math
from typing import List

import numpy as np
import pandas as pd


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


def make_train_test_set(dataset, split, training_set=[], test_set=[]):
    '''
    Om een algoritme te testen moet de data random in train en test data gesplit worden.
    De rede is omdat anders de volgorde van de dataset de accuraatheid van het algoritme
    kan bepalen.

    Dit wordt volbracht door de parameter "split". "split" kan varieren van 0 tot 1 en
    geeft daarmee aan hoe groot de trainings data ongeveer wordt (split = 0 = 0%,
    = 0.5 = 50%, 0,8 = 80%).

    De werking van split:
    In de functie zit een functie "random" die per aanroeping een nieuw cijfer geeft
    (van 0.0 tot 1.0). Als het cijfer onder de split is wordt de rij van de dataset
    ingedeelt bij de trainingsset en anders in de testset. Als split dus een hoger getal
    is is er meer kans dat de rij in de trainingset komt.
    '''
    for row in dataset:
        if random.random() < split:
            training_set.append(row)
        else:
            test_set.append(row)
    print('Total: ' + repr(len(dataset)))
    print('Train: ' + repr(len(training_set)))
    print('Test: ' + repr(len(test_set)))


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
    for x in range(1, length):
        if instance1[x] == -1 or instance2[x] == -1:
            pass
        else:
            distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(training_set, test_instance, k):
    '''
    Dit is de functie die de neighbours bij elkaar zet die het dichtst bij de
    test_instance zit. De hoeveelheid aan neighbours wordt bepaald door k.

    De werking:
    De variabel length krijgt de lengte van de test_instance -1. De rede hiervoor omdat
    dit getal wordt gebruikt in de euclidean_distance functie en dit nummer indiceerd
    de kolommen die gebruikt kunnen worden in de formule (oftewel alle kolommen behalve
    "season").

    Hierna wordt per test item in training_set de afstand tot de test_instance gemeten.
    Al deze data wordt in een list gestopt met tuples per trainings item de bijpassende
    afstand. Hierna worden all deze afstand gesorteerd op grote van het tweede item door
    het gebruik van itemgetter(1). Hiervan worden de hoeveelheid van K in de neighbors
    list toegevoegt.
    '''
    distances = []
    length = len(test_instance) - 1
    for trainings_item in training_set:
        dist = euclidean_distance(test_instance, trainings_item, length)
        distances.append((trainings_item, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    '''
        Deze functie telt alle neighbors seizoenen bij elkaar op en bekijkt welke van
        de vier seizoenen de meeste is. Dit seizoen wordt teruggegeven.

        De werking
        de class_votes wordt een dict die gevult wordt met de seizoenen als key en de
        hoe vaak het binnen de neighbours voorkomt. Hierdoor kan de dict class_votes
        sorteren op de items met het grootste aantal (oftewel die het meeste voorkomt)
        en de eerste key terug geven.
    '''
    class_votes = {}
    for neighbor in neighbors:
        response = neighbor[-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def get_accuracy(test_set, predictions):
    '''
        Dit is een simpele functie die de predictions en test_set met elkaar controleerd
        en daaruit een percentage haalt wat goed ging.
    '''
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0

def k_nearest_neighbour_algorithm(training_set, test_set, k):
    predictions = []
    for test_item in test_set:
        neighbors = get_neighbors(training_set, test_item, k)
        result = get_response(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(test_item[-1]))

    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

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

# Test met random split
complete_dataset = gen_csv_from_txt_to_ndarray(file='dataset1.csv', delimeter=';', dtype=dtype, converters=converters)
set_meteorological_seasons(complete_dataset)
training_set = []
test_set = []
split = 0.95
make_train_test_set(complete_dataset, split, training_set, test_set)
k = 20
k_nearest_neighbour_algorithm(training_set=training_set, test_set=test_set, k=k)

print("\n\n\n\n\n\n")
# Test met days.csv
complete_testset = gen_csv_from_txt_to_ndarray(file='days.csv', delimeter=';', dtype=dtype, converters=converters)

k_nearest_neighbour_algorithm(training_set=complete_dataset, test_set=complete_testset, k=k)
