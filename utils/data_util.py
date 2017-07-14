import os.path
import csv
import random
import numpy as np
from PIL import Image as img

NUM_CHANNELS = 3
IMAGES_PATH = 'datasets/posters/'
DEFAULT_HEIGHT = 268
DEFAULT_WIDTH = 182

def load_data(limit=100, size=None, shuffle=False, filename='datasets/MovieGenre.csv'):
    """
    Load images and their genres

    Input:
    - limit: number of row to load
    - size: new size of each image
    - shuffle: shuffle rows
    - filename: path to csv file contain genre information

    Output:
    - data: images of shape (N, C, H, W)
    - labels: genre of images
    - genre_decoder: dict to translate from label to genre name
    """

    genre_set = set()
    records = get_id_and_genre(filename)
    limit = min(len(records), limit)
    if shuffle:
        random.shuffle(records)
    for record in records:
        for genre in record[1]:
            genre_set.add(genre)

    genre_encoder = {}
    num_genre = 0
    for genre in genre_set:
        genre_encoder[genre] = num_genre
        num_genre += 1
    genre_decoder = {v: k for k, v in genre_encoder.iteritems()}


    width = DEFAULT_WIDTH
    height = DEFAULT_HEIGHT
    if size:
        width = height = size
    data = np.zeros([limit, NUM_CHANNELS, width, height])
    labels = np.empty((limit,), dtype=object)

    for row in range(limit):
        index = records[row][0]
        genre = records[row][1]
        filename = IMAGES_PATH + str(index) + '.jpg'
        image = img.open(filename)
        image = image.resize((width, height))
        data[row] = np.asarray(image).T
        labels[row] = [genre_encoder[i] for i in genre]

    return (data.transpose([0, 1, 3, 2]), labels, genre_decoder)


def get_id_and_genre(filename):
    
    records = []

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            index = row['imdbId']
            filename = 'datasets/posters/' + str(index) + '.jpg'
            if not os.path.isfile(filename):
                continue
            genres = row['Genre'].split('|')
            records.append((index, genres))

    return records
