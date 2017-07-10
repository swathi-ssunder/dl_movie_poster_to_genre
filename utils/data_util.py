import os.path
import csv
import numpy as np
from PIL import Image as img

NUM_CHANNELS = 3
IMAGES_PATH = 'datasets/posters/'

def load_data(filename='datasets/MovieGenre.csv', width=182,
        height=268, limit=100, resize=False):
    """
    Load images and their genres

    Input:
    - filename: path to csv file contain genre information
    - height: height of image in pixel
    - width: width of image in pixel
    - limit: number of row to load
    - resize: True if image should be reshaped to square, default to False

    Output:
    - data: images of shape (N, C, H, W)
    - labels: genre of images
    """

    genre_set = set()
    records = get_id_and_genre(filename, limit)
    limit = len(records)
    for record in records:
        for genre in record[1]:
            genre_set.add(genre)

    genre_encoder = {}
    num_genre = 0
    for genre in genre_set:
        genre_encoder[genre] = num_genre
        num_genre += 1

    side = min(width, height)
    data = np.zeros([limit, NUM_CHANNELS, width, height]) if not resize \
            else np.zeros([limit, NUM_CHANNELS, side, side])
    labels = np.zeros([limit])
    for row in range(limit):
        index = records[row][0]
        genre = records[row][1]
        filename = IMAGES_PATH + str(index) + '.jpg'
        image = img.open(filename)
        if resize:
            image = image.resize((side, side))
        data[row] = np.asarray(image).T
        labels[row] = genre_encoder[genre[0]]
    return (data.transpose([0, 1, 3, 2]), labels)


def get_id_and_genre(filename, limit):
    
    records = []

    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            if len(records) >= limit:
                break
            index = row['imdbId']
            filename = 'datasets/posters/' + str(index) + '.jpg'
            if not os.path.isfile(filename):
                continue
            genres = row['Genre'].split('|')
            records.append((index, genres))

    return records
