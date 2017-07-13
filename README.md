# dl_movie_poster_to_genre
This is a project undertaken during the Deep Learning for Computer Vision at TUM.

Goal:   Prediction of genres of a movie from its poster


## Setup
1. Run `./get_datasets.sh` to get the datasets
2. Run `python -m unittest test.data_util_test` to test `data_util`
3. To obtain datasets in remote machine, run `./transfer_data.sh s***` to 
    transfer images, make sure they are ready in your machine, and dir 
    `~/genre-predictor/datasets/posters` is available in the remote machine
    
    Note that running `./get_datasets.sh` in remote machine doesn't work since we
    don't have sudo access in there
