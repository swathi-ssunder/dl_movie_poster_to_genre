#!/bin/bash
echo "Password for user $1 to upload your images:"
echo "put datasets/posters/*
" | sftp -P 58022 $1@filecremers1.informatik.tu-muenchen.de:/$1/genre-predictor/datasets/posters/

