#!/bin/bash
curl https://kaggle2.blob.core.windows.net/forum-message-attachments/197363/6743/posters.7z --output posters.7z
mkdir -p datasets/posters
7z x posters.7z -odatasets/
rm posters.7z

# remove malformed images
cnt=0
for f in datasets/posters/*.jpg
do
    out="$(jpeginfo $f)"
    if [[ $out == *"ERROR"* ]]
    then
        let "cnt++"
        rm $f
    fi
done
echo "Done removing $cnt malformed images"
