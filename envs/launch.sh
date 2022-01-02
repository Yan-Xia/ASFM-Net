docker run -itd --rm --gpus all -p 6006:6006 -v $PWD/../:/ASFM-Net --name asfm-net asfm-net /bin/bash
docker exec -it asfm-net /bin/bash