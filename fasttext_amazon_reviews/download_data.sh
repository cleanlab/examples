# taken and adapted from: https://github.com/cleanlab/label-errors/releases/tag/amazon-reviews-dataset

# path to save the data
DATA_PATH=data/

# create data folder if it doesn't exist
mkdir -p $DATA_PATH

# download data
wget -P $DATA_PATH --continue https://github.com/cgnorthcutt/label-errors/releases/download/amazon-reviews-dataset/amazon5core.tar.gz-partaa;
wget -P $DATA_PATH --continue https://github.com/cgnorthcutt/label-errors/releases/download/amazon-reviews-dataset/amazon5core.tar.gz-partab;

# extract files
cat ${DATA_PATH}amazon5core.tar.gz-part?? | unpigz | tar -xv -C $DATA_PATH;

# pre-process data for training
cat ${DATA_PATH}amazon5core.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > ${DATA_PATH}amazon5core.preprocessed.txt