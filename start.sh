FILE_NAME=./logs/$(date "+%m%d_%H%M").log
echo $FILE_NAME
nohup python -u train.py >> $FILE_NAME 2>&1 &