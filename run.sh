if [ $# -lt 1 ]
then
    echo "input not enough, please give me a experiment name "
    exit -1
fi

cp main.py main_$1.py

for name in japanvowels natops fingermovement lsst wordrecognition conll2000 arabicdigit heartbeat facedetect brown_100 wingbeat
do
    # python -u main_$1.py data/$name/seqs.fdt $1/$name results/train/$name/models.pkl # inference
    python -u train.py data/$name/seqs.fdt $name  # for train
done

# rm main_$1.py
echo "Done."




