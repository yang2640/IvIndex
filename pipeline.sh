ext="mser.sift"
sample="data/cluster/samples.sift"
if [ -e "$sample" ]
then 
  rm "$sample"
fi
find data/Images -name "*.$ext"|sort|while read x;do echo $x;shuf -n 600 $x|sed '/^$/d' >> $sample;done

find data/Images -name "*.$ext"|sort>data/featlist

./kmeans

./quantize

./query > query.txt
