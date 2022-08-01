
projs=("." "ai_util" "insightface" "torchreid" "yolov5" "yolov5_face")

for proj in ${projs[@]}
do
cd $proj
echo $proj
git status
cd - > /dev/null
echo "-----"
done

for proj in ${projs[@]}
do
cd $proj
echo $proj
git pull
git push
cd - > /dev/null
echo "-----"
done
