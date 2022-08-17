cd yolov5
bash build.sh
cd -

cd torchreid
ln -s ../ai_util
ln -s ../yolov5
cd -

cd insightface
ln -s ../ai_util
ln -s ../yolov5_face
cd -