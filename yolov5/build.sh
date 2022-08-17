python setup.py build_ext --inplace

cd utils/mot/cmot_tools
python setup.py build_ext --inplace
cd -

cd utils/ui/
python setup.py build_ext --inplace
cd -


mkdir -p bytetrack/deploy/cbytetrack/build
cd bytetrack/deploy/cbytetrack/build
cmake ..
make -j8
cd -


cd bytetrack/
python setup.py develop
cd -
