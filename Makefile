all: compile


compile:
	mkdir -p build && cd build && cmake ../csrc/poker_dice -DCMAKE_BUILD_TYPE=Debug -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../cfvpy -DPYTHON_EXECUTABLE:FILEPATH=/projectnb/ece601/cliao25/rebel/.conda/envs/rebel/bin/python && make -j

compile_slow:
	mkdir -p build && cd build && cmake ../csrc/poker_dice -DCMAKE_BUILD_TYPE=Debug -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../cfvpy -DPYTHON_EXECUTABLE:FILEPATH=/projectnb/ece601/cliao25/rebel/.conda/envs/rebel/bin/python && make -j2

test: | compile
	make -C build test
	nosetests cfvpy/

clean:
	rm -rf build cfvpy/rela*so
