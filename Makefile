
lib/cv.o:
	gcc -fPIC -I./include/ -c src/cv.c -lm -o lib/cv.o

python: lib/cv.o
	cd python; python3 setup.py build_ext --inplace

test: python
	cd python; python3 test.py
