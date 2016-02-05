CC=g++
CFLAGS=-std=c++0x -Wall -DNDEBUG -O3
LDFLAGS=-L/usr/local/lib
ODIR=obj

DEPS=ivindex.h
LIBS=-lopencv_core -lopencv_highgui -lopencv_ml -lopencv_flann

OBJ1_o=ivindex.o quantize.o
OBJ1=$(patsubst %, $(ODIR)/%, $(OBJ1_o))

OBJ2_o=ivindex.o query.o
OBJ2=$(patsubst %, $(ODIR)/%, $(OBJ2_o))

OBJ3_o=kmeans.o
OBJ3=$(patsubst %, $(ODIR)/%, $(OBJ3_o))

all: ../quantize ../query ../kmeans

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

../quantize: $(OBJ1)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

../query: $(OBJ2)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

../kmeans: $(OBJ3)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o ../query ../quantize
