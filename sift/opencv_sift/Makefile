CC=g++
CFLAGS=-std=c++0x -Wall -DNDEBUG -O3 -I/usr/local/include
LDFLAGS=-L/usr/local/lib
ODIR=obj

DEPS=
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_nonfree -lopencv_features2d

OBJ1_o=sift.o 
OBJ1=$(patsubst %, $(ODIR)/%, $(OBJ1_o))

all: sift

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

sift: $(OBJ1)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)


.PHONY: clean

clean:
	rm -f $(ODIR)/*.o sift 
