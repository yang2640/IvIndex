CC=g++
CFLAGS=-std=c++0x -Wall -DNDEBUG -O3 -I/home/yzhou/HeIvIndex/sift/vlfeat
LDFLAGS=-L/usr/local/lib
ODIR=obj

DEPS=
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lvl 

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
