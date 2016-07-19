CC=g++
CFLAGS=-c -std=c++14 -g -O1
LDFLAGS=-larmadillo 
SOURCES=main.cpp chol.cpp conj_grad.cpp special_mat.cpp newton_cg.cpp trust_region.cpp bfgs.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=chap5

all: $(SOURCES) $(EXECUTABLE) 

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
