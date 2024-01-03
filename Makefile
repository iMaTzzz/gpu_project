CC = gcc
LD = nvcc

# -O0 désactive les optimisations à la compilation
# C'est utile pour débugger, par contre en "production"
# on active au moins les optimisations de niveau 2 (-O2).
CFLAGS = -Wall -Wextra -std=c99 -Iinclude -O2
LDFLAGS = -lm

# Par défaut, on compile tous les fichiers source (.c) qui se trouvent dans le
# répertoire src/
SRC_FILES=$(wildcard src/*.c)

# Par défaut, la compilation de src/toto.c génère le fichier objet obj/toto.o
OBJ_FILES=$(patsubst src/%.c,obj/%.o,$(SRC_FILES))

# Fichiers objet "prof". Lorsque vous aurez implémenté un de ces modules, il
# faudra le retirer de cette liste pour lier ppm2jpeg à votre implémentation du
# module en question. Le module htables_prof.o, qui contient la déclaration des tables
# de Huffman génériques sous forme de constantes, n'est pas à ré-implémenter.
OBJ_PROF_FILES = obj_prof/htables_prof.o \
                 obj_prof/bitstream_prof.o 

# Nouveau règle pour compiler les fichiers .cu
obj/%.cu.o: src/%.cu
    nvcc -c $(CFLAGS) $< -o $@

all: ppm2jpeg

ppm2jpeg: $(OBJ_FILES) $(OBJ_PROF_FILES) obj/quantification_gpu.cu.o
    nvcc $(OBJ_FILES) $(OBJ_PROF_FILES) obj/quantification_gpu.cu.o $(LDFLAGS) -o $@

.PHONY: clean

clean:
    rm -rf ppm2jpeg $(OBJ_FILES)