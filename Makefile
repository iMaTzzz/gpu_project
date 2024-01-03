CC = gcc
LD = gcc
# NVCC COMPILER OPTIONS:
NVCC = nvcc
NVCC_FLAGS=
NVCC_LIBS=


# -O0 désactive les optimisations à la compilation
# C'est utile pour débugger, par contre en "production"
# on active au moins les optimisations de niveau 2 (-O2).
CFLAGS = -Wall -Wextra -std=c99 -Iinclude -O2
LDFLAGS = -lm

# Par défaut, on compile tous les fichiers source (.c) qui se trouvent dans le
# répertoire src/
SRC_FILES=$(wildcard src/*.c)
CUDA_SRC_FILES=$(wildcard src/*.cu)

# Par défaut, la compilation de src/toto.c génère le fichier objet obj/toto.o
OBJ_FILES=$(patsubst src/%.c,obj/%.o,$(SRC_FILES))
CUDA_OBJ_FILES = $(patsubst src/%.cu,obj/%.o,$(CUDA_SRC_FILES))

# Fichiers objet "prof". Lorsque vous aurez implémenté un de ces modules, il
# faudra le retirer de cette liste pour lier ppm2jpeg à votre implémentation du
# module en question. Le module htables_prof.o, qui contient la déclaration des tables
# de Huffman génériques sous forme de constantes, n'est pas à ré-implémenter.
OBJ_PROF_FILES = obj_prof/htables_prof.o \
                 obj_prof/bitstream_prof.o

all: ppm2jpeg

ppm2jpeg: $(OBJ_FILES) $(OBJ_PROF_FILES) $(CUDA_OBJ_FILES)
	$(LD) $(OBJ_FILES) $(OBJ_PROF_FILES) $(CUDA_OBJ_FILES) $(LDFLAGS) -o $@

# Compile C source files to obejct files:
obj/%.o: src/%.c
	$(CC) -c $(CFLAGS) $< -o $@

# Compile CUDA source files to object files:
obj/%.o : src/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

.PHONY: clean

clean:
	rm -rf ppm2jpeg $(OBJ_FILES) $(CUDA_OBJ_FILES)
