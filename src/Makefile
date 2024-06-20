# Makefile

#FC = gfortran
#FFLAGS = -fopenmp -O3 -ftree-parallelize-loops=8

FC = nvfortran

FFLAGS = -stdpar=multicore -fast -Minfo -Mpreprocess
#FFLAGS = -stdpar=gpu -fast -Minfo -acc -Mpreprocess

SRCS = chll.f90

OBJS = $(SRCS:.f90=.o)

EXEC = chll
# Default target

all: standalone f2py

# linker for standalone executable
standalone: $(OBJS)
	$(FC) $(FFLAGS) -o $@ $(OBJS)

# compilation
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

# f2py rule
f2py: $(SRCS)
	f2py -c -m chll_module $(SRCS)

clean:
	rm -f $(EXEC) $(OBJS)


