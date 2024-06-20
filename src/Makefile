# Makefile
.SUFFIXES:
# Compiler and flags
FC = nvfortran
#FFLAGS = -stdpar=gpu -fast -Minfo -acc -Mpreprocess
FFLAGS = -stdpar=multicore -fast -Minfo -Mpreprocess -Mconcur=cncall
F2PYFLAGS = -stdpar=multicore -fast -Minfo -Mpreprocess -tp=native -mp=ompt,multicore -Mconcur=cncall -lomp

# Executable name
EXEC = chll

# Source files
SRCS = chll_mod.f90 chll.f90

# Object files
OBJS = $(SRCS:.f90=.o)

# Default target
all: $(EXEC)

# Linking step
$(EXEC): $(OBJS)
	$(FC) $(FFLAGS) $(OBJS) -o $(EXEC)

# Compilation of source files
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

# Dependencies
chll.o: chll_mod.o

# f2py rule to create Python module
f2py: chll_mod.f90
	FC="$(FC)" FFLAGS="$(F2PYFLAGS)" python3 -m numpy.f2py -c chll_mod.f90 -m chll_mod --backend meson

clean:
	rm -f $(EXEC) $(OBJS) *.mod *.so


