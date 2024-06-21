# Makefile
.SUFFIXES:
# Compiler and flags

#NVFORTRAN
FC = nvfortran
#FFLAGS = -acc=gpu -fast -Minfo=all -Mconcur -Mnobounds -acclibs -Bstatic-nvidia
FFLAGS = -acc=multicore -fast -Minfo=all -Mconcur=cncall -Minline=reshape -time=real -Mnobounds -acclibs
#F2PYFLAGS = -acc=gpu -fast -Minfo=all -Mconcur -Mnobounds -acclibs -Bstatic-nvidia
F2PYFLAGS = -acc=multicore -fast -Minfo=all -Mconcur=cncall -Minline=reshape -time=real -Mnobounds -acclibs

#GFORTRAN
#FC = gfortran
#FFLAGS = -fopenacc -foffload=nvptx-none -foffload=-lgfortran -Ofast -march=native -ffast-math -ffree-line-length-none
#FFLAGS = -fopenacc -foffload=nvptx-none -O2 -march=native -ffree-line-length-none
#F2PYFLAGS = -acc=gpu -fast -Minfo=all -Mconcur -Mnobounds -acclibs -Bstatic-nvidia
#F2PYFLAGS = -acc=multicore -fast -Minfo=all -Mconcur=cncall -Minline=reshape -time=real -Mnobounds -Bstatic-nvidia -acclibs

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
	#CC="nvc" FC="$(FC)" FFLAGS="$(F2PYFLAGS)" python3 -m numpy.f2py --verbose -c chll_mod.f90 -m chll_mod --backend meson --lower
	FC="$(FC)" FFLAGS="$(F2PYFLAGS)" python3 -m numpy.f2py --verbose -c chll_mod.f90 -m chll_mod --backend meson --lower

shared: chll_mod.f90
	$(FC) $(FFLAGS) -shared chll_mod.f90 -o libchll_mod.so
clean:
	rm -f $(EXEC) $(OBJS) *.mod *.so


