# Makefile
.SUFFIXES:
# Compiler and flags

#NVFORTRAN
#FC = nvfortran
#FFLAGS = -acc=gpu -fast -Minfo=all -Mconcur -Mnobounds -acclibs -Bstatic-nvidia
#FFLAGS = -acc=multicore -fast -Minfo=all -Mconcur=cncall -Minline=reshape -time=real -Mnobounds -Bstatic-nvidia -acclibs
#F2PYFLAGS = -acc=gpu -fast -Minfo=all -Mconcur -Mnobounds -acclibs -Bstatic-nvidia
#F2PYFLAGS = -acc=multicore -fast -Minfo=all -Mconcur=cncall -Minline=reshape -time=real -Mnobounds -Bstatic-nvidia -acclibs

#GFORTRAN
FC = gfortran
#FFLAGS = -fopenacc -foffload=nvptx-none -foffload=-lgfortran -Ofast -march=native -ffast-math -ffree-line-length-none
FFLAGS = -v -fopenacc -foffload=disable -foffload-options=-lgfortran -Ofast -march=native -ffast-math -ffree-line-length-none -fno-stack-protector
#F2PYFLAGS = -acc=gpu -fast -Minfo=all -Mconcur -Mnobounds -acclibs -Bstatic-nvidia
F2PYFLAGS = -acc=multicore -fast -Minfo=all -Mconcur=cncall -Minline=reshape -time=real -Mnobounds -Bstatic-nvidia -acclibs

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


clean:
	rm -f $(EXEC) $(OBJS) *.mod *.so


