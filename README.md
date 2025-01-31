# ChLLModel
Chiral version of Lebwohl Lasher Model

doi:10.5281/zenodo.14781847

Instructions:

To compile, change to `chll`, edit Makefile for system specific options (GPU/CPU) and compile shared library.

To run, import into Python script `from chll import chll` and instantiate a ChllSim class with necessary parameters and initial state. See examples in `scripts` directory.
