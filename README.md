# CUDA-Drill_Struct
Outputs a "carved" FCC structure with normally distributed holes inside it

INTRODUCTION

This software demonstrates the use of CUDA in building chemical structure files used in many applications (molecular dynamics, monte carlo simulations, ab initio optimizations, etc.). It is meant both as a template with different functions that someone enthusiastic about parallel computing may use to get started, and also as a base model for someone looking to build customized structures that they are unable to find and/or want to add to existing repositories.

This particular code builds simple FCC (Face-Centered Cubic) lattices with realistic defects, using the cuRAND library to create gaussian-distributed cavities or pores inside the lattice. These defects do not only arise from natural deterioration but may also be done in purpose by experimental means, in order to increase surface area of catalysts, for instance.

Because overhead is so low for this kernel, and because the focus is to expand on the use of cuRAND to build interesting patterns in existing structures, the "drilling" kernel is done with little regard to branch divergence, instead attempting to be optimized with the use of float4 variables and the use of the rsqrt() (or rather, a double precision version of it), which is faster than the intrinsic sqrt() function, and has enough accuracy for most purposes.

The lattice output consists of a .xyz file (VMD - http://www.ks.uiuc.edu/Research/vmd/ and related formats). The file looks like:

N

a

X1 x.xxxxx y.yyyyy z.zzzzz

X2 x.xxxxx y.yyyyy z.zzzzz

X3 x.xxxxx y.yyyyy z.zzzzz

X4 x.xxxxx y.yyyyy z.zzzzz

X5 x.xxxxx y.yyyyy z.zzzzz

X6 x.xxxxx y.yyyyy z.zzzzz

...

Where N is the total number of (x,y,z) triples, 'a' is a line containing the lattice parameter, and each of the following N lines contain the (x,y,z) triple of coordinates for each crystalline site, as well as atomic elements Xn. There should be no overlapping atoms at the same lattice site.

COMPILING THE SOURCE

The code requires a CUDA capable machine of compute capability 2.x or higher, and a working version of the CUDA developer toolkit(https://developer.nvidia.com/cuda-toolkit), as well as the cuRAND library. Recommended to compile with one of the following in terminal:

-> General optimization flags

nvcc -O3 -arch=sm_xx --ptxas-options=-v -o CUDA_Driller2.0.x CUDADriller_vs2.0.cu

-> In case compiler can't find your libraries

nvcc -O3 -arch=sm_xx --ptxas-options=-v -o CUDA_Driller2.0.x CUDADriller_vs2.0.cu -lm -lcurand

-> In case of memcpy_inline errors (usually with Ubuntu 16.04 and most versions of CUDA)

nvcc -O3 -arch=sm_XX --ptxas-options=-v -o CUDA_Driller2.0.x CUDADriller_vs2.0.cu -lm -lcurand -D_FORCE_INLINES

In all of the above, change the XX in -arch=sm_XX to the compute capability of the GPU you're using, e.g for a Tesla K20x (compute capability 3.5) we would use -arch=sm_35.

USING THE PROGRAM

This program needs to be run in the same folder as an input file called "cheesinput.dat" that must have the following parameters:

XX YY - Atomic Elements (supports binary alloys but support for more elements can be added in code)

X.XXXXX - Lattice parameter

XX.XXXX YY.YYYY ZZ.ZZZZ - Full size of structure (in Angstrons)

XX - Number of defects

XX.XXXX - Average size of defects

XX.XXXX - Standard deviation of defect size distribution

XXXXXXXXXXX - Long integer for pseudorandom number generation

A sample cheesinput.dat file is presented in this repository. This code may be used in combination with https://github.com/mgmonteiro/CUDA-crystal_surface in order to generate different kinds of structures. running it with ./ inside a folder with a valid input parameters file, will create two inputs, one being the coord_z.xyz corresponding to the structure with defects, and a file named porus.dat which is organized as:

XX YY ZZ WW

XX YY ZZ WW

...

Where each (x,y,z) is the position of the pores relative to (0,0,0), and in units of lattice parameters, and w is the size of the defect also in units of lattice parameters. Different lattices can be generated by using different "selection rules" during the allocation of the initial structure. For more information on how to build arbitrary structures see, for instance, DOI:10.5151/phypro-sic100-046.
