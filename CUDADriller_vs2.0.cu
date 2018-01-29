/*  Copyright 2018 Maxwel Gama Monteiro Junior

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
limitations under the License.

|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||							 ||||||||||||||||
|||||||||||||||| CUDADriller						 ||||||||||||||||
|||||||||||||||| 							 ||||||||||||||||
|||||||||||||||| Code Version: 2.0					 ||||||||||||||||
|||||||||||||||| Last updated: 06/2017					 ||||||||||||||||
|||||||||||||||| Author: Maxwel Gama Monteiro Junior			 ||||||||||||||||
|||||||||||||||| Built based on DOI:10.5151/phypro-sic100-046		 ||||||||||||||||
		 Contact:maxweljr@gmail.com			         ||||||||||||||||
								_____	 ||||||||||||||||
							       /.    \	 ||||||||||||||||
							      /   . . \	 ||||||||||||||||
							      \	.   . /	 ||||||||||||||||
							       \_____/	 ||||||||||||||||
                       A___A						 ||||||||||||||||
           A___A       |o o|						 ||||||||||||||||
     ____ / o o \      |='=|						 ||||||||||||||||
___/~____   ='= /_____/    |_________					 ||||||||||||||||
  (______)__m_m_)    /  ||||						 ||||||||||||||||
                    |___||||]						 ||||||||||||||||
									 ||||||||||||||||
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
*/

//LIBRARIES
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//GPU KERNEL


//DRILL: excludes sites inside random spheres - a whole lot of branching 
__global__ void drill(float *nran, float *nran2, float *x, float *y, float *z, int *in, int *natom, int l, int hole, int a, int b, int c, float p)
{

int n = threadIdx.x + blockIdx.x * gridDim.x;

extern __shared__ int cache[];
int temp = 0;

while(n < l)
{

natom[blockIdx.x] = 0;
int t_est = 0;
float4 r; //We will settle for single precision as this code does not pertain to energy values or order parameters, preserving precision even in large densities
float4 rn;
rn.x = x[n];
rn.y = y[n];
rn.z = z[n];
	 
	#pragma unroll
	for(int w = 0; w < hole; w++)
	{

	r.x = p*a*nran[3*w];
	r.y = p*b*nran[3*w+1];
	r.z = p*c*nran[3*w+2];
	r.w = p*nran2[w];
	r.w = r.w*r.w;

	rn.w  = (rn.x - r.x)*(rn.x - r.x);
	rn.w += (rn.y - r.y)*(rn.y - r.y);
	rn.w += (rn.z - r.z)*(rn.z - r.z);

	if( rn.w <= r.w) //Check atonm position against all the vacancies to see if it lies inside of any
	{
	t_est = 1;
	}
	
	}

	if(t_est == 0)
	{
	temp++; //atom is not inside any of the vacancies, must be counted
	in[n] = 0;
	}
	else in[n] = 1;

n += blockDim.x * gridDim.x;
}

cache[threadIdx.x] = temp;
__syncthreads();



int u = blockDim.x/2;

while(u != 0)
	{
		if (threadIdx.x < u)
			{
			cache[threadIdx.x] += cache[threadIdx.x + u];
			}
	__syncthreads();
	u /= 2;
	}

if (threadIdx.x == 0) natom[blockIdx.x] = cache[0];


}

//VARIABLES AND CONSTANTS

curandGenerator_t datagen; //info on holes to be drilled

unsigned int grid = 512; //CUDA blocks per grid (X x Y x Z) - 1 if implicit
unsigned int block = 512;//CUDA threads per block (X x Y x Z) - 1 if implicit
int *natom, *nat;	 //Effective number of lattice atoms excluding holes
int *in;		 //Integer array of binary values (1 = vacancy site, 0 = element)
int *data;		 //Integer array of binary values, host
int a,b,c; 		 //Integer lattice dimensions 
int i,j,k,l; 		 //Who doesn't love iterations
int count = 0;		 //Total count of lattice atoms
int hole;		 //Total count of holes to be drilled
int r;			 //(Integer) average size of holes i.e mean of gaussian hole
long long seed_uni;	 //Seed for uniform and gaussian random number generation
float p;		 //Lattice parameter
float *nran;		 //Random number distribution (uniform)
float *nran2;		 //Random number distribution (gaussian)
float *pore, *porer;	 //Vacancy data (x,y,z,r) format
float stddeva;		 //Standard deviation of gaussian distribution of hole sizes
float ang1, ang2, ang3;  //Angstron input value
int elem1, elem2; 	 //Lattice composition (elements)
double cpu_time_used;	 //Measured total simulation time
clock_t start_t, end_t;  //Ticks of CPU clock to measure time
float *x, *y, *z;	 //Buffer filling of lattice volume
float *fx, *fy, *fz;	 //Effective FCC filling of lattice volume
float *dx, *dy, *dz;	 //Memory block for CUDA device - if it doesn't fit now you can be sure you can't optimize/dynamics

FILE *inp_file;
FILE *out_file;
FILE *dat_file;


//MAIN PROGRAM
int main()
{

//Pseudo Random Number Generators

curandCreateGenerator(&datagen, CURAND_RNG_PSEUDO_MTGP32);


//I/O

if((inp_file=fopen("cheesinput.dat", "r")) == NULL){
printf("ERROR 404: input file not found\nPlease check and retry\n");
exit(1);
}
if((out_file=fopen("coord_z.xyz", "w")) == NULL){
printf("ERROR 777: output file could not be opened\nPlease check system and retry\n");
exit(1);
}
if ((dat_file=fopen("porus.dat", "w")) == NULL){
printf("ERROR  999: pore files could not be opened\nPlease check system and retry\n");
exit(1);
}


//READING FILES

fscanf(inp_file, "%d %d\n", &elem1, &elem2); //Reading strings is a mess and so is fscanf in general, please beware of this
fscanf(inp_file, "%f\n", &p);
fscanf(inp_file, "%f %f %f\n", &ang1, &ang2, &ang3);

p /= 2.0;
a = 0;
b = 0;
c = 0;

while( (float(a+1)*p) <= ang1) a++;
while( (float(b+1)*p) <= ang2) b++;
while( (float(c+1)*p) <= ang3) c++;



fscanf(inp_file, "%d\n", &hole);
fscanf(inp_file, "%f\n", &ang1);
fscanf(inp_file, "%f\n", &stddeva);

r = 0;
while( (float(r+1)*p) <= ang1) r++;
//printf("\n a is %d \n b is %d \n c is %d \n r is %d\n", a,b,c,r);

fscanf(inp_file, "%lld\n", &seed_uni);

curandSetPseudoRandomGeneratorSeed(datagen, seed_uni);

//FILLING THE VOLUME WITH A FCC LATTICE

fx = NULL;
fy = NULL;
fz = NULL;
l = 0;

start_t = clock();

for(i = 0; i < a; i++){
	for(j = 0; j < b; j++){
		for(k = 0; k < c; k++){
					//Filling space with the triplets U={(x,y,z)|x+y+z is even} creates the FCC Bravais Lattice, and the corresponding Space Group					
					if((i+j+k)%2 == 0)
					{

					l++;

					/*What is faster? One loop to count, allocate in one go, and another loop to fill vectors
					//Or fill in as allocate with realloc? Alternatively, realloc with bigger blocks and remove overhead
					//Alternatively use a kernel for this (calculating the elements which belong to U)
					//Note that this procedure is still fast enough, as realistically we will not have issues with structure-building
					//Compared to everything else that has to be done to the structure (optimization, dynamics, etc.)
					*/

					fx = (float*) realloc (x, sizeof(float)*l);
					fy = (float*) realloc (y, sizeof(float)*l);
					fz = (float*) realloc (z, sizeof(float)*l);

					if( (fx != NULL)&&(fy != NULL)&&(fz != NULL) )
					{	
					x = fx;
					y = fy;				
					z = fz;
					x[l-1] = p*i;
					y[l-1] = p*j;
					z[l-1] = p*k;
					}//index l linearizes memory to be acessed; 3 linear access loops are faster than 2 non linear ones!
					else
					{
					puts("Error allocating new memory block. Please try again with a smaller file size");
					exit(1);
					}

					}
				      }
				}
			}




gpuErrchk( cudaMalloc((void**)&dx, sizeof(float)*l) );
gpuErrchk( cudaMalloc((void**)&dy, sizeof(float)*l) );
gpuErrchk( cudaMalloc((void**)&dz, sizeof(float)*l) );
gpuErrchk( cudaMalloc((void**)&in, sizeof(int)*l) );

cudaMalloc((void**)&natom, sizeof(int)*block);
cudaMallocHost((void**)&nat, sizeof(int)*block);

cudaMalloc((void**)&nran, sizeof(float)*3*hole);
cudaMalloc((void**)&nran2, sizeof(float)*hole);
cudaMallocHost((void**)&data, sizeof(int)*l);
cudaMallocHost((void**)&pore, sizeof(float)*3*hole);
cudaMallocHost((void**)&porer, sizeof(float)*hole);

float rf = float(r);
float stand = float(stddeva);

curandGenerateUniform(datagen, nran, 3*hole);
curandGenerateNormal(datagen, nran2, hole, rf, stand);

//LOADING GPU

cudaMemcpyAsync(dx, x,  sizeof(float)*l, cudaMemcpyHostToDevice);
cudaMemcpyAsync(dy, y,  sizeof(float)*l, cudaMemcpyHostToDevice);
cudaMemcpyAsync(dz, z,  sizeof(float)*l, cudaMemcpyHostToDevice);

drill<<<grid,block,block*sizeof(int)>>>(nran, nran2, dx, dy, dz, in, natom, l, hole, a, b, c, p);

cudaMemcpy(nat, natom, sizeof(int)*block, cudaMemcpyDeviceToHost);
cudaMemcpy(data, in, sizeof(int)*l, cudaMemcpyDeviceToHost);
cudaMemcpy(pore, nran, sizeof(float)*3*hole, cudaMemcpyDeviceToHost);
cudaMemcpy(porer, nran2, sizeof(float)*hole, cudaMemcpyDeviceToHost);

for(i = 0; i < block; i++){
count+=nat[i];
}

//int testdat = 0;

//WRITING OUTPUT DATA
fprintf(out_file,"%d\n",count);
fprintf(out_file,"%f\n",2.0*p);


	//#pragma unroll
	for(i = 0; i < l; i++)
	{

	
		if(data[i] == 0 )
		{
		//testdat++;
		if(i%2==0)	   fprintf(out_file, "%d %.5f %.5f %.5f\n", elem1, x[i], y[i], z[i]); //String types are problematic
		else 		   fprintf(out_file, "%d %.5f %.5f %.5f\n", elem2, x[i], y[i], z[i]); //Prefer atomic number instead
		}


	}

	//printf("data was 0 %d times\n total number of elements is %d\n", testdat,l); //Test only

	#pragma unroll
	for(i = 0; i < hole; i++)
	{

	int xk = int(p*a*pore[3*i]);
	int yk = int(p*b*pore[3*i+1]);
	int zk = int(p*c*pore[3*i+2]);
	int rk = int(porer[i]*p);

	fprintf(dat_file, "%d %d %d %d\n", xk, yk, zk, rk );

	}


//WRAPPING UP

curandDestroyGenerator(datagen);

cudaFree(dx);
cudaFree(dy);
cudaFree(dz);
cudaFree(in);
cudaFree(natom);
cudaFree(nran);
cudaFree(nran2);

cudaFreeHost(pore);
cudaFreeHost(porer);
cudaFreeHost(data);
cudaFreeHost(nat);

free(x);
free(y);
free(z);

end_t = clock();
cpu_time_used = ((double) (end_t - start_t)) / CLOCKS_PER_SEC;

printf(">>>Ending Simulation\n");
printf("Total Time Elapsed (s) %13.3lf\n", cpu_time_used);
printf("======================================================================~\n");


return 0;
}

