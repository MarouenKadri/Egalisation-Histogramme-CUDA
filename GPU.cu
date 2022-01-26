#include "image.hpp"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <cstring>
#define __CLANG_CUDA_CMATH_H__

//#define "__clang_cuda_cmath.h"
//#include <cmath.h>
#include <iostream>
#include <cstdlib>
#include <cuda.h>
//#include "Common/helper_math.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Common/chronoGPU.hpp"

#define HANDLE_ERROR(_exp) do {											\
    const cudaError_t err = (_exp);										\
    if ( err != cudaSuccess ) {											\
        printf("ERREUR CUDA: %s in %s at line %d\n", cudaGetErrorString( err ), __FILE__, __LINE__);    \
        exit( EXIT_FAILURE );											\
    }																	\
} while (0)

//Pré-requis: inferieur à 1024 (maxThreadsPerBlock)
#define PRECISION_HIST 255

/*
source:
https://en.wikipedia.org/wiki/HSL_and_HSV
https://gist.github.com/mjackson/5311256
*/

//nvcc -o gpu1 GPU.cu image.cpp

using namespace std;
float kmax(float a, float b, float c) {
    return ((a > b)? (a > c ? a : c) : (b > c ? b : c));
}

float kmin(float a, float b, float c) {
    return ((a < b)? (a < c ? a : c) : (b < c ? b : c));
}

/*---------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------RGB TO HSV-----------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------*/

__global__ void rgb2hsv_v1(unsigned char* in, unsigned char*out, int n,int channels) {
    int tid= (threadIdx.x + blockIdx.x * blockDim.x)*3;
    float h, s, v,r,g,b;

    if(tid<n) {
        r =in[tid] / 255.0;
        g =in[tid+1] / 255.0;
        b =in[tid+2] / 255.0;

        float cmax = r > g ? (r > b ? r : b) : (g > b ? g : b);
        float cmin = r < g ? (r < b ? r : b) : (g < b ? g : b);
        float diff = cmax-cmin; // diff of cmax and cmin.
        if (cmax == cmin)
            h = 0;
        else if (cmax == r)
            h = fmodf((60 * ((g - b) / diff) + 360), 360.0);
        else if (cmax == g)
            h = fmodf((60 * ((b - r) / diff) + 120), 360.0);
        else if (cmax == b)
            h = fmodf((60 * ((r - g) / diff) + 240), 360.0);
        // if cmax equal zero
            if (cmax == 0)
                s = 0;
            else
                s = (diff / cmax) * 100;
        // compute v
        v = cmax * 100;
        out[tid]=h;
        out[tid+1]=s;
        out[tid+2]=v;
        tid=+channels;
    }
}

__global__ void rgb2hsv_v2(unsigned char* dev_rgb_in, float* dev_h, float* dev_s, float* dev_v, int n, int channels) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int id_rgb = tid*channels;
    float h, s, v,  r, g, b;

    while(id_rgb < n) {

        r = dev_rgb_in[ id_rgb ]     / 255.0;
        g = dev_rgb_in[ id_rgb + 1 ] / 255.0;
        b = dev_rgb_in[ id_rgb + 2 ] / 255.0;

        float cmax = r > g ? (r > b ? r : b) : (g > b ? g : b);
        float cmin = r < g ? (r < b ? r : b) : (g < b ? g : b);
        float diff = cmax-cmin;

        if (cmax == cmin)
            h = 0;
        else if (cmax == r)
            h = fmodf(( ((g - b) / diff) + 6), 6);
        else if (cmax == g)
            h = fmodf(( ((b - r) / diff) + 2), 6);
        else if (cmax == b)
            h = fmodf(( ((r - g) / diff) + 4), 6);

        h /= 6;
        s = (cmax == 0 ? 0 : diff / cmax);
        v = cmax;

        dev_h[tid] = h;
        dev_s[tid] = s;
        dev_v[tid] = v;

        tid += blockDim.x * gridDim.x;
        id_rgb = tid*channels;
    }
}

__global__ void rgb2hsv_v3(const unsigned char* dev_rgb_in, float* dev_h, float* dev_s, float* dev_v, int n, int channels, int buswidth_bit) {
    extern __shared__ unsigned char cache_rgb[]; // blockDim.x*sizeof(unsigned char)*3
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int id_rgb = tid*channels;
    float h, s, v,  r, g, b;
    int buswidth_octet = buswidth_bit >> 3, nbrgbperthread = buswidth_octet/3;

    if (tid < n/nbrgbperthread) {
        for (size_t i = 0; i < nbrgbperthread*3; i+=3) {
            cache_rgb[threadIdx.x*channels]   = dev_rgb_in[id_rgb];
            cache_rgb[threadIdx.x*channels+1] = dev_rgb_in[id_rgb+1];
            cache_rgb[threadIdx.x*channels+2] = dev_rgb_in[id_rgb+2];
        }
    }


    while(id_rgb < n) {
        r = dev_rgb_in[ id_rgb ]     / 255.0;
        g = dev_rgb_in[ id_rgb + 1 ] / 255.0;
        b = dev_rgb_in[ id_rgb + 2 ] / 255.0;

        float cmax = r > g ? (r > b ? r : b) : (g > b ? g : b);
        float cmin = r < g ? (r < b ? r : b) : (g < b ? g : b);
        float diff = cmax-cmin;

        if (cmax == cmin)
            h = 0;
        else if (cmax == r)
            h = fmodf(( ((g - b) / diff) + 6), 6);
        else if (cmax == g)
            h = fmodf(( ((b - r) / diff) + 2), 6);
        else if (cmax == b)
            h = fmodf(( ((r - g) / diff) + 4), 6);

        h /= 6;
        s = (cmax == 0 ? 0 : diff / cmax);
        v = cmax;

        dev_h[tid] = h;
        dev_s[tid] = s;
        dev_v[tid] = v;

        tid += blockDim.x * gridDim.x;
        id_rgb = tid*channels;
    }
}

/*---------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------HSV TO RGB-----------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------*/

__global__ void hsv2rgb_v1(unsigned char* dev_rgb_out, float* dev_h, float* dev_s, float* dev_v, int n, int channels) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x; //int bug, sans doute overflow -> valeur négative
    size_t id_rgb = tid*channels;                       //
    float fr, fg, fb;   //version flottante de r, g, b

    while (id_rgb < n) {
        //zone du disque HSV
        int num_tranche = (int)(dev_h[tid] * 6); // partie entière du (degré*6)
        //partie de la zone du disque (precision)
        float teinte_tranche = dev_h[tid] * 6 - num_tranche;  // partie décimale du (degré*6)

        float p = dev_v[tid] * (1 - dev_s[tid]);                          //
        float q = dev_v[tid] * (1 - teinte_tranche * dev_s[tid]);         // intensité des couleurs restantes
        float t = dev_v[tid] * (1 - (1 - teinte_tranche) * dev_s[tid]);   //

        //dans quelle tranche du disque de couleur HSV sommes nous ?
        switch (num_tranche % 6) {
            case 0: fr = dev_v[tid], fg = t, fb = p; break;  // rouge/jaune
            case 1: fr = q, fg = dev_v[tid], fb = p; break;  // jaune/vert
            case 2: fr = p, fg = dev_v[tid], fb = t; break;  // vert/cyan
            case 3: fr = p, fg = q, fb = dev_v[tid]; break;  // cyan/bleu
            case 4: fr = t, fg = p, fb = dev_v[tid]; break;  // bleu/magenta
            case 5: fr = dev_v[tid], fg = p, fb = q; break;  // magenta/rouge
        }

        //converion en nombre compris entre 0 et 255
        dev_rgb_out[id_rgb]   = (unsigned char)(fr * 255.f + 0.5);
        dev_rgb_out[id_rgb+1] = (unsigned char)(fg * 255.f + 0.5);
        dev_rgb_out[id_rgb+2] = (unsigned char)(fb * 255.f + 0.5);

        tid += blockDim.x * gridDim.x;
        id_rgb = tid*channels;
    }

}

/*---------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------HISTOGRAMME----------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------*/

//TODO histv[round(hsvc.v*100)]++;
__global__ void compute_hist_v1(float* dev_v, unsigned int* dev_hist, size_t size_dev_v){
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t index_hist;

    while (tid < size_dev_v) {
        index_hist = (size_t)(dev_v[tid] * PRECISION_HIST + 0.5);
        atomicAdd(&dev_hist[index_hist], 1);

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void compute_hist_v2(float* dev_v, unsigned int* dev_hist, size_t size_dev_v){
    __shared__ int cache_hist[PRECISION_HIST+1];

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t index_hist;
    int tidb = threadIdx.x;

    while (tidb < PRECISION_HIST+1) {
        cache_hist[tidb] = 0;
        tidb += blockDim.x;
    }
    __syncthreads();
    while (tid < size_dev_v) {
        index_hist = (size_t)(dev_v[tid] * PRECISION_HIST + 0.5);
        //atomicAdd(&cache_hist[index_hist], 1);
        atomicAdd(&cache_hist[index_hist], 1);
        tid += blockDim.x * gridDim.x;
    }
    __syncthreads();

    tidb = threadIdx.x;
    while (tidb < PRECISION_HIST+1) {
        atomicAdd(&dev_hist[tidb], cache_hist[tidb]);
        tidb += blockDim.x;
    }
}

//Pré-requis: 1024 threads par block
__global__ void compute_hist_v3(float* dev_v, unsigned int* dev_hist, size_t size_dev_v){
    __shared__ int cache_hist[PRECISION_HIST+1];
    extern __shared__ float cache_v[];

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t index_hist;
    int tidb = threadIdx.x;

    if (tid < size_dev_v) {
        cache_v[tidb] = dev_v[tid];
    }
    if (tidb < PRECISION_HIST+1) {
        cache_hist[tidb] = 0;
    }

    __syncthreads();
    while (tid < size_dev_v) {
        index_hist = (size_t)(cache_v[tidb] * PRECISION_HIST + 0.5);
        atomicAdd(&cache_hist[index_hist], 1);
        tid += blockDim.x * gridDim.x;
    }
    __syncthreads();

    while (tidb < PRECISION_HIST+1) {
        atomicAdd(&dev_hist[tidb], cache_hist[tidb]);
        tidb += blockDim.x;
    }
}

/*---------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------REPARTITION----------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------*/

__device__ size_t compute_repartition_v1(unsigned int* dev_hist, int l){
    int somme = 0;
    for (size_t i=0; i<=l; i++) {
      somme += dev_hist[i];
    }
    return somme;
}

__global__ void compute_repartition_v2(unsigned int* dev_hist, unsigned int* dev_rep){
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    int somme;

    while (tid < PRECISION_HIST+1) {
        somme = 0;
        for (size_t i=0; i<=tid; i++) {
            somme += dev_hist[i];
        }
        dev_rep[tid] = somme;
        tid += blockDim.x * gridDim.x;
    }
}

//Pré-requis: dimBlock = 1, dimGrid = 1
__global__ void compute_repartition_v3(unsigned int* dev_hist, unsigned int* dev_rep){
    int somme = 0;
    for (size_t i=0; i<=PRECISION_HIST+1; i++) {
        somme += dev_hist[i];
        dev_rep[i] = somme;
    }
}

//Pré-requis: nbthreads == PRECISION_HIST+1
__global__ void compute_repartition_v4(unsigned int* dev_hist, unsigned int* dev_rep){
    extern __shared__ int cache_hist[];
    size_t l = blockIdx.x;

    cache_hist[threadIdx.x] = dev_hist[threadIdx.x];
    __syncthreads();

    int stride = l;
    unsigned char impair;
    while (stride > 1) {
        impair = (0x01 & stride);
        stride = stride/2 + impair;
        if (threadIdx.x < stride-impair) {
            atomicAdd(&cache_hist[threadIdx.x], cache_hist[threadIdx.x + stride]);
        }
    }

    if (threadIdx.x == 0) {
        dev_rep[l] = cache_hist[0];
        //dev_rep[l] = 5;
    }
}

/*Utilisé pour compute_repartition_v5
Pré-requis: tid < 32 pour plus d'efficacité
source: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
Calcul de réduction en utilisant qu'un seul warp*/
__device__ void warpReduce(volatile int* cache_hist, int tid, size_t stride) {
    unsigned char impair;
    while (stride > 1){
        impair = (0x01 & stride);
        stride = stride/2 + impair;
        cache_hist[threadIdx.x] += cache_hist[threadIdx.x + stride];
    }
}

/*Pré-requis: nbthreads == PRECISION_HIST+1
  1 block par valeur du tableau dev_rep
  calcul par méthode de réduction*/
__global__ void compute_repartition_v5(unsigned int* dev_hist, unsigned int* dev_rep){
    extern __shared__ int cache_hist[];

    cache_hist[threadIdx.x] = dev_hist[threadIdx.x];
    __syncthreads();

    int stride = blockIdx.x;
    unsigned char impair;
    while (stride > 32) {
        impair = (0x01 & stride);
        stride = stride/2 + impair;
        if (threadIdx.x < stride-impair) {
            atomicAdd(&cache_hist[threadIdx.x], cache_hist[threadIdx.x + stride]);
        }
    }
    if (threadIdx.x < 32) warpReduce(cache_hist, threadIdx.x, stride);

    if (threadIdx.x == 0) {
        dev_rep[blockIdx.x] = cache_hist[0];
    }
}


/*---------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------EQUALIZATION---------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------*/

__global__ void compute_equalization_v1(float* dev_v, unsigned int* dev_hist, size_t width, size_t height){
    //newhsvc.v = ( 99.f / (100*width*height)) * (rep(round(hsvc.v*100), histv) );
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t size_dev_v = width*height, index_hist, resultat_rep;

    while (tid < size_dev_v){
        index_hist = (size_t)(dev_v[tid] * PRECISION_HIST + 0.5);
        resultat_rep = compute_repartition_v1(dev_hist, index_hist);
        dev_v[tid] = (float)(PRECISION_HIST-1) / (PRECISION_HIST*width*height) * resultat_rep;

        tid += blockDim.x * gridDim.x;
    }
}

//Pré-requis: précalculer la repartition dans dev_rep avant
__global__ void compute_equalization_v2(float* dev_v, unsigned int* dev_hist, unsigned int* dev_rep, size_t width, size_t height){
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t size_dev_v = width*height, index_hist;

    while (tid < size_dev_v){
        index_hist = (size_t)(dev_v[tid] * PRECISION_HIST + 0.5);
        dev_v[tid] = (float)(PRECISION_HIST-1) / (PRECISION_HIST*size_dev_v) * dev_rep[index_hist];

        tid += blockDim.x * gridDim.x;
    }
}

void affichertab(unsigned int* tab, size_t n){
    for (size_t i = 0; i < PRECISION_HIST+1; i++) {
        cout << tab[i] << " ";
        if ((i+1)%16 == 0) {
            cout << endl;
        }
    }
}


/*---------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------MAIN-----------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------*/

int main()
{
    size_t width, height, channels, n;
    size_t size_rgb_img, size_canal_hsv, size_hist;

    Image orig;
    orig.load("img/Chateau.png");
    //orig.load("img/photo4k2.jpg");

    width = orig._width, height = orig._height, channels = orig._nbChannels;
    n = width * height * channels;
    size_rgb_img = sizeof(unsigned char) * n;
    size_canal_hsv = sizeof(float) * width * height;
    size_hist = sizeof(unsigned int) * (PRECISION_HIST+1);

    Image out(width, height, channels);

    //création histogramme cpu
    unsigned int* hist = (unsigned int*)malloc(sizeof(unsigned int) * (PRECISION_HIST+1));   //de 0 à 256 (car arondi + 0.5)
    for (size_t i = 0; i < PRECISION_HIST+1; i++) { hist[i] = 0; }    //init histogramme à 0

    //déclaration variables gpu
    unsigned char *dev_rgb_in, *dev_rgb_out;
    unsigned int *dev_hist, *dev_rep;
    float *dev_h, *dev_s, *dev_v;

    cudaDeviceProp prop;
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) );

    //alloc matrice image input/output
    HANDLE_ERROR(cudaMalloc( (void**)&dev_rgb_in, size_rgb_img));
    HANDLE_ERROR(cudaMalloc( (void**)&dev_rgb_out, size_rgb_img));
    //alloc canaux HSV
    HANDLE_ERROR(cudaMalloc( (void**)&dev_h, size_canal_hsv));
    HANDLE_ERROR(cudaMalloc( (void**)&dev_s, size_canal_hsv));
    HANDLE_ERROR(cudaMalloc( (void**)&dev_v, size_canal_hsv));
    //alloc histogramme + tableau repartition
    HANDLE_ERROR(cudaMalloc( (void**)&dev_hist, size_hist));
    HANDLE_ERROR(cudaMalloc( (void**)&dev_rep, size_hist));


    //copie image input dans GPU
    HANDLE_ERROR(cudaMemcpy(dev_rgb_in, orig._pixels, size_rgb_img, cudaMemcpyHostToDevice));
    //copie histogramme initialisé dans GPU
    HANDLE_ERROR(cudaMemcpy(dev_hist, hist, size_hist, cudaMemcpyHostToDevice));


    int Num_threads = prop.maxThreadsPerBlock;                 //thread per block size
    int Num_blocks = n/Num_threads + 1 > prop.maxGridSize[0] ? prop.maxGridSize[0] : n/Num_threads + 1;

    cout << "Blocks: " << Num_blocks << ", Threads per blocks: " << Num_threads << endl;

    float h,s,v;
    unsigned int hist0[PRECISION_HIST+1];
    unsigned int tabrep0[PRECISION_HIST+1];

    ChronoGPU chr;
	chr.start();

        //Launch kernel
        //rgb2hsv_v2<<< Num_blocks, Num_threads >>>(dev_rgb_in  ,dev_h, dev_s, dev_v, n, channels);
        rgb2hsv_v3<<< Num_blocks, Num_threads, sizeof(unsigned char)*Num_threads*3 >>>(dev_rgb_in  ,dev_h, dev_s, dev_v, n, channels, prop.memoryBusWidth);

        //compute_hist_v1<<< Num_blocks, Num_threads >>>(dev_v, dev_hist, width * height);
        //compute_hist_v2<<< Num_blocks, Num_threads >>>(dev_v, dev_hist, width * height);
        compute_hist_v3<<< Num_blocks, Num_threads, sizeof(float)*Num_threads >>>(dev_v, dev_hist, width * height);

        //compute_repartition_v2<<< Num_blocks, Num_threads >>>(dev_hist, dev_rep);
        //compute_repartition_v3<<< 1, 1 >>>(dev_hist, dev_rep);
        compute_repartition_v4<<< PRECISION_HIST+1, PRECISION_HIST+1, sizeof(int)*PRECISION_HIST+1 >>>(dev_hist, dev_rep);
        //compute_repartition_v5<<< PRECISION_HIST+1, PRECISION_HIST+1, sizeof(int)*PRECISION_HIST+1 >>>(dev_hist, dev_rep);

        //compute_equalization_v1<<< Num_blocks, Num_threads >>>(dev_v, dev_hist, width, height);
        compute_equalization_v2<<< Num_blocks, Num_threads >>>(dev_v, dev_hist, dev_rep, width, height);

        hsv2rgb_v1<<< Num_blocks, Num_threads >>>(dev_rgb_out ,dev_h, dev_s, dev_v, n, channels);

	chr.stop();

        HANDLE_ERROR(cudaMemcpy(tabrep0, dev_rep, sizeof(unsigned int)*(PRECISION_HIST+1), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(out._pixels, dev_rgb_out, size_rgb_img, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(&h, dev_h, sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(&s, dev_s, sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(&v, dev_v, sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(hist0, dev_hist, sizeof(unsigned int)*(PRECISION_HIST+1), cudaMemcpyDeviceToHost));

        cout << "hist: " << endl;
        affichertab(hist0, PRECISION_HIST+1);
        cout << "DEV_REP" << endl;
        affichertab(tabrep0, PRECISION_HIST+1);
        cout << endl;

    out.save("outGPU.png");

    cout << endl << "Chrono GPU: " << chr.elapsedTime() << endl;

    HANDLE_ERROR(cudaFree(dev_rgb_in));
    HANDLE_ERROR(cudaFree(dev_rgb_out));
    HANDLE_ERROR(cudaFree(dev_h));
    HANDLE_ERROR(cudaFree(dev_s));
    HANDLE_ERROR(cudaFree(dev_v));

    return 0 ;
}



/*
__global__ void compute_hist_v2(float* dev_v, unsigned int* dev_hist, size_t size_dev_v){
    __shared__ int cache_hist[PRECISION_HIST+1];

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x, tid_block;
    size_t index_hist;

    while (tid < size_dev_v) {
    //if (tid < size_dev_v) {
        index_hist = (size_t)(dev_v[tid] * PRECISION_HIST + 0.5);
        //atomicAdd(&cache_hist[index_hist], 1);
        atomicAdd(&dev_hist[tid%(PRECISION_HIST+1)], 1);
        //atomicAdd(&dev_hist[index_hist], 1);

        tid += blockDim.x * gridDim.x;
    }
    __syncthreads();

    tid_block = threadIdx.x;
    //while (tid_block < PRECISION_HIST+1){
    /*if (tid_block < PRECISION_HIST+1){
        atomicAdd(&dev_hist[tid_block] ,cache_hist[tid_block]);
        //atomicAdd(&dev_hist[tid_block], 1);
        //tid_block += blockDim.x;
    }*/
    /*if (threadIdx.x == 0) {
        for (size_t i = 0; i < PRECISION_HIST+1; i++) {
            atomicAdd(&dev_hist[i] ,cache_hist[i]);
        }
    }
}
*/
