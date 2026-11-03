// Shim CPU de lo minimo de CUDA que usa prueba.cu (thrust con backend CPP).
#pragma once
#include <cstring>
#include <chrono>
#define __device__
#define __global__
#define __constant__
#ifndef __host__
#define __host__
#endif
struct dim3 { unsigned x, y, z; dim3(unsigned a=1, unsigned b=1, unsigned c=1): x(a), y(b), z(c) {} };
inline dim3 blockIdx, threadIdx, blockDim;
template<class T> inline int cudaMemcpyToSymbol(T& sym, const void* src, size_t n){ std::memcpy((void*)&sym, src, n); return 0; }
inline int cudaDeviceSynchronize(){ return 0; }
using cudaEvent_t = std::chrono::steady_clock::time_point*;
inline int cudaEventCreate(cudaEvent_t* e){ *e = new std::chrono::steady_clock::time_point; return 0; }
inline int cudaEventRecord(cudaEvent_t e){ *e = std::chrono::steady_clock::now(); return 0; }
inline int cudaEventSynchronize(cudaEvent_t){ return 0; }
inline int cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b){ *ms = std::chrono::duration<float, std::milli>(*b - *a).count(); return 0; }
inline int cudaEventDestroy(cudaEvent_t e){ delete e; return 0; }
#define LAUNCH(kern, B, T, ...) do { blockDim = T; \
  for (unsigned bz=0; bz<B.z; bz++) for (unsigned by=0; by<B.y; by++) for (unsigned bx=0; bx<B.x; bx++) \
  for (unsigned tz=0; tz<T.z; tz++) for (unsigned ty=0; ty<T.y; ty++) for (unsigned tx=0; tx<T.x; tx++) { \
    blockIdx = dim3(bx,by,bz); threadIdx = dim3(tx,ty,tz); kern(__VA_ARGS__); } } while(0)
