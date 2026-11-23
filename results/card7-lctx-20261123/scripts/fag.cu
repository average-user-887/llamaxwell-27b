// Card 7: standalone timing of the FA_GEMM prefill path (fattn-gemm.cu) at long depth.
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <vector>
#define CK(x) do{auto e=(x); if(e!=0){printf("err %d line %d\n",(int)e,__LINE__);exit(1);}}while(0)
constexpr int D=256;
template<typename ST> __global__ void __launch_bounds__(256) smax(const ST* S, half* P, const half* mk, size_t mrow, float* O, float2* ml, int nq,int H,int h0,int KCs,int kc,int c0,int first){
  const int q=blockIdx.x, hg=blockIdx.y, tid=threadIdx.x; const ST* s=S+((size_t)hg*nq+q)*KCs; half* p=P+((size_t)hg*nq+q)*KCs;
  const half* m=mk+(size_t)q*mrow+c0; __shared__ float red[8]; float v[16]; float mx=-INFINITY;
#pragma unroll
  for(int i=0;i<16;++i){int k=i*256+tid; float x=-INFINITY; if(k<kc) x=(float)s[k]+__half2float(m[k]); v[i]=x; mx=fmaxf(mx,x);}
  for(int o=16;o;o>>=1) mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,o)); if((tid&31)==0) red[tid>>5]=mx; __syncthreads();
  mx=red[0]; for(int i=1;i<8;++i) mx=fmaxf(mx,red[i]); __syncthreads();
  float2* mlp=ml+(size_t)hg*nq+q; float2 mlo=first?make_float2(-INFINITY,0.f):*mlp; float mn=fmaxf(mlo.x,mx); float al=mn==-INFINITY?1.f:__expf(mlo.x-mn); float sum=0;
#pragma unroll
  for(int i=0;i<16;++i){int k=i*256+tid; if(k<KCs){float e=(k<kc&&mn!=-INFINITY)?__expf(v[i]-mn):0.f; sum+=e; p[k]=__float2half(e);}}
  for(int o=16;o;o>>=1) sum+=__shfl_xor_sync(0xffffffff,sum,o); if((tid&31)==0) red[tid>>5]=sum; __syncthreads();
  if(tid==0){float t=0; for(int i=0;i<8;++i) t+=red[i]; *mlp=make_float2(mn,mlo.y*al+t);} if(!first) O[((size_t)q*H+h0+hg)*D+tid]*=al;
}
template<typename ST> __global__ void __launch_bounds__(256) smax1(const ST* S, half* P, const half* mk, size_t mrow, float* O, float2* ml, int nq,int H,int h0,int G,int KCs,int kc,int c0,int first){
  const int q=blockIdx.x, tid=threadIdx.x; const half* m=mk+(size_t)q*mrow+c0; __shared__ float red[2][8]; float mv[16];
#pragma unroll
  for(int i=0;i<16;++i){int k=i*256+tid; mv[i]=k<kc?__half2float(m[k]):-INFINITY;}
  for(int hg=0;hg<G;++hg){ const ST* s=S+((size_t)hg*nq+q)*KCs; half* p=P+((size_t)hg*nq+q)*KCs; float v[16]; float mx=-INFINITY;
#pragma unroll
    for(int i=0;i<16;++i){int k=i*256+tid; float x=-INFINITY; if(k<kc) x=(float)s[k]+mv[i]; v[i]=x; mx=fmaxf(mx,x);}
    for(int o=16;o;o>>=1) mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,o)); if((tid&31)==0) red[0][tid>>5]=mx; __syncthreads();
    mx=red[0][0]; for(int i=1;i<8;++i) mx=fmaxf(mx,red[0][i]);
    float2* mlp=ml+(size_t)hg*nq+q; float2 mlo=first?make_float2(-INFINITY,0.f):*mlp; float mn=fmaxf(mlo.x,mx); float al=mn==-INFINITY?1.f:__expf(mlo.x-mn); float sum=0;
#pragma unroll
    for(int i=0;i<16;++i){int k=i*256+tid; if(k<KCs){float e=(k<kc&&mn!=-INFINITY)?__expf(v[i]-mn):0.f; sum+=e; p[k]=__float2half(e);}}
    for(int o=16;o;o>>=1) sum+=__shfl_xor_sync(0xffffffff,sum,o); if((tid&31)==0) red[1][tid>>5]=sum; __syncthreads();
    if(tid==0){float t=0; for(int i=0;i<8;++i) t+=red[1][i]; *mlp=make_float2(mn,mlo.y*al+t);} if(!first) O[((size_t)q*H+h0+hg)*D+tid]*=al; }
}
int main(int argc,char**argv){
  int nk=atoi(argv[1]), nq=atoi(argv[2]), ns=atoi(argv[3]), KC=argc>4?atoi(argv[4]):4096, mode=argc>5?atoi(argv[5]):0; cublasGemmAlgo_t AL=(cublasGemmAlgo_t)(argc>6?atoi(argv[6]):-1);
  // mode 0 full path, 1 GEMMs only, 2 smax only
  const int H=24,HK=4,G=6; cublasHandle_t hb; CK(cublasCreate(&hb));
  half *K,*Q,*P,*mk; float *S,*O; float2* ml;
  size_t kn=(size_t)nk*HK*D; CK(cudaMalloc(&K,kn*2)); CK(cudaMemset(K,0,kn*2));
  CK(cudaMalloc(&Q,(size_t)nq*H*D*2)); CK(cudaMemset(Q,0,(size_t)nq*H*D*2));
  CK(cudaMalloc(&S,(size_t)G*nq*KC*4)); CK(cudaMalloc(&P,(size_t)G*nq*KC*2)); CK(cudaMalloc(&ml,(size_t)G*nq*8));
  CK(cudaMalloc(&O,(size_t)nq*H*D*4)); size_t mrow=nk; CK(cudaMalloc(&mk,(size_t)nq*mrow*2)); CK(cudaMemset(mk,0,(size_t)nq*mrow*2));
  const float sc=0.0625f, one=1, zero=0; const size_t kb1=HK*D*2, kb2=D*2;
  float* Ot; CK(cudaMalloc(&Ot,(size_t)G*nq*D*4));
  auto run2=[&](){ for(int s3=0;s3<ns;++s3) for(int g=0;g<HK;++g){ for(int c0=0,ci=0;c0<nk;c0+=KC,++ci){int kc=std::min(KC,nk-c0);
    if(mode==5) CK(cublasGemmStridedBatchedEx(hb,CUBLAS_OP_T,CUBLAS_OP_N,kc,nq,D,&sc,(char*)K+g*kb2+(size_t)c0*kb1,CUDA_R_16F,kb1/2,0,Q+(size_t)g*G*D,CUDA_R_16F,H*D,D,&zero,S,CUDA_R_32F,KC,(long long)nq*KC,G,CUBLAS_COMPUTE_32F,AL));
    float b=ci==0?0.f:1.f;
    if(mode==6) CK(cublasGemmStridedBatchedEx(hb,CUBLAS_OP_N,CUBLAS_OP_N,D,nq,kc,&one,(char*)K+g*kb2+(size_t)c0*kb1,CUDA_R_16F,kb1/2,0,P,CUDA_R_16F,KC,(long long)nq*KC,&b,O+(size_t)g*G*D,CUDA_R_32F,H*D,D,G,CUBLAS_COMPUTE_32F,AL));
    if(mode==7) CK(cublasGemmEx(hb,CUBLAS_OP_T,CUBLAS_OP_N,kc,G*nq,D,&sc,(char*)K+g*kb2+(size_t)c0*kb1,CUDA_R_16F,kb1/2,Q,CUDA_R_16F,D,&zero,S,CUDA_R_32F,KC,CUBLAS_COMPUTE_32F,AL));
    if(mode==8) CK(cublasGemmEx(hb,CUBLAS_OP_N,CUBLAS_OP_N,D,G*nq,kc,&one,(char*)K+g*kb2+(size_t)c0*kb1,CUDA_R_16F,kb1/2,P,CUDA_R_16F,KC,&b,Ot,CUDA_R_32F,D,CUBLAS_COMPUTE_32F,AL));
  }}};
  auto run=[&](){ for(int s3=0;s3<ns;++s3) for(int g=0;g<HK;++g){int h0=g*G; for(int c0=0,ci=0;c0<nk;c0+=KC,++ci){int kc=std::min(KC,nk-c0);
    if(mode!=2&&mode!=4) CK(cublasGemmStridedBatchedEx(hb,CUBLAS_OP_T,CUBLAS_OP_N,kc,nq,D,&sc,(char*)K+g*kb2+(size_t)c0*kb1,CUDA_R_16F,kb1/2,0,Q+(size_t)h0*D,CUDA_R_16F,H*D,D,&zero,S,CUDA_R_32F,KC,(long long)nq*KC,G,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    if(mode==3||mode==4) smax1<float><<<nq,256>>>(S,P,mk,mrow,O,ml,nq,H,h0,G,KC,kc,c0,ci==0); else if(mode!=1) smax<float><<<dim3(nq,G),256>>>(S,P,mk,mrow,O,ml,nq,H,h0,KC,kc,c0,ci==0);
    float b=ci==0?0.f:1.f;
    if(mode!=2&&mode!=4) CK(cublasGemmStridedBatchedEx(hb,CUBLAS_OP_N,CUBLAS_OP_N,D,nq,kc,&one,(char*)K+g*kb2+(size_t)c0*kb1,CUDA_R_16F,kb1/2,0,P,CUDA_R_16F,KC,(long long)nq*KC,&b,O+(size_t)h0*D,CUDA_R_32F,H*D,D,G,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));}}};
  auto R=[&](){ if(mode>=5) run2(); else run(); }; R(); CK(cudaDeviceSynchronize()); cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a); for(int r=0;r<3;++r) R(); cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms,a,b); ms/=3;
  double fma=2.0*ns*H*(double)nq*nk*D; printf("algo %d nk %d nq %d ns %d KC %d mode %d: %.2f ms/call  %.2f TFMA/s  x16 layers = %.3f s per ubatch\n",(int)AL,nk,nq,ns,KC,mode,ms,fma/ms/1e9,ms*16/1e3);
  CK(cudaGetLastError()); return 0;
}
