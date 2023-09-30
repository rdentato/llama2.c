
#ifndef LLAMA2_TENSOR_H
#define LLAMA2_TENSOR_H
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

typedef struct tensor_s {
  float *values;
  uint16_t nrows;
  uint16_t ncols;
} *tensor_t;

#ifndef VRG_VERSION
#define vrg_cnt(x1,x2,x3,x4,x5,x6,x7,x8,xN, ...) xN
#define vrg_cat5(_0, _1, _2, _3, _4) _0 ## _1 ## _2 ## _3 ## _4
#define vrg_empty0(_0, _1, _2, _3) vrg_commas(vrg_cat5(vrg_empty_, _0, _1, _2, _3))
#define vrg_empty____0 ,
#define vrg_commas(...) vrg_cnt(__VA_ARGS__, 0, 0, 0, 0, 0, 0, 0, _)
#define vrg_comma(...) ,
#define vrg_empty(...) vrg_empty0( vrg_commas(__VA_ARGS__)    , vrg_commas(vrg_comma __VA_ARGS__), \
                                   vrg_commas(__VA_ARGS__ ( )), vrg_commas(vrg_comma __VA_ARGS__ ( )) )
#define vrg_argn(...)  vrg_cnt(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define vrg_cat3_(x,y,z)  x ## y ## z
#define vrg_cat3(x,y,z)   vrg_cat3_(x,y,z)

#define vrg(vrg_f,...)  vrg_cat3(vrg_f, vrg_empty(__VA_ARGS__) , vrg_argn(__VA_ARGS__))(__VA_ARGS__)
#define VRG(vrg_f,...)  vrg_cat3(vrg_f, vrg_empty(__VA_ARGS__) , vrg_argn(__VA_ARGS__))(__VA_ARGS__)
#endif 

#define tensor(...) VRG(tensor,__VA_ARGS__)

#define tensor_2(v,r)      (&((struct tensor_s){v,r,1}))
#define tensor_3(v,r,c)    (&((struct tensor_s){v,r,c}))
#define tensor_4(v,r,c,l)  (&((struct tensor_s){(v) + (l) * (r) * (c),r,c}))

#define tensor_nrows(t)  ((t)->nrows)
#define tensor_ncols(t)  ((t)->ncols)
#define tensor_values(t) ((t)->values)

#define tensor_get(...) vrg(tensor_get,__VA_ARGS__)

#define tensor_get_2(t,r) tensor_get_3(t,r,0)
static  inline float tensor_get_3(tensor_t t, int r, int c) { return t->values[r * t->ncols + c]; }

#define tensor_set(...) vrg(tensor_set,__VA_ARGS__)

#define tensor_set_3(t,v,r) (((t)->values)[r] = v)
#define tensor_set_4(t,v,r,c) do { tensor_t t_ = t; t_->values[r * t_->ncols +c] = v } while (0)

#define tensor_add(...) vrg(tensor_add,__VA_ARGS__)

#define tensor_add_2(a,b)   tensor_add_4(a,a,b,1.0)
#define tensor_add_3(y,a,b) tensor_add_4(y,a,b,1.0)
static inline void tensor_add_4(tensor_t Y, tensor_t A, tensor_t B, float scale) { 
   for (int i = 0; i < A->ncols * A->nrows; i++) 
     Y->values[i] = A->values[i] + scale * B->values[i];
}

#define tensor_copy(...) vrg(tensor_cpy,__VA_ARGS__)
#define tensor_cpy_2(a,b) tensor_cpy_3(a,b,1.0)
static inline void tensor_cpy_3(tensor_t A, tensor_t B, float scale) { 
   int n = A->ncols * A->nrows;
   memcpy(A->values, B->values, n * sizeof(float));
   if (scale != 1.0)
       for (int i = n; --i >= 0;) 
           A->values[i] *= scale;
}

#define tensor_zero(t) do { tensor_t t_ = t; memset(t_->values, 0, (t_->ncols * t_->nrows) * sizeof(float)); } while (0)

static inline float tensor_dot(tensor_t A, tensor_t B)
{
    float ret = 0.0f;
    for (int i = 0; i< (A->nrows*A->ncols);i++) {
        ret += A->values[i] * B->values[i];
    }
    return ret;
}

static inline void tensor_mult(tensor_t y, tensor_t A, tensor_t x)
{
    int r = tensor_nrows(A);
    int c = tensor_ncols(A);

    // fprintf(stderr,"Xout[%d,%d] Xin[%d,%d] W[%d,%d]\n", tensor_nrows(Xout), tensor_ncols(Xout) , tensor_nrows(Xin), tensor_ncols(Xin), tensor_nrows(W), tensor_ncols(W));
    //assert(tensor_nrows(W) == tensor_nrows(Xout));
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < r; i++) {
        float val = 0.0f;
        for (int j = 0; j < c; j++) {
            val += tensor_get(A,i,j) * tensor_get(x,j);
        }
        tensor_set(y,val,i);
    }
}


typedef struct tensor_int8_s {
  int8_t *values;
  float  *scores;
  uint16_t nrows;
  uint16_t ncols;
  uint16_t gsize;
} *tensor_int8_t;



#endif