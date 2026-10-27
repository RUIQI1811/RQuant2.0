#ifndef RQ_TA_LIB_HPP
#define RQ_TA_LIB_HPP

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(__GNUC__) && !defined(__clang__)
/* GCC proves bounds for TA_MACD's warm-up loop only after inlining constant periods and warns about a path that
 * never runs. */
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif

namespace rq_ta_lib {

using std::ceil;
using std::cos;
using std::fabs;
using std::floor;
using std::fma;
using std::atan;
using std::isfinite;
using std::sin;
using std::sqrt;

/* ta_defs.h */
#define TA_INTEGER_MIN (INT_MIN+1)
#define TA_INTEGER_MAX (INT_MAX)
#define TA_REAL_MIN (-3e+37)
#define TA_REAL_MAX (3e+37)
#define TA_INTEGER_DEFAULT (INT_MIN)
#define TA_REAL_DEFAULT (-4e+37)
#define TA_MAX_INDEX 100000000
typedef enum {
   TA_SUCCESS = 0, TA_LIB_NOT_INITIALIZE = 1, TA_BAD_PARAM = 2, TA_ALLOC_ERR = 3, TA_GROUP_NOT_FOUND = 4,
   TA_FUNC_NOT_FOUND = 5, TA_INVALID_HANDLE = 6, TA_INVALID_PARAM_HOLDER = 7, TA_INVALID_PARAM_HOLDER_TYPE = 8,
   TA_INVALID_PARAM_FUNCTION = 9, TA_INPUT_NOT_ALL_INITIALIZE = 10, TA_OUTPUT_NOT_ALL_INITIALIZE = 11,
   TA_OUT_OF_RANGE_START_INDEX = 12, TA_OUT_OF_RANGE_END_INDEX = 13, TA_INVALID_LIST_TYPE = 14, TA_BAD_OBJECT = 15,
   TA_NOT_SUPPORTED = 16, TA_INSUFFICIENT_HISTORY = 17, TA_INTERNAL_ERROR = 5000, TA_UNKNOWN_ERR = 0xFFFF
} TA_RetCode;
enum { TA_COMPATIBILITY_DEFAULT = 0, TA_COMPATIBILITY_METASTOCK = 1 };
typedef int TA_Compatibility;
enum {
   TA_MAType_SMA = 0, TA_MAType_EMA = 1, TA_MAType_WMA = 2, TA_MAType_DEMA = 3, TA_MAType_TEMA = 4,
   TA_MAType_TRIMA = 5, TA_MAType_KAMA = 6, TA_MAType_MAMA = 7, TA_MAType_T3 = 8, TA_MAType_HMA = 9,
   TA_MAType_DISABLED = 10, TA_MAType_DEFAULT = 11, TA_MAType_ZLEMA = 12, TA_MAType_RMA = 13
};
typedef int TA_MAType;
#define TA_MATYPE_MIN 0
#define TA_MATYPE_MAX 13
enum {
   TA_FUNC_UNST_ADX = 0, TA_FUNC_UNST_UNUSED_1 = 1, TA_FUNC_UNST_ATR = 2, TA_FUNC_UNST_CMO = 3, TA_FUNC_UNST_DX = 4,
   TA_FUNC_UNST_EMA = 5, TA_FUNC_UNST_HT_DCPERIOD = 6, TA_FUNC_UNST_HT_DCPHASE = 7, TA_FUNC_UNST_HT_PHASOR = 8,
   TA_FUNC_UNST_HT_SINE = 9, TA_FUNC_UNST_HT_TRENDLINE = 10, TA_FUNC_UNST_HT_TRENDMODE = 11,
   TA_FUNC_UNST_UNUSED_12 = 12, TA_FUNC_UNST_KAMA = 13, TA_FUNC_UNST_MAMA = 14, TA_FUNC_UNST_UNUSED_15 = 15,
   TA_FUNC_UNST_MINUS_DI = 16, TA_FUNC_UNST_MINUS_DM = 17, TA_FUNC_UNST_NATR = 18, TA_FUNC_UNST_PLUS_DI = 19,
   TA_FUNC_UNST_PLUS_DM = 20, TA_FUNC_UNST_RSI = 21, TA_FUNC_UNST_UNUSED_22 = 22, TA_FUNC_UNST_T3 = 23,
   TA_FUNC_UNST_RMA = 24, TA_FUNC_UNST_HA = 25, TA_FUNC_UNST_RVI = 26, TA_FUNC_UNST_ALL = 65535
};
typedef int TA_FuncUnstId;
#define TA_FUNC_UNST_COUNT 27
enum { TA_RangeType_RealBody = 0, TA_RangeType_HighLow = 1, TA_RangeType_Shadows = 2 };
typedef int TA_RangeType;
enum {
   TA_BodyLong = 0, TA_BodyVeryLong = 1, TA_BodyShort = 2, TA_BodyDoji = 3, TA_ShadowLong = 4,
   TA_ShadowVeryLong = 5, TA_ShadowShort = 6, TA_ShadowVeryShort = 7, TA_Near = 8, TA_Far = 9, TA_Equal = 10,
   TA_AllCandleSettings = 11
};
typedef int TA_CandleSettingType;

/* ta_common.h, ta_global.h and ta_global.c: TA_Initialize's defaults, which the TA-Lib Python wrapper keeps */
#define TA_INTERNAL_ERROR(Id) ((TA_RetCode)(TA_INTERNAL_ERROR+Id))
typedef struct {
   TA_CandleSettingType settingType;
   TA_RangeType rangeType;
   int avgPeriod;
   double factor;
} TA_CandleSetting;
typedef struct {
   unsigned int unstablePeriod[TA_FUNC_UNST_COUNT];
   TA_CandleSetting candleSettings[TA_AllCandleSettings];
} TA_LibcPriv;
static const TA_LibcPriv ta_default_globals = {
   {0},
   {
      {TA_BodyLong, TA_RangeType_RealBody, 10, 1.0},
      {TA_BodyVeryLong, TA_RangeType_RealBody, 10, 3.0},
      {TA_BodyShort, TA_RangeType_RealBody, 10, 1.0},
      {TA_BodyDoji, TA_RangeType_HighLow, 10, 0.1},
      {TA_ShadowLong, TA_RangeType_RealBody, 0, 1.0},
      {TA_ShadowVeryLong, TA_RangeType_RealBody, 0, 2.0},
      {TA_ShadowShort, TA_RangeType_Shadows, 10, 1.0},
      {TA_ShadowVeryShort, TA_RangeType_HighLow, 10, 0.1},
      {TA_Near, TA_RangeType_HighLow, 5, 0.2},
      {TA_Far, TA_RangeType_HighLow, 5, 0.6},
      {TA_Equal, TA_RangeType_HighLow, 5, 0.05},
   },
};
static const TA_LibcPriv *const TA_Globals = &ta_default_globals;

/* ta_utility.h */
#define TA_FMA_MULTIVERSION
#define std_floor floor
#define std_ceil ceil
#define std_fabs fabs
#define std_atan atan
#define std_cos cos
#define std_sin sin
#define std_sqrt sqrt
#define TA_UNROLL(n)
#define round_pos(x) (std_floor((x)+0.5))
#define round_neg(x) (std_ceil((x)-0.5))
#define round_pos_2(x) ((std_floor((x*100.0)+0.5))/100.0)
#define round_neg_2(x) ((std_ceil((x*100.0)-0.5))/100.0)
#define TA_EPSILON (0.00000000000001)
#define TA_REAL_EQ(x,v,ep) (((v-ep)<=x)&&(x<=(v+ep)))
#define TA_IS_ZERO(v) (((-TA_EPSILON)<v)&&(v<TA_EPSILON))
#define TA_IS_ZERO_OR_NEG(v) (v<TA_EPSILON)
#define TA_IS_ZERO_SCALED(v,scale) (fabs(v) <= (TA_EPSILON*(scale)))
#define TA_IS_FINITE(v) (isfinite(v))
#define HILBERT_VARIABLES(varName) \
   ARRAY_LOCAL(varName##_Odd,3); \
   ARRAY_LOCAL(varName##_Even, 3); \
   double varName; \
   double prev_##varName##_Odd; \
   double prev_##varName##_Even; \
   double prev_##varName##_input_Odd; \
   double prev_##varName##_input_Even
#define INIT_HILBERT_VARIABLES(varName) { \
   varName##_Odd [0] = 0.0; \
   varName##_Odd [1] = 0.0; \
   varName##_Odd [2] = 0.0; \
   varName##_Even[0] = 0.0; \
   varName##_Even[1] = 0.0; \
   varName##_Even[2] = 0.0; \
   varName = 0.0; \
   prev_##varName##_Odd = 0.0; \
   prev_##varName##_Even = 0.0; \
   prev_##varName##_input_Odd = 0.0; \
   prev_##varName##_input_Even = 0.0; \
   }
#define DO_HILBERT_TRANSFORM(varName,input,OddOrEvenId) {\
   hilbertTempReal = a * input; \
   varName = -varName##_##OddOrEvenId[hilbertIdx]; \
   varName##_##OddOrEvenId[hilbertIdx] = hilbertTempReal; \
   varName += hilbertTempReal; \
   varName -= prev_##varName##_##OddOrEvenId; \
   prev_##varName##_##OddOrEvenId = b * prev_##varName##_input_##OddOrEvenId; \
   varName += prev_##varName##_##OddOrEvenId; \
   prev_##varName##_input_##OddOrEvenId = input; \
   varName *= adjustedPrevPeriod; \
   }
#define DO_HILBERT_ODD(varName,input) DO_HILBERT_TRANSFORM(varName,input,Odd)
#define DO_HILBERT_EVEN(varName,input) DO_HILBERT_TRANSFORM(varName,input,Even)
#define PER_TO_K(per) ((double)2.0 / ((double)(per + 1)))
#define PI 3.14159265358979323846
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define TA_REALBODY(IDX) ( std_fabs( (double)inClose[IDX] - (double)inOpen[IDX] ) )
#define TA_UPPERSHADOW(IDX) ( (double)inHigh[IDX] - ( inClose[IDX] >= inOpen[IDX] ? (double)inClose[IDX] : (double)inOpen[IDX] ) )
#define TA_LOWERSHADOW(IDX) ( ( inClose[IDX] >= inOpen[IDX] ? (double)inOpen[IDX] : (double)inClose[IDX] ) - (double)inLow[IDX] )
#define TA_HIGHLOWRANGE(IDX) ( (double)inHigh[IDX] - (double)inLow[IDX] )
#define TA_CANDLECOLOR(IDX) ( inClose[IDX] >= inOpen[IDX] ? 1 : -1 )
#define TA_CANDLERANGETYPE(SET) (TA_Globals->candleSettings[TA_##SET].rangeType)
#define TA_CANDLEAVGPERIOD(SET) (TA_Globals->candleSettings[TA_##SET].avgPeriod)
#define TA_CANDLEFACTOR(SET) (TA_Globals->candleSettings[TA_##SET].factor)
#define TA_CANDLERANGE(SET,IDX) \
   ( TA_CANDLERANGETYPE(SET) == TA_RangeType_RealBody ? TA_REALBODY(IDX) : \
   ( TA_CANDLERANGETYPE(SET) == TA_RangeType_HighLow ? TA_HIGHLOWRANGE(IDX) : \
   ( TA_CANDLERANGETYPE(SET) == TA_RangeType_Shadows ? TA_UPPERSHADOW(IDX) + TA_LOWERSHADOW(IDX) : \
   0 ) ) )
#define TA_CANDLEAVERAGE(SET,SUM,IDX) \
   ( TA_CANDLEFACTOR(SET) \
   * ( TA_CANDLEAVGPERIOD(SET) != 0.0? SUM / TA_CANDLEAVGPERIOD(SET) : TA_CANDLERANGE(SET,IDX) ) \
   / ( TA_CANDLERANGETYPE(SET) == TA_RangeType_Shadows ? 2.0 : 1.0 ) \
   )
#define TA_REALBODYGAPUP(IDX2,IDX1) ( min(inOpen[IDX2],inClose[IDX2]) > max(inOpen[IDX1],inClose[IDX1]) )
#define TA_REALBODYGAPDOWN(IDX2,IDX1) ( max(inOpen[IDX2],inClose[IDX2]) < min(inOpen[IDX1],inClose[IDX1]) )
#define TA_CANDLEGAPUP(IDX2,IDX1) ( inLow[IDX2] > inHigh[IDX1] )
#define TA_CANDLEGAPDOWN(IDX2,IDX1) ( inHigh[IDX2] < inLow[IDX1] )

/* ta_memory.h */
/* C converts malloc's void * implicitly; C++ needs this proxy. */
struct ta_malloc_result {
   void *pointer;
   template <class T> operator T *() const { return static_cast<T *>(pointer); }
};
#define TA_Malloc(a) (ta_malloc_result{malloc(a)})
#define TA_Free(a) free(a)
#define FREE_IF_NOT_NULL(x) { if((x)!=NULL) {TA_Free((void *)(x)); (x)=NULL;} }
#define ARRAY_VTYPE_REF(type,name) type *name
#define ARRAY_VTYPE_LOCAL(type,name,size) type name[size]
#define ARRAY_VTYPE_ALLOC(type,name,size) name = (type *)TA_Malloc( sizeof(type)*(size))
#define ARRAY_VTYPE_COPY(type,dest,src,size) memcpy(dest,src,sizeof(type)*(size))
#define ARRAY_VTYPE_MEMMOVE(type,dest,destIdx,src,srcIdx,size) memmove( &dest[destIdx], &src[srcIdx], (size)*sizeof(type) )
#define ARRAY_VTYPE_FREE(type,name) TA_Free(name)
#define ARRAY_VTYPE_FREE_COND(type,cond,name) if( cond ){ TA_Free(name); }
#define ARRAY_REF(name) ARRAY_VTYPE_REF(double,name)
#define ARRAY_LOCAL(name,size) ARRAY_VTYPE_LOCAL(double,name,size)
#define ARRAY_ALLOC(name,size) ARRAY_VTYPE_ALLOC(double,name,size)
#define ARRAY_COPY(dest,src,size) ARRAY_VTYPE_COPY(double,dest,src,size)
#define ARRAY_MEMMOVE(dest,destIdx,src,srcIdx,size) ARRAY_VTYPE_MEMMOVE(double,dest,destIdx,src,srcIdx,size)
#define ARRAY_FREE(name) ARRAY_VTYPE_FREE(double,name)
#define ARRAY_FREE_COND(cond,name) ARRAY_VTYPE_FREE_COND(double,cond,name)
#define ARRAY_INT_REF(name) ARRAY_VTYPE_REF(int,name)
#define ARRAY_INT_LOCAL(name,size) ARRAY_VTYPE_LOCAL(int,name,size)
#define ARRAY_INT_ALLOC(name,size) ARRAY_VTYPE_ALLOC(int,name,size)
#define ARRAY_INT_COPY(dest,src,size) ARRAY_VTYPE_COPY(int,dest,src,size)
#define ARRAY_INT_FREE(name) ARRAY_VTYPE_FREE(int,name)
#define ARRAY_MEMMOVEMIX_VAR int mmmixi, mmmixdestIdx, mmmixsrcIdx
#define ARRAY_MEMMOVEMIX(dest,destIdx,src,srcIdx,size) { \
   for( mmmixi=0, mmmixdestIdx=destIdx, mmmixsrcIdx=srcIdx; \
   mmmixi < size; \
   mmmixi++, mmmixdestIdx++, mmmixsrcIdx++ ) \
   { \
   dest[mmmixdestIdx] = src[mmmixsrcIdx]; \
   } \
   }
#define TA_GLOBALS_UNSTABLE_PERIOD(x,y) (TA_Globals->unstablePeriod[x])
#define CIRCBUF_PROLOG(Id,Type,Size) Type local_##Id[Size]; \
   int Id##_Idx; \
   Type *Id; \
   int maxIdx_##Id
#define CIRCBUF_PROLOG_CLASS(Id,Type,Size) CIRCBUF_PROLOG(Id,Type,Size)
#define CIRCBUF_INIT(Id,Type,Size) \
   { \
   if( Size < 1 ) \
   return TA_INTERNAL_ERROR(137); \
   if( (int)Size > (int)(sizeof(local_##Id)/sizeof(Type)) ) \
   { \
   Id = (Type *)TA_Malloc( sizeof(Type)*Size ); \
   if( !Id ) \
   return TA_ALLOC_ERR; \
   } \
   else \
   Id = &local_##Id[0]; \
   maxIdx_##Id = (Size-1); \
   Id##_Idx = 0; \
   }
#define CIRCBUF_INIT_CLASS(Id,Type,Size) CIRCBUF_INIT(Id,Type,Size)
#define CIRCBUF_INIT_LOCAL_ONLY(Id,Type) \
   { \
   Id = &local_##Id[0]; \
   maxIdx_##Id = (int)(sizeof(local_##Id)/sizeof(Type))-1; \
   Id##_Idx = 0; \
   }
#define CIRCBUF_DESTROY(Id) \
   { \
   if( Id != &local_##Id[0] ) \
   TA_Free( Id ); \
   }
#define CIRCBUF_REF(x) (x).
#define CIRCBUF_NEXT(Id) \
   { \
   Id##_Idx++; \
   if( Id##_Idx > maxIdx_##Id ) \
   Id##_Idx = 0; \
   }

/* @FUNCTIONS@ */

#undef TA_INTEGER_MIN
#undef TA_INTEGER_MAX
#undef TA_REAL_MIN
#undef TA_REAL_MAX
#undef TA_INTEGER_DEFAULT
#undef TA_REAL_DEFAULT
#undef TA_MAX_INDEX
#undef TA_MATYPE_MIN
#undef TA_MATYPE_MAX
#undef TA_FUNC_UNST_COUNT
#undef TA_INTERNAL_ERROR
#undef TA_FMA_MULTIVERSION
#undef std_floor
#undef std_ceil
#undef std_fabs
#undef std_atan
#undef std_cos
#undef std_sin
#undef std_sqrt
#undef TA_UNROLL
#undef round_pos
#undef round_neg
#undef round_pos_2
#undef round_neg_2
#undef TA_EPSILON
#undef TA_REAL_EQ
#undef TA_IS_ZERO
#undef TA_IS_ZERO_OR_NEG
#undef TA_IS_ZERO_SCALED
#undef TA_IS_FINITE
#undef HILBERT_VARIABLES
#undef INIT_HILBERT_VARIABLES
#undef DO_HILBERT_TRANSFORM
#undef DO_HILBERT_ODD
#undef DO_HILBERT_EVEN
#undef PER_TO_K
#undef PI
#undef min
#undef max
#undef TA_REALBODY
#undef TA_UPPERSHADOW
#undef TA_LOWERSHADOW
#undef TA_HIGHLOWRANGE
#undef TA_CANDLECOLOR
#undef TA_CANDLERANGETYPE
#undef TA_CANDLEAVGPERIOD
#undef TA_CANDLEFACTOR
#undef TA_CANDLERANGE
#undef TA_CANDLEAVERAGE
#undef TA_REALBODYGAPUP
#undef TA_REALBODYGAPDOWN
#undef TA_CANDLEGAPUP
#undef TA_CANDLEGAPDOWN
#undef TA_Malloc
#undef TA_Free
#undef FREE_IF_NOT_NULL
#undef ARRAY_VTYPE_REF
#undef ARRAY_VTYPE_LOCAL
#undef ARRAY_VTYPE_ALLOC
#undef ARRAY_VTYPE_COPY
#undef ARRAY_VTYPE_MEMMOVE
#undef ARRAY_VTYPE_FREE
#undef ARRAY_VTYPE_FREE_COND
#undef ARRAY_REF
#undef ARRAY_LOCAL
#undef ARRAY_ALLOC
#undef ARRAY_COPY
#undef ARRAY_MEMMOVE
#undef ARRAY_FREE
#undef ARRAY_FREE_COND
#undef ARRAY_INT_REF
#undef ARRAY_INT_LOCAL
#undef ARRAY_INT_ALLOC
#undef ARRAY_INT_COPY
#undef ARRAY_INT_FREE
#undef ARRAY_MEMMOVEMIX_VAR
#undef ARRAY_MEMMOVEMIX
#undef TA_GLOBALS_UNSTABLE_PERIOD
#undef CIRCBUF_PROLOG
#undef CIRCBUF_PROLOG_CLASS
#undef CIRCBUF_INIT
#undef CIRCBUF_INIT_CLASS
#undef CIRCBUF_INIT_LOCAL_ONLY
#undef CIRCBUF_DESTROY
#undef CIRCBUF_REF
#undef CIRCBUF_NEXT

}  // namespace rq_ta_lib

#endif
