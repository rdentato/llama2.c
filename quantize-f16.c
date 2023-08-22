#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <errno.h>

// #define DEBUG 

#ifdef DEBUG
#define dbgmsg(...)  (fprintf(stderr,"DBG: " __VA_ARGS__), fprintf(stderr," \xF%s:%d\n",__FILE__,__LINE__))
#else
#define dbgmsg(...)
#endif
#define _dbgmsg(...)

#define errmsg(...) (fprintf(stderr, "ERR: " __VA_ARGS__),fputc('\n',stderr))

#define PAGESIZE 32
#define HEADERSIZE 28


typedef uint16_t fp16_t;

static inline fp16_t fp16_from_float(float value)
{
    fp16_t fp16;
    uint32_t fp32 = *((uint32_t*)&value);
    uint32_t mant = (fp32 & 0x007FFCFF);
    int16_t exp = (int16_t)((fp32 & 0x7F800000) >> 23) - 127 + 15;

    // Handle special cases
    if (exp <= 0) { // Denormals and underflows
        mant = (mant | 0x00800000) >> (1 - exp);
        exp = 0;
    } else if (exp >= 0x1F) { // Overflows and infinities
        exp = 0x1F;
        mant = 0;
    }
    
    fp16  = (fp32 & 0x80000000)? 0x8000 : 0x0000;   // sign
    fp16 |= ((exp << 10));
    fp16 |= ((mant >> 13) & 0x03FF) ;

    // Round up as it seems that it is what ARM processor does with __fp16
    if (mant & 0x00001000) fp16++;

    return fp16;
}

float   buf_float[PAGESIZE];
fp16_t  buf_fp16[PAGESIZE];
uint8_t buf_header[HEADERSIZE];


void usage(char *progname)
{
  errmsg("Usage: %s your/model.bin\n            Creates a `model.f16` file with weights converted in fp16",progname);
  exit(1);
}


int main(int argc, char *argv[])
{

  if (argc < 2 || argv[1][0] == '\0') usage(argv[0]);
   
  char *infname = argv[1];
  char *outfname = "data.f16";

  dbgmsg("in: '%s' out: '%s'",infname,outfname);

  FILE *infile  = NULL;
  FILE *outfile = NULL;

  infile = fopen(infname,"rb");
  if (infile == NULL)  {errmsg("Unable to open input file `%s`",infname); goto end;}

  outfile = fopen(outfname,"wb");
  if (outfile == NULL)  {errmsg("Unable to open output file `%s`",outfname); goto end;}

  // Write the header
  int n;
  int i;
  n = fread(buf_header, HEADERSIZE, 1, infile);
  if (n<1) { errmsg("Unable to read header"); goto end;}

  n = fwrite(buf_header, HEADERSIZE, 1, outfile);
  if (n<1) { errmsg("Unable to write header"); goto end;}

  while(1) {
    n = fread(buf_float, sizeof(float), PAGESIZE, infile);
    if (n == 0) break;

    for (i = 0; i < n; i++) {
      buf_fp16[i] = fp16_from_float(buf_float[i]);
     _dbgmsg("float: %08f fp_16: %08f",buf_float[i],fp16_to_float(buf_fp16[i]));
    }

    fwrite(buf_fp16, sizeof(fp16_t), n, outfile);
    if (errno) break;
  }
  if (errno) {errmsg("I/O error: %d", errno); goto end;}

  printf("model converted and saved to data.f16\n");
end:
  if (outfile) fclose(outfile);
  if (infile)  fclose(infile);

  exit(0);
}
