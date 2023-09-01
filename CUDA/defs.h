
///// double precision: 
//#define REAL_T double 
//#define ERROR_THRESHOLD 1e-9 

///// simple precision:
#define REAL_T float 
#define ERROR_THRESHOLD 1e-5 

///// Generateur de nombres aleatoires :
#define PRNG_MAX 0x0007FFFFFFFll
#define PRNG_1   0x00473EE661Dll
#define PRNG_2   0x024719D0275ll
#define RANGE    101

///// Mesures de temps : 
static double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

// Initialise la ligne N°i (pointée par A_i) de la matrice de taille n x n : 
static void init_ligne(REAL_T *A_i, long i, long n){
  for (long j = 0; j < n; j++) {
    A_i[j] = (((REAL_T)((i * i * PRNG_1 + j * j * PRNG_2) & PRNG_MAX)) / PRNG_MAX) / n;
  }
  for (long k = 1; k < n; k *= 2) {
    if (i + k < n) {
      A_i[i + k] = ((i - k) * PRNG_2 + i * PRNG_1) % RANGE;
    }
    if (i - k >= 0) {
      A_i[i - k] = ((i + k) * PRNG_2 + i * PRNG_1) % RANGE;
    }
  }
}
