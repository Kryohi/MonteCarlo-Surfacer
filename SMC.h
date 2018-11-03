#ifndef SMC
#define SMC


#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <signal.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <complex.h>
#include <fftw3.h>
// + usare argp?


// number of gridpoints for the potential of the walls, along one dimension (M^2 total):
#define M 3

// number of particles:
#define N 32

/* NON USATI
// Number of simulation steps (all particles) after the equilibration MEASUREMENT_PERIOD
#define MAXSTEPS 1000000
// frequency of acquisition and storage of thermodinamic variables
#define MEASUREMENT_PERIOD 100 */


typedef struct DoubleArray { 
    size_t length;
    double *data;
} DoubleArray;

typedef struct Sim {    // struct containing all the useful results of one simulation
    double E;
    double dE;
    double P;
    double dP;
    double acceptance_ratio;
    double cv;
    double tau;
    double Rfinal[3*N];
    struct DoubleArray ACF;
} Sim;



struct Sim sMC(double L, double Lz, double T, const double *W, const double *R0, int maxsteps, int gather_lapse, int eqsteps);
void vecBoxMuller(double sigma, size_t length, double *A);
void shiftSystem(double * r, double L);
void shiftSystem2D(double * r, double L);
void shiftSystem3D(double * r, double L, double Lz);
void initializeWalls(double x0m, double x0sigma, double ym, double ymsigma, double *W, FILE *wall);
void initializeBox(double L, double Lz, int n, double * X);

//void markovProbability(const double * R, double * Rn, double L, double T, double s, double d, double *ap);
void oneParticleMoves(double * R, double * Rn, const double * W, double L, double Lz, double A, double T, int * j);

double energySingle(const double *r, double L, int i);
void forceSingle(const double *r, double L, int i, double *Fx, double *Fy, double *Fz);
void forces(const double *r, double L, double *F);
//void wallsForces(const double *r, const double *W, double L, double *F);
double energy(const double *r, double L);
double pressure(const double *r, double L, double Lz);
double wallsEnergy(const double *r, const double *W, double L, double Lz);
double wallsEnergySingle(double rx, double ry, double rz, const double * W, double L, double Lz);
void wallsForce(double rx, double ry, double rz, const double * W, double L, double Lz, double *Fx, double *Fy, double *Fz);
double wallsPressure(const double *r, const double * W, double L, double Lz);
void localDensity(const double *r, double L, double Lz, int Nv, unsigned long int *D);

void simple_acf(const double *H, size_t length, int k_max, double * acf);
void fft_acf(const double *H, size_t length, int k_max, double * acf);
double sum(const double *A, size_t length);
int intsum(const int * A, size_t length);
double mean(const double * A, size_t length);
double intmean(const int * A, size_t length);
double variance(const double * A, size_t length);
double variance_corr(const double * A, double tau, size_t length);
void zeros(size_t length, double *A);
void elforel(const double *A, const double * B, double * C, size_t length);
bool isApproxEqual(double a, double b);

int * currentTime();
void make_directory(const char* name) 


#endif


