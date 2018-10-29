#ifndef SMC_noMPI
#define SMC_noMPI


#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <signal.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <complex.h>
#include <fftw3.h>
// + usare argp?


// number of slices for the potential of the walls
#define M 40
// number of particles:
#define N 32

/* NON USATI
// Number of simulation steps (all particles) after the equilibration MEASUREMENT_PERIOD
#define MAXSTEPS 1000000
// frequency of acquisition and storage of thermodinamic variables
#define MEASUREMENT_PERIOD 100 */


struct DoubleArray { 
    size_t length;
    double *data;
};

struct Sim {    // struct containing all the useful results of one simulation
    double E;
    double dE;
    double P;
    double dP;
    double acceptance_ratio;
    double cv;
    double tau;
    double Rfinal[3*N];
    struct DoubleArray ACF;
} sim;



struct Sim sMC(double rho, double T, const double *W, const double *R0, int maxsteps, int gather_lapse, int eqsteps);
void vecBoxMuller(double sigma, size_t length, double *A);
void shiftSystem(double * r, double L);
void initializeWalls(double L, double x0m, double x0sigma, double ym, double ymsigma, double *W);
void initializeBox(double L, int n, double * X);
void markovProbability(const double * R, double * Rn, double L, double T, double s, double d, double *ap);
void oneParticleMoves(double * R, double * Rn, const double * W, double L, double A, double T, int * j);
void force(const double *r, double L, int i, double *Fx, double *Fy, double *Fz);
void forces(const double *r, double L, double *F);
//void wallsForces(const double *r, const double *W, double L, double *F);
void wallsForce(double rx, double ry, double rz, const double * W, double L, double *Fx, double *Fy, double *Fz);
double energySingle(const double *r, double L, int i);
double energy(const double *r, double L);
double wallsEnergy(const double *r, const double *W, double L);
double wallsEnergySingle(double rx, double ry, double rz, const double * W, double L);
double pressure(const double *r, double L);
double sum(const double *A, size_t length);
int intsum(const int * A, size_t length);
double mean(const double * A, size_t length);
double intmean(const int * A, size_t length);
double variance(const double * A, size_t length);
double variance2(const double * A, int intervallo, size_t length);
void zeros(size_t length, double *A);
void elforel(const double *A, const double * B, double * C, size_t length);
void simple_acf(const double *H, size_t length, int k_max, double * acf);
void fft_acf(const double *H, size_t length, int k_max, double * acf);



#endif
