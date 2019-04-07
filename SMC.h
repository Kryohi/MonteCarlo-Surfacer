
#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <signal.h>
#include <stdint.h> // for uint defs
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <complex.h>
#include <fftw3.h>
#include "matematicose.c"
#include "misccose.c"
// + usare argp?



// number of gridpoints for the potential of the walls, along one dimension (M^2 total):
#define M 3

// number of particles:
#define N 108

// Parameters of the "default", omnipresent wall (current 0.25 0.1)
#define a0 5.960464477539063e-9
#define b0 2.44140625e-5

// Lennard-Jones cut-off (units of sigma) //TODO use this
#define LJ_CUTOFF 3.0

// Lapse for writing to positions.csv and local_temp.csv (gets multiplied by gather_lapse)
#define STORAGE_TIME 1000 //25000

// Lapse between calculation of LCA_cutoff (gets multiplied by gather_lapse)
#define LCA_TIME 10    //100
// Distance cutoff as near neighbors
#define LCA_cutoff 1.7  //1.8

// number of cells for local data along x and y
#define Ncx 33
// number of cells for local data along x and y
#define Ncz 33

// Thickness of the layers of cells near the walls (should be small, but bigger than a few interatomic distance
#define LAYER_DEPTH 5.0

// maximum autocorrelation distance used in fft_acf
#define KMAX 2500000

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
    double l2[7];
    double l3[7];
    struct DoubleArray ACF;
} Sim;



struct Sim sMC(double L, double Lz, double T, double A, const double *W, const double *R0, int maxsteps, int gather_lapse, int eqsteps);
void vecBoxMuller(double sigma, size_t length, double *A);
void shiftSystem(double * r, double L);
void shiftSystem2D(double * r, double L);
void shiftSystem3D(double * r, double L, double Lz);
void createZRange(double Lz, double * z_cells);
void initializeWalls(double x0m, double x0sigma, double ym, double ymsigma, double *W, FILE *wall);
void initializeBox(double L, double Lz, int n, double * X);

//void markovProbability(const double * R, double * Rn, double L, double T, double s, double d, double *ap);
void oneParticleMoves(double * R, double * Rn, const double * W, double L, double Lz, double A, double T, int * j, double * U);

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
void localDensityAndMobility(const double *r, double L, double Lz, unsigned long int *D, int *Rbin, unsigned long int *Mu);
void localDensityAndMobility_nonuniz(const double *r, double L, double Lz, double *z_cells, unsigned long int *D, int *Rbin, unsigned long int *Mu);
void clusterAnalysis(const double *r, int N_, double L, int *LCA);
int boundsCheck(double *r, double L, double Lz);

void simple_acf(const double *H, size_t length, int k_max, double * acf);
DoubleArray fft_acf(const double *H, size_t length, int k_max);
double variance_corr(const double * A, double tau, size_t length);




