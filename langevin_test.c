#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <signal.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>


void initializeSystem(double L, int N, double *A);
void vecboxMuller(double sigma, size_t N, double *A);
void shiftSystem(double *r, double L, int N);
double energy(double *r, double L, int N);
double pressure(double *r, double L, int N);
double mean(double *A, size_t length);
void elforel(double *A, double * B, double * C, size_t length);


int main(int argc, char** argv)
{
    srand(time(NULL));  // metterne uno per processo MPI
    clock_t start, end;
    double sim_time;
    const int N = 32;
    const int maxsteps = 10000;
    double * X = malloc(3*N * sizeof(double));
    double * Y = malloc(3*N * sizeof(double));
    double * E = malloc(maxsteps * sizeof(double));
    double * P = malloc(maxsteps * sizeof(double));
    double * jj = malloc(maxsteps * sizeof(double));

    FILE *positions = fopen("positions.csv", "w");
    for (int n=0; n<N; n++)
        fprintf(positions, "x%d,y%d,z%d,", n+1, n+1, n+1);
    fprintf(positions, "\n");

    double rho = 0.1;
    double L = cbrt(N/rho);
    double a = L/(int)(cbrt(N/4));

    initializeSystem(L, N, X);

    // Thermalization


    // Actual simulation
    start = clock();

    for (int i=0; i<3; i++)
    {

        //vecboxMuller(1.0, N, X);

        printf("%f\t%f\t%f\n", E[i], P[i], jj[i]);
        for (int n=0; n<3*N; n++)
            fprintf(positions, "%0.18lf,", X[n]);

        fprintf(positions, "\n");

    }
    end = clock();
    sim_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nTime: %lf s\n\n", sim_time);  // mettere in funzione

    // Opens csv file where it then writes a table with the data
    FILE *data = fopen("data.csv", "w");
    if (data == NULL || positions == NULL) return -1;

    for (int i=0; i<maxsteps; i++)
        fprintf(data, "%0.18lf,%0.18lf,%0.18lf\n", E[i], P[i], jj[i]);    // writes in the .csv file


    // frees the allocated memory
    free(X); free(Y); free(E); free(P); free(jj);
    return 0;
}


void initializeSystem(double L, int N, double *X) {
    int Na = (int)(cbrt(N/4)); // number of cells per dimension
    double a = L / Na;  // passo reticolare
    (Na != cbrt(N/4)) && printf("Can't make a cubic FCC crystal with this N :(");


    for (int i=0; i<Na; i++)    {   // loop over every cell of the fcc lattice
        for (int j=0; j<Na; j++)    {
            for (int k=0; k<Na; k++)    {
                int n = i*Na*Na + j*Na + k; // unique number for each triplet i,j,k
                X[n*12+0] = a*i;
                X[n*12+1] = a*j;
                X[n*12+2] = a*k;

                X[n*12+3] = a*i + a/2;
                X[n*12+4] = a*j + a/2;
                X[n*12+5] = a*k;

                X[n*12+6] = a*i + a/2;
                X[n*12+7] = a*j;
                X[n*12+8] = a*k + a/2;

                X[n*12+9] = a*i;
                X[n*12+10] = a*j + a/2;
                X[n*12+11] = a*k + a/2;
            }
        }
    }

    for (int n=0; n<3*N; n++)
        X[n] += a/4;   // needed to avoid particles exactly at the edges of the box

    shiftSystem(X,L,N);
}


void shiftSystem(double *r, double L, int N)  {  // da sistemare
    for (int j=0; j<3*N; j++)
        r[j] = r[j] - L*rint(r[j]/L);
}


double energy(double *r, double L, int N)  {
    double V = 0.0;
    double dx, dy, dz, dr2;
    for (int l=0; l<N; l++)  {
        for (int i=0; i<l; i++)   {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            dz = dz - L*rint(dz/L);
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)
                V += 4*(1.0/pow(pow(dr2,3.),2.) - 1.0/(dr2*dr2*dr2));
        }
    }
    return V;
}

double pressure(double *r, double L, int N)  {
    double P = 0.0;
    double dx, dy, dz, dr2;
    for (int l=0; l<N; l++)  {
        for (int i=0; i<l; i++)   {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            dz = dz - L*rint(dz/L);
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)
                P += 4*(6*pow(dr2,-3.) - 12*pow(dr2,-6.));   // ricontrollare
        }
    }
    return -P/(3*L*L*L);
}


void vecboxMuller(double sigma, size_t N, double * A)  {   // confrontare con dSFMT
    double x1, x2;

    for (int i=0; i<round(3*N/2); i++) {
        x1 = (double)rand() / (double)RAND_MAX;
        x2 = (double)rand() / (double)RAND_MAX;
        A[2*i] = sqrt(-2*sigma*log(1-x1))*cos(2*M_PI*x2);
        A[2*i+1] = sqrt(-2*sigma*log(1-x2))*sin(2*M_PI*x1);
    }
}

void fft_acf(double *H, int k_max, size_t length, double * acf)   {

    fftw_plan p;
    int lfft = length/2+1;
    double * Z = fftw_malloc(length * sizeof(double));
    complex double * fvi = fftw_malloc(lfft * sizeof(fftw_complex));
    double * temp = fftw_malloc(length * sizeof(double));
    complex double * C_H = fftw_malloc(lfft * sizeof(fftw_complex));

    double meanH = mean(H, length);
    for (int i=0; i<length; i++)
        Z[i] = H[i] - meanH;

    p = fftw_plan_dft_r2c_1d(lfft, Z, fvi, FFTW_FORWARD);
    fftw_execute(p);

    for (int i=0; i<lfft; i++)  // compute the abs2 of the transform
        temp[i] = fvi[i] * conj(fvi[i]);

    p = fftw_plan_dft_r2c_1d(lfft, temp, C_H, FFTW_BACKWARD);
    fftw_execute(p);

    for (int i=0; i<k_max; i++)
        acf[i] = creal(C_H[i]) / creal(C_H[1]);


    fftw_destroy_plan(p);
    fftw_free(fvi); fftw_free(C_H); free(temp); free(Z);
}



double sum(double * A, size_t length)   {
    double s = 0.;
    for (int i=0; i<length; i++)
        s += A[i];

    return s;
}

double dot(double * A, double * B, size_t length)  {
    double result = 0.0;

    for (int i=0; i<length; i++)
        result += A[i]*B[i];

    return result;
}

void elforel(double * A, double * B, double * C, size_t length)  {
    for (int i=0; i<length; i++)
        C[i] = A[i]*B[i];
}

double mean(double * A, size_t length) {
    return sum(A,length)/length;
}

double media(double * A, size_t length) {
    double birra = 0.4;
    return birra;
}

double variance(double * A, size_t length)  {
    double * A2 = malloc(length * sizeof(double));
    elforel(A,A,A2,length);
    double var = mean(A2,length) - mean(A,length)*mean(A,length);
    free(A2);
    return var;
}
