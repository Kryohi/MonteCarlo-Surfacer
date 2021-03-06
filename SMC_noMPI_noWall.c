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
#define M 20
// number of particles:
#define N 32

/* NON USATI
// Number of simulation steps (all particles) after the equilibration MEASUREMENT_PERIOD
#define MAXSTEPS 1000000
// frequency of acquisition and storage of thermodinamic variables
#define MEASUREMENT_PERIOD 100 */


struct Sim sMC(double rho, double T, const double *W, const double *R0, int maxsteps, int gather_lapse, int eqsteps);
void vecBoxMuller(double sigma, size_t length, double *A);
void shiftSystem(double * r, double L);
void initializeWalls(double L, double x0m, double x0sigma, double ym, double ymsigma, double *W);
void initializeBox(double L, int n, double * X);
void markovProbability(const double * R, double * Rn, double L, double T, double s, double d, double *ap);
void oneParticleMoves(double * R, double * Rn, double L, double A, double T, int * j);
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



int main(int argc, char** argv)
{
    // In the main all the variables common to the simulations in every process are declared
    int maxsteps = 1000000;
    int gather_lapse = 10;
    int eqsteps = 10000;    // number of steps for the equilibrium pre-simulation
    double rho = 0.1;
    double T = 0.4;
    double L = cbrt(N/rho);
    
   
    /* Initialize Particle Positions */
    
    double * R0 = calloc(3*N, sizeof(double));
    char filename[32]; 
    snprintf(filename, 32, "last_state_n%d_r%0.2f_T%0.2f.csv", N, rho, T);
    
    if (access( filename, F_OK ) != -1) 
    {
        printf("\nUsing previously saved particle configuration...\n");
        FILE * last_state;
        last_state = fopen(filename, "r");  // o cercare ultima riga di positions con fseek?
        for (int i=0; i<3*N; i++)
            fscanf(last_state, "%lf,", &R0[i]);
        
        fclose(last_state);
    
    } else {
        printf("\nInitializing system...\n");
        initializeBox(L, N, R0); // da sostituire con cavity?
    }
    
    
    /* Initialize Walls */
    
    // parameters c and d for every piece of the wall
    double * W = calloc(2*M, sizeof(double));
    
    // parameters of Lennard-Jones potentials of the walls (average and sigma of a gaussian)
    double x0m = 1.0;   // average width of the wall
    double x0sigma = 0.2;
    double ym = 1.5;    // average bounding energy
    double ymsigma = 0.3;
    
    initializeWalls(L, x0m, x0sigma, ym, ymsigma, W);
    
    
    /* Prepare the results and start the simulations */
    
    struct Sim MC1;
    
    MC1 = sMC(rho, T, W, R0, maxsteps, gather_lapse, eqsteps);
    
    printf("\nMean energy: %f ± %f", MC1.E, MC1.dE);
    printf("\nMean pressure: %f ± %f", MC1.P, MC1.dP);
    printf("\nApproximate heat capacity: %f", MC1.cv);
    printf("\nAverage autocorrelation time: %f", MC1.tau);
    printf("\nAverage acceptance ratio: %f\n", MC1.acceptance_ratio);
    printf("\n");
    
    // save the last position of every particle, to use in a later run
    FILE * last_state;
    last_state = fopen(filename, "w");
    for (int i=0; i<3*N; i++)
        fprintf(last_state, "%0.12f,", MC1.Rfinal[i]);
    
    fclose(last_state);
    free(W);
    return 0;
}


struct Sim sMC(double rho, double T, const double *W, const double *R0, int maxsteps, int gather_lapse, int eqsteps)   
{
    printf("Starting new run with %d particles, T = %0.2f, rho = %0.2f, A = %f, for %d steps...\n", N, T, rho, 5e-8, maxsteps);
    srand(time(NULL));  // metterne uno per processo MPI
    clock_t start, end;
    
    double sim_time;
    int gather_steps = (int)(maxsteps/gather_lapse);
    int kmax = 42000;
    
    //copy the initial positions R0 (common to all the simulations) to the local array R
    double *R = malloc(3*N * sizeof(double));
    memcpy(R, R0, 3*N * sizeof(double));
    double * Rn = calloc(3*N, sizeof(double)); // contains the proposed particle positions
    double * ap = calloc(N, sizeof(double)); // solo per multi-particle moves, TODO
    double * E = calloc(gather_steps, sizeof(double));
    double * P = calloc(gather_steps, sizeof(double));
    int * jj = calloc(maxsteps, sizeof(int)); // usare solo in termalizzazione?
    double * acf = calloc(kmax, sizeof(double));    // autocorrelation function
    double * acf2 = calloc(kmax, sizeof(double));

    // Initialize csv files
    char filename[32]; 
    snprintf(filename, 32, "positions_n%d_r%0.2f_T%0.2f.csv", N, rho, T);
    FILE *positions = fopen(filename, "w");
    if (positions == NULL)
        perror("error while writing on positions.csv");
    
    for (int n=0; n<N; n++)
        fprintf(positions, "x%d,y%d,z%d,", n+1, n+1, n+1);
    fprintf(positions, "\n");
    
    snprintf(filename, 32, "data_n%d_r%0.2f_T%0.2f.csv", N, rho, T);
    FILE *data = fopen(filename, "w");
    if (data == NULL)
        perror("error while writing on data.csv");
    
    fprintf(data, "E, P, jj\n");
    
    
    // System properties
    double L = cbrt(N/rho);
    //double D = 2e-3;
    //double g = 0.065;
    //double A = g*T;
    //double s = sqrt(4*A*D)/g;
    double A = 4e-8;
    
    
    
    // Thermalization


    
    // Actual simulation
    start = clock();

    for (int n=0; n<maxsteps; n++)
    {
        if (n % gather_lapse == 0)  {
            int k = (int)(n/gather_lapse);
            E[k] = energy(R, L);
            P[k] = pressure(R, L);
            
            /*printf("%f\t%f\t%f\n", E[k], P[k], (float)jj[k]/N);
            for (int i=0; i<3*N; i++)
                fprintf(positions, "%0.12lf,", R[i]);

            fprintf(positions, "\n");*/
            // aggiungere indicatore di progresso ? [fflush(stdout);]
        }
        
        oneParticleMoves(R, Rn, L, A, T, &jj[n]);
    }
    
    end = clock();
    sim_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nTime: %f s\n", sim_time);

    // autocorrelation calculation
    fft_acf(E, gather_steps, kmax, acf);
    simple_acf(E, gather_steps, kmax, acf2);
    printf("TauSimple: %f \n", sum(acf2,kmax)*gather_lapse);

    
    // save temporal data of the system (gather_steps arrays of energy, pressure and acceptance ratio)
    for (int k=0; k<gather_steps; k++)
        fprintf(data, "%0.9f,%0.9f,%0.4f\n", E[k], P[k], (float)(jj[k]/N));
    

    // Create struct of the mean values and deviations to return
    struct Sim results;
    results.E = mean(E, gather_steps) + 3*N*T/2;
    results.dE = sqrt(variance(E, gather_steps));
    results.P = mean(P, gather_steps);
    results.dP = sqrt(variance(P, gather_steps));
    results.acceptance_ratio = intmean(jj, maxsteps)/N;
    results.tau = sum(acf, kmax) * gather_lapse;
    results.cv = variance2(E, (int)rint(results.tau/2), gather_steps) / (T*T);
    
    struct DoubleArray ACF;
    ACF.length = kmax;
    ACF.data = acf;
    results.ACF = ACF;
    memcpy(results.Rfinal, R, 3*N * sizeof(double));

    // frees the allocated memory
    free(R); free(Rn); free(E); free(P); free(jj); free(ap); free(acf);
    int fclose(FILE *positions); int fclose(FILE *data);

    return results;
}


/*
 * Execute a single particle Smart Monte Carlo step for each of the N particles.
 * It modifies the passed pointers R, Rn and j (the last one containing the ratio of accepted steps).
 * 
*/

void oneParticleMoves(double * R, double * Rn, double L, double A, double T, int * j)
{
    double * displ = malloc(3*N * sizeof(double));
    double Um, Un, deltaX, deltaY, deltaZ, Fmx, Fmy, Fmz, Fnx, Fny, Fnz, deltaW, ap;
        
    vecBoxMuller(sqrt(2*A), 3*N, displ);
    
    for (int i=0; i<3*N; i++)   // è necessario qua o si può usare solo una volta all'inizio?
        Rn[i] = R[i];
    
    //startingParticle = rand()%N;
    //for (int n = startingParticle; n != (startingParticle+N-1)%N; n = (n++)%N)
    for (int n=0; n<N; n++)
    {
        Um = energySingle(R, L, n);
        force(R, L, n, &Fmx, &Fmy, &Fmz);

        deltaX = Fmx*(A/T) + displ[3*n];
        deltaY = Fmy*(A/T) + displ[3*n+1];
        deltaZ = Fmz*(A/T) + displ[3*n+2];
        
        Rn[3*n] = R[3*n] + deltaX;
        Rn[3*n+1] = R[3*n+1] + deltaY;
        Rn[3*n+2] = R[3*n+2] + deltaZ;

        Un = energySingle(Rn, L, n);
        force(Rn, L, n, &Fnx, &Fny, &Fnz);

        shiftSystem(Rn,L);   // probably useless here

        deltaW = ((Fnx-Fmx)*(Fnx-Fmx) + (Fny-Fmy)*(Fny-Fmy) + (Fnz-Fmz)*(Fnz-Fmz) + 2*((Fnx-Fmx)*Fmx + (Fny-Fmy)*Fmy + (Fnz-Fmz)*Fmz)) * A/(4*T);

        ap = exp(-(Un-Um + (deltaX*(Fnx+Fmx) + deltaY*(Fny+Fmy) + deltaZ*(Fnz+Fmz))/2 + deltaW)/T);

        
        if ((double)rand()/(double)RAND_MAX < ap)
        {
            R[3*n] = Rn[3*n];
            R[3*n+1] = Rn[3*n+1];
            R[3*n+2] = Rn[3*n+2];
            *j += 1;
        }
        else    {   // riporta proposta Y a stato iniziale X
            Rn[3*n] = R[3*n];
            Rn[3*n+1] = R[3*n+1];
            Rn[3*n+2] = R[3*n+2];
        }
    }
    
    free(displ);
}

/*
 * Old SMC step, doesn't work ¯\_(ツ)_/¯
 * 
*/
/*
void markovProbability(const double *X, double *Y, double L, double T, double s, double D, double *ap)  
{
    double * gauss = malloc(3*N * sizeof(double));
    double * FX = calloc(3*N, sizeof(double));
    double * FY = calloc(3*N, sizeof(double));
    double * displacement = malloc(3*N * sizeof(double));
    double * WX = malloc(3*N * sizeof(double));
    double * WY = malloc(3*N * sizeof(double));
    
    vecBoxMuller(s, 3*N, gauss);
    forces(X, L, FX);
    
    // Proposal
    for (int i=0; i<3*N; i++)
    {
        Y[i] = X[i] + D*FX[i] + gauss[i]*s;     // force da dividere per γ?
        displacement[i] = Y[i] - X[i];   // usare shiftSystem() anche su questo?
    }
    
    shiftSystem(Y, L);
    forces(Y, L, FY);
    double Uxy = energy(Y,L) - energy(X,L); // da cambiare in modo da non dover ricalcolare l'energia salvata
    
    // Acceptance probability calculation
    for (int i=0; i<3*N; i++)     // DA SISTEMARE
    {
        WX[i] = (displacement[i] - FX[i]*D) * (displacement[i] - FX[i]*D);  // controllare segni
        WY[i] = (- displacement[i] - FY[i]*D) * (- displacement[i] - FY[i]*D);
        
        ap[i] = exp(-Uxy/T + (WX[i]-WY[i])/(4*D*T));
    }
    
    free(gauss); free(FX); free(FY); free(displacement); free(WX); free(WY); 
}
*/

void initializeBox(double L, int N_, double *X) 
{
    int Na = (int)(cbrt(N_/4)); // number of cells per dimension
    double a = L / Na;  // interparticle distance
    if (Na != cbrt(N_/4))
        perror("Can't make a cubic FCC crystal with this N :(");


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

    shiftSystem(X,L);
}


// TODO
void initializeCavity(double L, int N_, double *X) // da rendere rettangolare?
{
    int Na = (int)(cbrt(N_/4)); // number of cells per dimension
    double a = L / Na;  // interparticle distance
    if (Na != cbrt(N_/4))
        perror("Can't make a cubic FCC crystal with this N :(");


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

    shiftSystem(X,L);
}


/*
 Takes average and standard deviation of the distance from the y-axis and of the maximum binding energy
 and puts in the W array two gaussian distributions of those parameters
*/
// Per ora divide la parete in M fettine rettangolari

void initializeWalls(double L, double x0m, double x0sigma, double ym, double ymsigma, double *W)    
{
    double * X0 = malloc(M * sizeof(double));
    double * YM = malloc(M * sizeof(double));
    
    vecBoxMuller(x0sigma, M, X0);
    vecBoxMuller(ymsigma, M, YM);
    
    for (int l=0; l<M; l++)  {
        W[2*l] = 2*pow(X0[l]+x0m, 12.) * (YM[l]+ym)*(YM[l]+ym);
        W[2*l+1] = pow(X0[l]+x0m, 6.) * (YM[l]+ym);
    }
    
    free(X0); free(YM);
}


/*
 * Calculate the 3N array of forces acting on each particles due to the interaction with other particles
 * da rendere "2D" in simulazione finale, ovvero condizione diventa dx^2 + dy^2 < L*L/4 (?)
*/

void forces(const double *r, double L, double *F) 
{
    double dx, dy, dz, dr2, dr8, dV;

    for (int l=1; l<N; l++)  
    {
         for (int i=0; i<l; i++)   
         {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            dz = dz - L*rint(dz/L); // le particelle oltre la parete vanno sentite?
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)    
            {
                //dV = der_LJ(sqrt(dr2))
                dr8 = dr2*dr2*dr2*dr2;
                dV = 24.0/dr8 - 48.0/(dr8*dr2*dr2*dr2);
                F[3*l+0] -= dV*dx;
                F[3*l+1] -= dV*dy;
                F[3*l+2] -= dV*dz;
                F[3*i+0] += dV*dx;
                F[3*i+1] += dV*dy;
                F[3*i+2] += dV*dz;
            }
        }
    }
}


/*
 * Calculate the forces acting on particle i due to the interaction with other particles
 * da rendere "2D" in simulazione finale, ovvero condizione diventa dx^2 + dy^2 < L*L/4 (?)
*/

void force(const double *r, double L, int i, double *Fx, double *Fy, double *Fz) 
{
    double dx, dy, dz, dr2, dr8, dV;
    *Fx = 0.0;
    *Fy = 0.0;
    *Fz = 0.0;

    for (int l=1; l<N; l++)  
    {
         if (l != i)   
         {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            dz = dz - L*rint(dz/L); // le particelle oltre la parete vanno sentite?
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)
            {
                dr8 = dr2*dr2*dr2*dr2;
                dV = 24.0/dr8 - 48.0/(dr8*dr2*dr2*dr2);
                *Fx -= dV*dx;
                *Fy -= dV*dy;
                *Fz -= dV*dz;
            }
        }
    }
}



/*
 * Calculate the force exerted on one particle by the surface.
 * Be careful, it doesn't initialize F to zero, it only adds.
 * 
*/
// TODO
// sentire forza da tutti i segmenti? Solo quelli più vicini?

void wallsForce(double rx, double ry, double rz, const double * W, double L, double *Fx, double *Fy, double *Fz) 
{ 
    double dx, dy, dz, dr2, dr8, dV;
    double dw = L/M;
    
    for (int m=0; m<M; m++)  
    {
        //dividendo parete anche lungo x, aggiungere dy come dx, aggiungere dy^2 a dr2
        dx = rx - m*dw;
        dx = dx - L*rint(dx/L);
        dy = ry;
        dy = dy - L*rint(dy/L);
        dz = rz + L/2;
        dr2 = dx*dx + dz*dz; //miriiiiiii loveeee <3
        
        if (dr2 < L*L/4)
        {
            dr8 = dr2*dr2*dr2*dr2;
            dV = 24.0 * W[2*m] / dr8 - 48.0 * W[2*m+1] /(dr8*dr2*dr2*dr2);
            *Fx -= dV*dx;
            *Fy -= dV*dy;
            *Fz -= dV*dz;
        }
    }
}


/*
 * Calculate the interparticle potential energy of the system
 * da rendere "2D" in simulazione finale, ovvero condizione diventa dx^2 + dy^2 < L*L/4 (?)
*/

double energy(const double *r, double L)  
{
    double V = 0.0;
    double dx, dy, dz, dr2;
    for (int l=1; l<N; l++)  {
        for (int i=0; i<l; i++)   {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            dz = dz - L*rint(dz/L); // le particelle oltre la parete vanno sentite?
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)
                V += 1.0/(dr2*dr2*dr2*dr2*dr2*dr2) - 1.0/(dr2*dr2*dr2);
        }
    }
    return V*4;
}


/*
 * Calculate the potential energy of particle i with respect to other particles
 * da rendere "2D" in simulazione finale, ovvero condizione diventa dx^2 + dy^2 < L*L/4
*/

double energySingle(const double *r, double L, int i)
{
    double V = 0.0;
    double dx, dy, dz, dr2, dr6;
    for (int l=1; l<N; l++)  {
        if (l != i)   {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            dz = dz - L*rint(dz/L);
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)    {
                dr6 = dr2*dr2*dr2;
                V += 1.0/(dr6*dr6) - 1.0/dr6;
            }
        }
    }
    return V*4;
}

/*
 * Calculate the potential energy between the particles and the wall
 * 
*/
/*
double wallsEnergy(const double *r, const double *W, double L)  
{
    double V = 0.0;
    double dz2;
    for (int n=0; n<N; n++)  {
            dz2 = r[3*n+2] * r[3*n+2];
            V += W[1] / pow(pow(dz2,3.),2.) - W[2] / (dz2*dz2*dz2);
    }
    return V*4;
}*/

double wallsEnergySingle(double rx, double ry, double rz, const double * W, double L)  
{
    double V = 0.0;
    double dx, dy, dz, dr2, dr6;
    double dw = L/M;
    
    for (int m=0; m<M; m++)  
    {
        //dividendo parete anche lungo x, aggiungere dy come dx, aggiungere dy^2 a dr2
        dx = rx - m*dw;
        dx = dx - L*rint(dx/L);
        dy = ry;
        dy = dy - L*rint(dy/L);
        dz = rz + L/2;
        dr2 = dx*dx + dz*dz; 
        
        if (dr2 < L*L/4)
        {
            dr6 = dr2*dr2*dr2;
            V += W[2*m]/(dr6*dr6) - W[2*m+1]/dr6;
        }
    }
    return V*4;
}



double pressure(const double *r, double L)  
{
    double P = 0.0;
    double dx, dy, dz, dr2;
    for (int l=1; l<N; l++)  {
        for (int i=0; i<l; i++)   {
            dx = r[3*l] - r[3*i];
            dx = dx - L*rint(dx/L);
            dy = r[3*l+1] - r[3*i+1];
            dy = dy - L*rint(dy/L);
            dz = r[3*l+2] - r[3*i+2];
            dz = dz - L*rint(dz/L);
            dr2 = dx*dx + dy*dy + dz*dz;
            if (dr2 < L*L/4)    {
                //P += 24*pow(dr2,-3.) - 48*pow(dr2,-6.);
                P += 24.0/(dr2*dr2*dr2) - 48.0/(dr2*dr2*dr2*dr2*dr2*dr2);
            }
        }
    }
    return -P/(3*L*L*L);
}


inline void shiftSystem(double *r, double L)  // da ricontrollare
{
    for (int j=0; j<3*N; j++)
        r[j] = r[j] - L*rint(r[j]/L);
}

inline void shiftSystem2D(double *r, double L)  // da ricontrollare
{
    for (int j=0; j<N; j++) {
        r[3*j] = r[3*j] - L*rint(r[3*j]/L);
        r[3*j+1] = r[3*j+1] - L*rint(r[3*j+1]/L);
    }
}


/*
 * Put in the array A gaussian-distributed numbers around 0, with standard deviation sigma
 * // confrontare con dSFMT
*/

inline void vecBoxMuller(double sigma, size_t length, double * A)
{
    double x1, x2;

    for (int i=0; i<round(length/2); i++) {
        x1 = (double)rand() / (double)RAND_MAX;
        x2 = (double)rand() / (double)RAND_MAX;
        A[2*i] = sqrt(-2*sigma*log(1-x1))*cos(2*M_PI*x2);
        A[2*i+1] = sqrt(-2*sigma*log(1-x2))*sin(2*M_PI*x1);
    }
}


/*
 * Calculate the autocorrelation function
 * 
*/

void fft_acf(const double *H, size_t length, int k_max, double * acf)   
{
    if (length < k_max*2)
        perror("error: number of datapoints too low to calculate autocorrelation");

    fftw_plan p;
    fftw_complex *fvi, *C_H, *temp;
    int lfft = (int)(length/2)+1;
    double * Z = fftw_malloc(length * sizeof(double));
    fvi = (fftw_complex *) fftw_malloc(lfft * sizeof(fftw_complex));
    temp = (fftw_complex *) fftw_malloc(lfft * sizeof(fftw_complex));
    C_H = (fftw_complex *) fftw_malloc(lfft * sizeof(fftw_complex));
    
    double meanH = mean(H, length);
    for (int i=0; i<length; i++)
        Z[i] = H[i] - meanH;

    p = fftw_plan_dft_r2c_1d(lfft, Z, fvi, FFTW_ESTIMATE);
    fftw_execute(p);

    for (int i=0; i<lfft; i++)  // compute the abs2 of the transform
        temp[i] = fvi[i] * conj(fvi[i]) + 0.0I;
    
    p = fftw_plan_dft_1d(lfft, temp, C_H, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    for (int i=0; i<k_max; i++)
        acf[i] = creal(C_H[i]) / creal(C_H[1]);


    fftw_destroy_plan(p);
    fftw_free(fvi); fftw_free(C_H); fftw_free(temp); free(Z);
}


void simple_acf(const double *H, size_t length, int k_max, double * acf)   
{
    if (length < k_max*2)
        perror("error: number of datapoints too low to calculate autocorrelation");
    
    double C_H_temp;
    double * Z = fftw_malloc(length * sizeof(double));
    double meanH = mean(H, length);
    
    for (int i=0; i<length; i++)
        Z[i] = H[i] - meanH;
    
    
    for (int k=0; k<k_max; k++) {
        C_H_temp = 0.0;
        
        for (int i=0; i<length-k_max; i++)
            C_H_temp += Z[i] * Z[i+k];

        acf[k] = C_H_temp/length;
    }
    
    for (int k=0; k<k_max; k++)
        acf[k] = acf[k] / acf[0]; // unbiased and normalized autocorrelation function

    free(Z);
}


/*
    Mathematics rubbish
*/

inline double sum(const double * A, size_t length)   
{
    double s = 0.;
    for (int i=0; i<length; i++)
        s += A[i];

    return s;
}

inline int intsum(const int * A, size_t length)   
{
    int s = 0;
    for (int i=0; i<length; i++)
        s += A[i];

    return s;
}


inline double dot(const double * A, double * B, size_t length)  
{
    double result = 0.0;

    for (int i=0; i<length; i++)
        result += A[i]*B[i];

    return result;
}

inline void elforel(const double * A, const double * B, double * C, size_t length)  
{
    for (int i=0; i<length; i++)
        C[i] = A[i]*B[i];
}

inline double mean(const double * A, size_t length) 
{
    return sum(A,length)/length;
}

inline double intmean(const int * A, size_t length) 
{
    int s = 0;
    for (int i=0; i<length; i++)
        s += A[i];
    
    return (double)s/length;
}

inline void zeros(size_t length, double *A)
{
    for (int i=length; i!=0; i--)
        A[i] = 0.0;
}

inline double variance(const double * A, size_t length)
{
    double * A2 = malloc(length * sizeof(double));
    elforel(A,A,A2,length);
    double var = mean(A2,length) - mean(A,length)*mean(A,length);
    free(A2);
    return var;
}

inline double variance2(const double * A, int buco, size_t length)
{
    double var = 0.0;
    double mean_A = mean(A,length);
    int newlength = (int)(length/buco);
    
    for (int i = 0; i < newlength; i++)
        var += (A[i*buco] - mean_A)*(A[i*buco] - mean_A);
    
    return var/newlength;
}


/*
    Other rubbish
*/


/* TODO
 * 
 * mettere funzioni in file.h esterno
 * cambiare ordine di molecola da spostare ad ogni ciclo?
 * deltaX gaussiano sferico o in ogni direzione?
 * decidere cosa mettere come macro e cosa come variabiel passata a simulation (tipo gather_lapse)
 * 
 * Prestazioni: 
 *
 * FX, FY, WX, WY preallocati?
 * dSMT al post di rand()
 * sostituire funzioni stupide con macro (probabilmente inutile - compilatore + inlining)
 * confrontare prestazioni con array in stack
 * provare loop-jamming
 * provare loop al contrario  (probabilmente inutile - compilatore)
 * provare uint_fast8_t (probabilmente inutile)
 * aliases per funzioni dentro funzioni con array in lettura
 * bitwise shift per moltiplicare/dividere per 2  (probabilmente inutile - compilatore)
 * provare c++ con vectorizer di Agner
 * 
 */

