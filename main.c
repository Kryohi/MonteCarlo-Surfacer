

#include "SMC.h"
#include "SMC.c"


int main(int argc, char** argv)
{
    // variables common to the simulations in every process
    int maxsteps = 5000000;
    int gather_lapse = (int) maxsteps/100000;     // number of steps between each acquisition of data
    int eqsteps = 1000000;       // number of steps for the equilibrium pre-simulation
    double L = 16;
    double Lz = 28;
    double rho = N / (L*L*Lz);
    double T = 0.7;
    
    // creates data folder and common filename suffix to save data
    make_directory("Data");
    chdir("Data");
    char filename[64];
    snprintf(filename, 64, "data_N%d_M%d_r%0.4f_T%0.2f", N, M, rho, T);
    make_directory(filename);
    chdir(filename);
    
    int *now = currentTime();
    printf("\n\n----  Starting the simulation at local time %02d:%02d  ----\n", now[0], now[1]);
    
    
    /* Initialize Walls:
    */
    
    // parameters a and b for every piece of the wall
    double * W = calloc(2*M*M, sizeof(double));
    
    // parameters of Lennard-Jones potentials of the walls (average and sigma of a gaussian)
    double x0m = 0.7;       // average width of the wall (distance at which the Lennard-Jones potential is 0)
    double x0sigma = 0.0;
    double ym = 1.8;        // average bounding energy
    double ymsigma = 0.3;
    
    // save the wall potentials to a csv file     
    snprintf(filename, 64, "./wall_N%d_M%d_r%0.4f_T%0.2f.csv", N, M, rho, T);
    FILE * wall;
    wall = fopen(filename, "w");
    if (wall == NULL)
        perror("error while writing on wall.csv");
    
    
    initializeWalls(x0m, x0sigma, ym, ymsigma, W, wall);
    
    
    
    /* Initialize particle positions:
       if a previous simulation was run with the same N, M, rho and T parameters,
       the last position configuration of that system is picked as a starting configuration.
    */
    
    double * R0 = calloc(3*N, sizeof(double));
   
    snprintf(filename, 64, "./last_state_N%d_M%d_r%0.4f_T%0.2f.csv", N, M, rho, T);
    
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
        initializeBox(L, Lz, N, R0); // da sostituire con cavity?
    }
    
    
    
    /* Prepare the results and start the simulation(s) */
        
    struct Sim MC1;
    
    MC1 = sMC(L, Lz, T, W, R0, maxsteps, gather_lapse, eqsteps);
    
    
    printf("\n###  Final results  ###");
    printf("\nMean energy: %f ± %f", MC1.E, MC1.dE);
    printf("\nMean pressure: %f ± %f", MC1.P, MC1.dP);
    printf("\nApproximate heat capacity: %f", MC1.cv);
    printf("\nAverage autocorrelation time: %f", MC1.tau);
    printf("\nAverage acceptance ratio: %f\n", MC1.acceptance_ratio);
    printf("\n");
    
    
    // save the last position of every particle, to use in a later run
    FILE * last_state;
    snprintf(filename, 64, "./last_state_N%d_M%d_r%0.4f_T%0.2f.csv", N, M, rho, T);
    last_state = fopen(filename, "w");
    if (last_state == NULL)
        perror("error while writing on last_state.csv");
    
    for (int i=0; i<3*N; i++)
        fprintf(last_state, "%0.12f,", MC1.Rfinal[i]);
    
    fclose(last_state); fclose(wall);
    free(R0); free(W);
    
    return 0;
}

