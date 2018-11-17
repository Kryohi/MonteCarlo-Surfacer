

#include "SMC.h"
#include "SMC.c"

#define rank 0


int main(int argc, char** argv)
{
    // variables common to the simulations in every process
    int maxsteps = 18000000;
    int gather_lapse = (int) maxsteps/500000;     // number of steps between each acquisition of data
    int eqsteps = 3000000;       // number of steps for the equilibrium pre-simulation
    int numbins = 33*33*33;
    double L, Lz;
    // oppure fissare densità e rapporto Lz/L ?
    #if N==32
        L = 20; // 30, 70
        Lz = 120;
    #elif N<120
        L = 33;//60;
        Lz = 200;//100;
    #else 
        L = 33;
        Lz = 200;
    #endif

    double rho = N / (L*L*Lz);
    double T = 1.8;
    double gamma = 1.0;
    //double dT = 2e-2;
    //double s = sqrt(4*A*D)/dT;
    double A = gamma*T; // legata a L?
    
    
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
    double x0m = 2.0;       // average width of the wall (distance at which the potential is 0) 
    double x0sigma = 0.0;
    double ym = 3.0;        // average bounding energy //da sinistra:3.5, 3.0, 4.0
    double ymsigma = 0.5;   // 0.5, 0.5, 0.5
    
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
        initializeBox(L, Lz, N, R0);
    }
    
    
    
    /* Prepare the results and start the simulation(s) */
        
    struct Sim MC1;
    
    MC1 = sMC(L, Lz, T, A, W, R0, maxsteps, gather_lapse, eqsteps, numbins);
    
    
    printf("\n###  Final results  ###");
    printf("\nMean energy: %f ± %f", MC1.E, MC1.dE);
    printf("\nMean pressure: %f ± %f", MC1.P, MC1.dP);
    printf("\nApproximate heat capacity: %f", MC1.cv);
    printf("\nAverage autocorrelation time: %f", MC1.tau);
    printf("\nAverage acceptance ratio: %f\n", MC1.acceptance_ratio);
    printf("\n");
    
    
    /* Save data and free variables and files */
    FILE * info;
    snprintf(filename, 64, "./info_N%d_M%d_r%0.4f_T%0.2f.csv", N, M, rho, T);
    info = fopen(filename, "w");
    fprintf(info, "\nBox dimensions: %0.1f * %0.1f * %0.1f", (double)L, (double)L, (double)Lz);
    fprintf(info, "\nParticle density: %0.4f", N / (L*L*Lz));
    fprintf(info, "\nAverage interparticle distance: ~%0.3f", cbrt((L*L*Lz)/N)/2);
    fprintf(info, "\nWall elements distance / interparticle distance: ~%0.3f", (L/M) / (cbrt((L*L*Lz)/N)) / 2);
    fprintf(info, "\nA used: %0.3f (%0.3f * kT)", A, gamma);
    fprintf(info, "\nMean energy: %f ± %f", MC1.E, MC1.dE);
    fprintf(info, "\nMean pressure: %f ± %f", MC1.P, MC1.dP);
    fprintf(info, "\nApproximate heat capacity: %f", MC1.cv);
    fprintf(info, "\nAverage autocorrelation time: %f", MC1.tau);
    fprintf(info, "\nAverage acceptance ratio: %f\n", MC1.acceptance_ratio);

    
    // save the last position of every particle, to use in a later run
    FILE * last_state;
    snprintf(filename, 64, "./last_state_N%d_M%d_r%0.4f_T%0.2f.csv", N, M, rho, T);
    last_state = fopen(filename, "w");
    if (last_state == NULL)
        perror("error while writing on last_state.csv");
    
    for (int i=0; i<3*N; i++)
        fprintf(last_state, "%0.12f,", MC1.Rfinal[i]);
    
    fclose(last_state); fclose(wall); fclose(info);
    free(R0); free(W);
    
    return 0;
}


