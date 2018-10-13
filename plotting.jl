using CSVFiles, DataFrames, Plots
gr()

run(`gcc -Wall -lm -lfftw3 -O3 langevin_test.c -o exe`)
run(`./exe`)

dfp = DataFrame(load("positions.csv"))
N = Int((size(dfp,2)-1)/3)

X0 = [dfp[1, col] for col in 1:3N] # subset of columns
#X0, a = MCs.initializeSystem(N, cbrt(320))
make3Dplot(X0)



function make3Dplot(A::Array{Float64}; T = -1.0, rho = -1.0)
    Plots.default(size=(800,600))
    N = Int(length(A)/3)
    if rho == -1.0
        Plots.scatter(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(7,0.9,:blue,Plots.stroke(0)),w=7,
         xaxis=("x"), yaxis=("y"), zaxis=("z"), leg=false)
    else
        L = cbrt(N/rho)
        Plots.scatter(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(7,0.9,:blue,Plots.stroke(0)),w=7,
         xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-L/2,L/2)), leg=false)
    end
    gui()
end

function make2DtemporalPlot(M::Array{Float64,2}; T=-1.0, rho=-1.0, save=true)
    Plots.default(size=(800,600))
    N = Int(size(M,1)/3)
    L = cbrt(N/rho)
    Na = round(Int,∛(N/4)) # number of cells per dimension
    a = L / Na  # passo reticolare
    X = M[1:3:3N,1]
    #pick the particles near the plane x=a/4
    I = findall(abs.(X .-a/4) .< a/4)
    # now the X indices of M in the choosen plane are 3I-2
    scatter(M[3I.-1,1], M[3I,1], m=(7,0.7,:red,Plots.stroke(0)),w=7, xaxis=("x",(-L/2,L/2)),
     yaxis=("y",(-L/2,L/2)), leg=false)
    for i =2:size(M,2)
        scatter!(M[3I.-1,i], M[3I,i], m=(7,0.05,:blue,Plots.stroke(0)), markeralpha=0.05)
    end
    file = string("./Plots/temporal2D_",N,"_T",T,"_d",rho,".pdf")
    save && savefig(file)
    gui()
end
