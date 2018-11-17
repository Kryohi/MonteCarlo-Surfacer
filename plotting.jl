using Statistics, CSVFiles, DataFrames, Printf, PyCall, ProgressMeter, FFTW
pygui(:qt); import PyPlot: pygui
using Plots, StatPlots
pygui(:qt)
pyplot(size=(800, 600))
fnt = "overpass-regular"
default(titlefont=Plots.font(fnt,24), guidefont=Plots.font(fnt,20), tickfont=Plots.font(fnt,12),
 legendfont=Plots.font(fnt,12))
PyPlot.pygui(true)
using Plots.PlotMeasures
#plotlyjs(size=(800,600))

#run(`clang -Wall -lm -lfftw3 -O3 -march=native ./SMC_noMPI.c -o smc`)
#run(`./smc`)


function make3Dplot(A::Array{Float64,1}; T=-1.0, L=-1.0, Lz=-1.0, we=zeros(3,3), surface=true,reuse=true)

    N = Int(length(A)/3)
    if L == -1.0
        Plots.scatter3d(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(6,0.7,:blue,Plots.stroke(0)),
         w=7, xaxis=("x"), yaxis=("y"), zaxis=("z"), leg=false, reuse=reuse)
    else
        Plots.scatter3d(A[1:3:3N-2], A[2:3:3N-1], A[3:3:3N], m=(6,0.7,:blue, Plots.stroke(0)), w=7,
        xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-Lz/2,Lz/2)), leg=false, reuse=reuse)
    end
    if surface
        x = LinRange(-L/2, L/2, 3)
        y = x
        surface!( [ (x[1],y[1],-Lz/2+1), (x[2],y[2],-Lz/2+1), (x[3],y[3],-Lz/2+1),
        (x[2],y[1],-Lz/2+1), (x[1],y[2],-Lz/2+1), (x[1],y[3],-Lz/2+1),
        (x[3],y[1],-Lz/2+1), (x[2],y[3],-Lz/2+1), (x[3],y[2],-Lz/2+1) ] )
        surface!( [ (x[1],y[1],Lz/2-1), (x[2],y[2],Lz/2-1), (x[3],y[3],Lz/2-1),
        (x[2],y[1],Lz/2-1), (x[1],y[2],Lz/2-1), (x[1],y[3],Lz/2-1),
        (x[3],y[1],Lz/2-1), (x[2],y[3],Lz/2-1), (x[3],y[2],Lz/2-1) ] )
    end
end


function make2DtemporalPlot(M::Array{Float64,2}; T=-1.0, L = -1.0, Lz = -1.0, save=true, reuse=true)
    #Plots.default(size=(800,600))
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
end

# makes an mp4 video made by a lot of 3D plots (can be easily modified to produce a gif instead)
# don't run this with more than ~1000 frames unless you have a lot of spare time...
function makeVideo(M; T=-1, L = -1.0, Lz = -1.0, fps = 20)
    close("all")
    Plots.default(size=(1280,1080))
    N = Int(size(M,1)/3)
    rho = round(Int, 10000*N/(L*L*Lz)) / 10000
    #rho==-1 ? L = cbrt(N/(2*maximum(M))) : L = cbrt(N/rho)
    println("\nI'm cooking pngs to make a nice video. It will take some time...")
    prog = Progress(size(M,2), dt=1, barglyphs=BarGlyphs("[=> ]"), barlen=50)  # initialize progress bar

    anim = @animate for i =1:size(M,2)
        Plots.scatter(M[1:3:3N-2,i], M[2:3:3N-1,i], M[3:3:3N,i], m=(10,0.9,:blue,Plots.stroke(0)), w=7,
         xaxis=("x",(-L/2,L/2)), yaxis=("y",(-L/2,L/2)), zaxis=("z",(-Lz/2,Lz/2)), leg=false)
        next!(prog) # increment the progress bar
    end
    file = string("./Video/LJ",N,"_T",T,"_d",rho,".mp4")
    mp4(anim, file, fps = fps)
    gui() #show the last frame in a separate window
end

function acf(H::Array{Float64,1}, k_max::Int64)
    C_H = zeros(k_max)
    Z = H .- mean(H)
    if k_max>20000
        bar = Progress(k_max, dt=1.0, desc="Calculating acf...", barglyphs=BarGlyphs("[=> ]"), barlen=45)
    end
    @fastmath for k = 1:k_max
        C_H_temp = 0.0
        for i = 1:length(H)-k_max-1
            @inbounds C_H_temp += Z[i] * Z[i+k-1]
        end
        C_H[k] = C_H_temp
        (k_max>20000) && next!(bar)
    end
    CH1 = C_H[1]/(length(H))
    return C_H ./ (CH1*length(H))
end

function fft_acf(H::Array{Float64,1}, k_max::Int)

    Z = H .- mean(H)
    fvi = rfft(Z)
    acf = ifft(fvi .* conj.(fvi))
    acf = real.(acf)

    return acf[1:k_max]./acf[1]
end

function acf_spectrum(H::Array{Float64,1}, k_max::Int)

    Z = H .- mean(H)
    fvi = rfft(Z)
    acf = real.(fvi .* conj.(fvi))

    return acf[1:k_max]
end

mass_mol = 536.438          # g/mol
mass = mass_mol / 6.022e23  # g

N = 108
M = 3
if N==108
    L = 33#30
    Lz = 200#200#70
else
    L = 34#30
    Lz = 70#70
end
gamma = 0.7
rho = round(Int, 10000*N/(L*L*Lz)) / 10000
T = 2.1

parameters = @sprintf "_N%d_M%d_r%0.4f_T%0.2f" N M rho T;
cd(string("$(ENV["HOME"])/Programming/C/MonteCarlo-Surfacer/Data/data", parameters))

dfw = DataFrame(load(string("./wall", parameters, ".csv")))
dfp = DataFrame(load(string("./positions", parameters, "_rank0.csv")))
dfd = DataFrame(load(string("./data", parameters, "_rank0.csv")))
C_H = DataFrame(load(string("./autocorrelation", parameters, "_rank0.csv")))
#lD = DataFrame(load(string("./localdensity", parameters, ".csv")))
lD = DataFrame(load(string("./local", parameters, "_rank0.csv")))
sum(lD.n) // length(dfd.E) # check sul numero totale di particelle raccolte

gather_length = length(dfd.E)
gather_lapse = round(Int, 18*10^6/gather_length)
#lD[:n] = lD[:n] / numData


##################################
## Local density and mobility plots
##################################

nd = Int(cbrt(length(lD.n)))
LD = zeros(nd, nd, nd)
LD_impilata = zeros(nd, nd)
LD_parz_impilata = zeros(nd,nd,7)
mobility_part = zeros(nd,nd,7)


for i = 0:nd-1
    for j = 0:nd-1
        v = i*nd*nd + j*nd;
        LD[i+1,j+1,:] = lD.n[v .+ (1:nd)]
        LD_impilata[i+1, j+1] = sum(lD.n[ (lD.nx .== i) .& (lD.ny .== j) ])
        LD_parz_impilata[i+1, j+1, 1] = sum(lD.n[ v .+ (1:1) ]);
        LD_parz_impilata[i+1, j+1, 2] = sum(lD.n[ v .+ (2:2) ]);
        LD_parz_impilata[i+1, j+1, 3] = sum(lD.n[ v .+ (3:11) ]);
        LD_parz_impilata[i+1, j+1, 4] = sum(lD.n[ v .+ (12:19) ]);
        LD_parz_impilata[i+1, j+1, 5] = sum(lD.n[ v .+ (20:31) ]);
        LD_parz_impilata[i+1, j+1, 6] = sum(lD.n[ v .+ (32:32) ]);
        LD_parz_impilata[i+1, j+1, 7] = sum(lD.n[ v .+ (33:nd) ]);
        mobility_part[i+1, j+1, 1] = sum(lD.mu[ v .+ (1:1) ]);
        mobility_part[i+1, j+1, 2] = sum(lD.mu[ v .+ (2:2) ]);
        mobility_part[i+1, j+1, 3] = sum(lD.mu[ v .+ (3:11) ]);
        mobility_part[i+1, j+1, 4] = sum(lD.mu[ v .+ (12:19) ]);
        mobility_part[i+1, j+1, 5] = sum(lD.mu[ v .+ (20:31) ]);
        mobility_part[i+1, j+1, 6] = sum(lD.mu[ v .+ (32:32) ]);
        mobility_part[i+1, j+1, 7] = sum(lD.mu[ v .+ (33:nd) ]);
        #LD_parz_impilata[i+1, j+1, 2] = sum(lD.n[ (lD.nx .== i) .& (lD.ny .== j)
        #  .& ( (lD.nz .== 2) .| (lD.nz .== 3) | (lD.nz .== 4) )]);
    end
end
LD = LD./gather_length
LD_impilata = LD_impilata./gather_length
LD_parz_impilata = LD_parz_impilata./gather_length
mobility_part = mobility_part./gather_length
mobility_part = mobility_part ./ LD_parz_impilata

z_distr = zeros(7)
for n = 1:7
    z_distr[n] = sum(LD_parz_impilata[:,:,n])
end

X = LinRange(-L/2, L/2, nd);

contourf(X, X, LD_parz_impilata[:,:,4], aspect_ratio=1, title="Z 4/7", reuse=false)
#contour(X, X, LD_parz_impilata[:,:,2], reuse=false)

## Grafico con fette in subplots di densità
data = [LD_parz_impilata[:,:,i] for i in [1,2,3,5,6,7]]
p1 = contourf(X, X, LD_parz_impilata[:,:,1], aspect_ratio=1);
p2 = contourf(X, X, LD_parz_impilata[:,:,2], aspect_ratio=1);
p3 = contourf(X, X, LD_parz_impilata[:,:,3], aspect_ratio=1);
p5 = contourf(X, X, LD_parz_impilata[:,:,5], aspect_ratio=1);
p6 = contourf(X, X, LD_parz_impilata[:,:,6], aspect_ratio=1);
p7 = contourf(X, X, LD_parz_impilata[:,:,7], aspect_ratio=1);
plot(p1, p2, p3, p7, p6, p5, layout=(2, 3), title=["Z 1/7" "Z 2/7" "Z 3/7" "Z 7/7" "Z 6/7" "Z 5/7"],
title_location=:center, left_margin=[0mm 0mm], bottom_margin=16px, xrotation=60, reuse=false)

## Grafico con fette in subplots di mobility
data = [mobility_part[:,:,i] for i in [1,2,3,5,6,7]]
minm = minimum(mobility_part[:])
p1 = contourf(X, X, mobility_part[:,:,1], clims=[minm,1.], aspect_ratio=1);
p2 = contourf(X, X, mobility_part[:,:,2], clims=[minm,1.], aspect_ratio=1);
p3 = contourf(X, X, mobility_part[:,:,3], clims=[minm,1.], aspect_ratio=1);
p5 = contourf(X, X, mobility_part[:,:,5], clims=[minm,1.], aspect_ratio=1);
p6 = contourf(X, X, mobility_part[:,:,6], clims=[minm,1.], aspect_ratio=1);
p7 = contourf(X, X, mobility_part[:,:,7], clims=[minm,1.], aspect_ratio=1);
plot(p1, p2, p3, p7, p6, p5, layout=(2, 3), clims=[minm,1.],
 title=["Z 1/7" "Z 2/7" "Z 3/7" "Z 7/7" "Z 6/7" "Z 5/7"],
 title_location=:center, left_margin=[0mm 0mm], bottom_margin=16px, xrotation=60, reuse=false)


## wall potential
Ebound = zeros(M,M)
WallWidth = zeros(M,M)
A = zeros(M,M)
B = zeros(M,M)
for i = 0:M-1
    for j = 0:M-1
        Ebound[i+1,j+1] = dfw.ymin[findfirst( (dfw.nx .== i) .& (dfw.ny .== j) )]
        WallWidth[i+1,j+1] = dfw.x0[findfirst( (dfw.nx .== i) .& (dfw.ny .== j) )]
        A[i+1,j+1] = (WallWidth[i+1,j+1])^12* Ebound[i+1,j+1]
        B[i+1,j+1] = (WallWidth[i+1,j+1])^6* Ebound[i+1,j+1]
    end
end

X0 = [dfp[4, col] for col in 1:3N] # subset of columns
make3Dplot(X0, L=L, Lz=Lz, T=T, we = WallWidth, reuse=false)
#
# x = y = range(-5, stop = 5, length = 40)
# f(x,y) = sin(x + 10sin(4)) + cos(y)
# l = @layout [a{0.7w} b; c{0.2h}]
# p = plot(x, y, f, st = [:surface, :contourf], layout=l)

# xrange = 0.0:0.001:3
# w1 = plot(xrange,xrange.^0 .-1, xaxis=("x",(xrange[1], xrange[end])),yaxis=("V",(-2.5, 4)),reuse=false)
# for i = 1:M
#     for j = 1:M
#         plot!(w1, xrange, 4 .* (A[i,j].*xrange.^-12 - B[i,j].*xrange.^-6))
#     end
# end

W = LinRange(-L/2, L/2, M)
contour(W, W, Ebound, fill=true, reuse=false, legend=true)


## Ebound - Condensed particles
approx_ratio = (sum(LD_parz_impilata[:,:,1])+
 sum(LD_parz_impilata[:,:,7]))/sum(LD_parz_impilata[:,:,2:end-1])
#first thing to do is finding the LD boxes corresponding to the wall potential

# tentative plot
plot(vcat(Ebound...), vcat(nuclei...))



## Check energy, pressure and ar
pyplot(size=(400, 300))
Plots.plot(1:100*gather_lapse:length(dfd.E)*gather_lapse, dfd.E[1:100:end],
 linewidth=0.5, xlabel = "n", ylabel="E",reuse=false, legend=false)
Plots.plot(dfd.P[1:10:end], linewidth=0.5, reuse=false, legend=false)
Plots.scatter(dfd.jj[1:10:end]./N, linewidth=0.5, reuse=false, legend=false)
gui()

plot(1:20:length(C_H.CH), C_H.CH[1:20:end], xaxis = ("k", (-2*10^4,length(C_H.CH))), legend=false)
kmax = floor(Int, gather_length/2)
plot(1:kmax, acf_spectrum(dfd.E, kmax), xaxis = (:log10, (1,kmax)), yaxis = (:log10, (1,Inf)))
acf_pds = abs.(rfft(C_H.CH))
plot(1:10:length(acf_pds), acf_pds[1:10:end], yaxis = (:log10, (0.1,Inf)))
gui()
plot([1:1:round(Int,length(C_H.CH)/2)], abs.(fft(C_H.CH))[1:1:round(Int,length(C_H.CH)/2)],
 xaxis = ("k", (1,length(C_H.CH)/2-1)), yaxis = ("pds", (0,80)))


#acfsimple = acf(dfd.E, 10000)
#acffast = fft_acf(dfd.E, 5000)
#tausimple = sum(acfsimple)
#tau = sum(acffast)


using Makie

vx = -1:0.01:1
vy = -1:0.01:1

f(x, y) = (sin(x*10) + cos(y*10)) / 4

p1 = Makie.surface(vx, vy, f)
p2 = Makie.contour3d(vx, vy, (x, y) -> f(x,y), levels = 15, linewidth = 3)

scene = vbox(p1, p2)
text!(campixel(p1), "surface", position = widths(p1) .* Vec(0.5, 1), align=(:center,:top), raw=true)
text!(campixel(p2), "contour3d", position = widths(p2) .* Vec(0.5, 1), align=(:center,:top), raw=true)
scene

gui()
volume(LD./maximum(LD), algorithm = :mip)
