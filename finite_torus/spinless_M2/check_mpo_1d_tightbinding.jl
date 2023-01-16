using LinearAlgebra
using JSON
using HDF5, JLD2
using Random
using TensorKit
using MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\GfPEPS_ctmrg")

include("build_mpo.jl")
include("fmps_funs.jl")

L=4;#sites
N=3;#number of filled particles
phi=0;#boundary hopping phase
@assert L>2
t=1;
H=zeros(L,L)*im;
for c1=1:L
    for c2=1:L
        if abs(c1-c2)==1
            H[c1,c2]=-t;
        elseif (c1==1)&(c2==L)
            H[c1,c2]=-t*exp(im*phi*pi);
        elseif (c1==L)&(c2==1)
            H[c1,c2]=-t*exp(-im*phi*pi);
        end
    end
end

eu,ev=eigen(H);
Eg=sum(eu[1:N]);
println("Exact energy: "*string(Eg));









A_set=create_vaccum(L);
for cc=1:N
    mpo_set=build_mpo_set(ev[:,cc]);
    A_set=mpo_mps(mpo_set,A_set);
    #@tensor A_total[:]:=A_set_new[1][-1,1,-2]*A_set_new[2][1,2,-3]*A_set_new[3][2,-5,-4]
end

Ag=deepcopy(A_set);



E_mps=0;
for cc=1:L-1
    E_mps=E_mps+(-t)*overlap_1D(Ag,mpo_mps(hop_mpo(cc,cc+1,L),Ag))/overlap_1D(Ag,Ag);
end
E_mps=E_mps+(-t*exp(im*phi*pi))*overlap_1D(Ag,mpo_mps(hop_mpo(1,L,L),Ag))/overlap_1D(Ag,Ag);
E_mps=E_mps+E_mps';

println("mps energy: "*string(E_mps))




@tensor A_total[:]:=Ag[1][-1,1,-2]*Ag[2][1,2,-3]*Ag[3][2,-5,-4]

# cc=1;
# Ae=mpo_mps(hop_mpo(cc,cc+1,L),Ag)
# @tensor A_total2[:]:=Ae[1][-1,1,-2]*Ae[2][1,2,-3]*Ae[3][2,-5,-4]
#save("A.jld2", "A",A);
