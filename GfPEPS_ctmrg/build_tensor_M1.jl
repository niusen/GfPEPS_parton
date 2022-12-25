using LinearAlgebra
using JSON
using HDF5, JLD2
using Random
using TensorKit
using MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\GfPEPS_ctmrg")

include("build_mpo.jl")


#PEPS parameters
filling=1;
P=2;#number of physical fermion modes every unit-cell
M=1;#number of virtual modes per bond
#each site has 4M virtual fermion modes
Q=2*M+filling;#total number of physical and virtual fermions on a site; 
#size of W matrix: (P+4M, Q)
init_state="Hofstadter_N2_M"*string(M)*".jld";#initialize: nothing
#init_state=nothing





W=load(init_state)["W"];
E0=load(init_state)["E0"];
#basis of W matrix: P, L,R,D,U

# Vf=ℂ[FermionNumber](0=>1, 1=>1);
# U2=unitary(fuse(Vf ⊗ Vf), Vf ⊗ Vf)

# A_set=create_vaccum(2);
# mpo_set=build_mpo_set([0.6,0.4]);
# A_set_new=mpo_mps(mpo_set,A_set)
# @tensor A_total[:]:=A_set_new[1][-1,1,-2]*A_set_new[2][1,-4,-3]


# A_set=create_vaccum(3);
# mpo_set=build_mpo_set([0.6,0.4,0.2]);
# A_set_new=mpo_mps(mpo_set,A_set)
# @tensor A_total[:]:=A_set_new[1][-1,1,-2]*A_set_new[2][1,2,-3]*A_set_new[3][2,-5,-4]

# A_set=create_vaccum(4);
# mpo_set=build_mpo_set([0.6,0.4,0.2,0.1]);
# A_set_new=mpo_mps(mpo_set,A_set)
# @tensor A_total[:]:=A_set_new[1][-1,1,-2]*A_set_new[2][1,2,-3]*A_set_new[3][2,3,-4]*A_set_new[4][3,-6,-5]




A_set=create_vaccum(P+4M);
for cc=1:Q
    mpo_set=build_mpo_set(W[:,cc]);
    A_set=mpo_mps(mpo_set,A_set);
    #@tensor A_total[:]:=A_set_new[1][-1,1,-2]*A_set_new[2][1,2,-3]*A_set_new[3][2,-5,-4]
end


T=A_set[1];


@tensor T[:]:=T[-1,1,-2]*A_set[2][1,-4,-3];
U_phy=unitary(fuse(space(T,2)⊗space(T,3)),space(T,2)⊗space(T,3));#fuse the two physical indices
@tensor T[:]:=T[-1,1,2,-3]*U_phy[-2,1,2];

@tensor T[:]:=T[-1,-2,1]*A_set[3][1,-3,-4];
@tensor T[:]:=T[-1,-2,1,-3]*A_set[4][1,-5,-4];
@tensor T[:]:=T[-1,-2,-3,-4,1]*A_set[5][1,-6,-5];
@tensor T[:]:=T[-1,-2,-3,-4,-5,1]*A_set[6][1,-7,-6];
#index order: dummy, P, L,R,D,U, dummy

save("Tensor_M1_intermediate.jld2", "T",T,"U_phy",U_phy);

T=permute(T,(3,5,4,6,2,1,7,));
#index order:  L,D,R,U,P, dummy, dummy
U=unitary(fuse(space(T,6)⊗space(T,7)),space(T,6)⊗space(T,7));
@tensor T[:]:=T[-1,-2,-3,-4,-5,1,2]*U[-6,1,2];


#attatch maximally entangled bonds, which are destruction operators
bond_mpo=build_mpo_set([1,1]);
@tensor bond_mpo[:]:=bond_mpo[1][-1,-4,1,-2]*bond_mpo[2][1,-5,-6,-3];

vacc=create_vaccum(2);
@tensor bond_mpo[:]:=bond_mpo[-1,-2,-3,1,2,-4]*vacc[1][-5,3,1]*vacc[2][3,-6,2];

U=unitary(fuse(space(bond_mpo,1)⊗space(bond_mpo,4)⊗space(bond_mpo,5)⊗space(bond_mpo,6)), space(bond_mpo,1)⊗space(bond_mpo,4)⊗space(bond_mpo,5)⊗space(bond_mpo,6))
@tensor bond_mpo[:]:=bond_mpo[1,-2,-3,2,3,4]*U[-1,1,2,3,4];

@tensor T[:]:=T[-1,-2,1,2,-5,-6]*bond_mpo'[-7,1,-3]*bond_mpo'[-8,2,-4];
U=unitary(fuse(space(T,6)⊗space(T,7)⊗space(T,8)), space(T,6)⊗space(T,7)⊗space(T,8));
@tensor T[:]:=T[-1,-2,-3,-4,-5,1,2,3]*U[-6,1,2,3];






save("fermionic_Tensor_M1.jld2", "T",T,"U_phy",U_phy);


