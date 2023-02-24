using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
using Combinatorics
cd(@__DIR__)

chi=80
tol=1e-6
CTM_ite_nums=500;
CTM_trun_tol=1e-10;


include("mpo_mps_funs.jl");

#PEPS parameters
filling=2;
P=4;#number of physical fermion modes every unit-cell
M=2;#number of virtual modes per bond

#each site has 4M virtual fermion modes
Q=2*M+filling;#total number of physical and virtual fermions on a site; 
#size of W matrix: (P+4M, Q)
#init_state="Hofstadter_N2_M"*string(M)*".jld";#initialize: nothing
init_state="Hofstadter_N4_fil2_M"*string(M)*".jld";#initialize: nothing
#init_state="QWZ_M"*string(M)*".jld";#initialize: nothing
#init_state="C2_model1_correct_M"*string(M)*".jld";#initialize: nothing
#init_state="C2_model1_incorrect_M"*string(M)*".jld";#initialize: nothing

function build_tensor_swap(init_state,filling,P,M,Q)
    W=load(init_state)["W"];
    E0=load(init_state)["E0"];


    W_set=create_mpo(W);


    mps_set=create_vaccum_mps(P+4*M);
    for cc=1:Q
        mps_set=mpo_mps(W_set[cc,:],mps_set);
    end

    psi_G=mps_set;



    bond_coe=zeros(2,1);bond_coe[1,1]=1;bond_coe[2,1]=1;
    bond_set=create_mpo(bond_coe);
    #conjugate
    for cc=1:2
        bond_set[cc]=permute(bond_set[cc],(1,2,3,4,))
        bond_set[cc]=bond_set[cc]';
        #bond_set[cc]=permute(bond_set[cc]',(1,4,3,2,))
    end
    vaccums=create_vaccum_mps(2);
    for cc=1:2
        vaccums[cc]=permute(vaccums[cc],(1,2,3,))
        vaccums[cc]=vaccums[cc]';
    end
    for cc=1:1
        vaccums=mpo_mps(bond_set[cc,:],vaccums);
    end

    @tensor bond[:]:=vaccums[1][-1,1,-2]*vaccums[2][1,-4,-3];
    U=unitary(fuse(space(bond,1)⊗space(bond,4)),space(bond,1)⊗space(bond,4));
    @tensor bond[:]:=bond[1,-2,-3,2]*U[-1,1,2];



    bond_m=zeros(1,1,2,2,2,2)*im;
    bond_m[1,1,2,2,1,1]=1;
    bond_m[1,1,1,2,2,1]=-1;
    bond_m[1,1,2,1,1,2]=1;
    bond_m[1,1,1,1,2,2]=-1;
    bond_m_U1=TensorMap(bond_m,Rep[U₁](1=>1)' ⊗ Rep[U₁](1=>1)' ⊗ Rep[U₁](0=>1, 1=>1) ⊗ Rep[U₁](0=>1, 1=>1), Rep[U₁](0=>1, 1=>1)' ⊗ Rep[U₁](0=>1, 1=>1)' );
    bond_m_U1=permute(bond_m_U1,(1,2,3,4,5,6,))

    bond_m=reshape(bond_m,1,4,4)
    Pm=zeros(4,4)*im;Pm[1,1]=1;Pm[2,4]=1;Pm[3,2]=1;Pm[4,3]=1;
    @tensor bond_m[:]:=bond_m[-1,1,2]*Pm[-2,1]*Pm[-3,2];
    V=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1,(1,1/2)=>1,(2,0)=>1);#element order after converting to dense: <0,0>, <up,down>, <up,0>, <0,down>, 
    Vdummy=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2,0)=>1);
    bond_gate=TensorMap(bond_m,Vdummy ⊗ V ← V');
    bond_gate=permute(bond_gate,(1,2,3,));




    #gutzwiller projector
    P=zeros(2,2,2)*im;
    P[1,2,1]=1;
    P[2,1,2]=1;
    P=reshape(P,2,4);
    @tensor P[:]:=P[-1,1]*Pm[-2,1];
    Vspin=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1);
    P_G=TensorMap(P, Vspin'  ←  V');
    return psi_G,P_G

end

psi_G,P_G=build_tensor_swap(init_state,filling,P,M,Q);


#####################################################################

include("swap_funs.jl")
include("fermi_permute.jl")
include("double_layer_funs.jl")
include("Projector_funs.jl")
include("CTMRG_funs.jl")
include("ES_algorithms.jl")





M=2;
Guztwiller=true;#add projector

M1=psi_G[1];
M2=psi_G[2];
M3=psi_G[3];
M4=psi_G[4];
M5=psi_G[5];
M6=psi_G[6];
M7=psi_G[7];
M8=psi_G[8];
M9=psi_G[9];
M10=psi_G[10];

if Guztwiller
    @tensor M1[:]:=M1[-1,-2,1]*P_G[-3,1];
    @tensor M2[:]:=M2[-1,-2,1]*P_G[-3,1];
    @tensor M3[:]:=M3[-1,-2,1]*P_G[-3,1];
    @tensor M4[:]:=M4[-1,-2,1]*P_G[-3,1];
else
end

U_phy1=unitary(fuse(space(M1,1)⊗space(M1,3)⊗space(M2,3)⊗space(M3,3)⊗space(M4,3)), space(M1,1)⊗space(M1,3)⊗space(M2,3)⊗space(M3,3)⊗space(M4,3));

@tensor A[:]:=M1[8,1,4]*M2[1,2,5]*M3[2,3,6]*M4[3,-2,7]*U_phy1[-1,8,4,5,6,7];
@tensor A[:]:=A[-1,1]*M5[1,-3,-2];
@tensor A[:]:=A[-1,-2,1]*M6[1,-4,-3];
@tensor A[:]:=A[-1,-2,-3,1]*M7[1,-5,-4];
@tensor A[:]:=A[-1,-2,-3,-4,1]*M8[1,-6,-5];
@tensor A[:]:=A[-1,-2,-3,-4,-5,1]*M9[1,-7,-6];
@tensor A[:]:=A[-1,-2,-3,-4,-5,-6,1]*M10[1,-8,-7];
@tensor A[:]:=A[-1,-2,-3,-4,-5,-6,-7,1]*M11[1,-9,-8];
@tensor A[:]:=A[-1,-2,-3,-4,-5,-6,-7,-8,1]*M12[1,-10,-9];

U_phy_dummy=unitary(fuse(space(A,1)⊗space(A,10)), space(A,1)⊗space(A,10));#this doesn't do anything
@tensor A[:]:=A[1,-2,-3,-4,-5,-6,-7,-8,-9,2]*U_phy_dummy[-1,1,2];
# P,L,R,D,U


bond=data["bond_gate"];#dummy, D1, D2 

#Add bond:both parity gate and bond operator
@tensor A[:]:=A[-1,-6,-7,1,2,3,4,-12,-13]*bond[-2,-8,1]*bond[-3,-9,2]*bond[-4,-10,3]*bond[-5,-11,4];

U_phy2=unitary(fuse(space(A,1)⊗space(A,2)⊗space(A,3)⊗space(A,4)⊗space(A,5)), space(A,1)⊗space(A,2)⊗space(A,3)⊗space(A,4)⊗space(A,5));
@tensor A[:]:=A[1,2,3,4,5,-2,-3,-4,-5,-6,-7,-8,-9]*U_phy2[-1,1,2,3,4,5];
#P,L,R,D,U





#swap between spin up and spin down modes, since |L,U,P><D,R|====L,U,P|><|R,D
special_gate=special_parity_gate(A,4);
@tensor A[:]:=A[-1,-2,-3,1,-5,-6,-7,-8,-9]*special_gate[-4,1];
@tensor A[:]:=A[-1,-2,-3,-4,1,-6,-7,-8,-9]*special_gate[-5,1];
@tensor A[:]:=A[-1,-2,-3,-4,-5,1,-7,-8,-9]*special_gate[-6,1];
@tensor A[:]:=A[-1,-2,-3,-4,-5,-6,1,-8,-9]*special_gate[-7,1];

gate=swap_gate(A,4,5);@tensor A[:]:=A[-1,-2,-3,1,2,-6,-7,-8,-9]*gate[-4,-5,1,2];  
gate=swap_gate(A,6,7);@tensor A[:]:=A[-1,-2,-3,-4,-5,1,2,-8,-9]*gate[-6,-7,1,2];  



#group virtual legs on the same legs
U1=unitary(fuse(space(A,2)⊗space(A,3)),space(A,2)⊗space(A,3)); 
U2=unitary(fuse(space(A,8)⊗space(A,9)),space(A,8)⊗space(A,9));
@tensor A[:]:=A[-1,1,2,-3,-4,-5,-6,-7,-8]*U1[-2,1,2];
@tensor A[:]:=A[-1,-2,1,2,-4,-5,-6,-7]*U2'[1,2,-3];
@tensor A[:]:=A[-1,-2,-3,1,2,-5,-6]*U1'[1,2,-4];
@tensor A[:]:=A[-1,-2,-3,-4,1,2]*U2[-5,1,2];



A1=deepcopy(A);


gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2]*gate[-4,-5,1,2];           
A=permute(A,(1,2,3,5,4,));#P,L,R,U,D

gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5]*gate[-3,-4,1,2]; 
A=permute(A,(1,2,4,3,5,));#P,L,U,R,D

gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5]*gate[-1,-2,1,2]; 
A=permute(A,(2,1,3,4,5,));#L,P,U,R,D

gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5]*gate[-2,-3,1,2]; 
A=permute(A,(1,3,2,4,5,));#L,U,P,R,D

#convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D


A_origin=deepcopy(A);




y_anti_pbc=false;
boundary_phase_y=0.5;

if y_anti_pbc
    gauge_gate1=gauge_gate(A,2,2*pi/6*boundary_phase_y);
    @tensor A[:]:=A[-1,1,-3,-4,-5]*gauge_gate1[-2,1];
end

#############################
# #convert to the order of PEPS code
A=permute(A,(1,5,4,2,3,));

#############################


conv_check="singular_value";
CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A,"PBC",true);
@time CTM, AA_fused, U_L,U_D,U_R,U_U=CTMRG(AA_fused,chi,conv_check,tol,CTM,CTM_ite_nums,CTM_trun_tol);

N=6;
EH_n=30;
decomp=false;
ES_CTMRG_ED(CTM,U_L,U_D,U_R,U_U,M,chi,N,decomp,EH_n);



