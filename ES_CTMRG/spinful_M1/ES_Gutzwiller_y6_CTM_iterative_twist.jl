using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)


include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\ES_CTMRG\\swap_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\ES_CTMRG\\fermi_permute.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\ES_CTMRG\\double_layer_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\Projector_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\ES_CTMRG\\CTMRG_funs.jl")

include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\ES_CTMRG\\ES_algorithms.jl")


chi=30
tol=1e-6
CTM_ite_nums=500;
CTM_trun_tol=1e-10;


M=1;
Guztwiller=true;#add projector


data=load("swap_gate_Tensor_M"*string(M)*".jld2");
P_G=data["P_G"];

psi_G=data["psi_G"];   #P1,P2,L,R,D,U
M1=psi_G[1];
M2=psi_G[2];
M3=psi_G[3];
M4=psi_G[4];
M5=psi_G[5];
M6=psi_G[6];

if Guztwiller
    @tensor M1[:]:=M1[-1,-2,1]*P_G[-3,1];
    @tensor M2[:]:=M2[-1,-2,1]*P_G[-3,1];
    SS_op=data["SS_op_S"];
else
    SS_op=data["SS_op_F"];
end


U_phy1=unitary(fuse(space(M1,1)⊗space(M1,3)⊗space(M2,3)), space(M1,1)⊗space(M1,3)⊗space(M2,3));

@tensor A[:]:=M1[4,1,2]*M2[1,-2,3]*U_phy1[-1,4,2,3];
@tensor A[:]:=A[-1,1]*M3[1,-3,-2];
@tensor A[:]:=A[-1,-2,1]*M4[1,-4,-3];
@tensor A[:]:=A[-1,-2,-3,1]*M5[1,-5,-4];
@tensor A[:]:=A[-1,-2,-3,-4,1]*M6[1,-6,-5];

U_phy2=unitary(fuse(space(A,1)⊗space(A,6)), space(A,1)⊗space(A,6));
@tensor A[:]:=A[1,-2,-3,-4,-5,2]*U_phy2[-1,1,2];
# P,L,R,D,U


bond=data["bond_gate"];#dummy, D1, D2 

#Add bond:both parity gate and bond operator
@tensor A[:]:=A[-1,-2,1,2,-5]*bond[-6,-3,1]*bond[-7,-4,2];
U_phy2=unitary(fuse(space(A,1)⊗space(A,6)⊗space(A,7)), space(A,1)⊗space(A,6)⊗space(A,7));
@tensor A[:]:=A[1,-2,-3,-4,-5,2,3]*U_phy2[-1,1,2,3];
#P,L,R,D,U





#swap between spin up and spin down modes, since |L,U,P><D,R|====L,U,P|><|R,D
special_gate=special_parity_gate(A,3);
@tensor A[:]:=A[-1,-2,1,-4,-5]*special_gate[-3,1];
special_gate=special_parity_gate(A,4);
@tensor A[:]:=A[-1,-2,-3,1,-5]*special_gate[-4,1];



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

#small twist
small_twist=true;
if small_twist
    gauge_gate1=gauge_gate(A,2,2*pi/6*0.15);
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
ES_CTMRG_ED(CTM,U_L,U_D,U_R,U_U,M,chi,N,EH_n,decomp,y_anti_pbc);




