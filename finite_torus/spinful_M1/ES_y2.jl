using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)


include("swap_funs.jl")
include("fermi_permute.jl")







M=1;
Guztwiller=false;#add projector


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



#############################
# #convert to the order of PEPS code
# A=permute(A,(1,5,4,2,3,));
#############################



y_anti_pbc=true;
boundary_phase_y=0.0;


#########################################
A1=deepcopy(A_origin);#L1,U1,P1,R1,D1
A2=deepcopy(A_origin);#L2,U2,P2,R2,D2

if y_anti_pbc
    gauge_gate1=gauge_gate(A1,2,pi*boundary_phase_y);
    @tensor A1[:]:=A1[-1,1,-3,-4,-5]*gauge_gate1[-2,1];
    gauge_gate2=gauge_gate(A2,2,pi*boundary_phase_y);
    @tensor A2[:]:=A2[-1,1,-3,-4,-5]*gauge_gate2[-2,1];

end

A2=permute_neighbour_ind(A2,1,2,5);#U2,L2,P2,R2,D2

@tensor A1A2[:]:=A1[-1,-2,-3,-4,1]*A2[1,-5,-6,-7,-8];#L1,U1,P1,R1,L2,P2,R2,D2
A1A2=permute_neighbour_ind(A1A2,2,3,8);#L1,P1,U1,R1,L2,P2,R2,D2
A1A2=permute_neighbour_ind(A1A2,3,4,8);#L1,P1,R1,U1,L2,P2,R2,D2
A1A2=permute_neighbour_ind(A1A2,4,5,8);#L1,P1,R1,L2,U1,P2,R2,D2
A1A2=permute_neighbour_ind(A1A2,5,6,8);#L1,P1,R1,L2,P2,U1,R2,D2
A1A2=permute_neighbour_ind(A1A2,6,7,8);#L1,P1,R1,L2,P2,R2,U1,D2
A1A2=permute_neighbour_ind(A1A2,7,8,8);#L1,P1,R1,L2,P2,R2,D2,U1
@tensor A1A2[:]:=A1A2[-1,-2,-3,-4,-5,-6,1,1];#L1,P1,R1,L2,P2,R2
#########################################

A1A2=permute_neighbour_ind(A1A2,3,4,6);#L1,P1,L2,R1,P2,R2
A1A2=permute_neighbour_ind(A1A2,2,3,6);#L1,L2,P1,R1,P2,R2
A1A2=permute_neighbour_ind(A1A2,4,5,6);#L1,L2,P1,P2,R1,R2

U_phy=unitary(fuse(space(A1A2,3)⊗space(A1A2,4)),space(A1A2,3)⊗space(A1A2,4));
@tensor A1A2[:]:=A1A2[-1,-2,1,2,-4,-5]*U_phy[-3,1,2];#L1,L2,P,R1,R2

###############
#the below line seems to be not necessary???????
gate=swap_gate(A1A2,4,5); @tensor A1A2[:]:=A1A2[-1,-2,-3,1,2]*gate[-4,-5,1,2]; 
###############
U=unitary(fuse(space(A1A2,1)⊗space(A1A2,2)),space(A1A2,1)⊗space(A1A2,2));
@tensor A1A2[:]:=A1A2[1,2,-2,3,4]*U[-1,1,2]*U'[3,4,-3];

@tensor AA[:]:=A1A2'[-1,1,-3]*A1A2[-2,1,-4];



eu,ev=eig(AA,(1,2,),(3,4,))
println(diag(convert(Array,eu)))

@assert norm(ev*eu*inv(ev)-permute(AA,(1,2,),(3,4,)))/norm(AA)<1e-12;
pp=zeros(1,dim(space(eu,1)));
pp[findmax(abs.(diag(convert(Array,eu))))[2]]=1;
P=TensorMap(pp,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1),space(eu,1));
@assert abs(convert(Array,P*eu*P')[1])==maximum(abs.(diag(convert(Array,eu))))
# U=unitary(fuse(space(AA,1)⊗space(AA,2)),space(AA,1)⊗space(AA,2));

# @tensor AA[:]:=AA[1,2,3,4]*U[-1,1,2]*U'[3,4,-2];

VL=permute(ev*P',(2,1,),(3,));#R,R'
VR=P*inv(ev);#L',L

@tensor H[:]:=VL[-1,1,2]*VR[2,1,-2];
H=convert(Array,H);
eu,ev=eigen(H);
println(eu)



