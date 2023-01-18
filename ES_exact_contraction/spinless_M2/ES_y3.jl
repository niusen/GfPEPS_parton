using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)


include("swap_funs.jl")
include("fermi_permute.jl")
include("gauge_flux.jl")








data=load("swap_gate_Tensor_M2.jld2")
A=data["A"];   #P1,P2,L,R,D,U

A_new=zeros(1,2,2,2,2,2,2,2,2,2,2)*im;
A_new[1,:,:,:,:,:,:,:,:,:,:]=A;
Vdummy=ℂ[U1Irrep](-5=>1);
V=ℂ[U1Irrep](0=>1,1=>1);
# Vdummy=GradedSpace[Irrep[U₁]⊠Irrep[ℤ₂]]((-3,1)=>1);
# V=GradedSpace[Irrep[U₁]⊠Irrep[ℤ₂]]((0,0)=>1,(1,1)=>1);
#A_new = TensorMap(A_new, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ← V'⊗ V'⊗ V'⊗ V');
A_new = TensorMap(A_new, Vdummy' ⊗ V' ⊗ V' ⊗ V' ⊗ V' ⊗ V' ⊗ V' ← V⊗ V⊗ V⊗ V);


@assert norm(convert(Array,A_new)[1,:,:,:,:,:,:,:,:,:,:]-A)/norm(A)<1e-14
A=A_new; # dummy,P1,P2,L,R,D,U


U_phy1=unitary(fuse(space(A,1)⊗space(A,2)⊗space(A,3)), space(A,1)⊗space(A,2)⊗space(A,3));
@tensor A[:]:=A[1,2,3,-2,-3,-4,-5,-6,-7,-8,-9]*U_phy1[-1,1,2,3]; # P,L,R,D,U


#Add bond:both parity gate and bond operator
bond=zeros(1,2,2); bond[1,1,2]=1;bond[1,2,1]=1; bond=TensorMap(bond, ℂ[U1Irrep](1=>1)' ← V' ⊗ V');
gate=parity_gate(A,4); @tensor A[:]:=A[-1,-2,-3,1,-5,-6,-7,-8,-9]*gate[-4,1];
gate=parity_gate(A,6); @tensor A[:]:=A[-1,-2,-3,-4,-5,1,-7,-8,-9]*gate[-6,1];
@tensor total_bond[:]:=bond[-1,-5,-9]*bond[-2,-6,-10]*bond[-3,-7,-11]*bond[-4,-8,-12];
@tensor A[:]:=A[-1,-2,-3,1,2,3,4,-8,-9]*total_bond[-10,-11,-12,-13,-4,-5,-6,-7,1,2,3,4];
U_phy2=unitary(fuse(space(A,1)⊗space(A,10)⊗space(A,11)⊗space(A,12)⊗space(A,13)), space(A,1)⊗space(A,10)⊗space(A,11)⊗space(A,12)⊗space(A,13));
@tensor A[:]:=A[1,-2,-3,-4,-5,-6,-7,-8,-9,2,3,4,5]*U_phy2[-1,1,2,3,4,5];
#P,L,R,D,U


###################
#|><R1R2|=|><|R2R1
gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2,-6,-7,-8,-9]*gate[-4,-5,1,2];  
gate=swap_gate(A,6,7); @tensor A[:]:=A[-1,-2,-3,-4,-5,1,2,-8,-9]*gate[-6,-7,1,2];  
###################


#group virtual legs on the same legs
U1=unitary(fuse(space(A,2)⊗space(A,3)),space(A,2)⊗space(A,3)); 
U2=unitary(fuse(space(A,8)⊗space(A,9)),space(A,8)⊗space(A,9));

@tensor A[:]:=A[-1,1,2,-3,-4,-5,-6,-7,-8]*U1[-2,1,2];
@tensor A[:]:=A[-1,-2,1,2,-4,-5,-6,-7]*U2'[1,2,-3];
@tensor A[:]:=A[-1,-2,-3,1,2,-5,-6]*U1'[1,2,-4];
@tensor A[:]:=A[-1,-2,-3,-4,1,2]*U2[-5,1,2];


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
A3=deepcopy(A_origin);#L3,U3,P3,R3,D3

if y_anti_pbc
    gauge_gate1=gauge_gate(A1,2,2*pi/3*boundary_phase_y);
    @tensor A1[:]:=A1[-1,1,-3,-4,-5]*gauge_gate1[-2,1];
    gauge_gate2=gauge_gate(A2,2,2*pi/3*boundary_phase_y);
    @tensor A2[:]:=A2[-1,1,-3,-4,-5]*gauge_gate2[-2,1];
    gauge_gate3=gauge_gate(A3,2,2*pi/3*boundary_phase_y);
    @tensor A3[:]:=A3[-1,1,-3,-4,-5]*gauge_gate3[-2,1];
end

A2=permute_neighbour_ind(A2,1,2,5);#U2,L2,P2,R2,D2

@tensor A1A2[:]:=A1[-1,-2,-3,-4,1]*A2[1,-5,-6,-7,-8];#L1,U1,P1,R1,L2,P2,R2,D2


A1A2=permute_neighbour_ind(A1A2,2,3,8);#L1,P1,U1,R1,L2,P2,R2,D2
A1A2=permute_neighbour_ind(A1A2,3,4,8);#L1,P1,R1,U1,L2,P2,R2,D2
A1A2=permute_neighbour_ind(A1A2,4,5,8);#L1,P1,R1,L2,U1,P2,R2,D2
A1A2=permute_neighbour_ind(A1A2,5,6,8);#L1,P1,R1,L2,P2,U1,R2,D2
A1A2=permute_neighbour_ind(A1A2,6,7,8);#L1,P1,R1,L2,P2,R2,U1,D2

A1A2=permute_neighbour_ind(A1A2,3,4,8);#L1,P1,L2,R1,P2,R2,U1,D2
A1A2=permute_neighbour_ind(A1A2,2,3,8);#L1,L2,P1,R1,P2,R2,U1,D2
A1A2=permute_neighbour_ind(A1A2,4,5,8);#L1,L2,P1,P2,R1,R2,U1,D2

U_phy=unitary(fuse(space(A1A2,3)⊗space(A1A2,4)),space(A1A2,3)⊗space(A1A2,4));
@tensor A1A2[:]:=A1A2[-1,-2,1,2,-4,-5,-6,-7]*U_phy[-3,1,2];#L1,L2,P,R1,R2,U1,D2

A1A2=permute_neighbour_ind(A1A2,5,6,7);#L1,L2,P,R1,U1,R2,D2
A1A2=permute_neighbour_ind(A1A2,4,5,7);#L1,L2,P,U1,R1,R2,D2

A3=permute_neighbour_ind(A3,1,2,5);#U3,L3,P3,R3,D3
A3=permute_neighbour_ind(A3,3,4,5);#U3,L3,R3,P3,D3
A3=permute_neighbour_ind(A3,2,3,5);#U3,R3,L3,P3,D3

@tensor A1A2A3[:]:=A1A2[-1,-2,-3,-4,-5,-6,1]*A3[1,-7,-8,-9,-10];#L1,L2,P,U1,R1,R2,R3,L3,P3,D3





U=unitary(fuse(space(A1A2A3,5)⊗space(A1A2A3,6)⊗space(A1A2A3,7)), space(A1A2A3,5)⊗space(A1A2A3,6)⊗space(A1A2A3,7));
@tensor A1A2A3[:]:=A1A2A3[-1,-2,-3,-4,1,2,3,-6,-7,-8]*U[-5,1,2,3];#L1,L2,P,U1,R,L3,P3,D3

A1A2A3=permute_neighbour_ind(A1A2A3,4,5,8);#L1,L2,P,R,U1,L3,P3,D3
A1A2A3=permute_neighbour_ind(A1A2A3,5,6,8);#L1,L2,P,R,L3,U1,P3,D3
A1A2A3=permute_neighbour_ind(A1A2A3,6,7,8);#L1,L2,P,R,L3,P3,U1,D3
A1A2A3=permute_neighbour_ind(A1A2A3,7,8,8);#L1,L2,P,R,L3,P3,D3,U1

@tensor A1A2A3[:]:=A1A2A3[-1,-2,-3,-4,-5,-6,1,1];#L1,L2,P,R,L3,P3

A1A2A3=permute_neighbour_ind(A1A2A3,4,5,6);#L1,L2,P,L3,R,P3
A1A2A3=permute_neighbour_ind(A1A2A3,3,4,6);#L1,L2,L3,P,R,P3
A1A2A3=permute_neighbour_ind(A1A2A3,5,6,6);#L1,L2,L3,P,P3,R


##################
A1A2A3=permute_neighbour_ind(A1A2A3,2,3,6);#L1,L3,L2,P,P3,R
A1A2A3=permute_neighbour_ind(A1A2A3,1,2,6);#L3,L1,L2,P,P3,R
A1A2A3=permute_neighbour_ind(A1A2A3,2,3,6);#L3,L2,L1,P,P3,R
A1A2A3=permute(A1A2A3,(3,2,1,4,5,6,))
##################

@tensor A1A2A3[:]:=A1A2A3[1,2,3,-2,-3,-4]*U'[1,2,3,-1];#L,P,P3,R

U_phy2=unitary(fuse(space(A1A2A3,2)⊗space(A1A2A3,3)),space(A1A2A3,2)⊗space(A1A2A3,3));
@tensor A1A2A3[:]:=A1A2A3[-1,1,2,-3]*U_phy2[-2,1,2];#L,P,R





@tensor AA[:]:=A1A2A3'[-1,1,-3]*A1A2A3[-2,1,-4];



eu,ev=eig(AA,(1,2,),(3,4,))


@assert norm(ev*eu*inv(ev)-permute(AA,(1,2,),(3,4,)))/norm(AA)<1e-12;
pp=zeros(1,dim(space(eu,1)));
pp[findmax(abs.(diag(convert(Array,eu))))[2]]=1;
P=TensorMap(pp,Rep[U₁](0=>1),space(eu,1));
@assert abs(convert(Array,P*eu*P')[1])==maximum(abs.(diag(convert(Array,eu))))
# U=unitary(fuse(space(AA,1)⊗space(AA,2)),space(AA,1)⊗space(AA,2));

# @tensor AA[:]:=AA[1,2,3,4]*U[-1,1,2]*U'[3,4,-2];

VR=ev*P';#L',L,dummy
VR_b=deepcopy(permute(VR,(1,2,3,)));#L',L,dummy

VR=permute(VR,(2,1,),(3,));#L,L',dummy
VL=P*inv(ev);#dummy,R',R
VL_b=deepcopy(VL);
VL_b=VL_b/norm(VL_b);
@tensor VL_b[:]:=VL_b[-1,1,2]*U[1,-2,-3,-4]*U'[-5,-6,-7,2];#dummy,R1',R2',R3',R1,R2,R3


@tensor H[:]:=VR[-1,1,2]*VL[2,1,-2];
H=convert(Array,H);
eu,ev=eigen(H);
println(eu)



