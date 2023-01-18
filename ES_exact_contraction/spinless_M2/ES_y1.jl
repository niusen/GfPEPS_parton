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

if y_anti_pbc
    gauge_gate1=gauge_gate(A,2,2*pi*boundary_phase_y);
    @tensor A[:]:=A[-1,1,-3,-4,-5]*gauge_gate1[-2,1];
end


A=permute_neighbour_ind(A,2,3,5);#L,P,U,R,D
A=permute_neighbour_ind(A,3,4,5);#L,P,R,U,D
A=permute_neighbour_ind(A,4,5,5);#L,P,R,D,U


@tensor A[:]:=A[-1,-2,-3,1,1];#L,P,R

A_origin1=deepcopy(A);#L,P,R

@tensor AA[:]:=A_origin1'[-1,1,-3]*A_origin1[-2,1,-4];#L',L,R',R



eu,ev=eig(AA,(1,2,),(3,4,))


@assert norm(ev*eu*inv(ev)-permute(AA,(1,2,),(3,4,)))/norm(AA)<1e-12;
eu_dense=abs.(diag(convert(Array,eu)));
Pm=zeros(1,length(eu_dense))*im;
Pm[findmax(abs.(eu_dense))[2]]=1;
P=TensorMap(Pm,Rep[U₁](0=>1),space(eu,1));
@assert abs(convert(Array,P*eu*P')[1])==maximum(abs.(diag(convert(Array,eu))))
# U=unitary(fuse(space(AA,1)⊗space(AA,2)),space(AA,1)⊗space(AA,2));

# @tensor AA[:]:=AA[1,2,3,4]*U[-1,1,2]*U'[3,4,-2];

VL=permute(ev*P',(2,1,),(3,));#R,R'
VR=P*inv(ev);#L',L

@tensor H[:]:=VL[-1,1,2]*VR[2,1,-2];
H=convert(Array,H);
eu,ev=eigen(H);
println(eu)



