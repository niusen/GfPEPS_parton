using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)


include("swap_funs.jl")
include("fermi_permute.jl")
include("gauge_flux.jl")







data=load("swap_gate_Tensor_M1.jld2")
A=data["A"];   #P1,P2,L,R,D,U

A_new=zeros(1,2,2,2,2,2,2)*im;
A_new[1,:,:,:,:,:,:]=A;
Vdummy=ℂ[U1Irrep](-3=>1);
V=ℂ[U1Irrep](0=>1,1=>1);
# Vdummy=GradedSpace[Irrep[U₁]⊠Irrep[ℤ₂]]((-3,1)=>1);
# V=GradedSpace[Irrep[U₁]⊠Irrep[ℤ₂]]((0,0)=>1,(1,1)=>1);
A_new = TensorMap(A_new, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ← V');

@assert norm(convert(Array,A_new)[1,:,:,:,:,:,:]-A)/norm(A)<1e-14
A=A_new; # dummy,P1,P2,L,R,D,U


U_phy1=unitary(fuse(space(A,1)⊗space(A,2)⊗space(A,3)), space(A,1)⊗space(A,2)⊗space(A,3));
@tensor A[:]:=A[1,2,3,-2,-3,-4,-5]*U_phy1[-1,1,2,3]; # P,L,R,D,U

#Add bond:both parity gate and bond operator
bond=zeros(1,2,2); bond[1,1,2]=1;bond[1,2,1]=1; bond=TensorMap(bond, ℂ[U1Irrep](1=>1) ← V ⊗ V);
gate=parity_gate(A,3); @tensor A[:]:=A[-1,-2,1,-4,-5]*gate[-3,1];
@tensor A[:]:=A[-1,-2,1,2,-5]*bond[-6,-3,1]*bond[-7,-4,2];
U_phy2=unitary(fuse(space(A,1)⊗space(A,6)⊗space(A,7)), space(A,1)⊗space(A,6)⊗space(A,7));
@tensor A[:]:=A[1,-2,-3,-4,-5,2,3]*U_phy2[-1,1,2,3];
#P,L,R,D,U




gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2]*gate[-4,-5,1,2];           
A=permute(A,(1,2,3,5,4,));#P,L,R,U,D

gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5]*gate[-3,-4,1,2]; 
A=permute(A,(1,2,4,3,5,));#P,L,U,R,D

gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5]*gate[-1,-2,1,2]; 
A=permute(A,(2,1,3,4,5,));#L,P,U,R,D

gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5]*gate[-2,-3,1,2]; 
A=permute(A,(1,3,2,4,5,));#L,U,P,R,D

A_origin=deepcopy(A);

#convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D

#############################
# #convert to the order of PEPS code
# A=permute(A,(1,5,4,2,3,));
#############################



y_anti_pbc=true;
boundary_phase_y=0;

if y_anti_pbc
    gauge_gate1=gauge_gate(A,2,2*pi*boundary_phase_y);
    @tensor A[:]:=A[-1,1,-3,-4,-5]*gauge_gate1[-2,1];
end


A=permute_neighbour_ind(A,2,3,5);#L,P,U,R,D
A=permute_neighbour_ind(A,3,4,5);#L,P,R,U,D
A=permute_neighbour_ind(A,4,5,5);#L,P,R,D,U


@tensor A[:]:=A[-1,-2,-3,1,1];#L,P,R

A_origin1=deepcopy(A);#L,P,R

method=2;
if method==1
    Adag=permute(A_origin1',(3,2,1,));#R',P',L'
    A=permute_neighbour_ind(A,1,2,3);#P,L,R
    Adag=permute_neighbour_ind(Adag,2,3,3);#R',L',P'
    @tensor AA[:]:=Adag[-1,-2,1]*A[1,-3,-4];#R',L',L,R

    AA=permute_neighbour_ind(AA,1,2,4);#L',R',L,R
    AA=permute_neighbour_ind(AA,2,3,4);#L',L,R',R


        
    gate=swap_gate(AA,3,4); @tensor AA[:]:=AA[-1,-2,1,2]*gate[-3,-4,1,2]; 

    eu,ev=eig(AA,(1,2,),(3,4,))


    @assert norm(ev*eu*inv(ev)-permute(AA,(1,2,),(3,4,)))/norm(AA)<1e-12;

    P=TensorMap([0 1 0 0],Rep[U₁](0=>1),space(eu,1));
    @assert abs(convert(Array,P*eu*P')[1])==maximum(abs.(diag(convert(Array,eu))))
    # U=unitary(fuse(space(AA,1)⊗space(AA,2)),space(AA,1)⊗space(AA,2));

    # @tensor AA[:]:=AA[1,2,3,4]*U[-1,1,2]*U'[3,4,-2];

    VL=permute(ev*P',(2,1,),(3,));#R,R'
    VR=P*inv(ev);#L',L

    @tensor H[:]:=VL[-1,1,2]*VR[2,1,-2];
    H=convert(Array,H);
    eu,ev=eigen(H);
    println(eu)

elseif method==2
    @tensor AA[:]:=A_origin1'[-1,1,-3]*A_origin1[-2,1,-4];#L',L,R',R


    
    eu,ev=eig(AA,(1,2,),(3,4,))


    @assert norm(ev*eu*inv(ev)-permute(AA,(1,2,),(3,4,)))/norm(AA)<1e-12;

    P=TensorMap([0 1 0 0],Rep[U₁](0=>1),space(eu,1));
    @assert abs(convert(Array,P*eu*P')[1])==maximum(abs.(diag(convert(Array,eu))))
    # U=unitary(fuse(space(AA,1)⊗space(AA,2)),space(AA,1)⊗space(AA,2));

    # @tensor AA[:]:=AA[1,2,3,4]*U[-1,1,2]*U'[3,4,-2];

    VL=permute(ev*P',(2,1,),(3,));#R,R'
    VR=P*inv(ev);#L',L

    @tensor H[:]:=VL[-1,1,2]*VR[2,1,-2];
    H=convert(Array,H);
    eu,ev=eigen(H);
    println(eu)

end

