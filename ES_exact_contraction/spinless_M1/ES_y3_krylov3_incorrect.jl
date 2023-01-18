using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)


include("swap_funs.jl")
include("fermi_permute.jl")
include("gauge_flux.jl")
include("double_layer_funs.jl")








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




y_anti_pbc=true;
boundary_phase_y=0;

if y_anti_pbc
    gauge_gate1=gauge_gate(A,2,2*pi/3*boundary_phase_y);
    @tensor A[:]:=A[-1,1,-3,-4,-5]*gauge_gate1[-2,1];
end

#############################
# #convert to the order of PEPS code
A=permute(A,(1,5,4,2,3,));
A_fused=deepcopy(A);
AA_fused, U_L,U_D,U_R,U_U=build_double_layer_swap(deepcopy(A_fused'),deepcopy(A_fused));

AA=permute(AA_fused,(1,4,3,2,));#L,U,R,D
#############################



function M_vr(AA,vr,U3,U2)
    vr=deepcopy(vr);#L',L,dummy

    @tensor vr[:]:=vr[1,2,-7]*U3[1,-1,-2,-3]*U3'[-4,-5,-6,2];#L1',L2',L3',L1,L2,L3,dummy
    vr=permute_neighbour_ind(vr,3,4,7);#L1',L2',L1,L3',L2,L3,dummy
    vr=permute_neighbour_ind(vr,2,3,7);#L1',L1,L2',L3',L2,L3,dummy
    vr=permute_neighbour_ind(vr,4,5,7);#L1',L1,L2',L2,L3',L3,dummy
    @tensor vr[:]:=vr[1,2,3,4,5,6,-4]*U2[-1,1,2]*U2[-2,3,4]*U2[-3,5,6];#L1,L2,L3,dummy



    AA1=deepcopy(AA);#L1,U1,R1,D1
    AA2=deepcopy(AA);#L2,U2,R2,D2
    AA3=deepcopy(AA);#L3,U3,R3,D3

    AA1=permute_neighbour_ind(AA1,3,4,4);#L1,U1,D1,R1
    AA2=permute_neighbour_ind(AA2,3,4,4);#L2,U2,D2,R2
    AA3=permute_neighbour_ind(AA3,3,4,4);#L3,U3,D3,R3



    vr=permute_neighbour_ind(vr,2,3,4);#L1,L3,L2,dummy
    vr=permute_neighbour_ind(vr,1,2,4);#L3,L1,L2,dummy
    vr=permute_neighbour_ind(vr,2,3,4);#L3,L2,L1,dummy

    @tensor vr[:]:=AA3[-1,-2,-3,1]*vr[1,-4,-5,-6];#L3,U3,D3,L2,L1,dummy
    vr=permute_neighbour_ind(vr,1,2,6);#U3,L3,D3,L2,L1,dummy
    vr=permute_neighbour_ind(vr,3,4,6);#U3,L3,L2,D3,L1,dummy
    vr=permute_neighbour_ind(vr,2,3,6);#U3,L2,L3,D3,L1,dummy
    vr=permute_neighbour_ind(vr,1,2,6);#L2,U3,L3,D3,L1,dummy

    @tensor vr[:]:=AA2[-1,-2,1,2]*vr[2,1,-3,-4,-5,-6];#L2,U2,L3,D3,L1,dummy
    vr=permute_neighbour_ind(vr,4,5,6);#L2,U2,L3,L1,D3,dummy
    vr=permute_neighbour_ind(vr,3,4,6);#L2,U2,L1,L3,D3,dummy
    vr=permute_neighbour_ind(vr,2,3,6);#L2,L1,U2,L3,D3,dummy
    vr=permute_neighbour_ind(vr,1,2,6);#L1,L2,U2,L3,D3,dummy
    vr=permute_neighbour_ind(vr,2,3,6);#L1,U2,L2,L3,D3,dummy
    vr=permute_neighbour_ind(vr,4,5,6);#L1,U2,L2,D3,L3,dummy
    vr=permute_neighbour_ind(vr,3,4,6);#L1,U2,D3,L2,L3,dummy


    gate=parity_gate(AA1,2); @tensor AA1[:]:=AA1[-1,1,-3,-4]*gate[-2,1];
    @tensor vr[:]:=AA1[-1,1,2,3]*vr[3,2,1,-2,-3,-4];#L1,L2,L3,dummy


    vr=permute(vr,(1,2,3,4,));  #L1,L2,L3,dummy



    @tensor vr[:]:=vr[1,2,3,-7]*U2'[-1,-2,1]*U2'[-3,-4,2]*U2'[-5,-6,3];#L1',L1,L2',L2,L3',L3,dummy
    vr=permute_neighbour_ind(vr,4,5,7);#L1',L1,L2',L3',L2,L3,dummy
    vr=permute_neighbour_ind(vr,2,3,7);#L1',L2',L1,L3',L2,L3,dummy
    vr=permute_neighbour_ind(vr,3,4,7);#L1',L2',L3',L1,L2,L3,dummy


    @tensor vr[:]:=vr[1,2,3,4,5,6,-7]*U3'[1,2,3,-1]*U3[-2,4,5,6];#L',L,dummy


    return vr;
end

function vl_M(AA,vl)
    vl=deepcopy(vl);#dummy,R1,R2,R3
    AA1=deepcopy(AA);#L1,U1,R1,D1
    AA2=deepcopy(AA);#L2,U2,R2,D2
    AA3=deepcopy(AA);#L3,U3,R3,D3

    vl=permute_neighbour_ind(vl,3,4,4);#dummy,R1,R3,R2
    vl=permute_neighbour_ind(vl,2,3,4);#dummy,R3,R1,R2
    vl=permute_neighbour_ind(vl,3,4,4);#dummy,R3,R2,R1


    @tensor vl[:]:=vl[-1,-2,-3,1]*AA1[1,-4,-5,-6];#dummy,R3,R2,U1,R1,D1

    vl=permute_neighbour_ind(vl,3,4,6);#dummy,R3,U1,R2,R1,D1
    vl=permute_neighbour_ind(vl,4,5,6);#dummy,R3,U1,R1,R2,D1
    vl=permute_neighbour_ind(vl,5,6,6);#dummy,R3,U1,R1,D1,R2

    @tensor vl[:]:=vl[-1,-2,-3,-4,1,2]*AA2[2,1,-5,-6];#dummy,R3,U1,R1,R2,D2

    vl=permute_neighbour_ind(vl,2,3,6);#dummy,U1,R3,R1,R2,D2
    vl=permute_neighbour_ind(vl,3,4,6);#dummy,U1,R1,R3,R2,D2
    vl=permute_neighbour_ind(vl,4,5,6);#dummy,U1,R1,R2,R3,D2
    vl=permute_neighbour_ind(vl,5,6,6);#dummy,U1,R1,R2,D2,R3
    vl=permute_neighbour_ind(vl,2,3,6);#dummy,R1,U1,R2,D2,R3
    vl=permute_neighbour_ind(vl,3,4,6);#dummy,R1,R2,U1,D2,R3

    AA3=permute_neighbour_ind(AA3,3,4,4);#L3,U3,D3,R3
    gate=parity_gate(AA3,3); @tensor AA3[:]:=AA3[-1,-2,1,-4]*gate[-3,1];

    @tensor vl[:]:=vl[-1,-2,-3,1,2,3]*AA3[3,2,1,-4];#dummy,R1,R2,R3


    vl=permute(vl,(1,2,3,4,));
    return vl;
end

U2=unitary(fuse(space(U_L,2)*space(U_L,3)), space(U_L,2)*space(U_L,3));
v_init=TensorMap(randn, fuse(space(U_L,2)*space(U_L,2)*space(U_L,2))⊗ fuse(space(U_L,2)*space(U_L,2)*space(U_L,2))',Rep[U₁](0=>1));
U3=unitary(space(v_init,1)', space(U_L,3)*space(U_L,3)*space(U_L,3));
v_init=permute(v_init,(1,2,3,),());#L',L,dummy
contraction_fun_R(x)=M_vr(AA,x,U3,U2);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 1,:LM,Arnoldi(krylovdim=10));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,dummy



@tensor VR[:]:=VR[1,2,-7]*U3[1,-1,-2,-3]*U3'[-4,-5,-6,2];#L1',L2',L3',L1,L2,L3,dummy
VR=permute_neighbour_ind(VR,3,4,7);#L1',L2',L1,L3',L2,L3,dummy
VR=permute_neighbour_ind(VR,2,3,7);#L1',L1,L2',L3',L2,L3,dummy
VR=permute_neighbour_ind(VR,4,5,7);#L1',L1,L2',L2,L3',L3,dummy
@tensor VR[:]:=VR[1,2,3,4,5,6,-4]*U2[-1,1,2]*U2[-2,3,4]*U2[-3,5,6];#L1,L2,L3,dummy
VR_cc=deepcopy(VR);







v_init=TensorMap(randn, space(AA,3)*space(AA,3)*space(AA,3),Rep[U₁](0=>1)');
v_init=permute(v_init,(4,1,2,3,),());#dummy,R1,R2,R3
contraction_fun_L(x)=vl_M(AA,x);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 1,:LM,Arnoldi(krylovdim=10));
VL=evl[findmax(abs.(eul))[2]];#dummy,R1,R2,R3






@tensor VL[:]:=VL[-1,1,2,3]*U_L[1,-2,-3]*U_L[2,-4,-5]*U_L[3,-6,-7];#dummy, R1',R1,R2',R2,R3',R3
VL=permute(VL,(1,3,2,5,4,7,6,));#dummy, R1,R1',R2,R2',R3,R3'






@tensor VR[:]:=VR[1,2,3,-7]*U_L'[-1,-2,1]*U_L'[-3,-4,2]*U_L'[-5,-6,3];#L1',L1,L2',L2,L3',L3,dummy



VR=permute_neighbour_ind(VR,2,3,7);#L1',L2',L1,L2,L3',L3,dummy
VR=permute_neighbour_ind(VR,4,5,7);#L1',L2',L1,L3',L2,L3,dummy
VR=permute_neighbour_ind(VR,3,4,7);#L1',L2',L3',L1,L2,L3,dummy


# U1=unitary(fuse(space(VR,1)*space(VR,2)*space(VR,3)), space(VR,1)*space(VR,2)*space(VR,3));
# U2=unitary(fuse(space(VR,4)*space(VR,5)*space(VR,6)), space(VR,4)*space(VR,5)*space(VR,6));

# @tensor VR_a[:]:=VR[1,2,3,4,5,6,-3]*U1[-1,1,2,3]*U2[-2,4,5,6];#L',L,dummy


VL_a=deepcopy(VL);

VL=permute_neighbour_ind(VL,5,6,7);#dummy, R1,R1',R2,R3,R2',R3'
VL=permute_neighbour_ind(VL,4,5,7);#dummy, R1,R1',R3,R2,R2',R3'
VL=permute_neighbour_ind(VL,3,4,7);#dummy, R1,R3,R1',R2,R2',R3'
VL=permute_neighbour_ind(VL,2,3,7);#dummy, R3,R1,R1',R2,R2',R3'
VL=permute_neighbour_ind(VL,4,5,7);#dummy, R3,R1,R2,R1',R2',R3'
VL=permute_neighbour_ind(VL,3,4,7);#dummy, R3,R2,R1,R1',R2',R3'
VL=permute_neighbour_ind(VL,6,7,7);#dummy, R3,R2,R1,R1',R3',R2'
VL=permute_neighbour_ind(VL,5,6,7);#dummy, R3,R2,R1,R3',R1',R2'
VL=permute_neighbour_ind(VL,6,7,7);#dummy, R3,R2,R1,R3',R2',R1'



VL_a=permute_neighbour_ind(VL_a,3,4,7);#dummy, R1,R2,R1',R2',R3,R3'
VL_a=permute_neighbour_ind(VL_a,5,6,7);#dummy, R1,R2,R1',R3,R2',R3'
VL_a=permute_neighbour_ind(VL_a,4,5,7);#dummy, R1,R2,R3,R1',R2',R3'

@tensor VL_a[:]:=VL_a[-1,1,2,3,4,5,6,]*U2'[1,2,3,-2]*U1'[4,5,6,-3];#dummy,R,R'




@tensor H[:]:=VR[-1,-2,-3,1,2,3,4]*VL[4,3,2,1,-4,-5,-6];#L1',L2',L3' ,R3',R2',R1'
H=permute(H,(1,2,3,6,5,4,));


eu,ev=eig(H,(1,2,3,),(4,5,6,))
eu=diag(convert(Array,eu));
eu=eu/sum(eu)



# ev=permute(ev,(1,2,3,));#L1',L2',dummy
# ev_translation=VL=permute_neighbour_ind(ev',1,2,3);#L2',L1',dummy

# @tensor k_phase[:]:=ev_translation[1,2,-1]*ev[1,2,-3];
# k_phase=convert(Array,k_phase);
# @assert norm(diagm(diag(k_phase))-k_phase)/norm(k_phase)<1e-10;



@tensor H[:]:=VL[1,-1,-2,-3,2,3,4]*VR[4,3,2,-4,-5,-6,1];#R3,R2,R1, L1,L2,L3



function H_v(H,v)
    H=deepcopy(H);#R3,R2,R1, L1,L2,L3
    v=deepcopy(v);#R3,R2,R1,dummy
    @tensor v[:]:=H[-1,-2,-3,1,2,3]*v[3,2,1,-4];#R3,R2,R1, L1,L2,L3  R3,R2,R1,dummy

    v=permute(v,(1,2,3,4,));
    return v;
end
v_init=TensorMap(randn, space(H,1)*space(H,1)*space(H,1),Rep[U₁](0=>1));
v_init=permute(v_init,(1,2,3,4,),());#R3,R2,R1,dummy
contraction_fun(x)=H_v(H,x);
@time eu,ev=eigsolve(contraction_fun, v_init, 5,:LM,Arnoldi(krylovdim=10));

v_init=TensorMap(randn, space(H,1)*space(H,1)*space(H,1),Rep[U₁](-2=>1));
v_init=permute(v_init,(1,2,3,4,),());#R3,R2,R1,dummy
contraction_fun(x)=H_v(H,x);
@time eu,ev=eigsolve(contraction_fun, v_init, 5,:LM,Arnoldi(krylovdim=20));


