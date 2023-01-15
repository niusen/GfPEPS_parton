using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)


include("swap_funs.jl")
include("fermi_permute.jl")
include("gauge_flux.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\GfPEPS_ctmrg\\swap_gate_ctmrg\\M1\\GfPEPS_CTMRG.jl")








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
    gauge_gate1=gauge_gate(A,2,pi*boundary_phase_y);
    @tensor A[:]:=A[-1,1,-3,-4,-5]*gauge_gate1[-2,1];
end

#############################
# #convert to the order of PEPS code
A=permute(A,(1,5,4,2,3,));
A_fused=deepcopy(A);
CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(1,A_fused,"PBC",true);

AA=permute(AA_fused,(1,4,3,2,));#L,U,R,D
#############################



function M_vr(AA,vr)
    vr=deepcopy(vr);#L1,L2,dummy
    AA1=deepcopy(AA);#L1,U1,R1,D1
    AA2=deepcopy(AA);#L2,U2,R2,D2

    AA1=permute_neighbour_ind(AA1,3,4,4);#L1,U1,D1,R1
    @tensor vr[:]:=AA1[-1,-2,-3,1]*vr[1,-4,-5];#L1,U1,D1, L2,dummy

    vr=permute_neighbour_ind(vr,2,3,5);#L1,D1,U1, L2,dummy
    vr=permute_neighbour_ind(vr,1,2,5);#D1,L1,U1, L2,dummy
    vr=permute_neighbour_ind(vr,3,4,5);#D1,L1,L2,U1,dummy
    vr=permute_neighbour_ind(vr,2,3,5);#D1,L2,L1,U1,dummy
    vr=permute_neighbour_ind(vr,1,2,5);#L2,D1,L1,U1,dummy

    AA2=permute_neighbour_ind(AA2,3,4,4);#L2,U2,D2,R2
    AA2=permute_neighbour_ind(AA2,2,3,4);#L2,D2,U2,R2
    gate=parity_gate(AA2,3); @tensor AA2[:]:=AA2[-1,-2,1,-4]*gate[-3,1];
    @tensor vr[:]:=AA2[-1,-2,1,2]*vr[2,1,-3,-4,-5];#L2,D2,L1,U1,dummy

    vr=permute_neighbour_ind(vr,2,3,5);#L2,L1,D2,U1,dummy

    @tensor vr[:]:=vr[-1,-2,1,1,-3];#L2,L1,dummy
    vr=permute_neighbour_ind(vr,1,2,3);#L1,L2,dummy

    vr=permute(vr,(1,2,3,));
    return vr;
end

function vl_M(AA,vl)
    vl=deepcopy(vl);#dummy,R1,R2
    AA1=deepcopy(AA);#L1,U1,R1,D1
    AA2=deepcopy(AA);#L2,U2,R2,D2

    @tensor vl[:]:=vl[-1,-2,1]*AA2[1,-3,-4,-5];#dummy,R1,U2,R2,D2

    vl=permute_neighbour_ind(vl,2,3,5);#dummy,U2,R1,R2,D2
    vl=permute_neighbour_ind(vl,3,4,5);#dummy,U2,R2,R1,D2
    vl=permute_neighbour_ind(vl,4,5,5);#dummy,U2,R2,D2,R1

    # gate=parity_gate(vl,4); @tensor vl[:]:=vl[-1,-2,-3,1,-5]*gate[-4,1];
    @tensor vl[:]:=vl[-1,-2,-3,1,2]*AA1[2,1,-4,-5];#dummy,U2,R2,R1,D1

    vl=permute_neighbour_ind(vl,2,3,5);#dummy,R2,U2,R1,D1
    vl=permute_neighbour_ind(vl,3,4,5);#dummy,R2,R1,U2,D1
    vl=permute_neighbour_ind(vl,4,5,5);#dummy,R2,R1,D1,U2
    @tensor vl[:]:=vl[-1,-2,-3,1,1];#dummy,R2,R1

    vl=permute_neighbour_ind(vl,2,3,3);#dummy,R1,R2

    vl=permute(vl,(1,2,3,));
    return vl;
end
v_init=TensorMap(randn, space(AA,1)*space(AA,1),Rep[U₁](0=>1));
v_init=permute(v_init,(1,2,3,),());#L1,L2,dummy
contraction_fun_R(x)=M_vr(AA,x);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 1,:LM,Arnoldi(krylovdim=10));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,dummy

v_init=TensorMap(randn, space(AA,3)*space(AA,3),Rep[U₁](0=>1)');
v_init=permute(v_init,(3,1,2,),());#dummy,R1,R2
contraction_fun_L(x)=vl_M(AA,x);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 1,:LM,Arnoldi(krylovdim=10));
VL=evl[findmax(abs.(eul))[2]];#dummy,R1,R2


@tensor VL[:]:=VL[-1,1,2]*U_L[1,-2,-3]*U_L[2,-4,-5];#dummy, R1',R1,R2',R2
VL=permute(VL,(1,3,2,5,4,));#dummy, R1,R1',R2,R2'
@tensor VR[:]:=VR[1,2,-5]*U_L'[-1,-2,1]*U_L'[-3,-4,2];#L1',L1,L2',L2,dummy







VR=permute_neighbour_ind(VR,2,3,5);#L1',L2',L1,L2,dummy

VL=permute_neighbour_ind(VL,3,4,5);#dummy, R1,R2,R1',R2'
VL=permute_neighbour_ind(VL,2,3,5);#dummy, R2,R1,R1',R2'
VL=permute_neighbour_ind(VL,4,5,5);#dummy, R2,R1,R2',R1'




@tensor H[:]:=VR[-1,-2,1,2,3]*VL[3,2,1,-3,-4];#L1',L2',R2',R1'
H=permute(H,(1,2,4,3,));
eu,ev=eig(H,(1,2,),(3,4,))
eu=diag(convert(Array,eu));
eu=eu/sum(eu)

ev=permute(ev,(1,2,3,));#L1',L2',dummy
ev_translation=VL=permute_neighbour_ind(ev',1,2,3);#L2',L1',dummy

@tensor k_phase[:]:=ev_translation[1,2,-1]*ev[1,2,-3];
k_phase=convert(Array,k_phase);
@assert norm(diagm(diag(k_phase))-k_phase)/norm(k_phase)<1e-10;

# @tensor H[:]:=VL[3,-1,-2,2,1]*VR[1,2,-3,-4,3];#R2,R1,L1,L2 
# H=permute(H,(1,2,4,3,));
# eu,ev=eig(H,(1,2,),(3,4,))
# eu=diag(convert(Array,eu));
# eu=eu/sum(eu)



# U1=unitary(fuse(space(AA1AA2,1)⊗space(AA1AA2,2)), space(AA1AA2,1)⊗space(AA1AA2,2));
# U2=unitary(fuse(space(AA1AA2,3)⊗space(AA1AA2,4)), space(AA1AA2,3)⊗space(AA1AA2,4));
# @tensor AA1AA2[:]:=AA1AA2[1,2,3,4,5,6,7,8]*U1[-1,1,2]*U2[-2,3,4]*U1'[5,6,-3]*U2'[7,8,-4];#L',L,R',R

# eu,ev=eig(AA1AA2,(1,2,),(3,4,))


# @assert norm(ev*eu*inv(ev)-permute(AA1AA2,(1,2,),(3,4,)))/norm(AA1AA2)<1e-12;

# pp=zeros(1,dim(space(eu,1)));
# pp[findmax(abs.(diag(convert(Array,eu))))[2]]=1;
# P=TensorMap(pp,Rep[U₁](0=>1),space(eu,1));
# @assert abs(convert(Array,P*eu*P')[1])==maximum(abs.(diag(convert(Array,eu))))
# # U=unitary(fuse(space(AA,1)⊗space(AA,2)),space(AA,1)⊗space(AA,2));

# # @tensor AA[:]:=AA[1,2,3,4]*U[-1,1,2]*U'[3,4,-2];

# VL=permute(ev*P',(2,1,),(3,));#R,R'
# VR=P*inv(ev);#L',L

# @tensor H[:]:=VL[-1,1,2]*VR[2,1,-2];
# H=convert(Array,H);
# eu,ev=eigen(H);
# println(eu)



