using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)


include("swap_funs.jl")
include("fermi_permute.jl")
include("gauge_flux.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\GfPEPS_ctmrg\\swap_gate_ctmrg\\M1\\GfPEPS_model.jl")








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




function hopping_3sites(U_phy1,U_phy2)

    Vdummy=ℂ[U1Irrep](-1=>1);
    V=ℂ[U1Irrep](0=>1,1=>1);

    Id=[1 0;0 1];
    sm=[0 1;0 0]; sp=[0 0;1 0]; sz=[1 0; 0 -1]; occu=[0 0; 0 1];
    

    Cdag_set=Vector(undef,6);

    @tensor cdag[:]:=sp[-1,-7]*Id[-2,-8]*Id[-3,-9]*Id[-4,-10]*Id[-5,-11]*Id[-6,-12];
    Cdag=zeros(1,2,2,2,2,2,2,2,2,2,2,2,2);
    Cdag[1,:,:,:,:,:,:,:,:,:,:,:,:]=cdag;
    Cdag=TensorMap(Cdag, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V);
    Cdag_set[1]=Cdag;

    @tensor cdag[:]:=sz[-1,-7]*sp[-2,-8]*Id[-3,-9]*Id[-4,-10]*Id[-5,-11]*Id[-6,-12];
    Cdag=zeros(1,2,2,2,2,2,2,2,2,2,2,2,2);
    Cdag[1,:,:,:,:,:,:,:,:,:,:,:,:]=cdag;
    Cdag=TensorMap(Cdag, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V);
    Cdag_set[2]=Cdag;

    @tensor cdag[:]:=sz[-1,-7]*sz[-2,-8]*sp[-3,-9]*Id[-4,-10]*Id[-5,-11]*Id[-6,-12];
    Cdag=zeros(1,2,2,2,2,2,2,2,2,2,2,2,2);
    Cdag[1,:,:,:,:,:,:,:,:,:,:,:,:]=cdag;
    Cdag=TensorMap(Cdag, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V);
    Cdag_set[3]=Cdag;

    @tensor cdag[:]:=sz[-1,-7]*sz[-2,-8]*sz[-3,-9]*sp[-4,-10]*Id[-5,-11]*Id[-6,-12];
    Cdag=zeros(1,2,2,2,2,2,2,2,2,2,2,2,2);
    Cdag[1,:,:,:,:,:,:,:,:,:,:,:,:]=cdag;
    Cdag=TensorMap(Cdag, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V);
    Cdag_set[4]=Cdag;

    @tensor cdag[:]:=sz[-1,-7]*sz[-2,-8]*sz[-3,-9]*sz[-4,-10]*sp[-5,-11]*Id[-6,-12];
    Cdag=zeros(1,2,2,2,2,2,2,2,2,2,2,2,2);
    Cdag[1,:,:,:,:,:,:,:,:,:,:,:,:]=cdag;
    Cdag=TensorMap(Cdag, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V);
    Cdag_set[5]=Cdag;

    @tensor cdag[:]:=sz[-1,-7]*sz[-2,-8]*sz[-3,-9]*sz[-4,-10]*sz[-5,-11]*sp[-6,-12];
    Cdag=zeros(1,2,2,2,2,2,2,2,2,2,2,2,2);
    Cdag[1,:,:,:,:,:,:,:,:,:,:,:,:]=cdag;
    Cdag=TensorMap(Cdag, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V);
    Cdag_set[6]=Cdag;

    hop_set=Matrix(undef,length(Cdag_set),length(Cdag_set));

    for cc=1:length(Cdag_set)
        for dd=1:length(Cdag_set)
            Cdag=deepcopy(Cdag_set[cc]);
            C=deepcopy(Cdag_set[dd]);
            C=permute(C,(1,2,3,4,5,6,7,8,9,10,11,12,13,));
            C=C';
            @tensor hop[:]:=Cdag[1,-1,-2,-3,-4,-5,-6,2,3,4,5,6,7]*C[1,-7,-8,-9,-10,-11,-12,2,3,4,5,6,7];

            @tensor hop[:]:=hop[1,2,6,7,11,12,4,5,9,10,14,15]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-4]*U_phy1[-2,8,6,7]*U_phy1'[8,9,10,-5]*U_phy1[-3,13,11,12]*U_phy1'[13,14,15,-6];
            @tensor hop[:]:=hop[3,7,11,4,8,12]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-4]*U_phy2[-2,7,5,6]*U_phy2'[8,5,6,-5]*U_phy2[-3,11,9,10]*U_phy2'[12,9,10,-6];

            hop_set[cc,dd]=hop;

        end
    end


    return hop_set
end


hop_set=hopping_3sites(U_phy1,U_phy2);


y_anti_pbc=true;
boundary_phase_y=0;



A=permute_neighbour_ind(A_origin,3,4,5);#L,U,R,P,D
A=permute_neighbour_ind(A,2,3,5);#L,R,U,P,D
###########
A=permute_neighbour_ind(A,1,2,5);#R,L,U,P,D
###########
@tensor A[:]:=A[1,1,-1,-2,-3];#U,P,D

A=permute_neighbour_ind(A,1,2,3);#P,U,D
A1=deepcopy(A);#P1,U1,D1
A2=deepcopy(A);#P2,U2,D2
A3=deepcopy(A);#P3,U3,D3

if y_anti_pbc
    ##################
    gauge_gate1=gauge_gate(A1,2,2*pi/3*boundary_phase_y);
    @tensor A1[:]:=A1[-1,1,-3]*gauge_gate1[-2,1];

    gauge_gate2=gauge_gate(A2,2,2*pi/3*boundary_phase_y);
    @tensor A2[:]:=A2[-1,1,-3]*gauge_gate2[-2,1];

    gauge_gate3=gauge_gate(A3,2,2*pi/3*boundary_phase_y);
    @tensor A3[:]:=A3[-1,1,-3]*gauge_gate3[-2,1];
    ##################

end


A2=permute_neighbour_ind(A2,1,2,3);#U2,P2,D2
@tensor A1A2[:]:=A1[-1,-2,1]*A2[1,-3,-4];#P1,U1,P2,D2

A3=permute_neighbour_ind(A3,1,2,3);#U3,P3,D3
@tensor A1A2A3[:]:=A1A2[-1,-2,-3,1]*A3[1,-4,-5];#P1,U1,P2,P3,D3


A1A2A3=permute_neighbour_ind(A1A2A3,2,3,5);#P1,P2,U1,P3,D3
A1A2A3=permute_neighbour_ind(A1A2A3,3,4,5);#P1,P2,P3,U1,D3
A1A2A3=permute_neighbour_ind(A1A2A3,4,5,5);#P1,P2,P3,D3,U1

@tensor A1A2A3[:]:=A1A2A3[-1,-2,-3,1,1];#P1,P2,P3

@tensor normA[:]:=A1A2A3'[1,2,3]*A1A2A3[1,2,3];
Norm=blocks(normA)[U1Irrep(0)][1];


println("all hopping terms:")
hop_matrix=zeros(size(hop_set,1),size(hop_set,2))*im
for cc=1:size(hop_set,1)
    for dd=1:size(hop_set,2)

        @tensor hop[:]:=A1A2A3'[1,2,3]*hop_set[cc,dd][1,2,3,4,5,6]*A1A2A3[4,5,6];
        hop=blocks(hop)[U1Irrep(0)][1];
        hop_matrix[cc,dd]=hop/Norm;

    end
end
println(hop_matrix)




