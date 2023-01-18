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




function hopping_2sites(U_phy1,U_phy2)

    Vdummy=ℂ[U1Irrep](-1=>1);
    V=ℂ[U1Irrep](0=>1,1=>1);

    Id=[1 0;0 1];
    sm=[0 1;0 0]; sp=[0 0;1 0]; sz=[1 0; 0 -1]; occu=[0 0; 0 1];
    

    Cdag_set=Vector(undef,4);

    @tensor cdag[:]:=sp[-1,-5]*Id[-2,-6]*Id[-3,-7]*Id[-4,-8];
    Cdag=zeros(1,2,2,2,2,2,2,2,2);
    Cdag[1,:,:,:,:,:,:,:,:]=cdag;
    Cdag=TensorMap(Cdag, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V);
    Cdag_set[1]=Cdag;

    @tensor cdag[:]:=sz[-1,-5]*sp[-2,-6]*Id[-3,-7]*Id[-4,-8];
    Cdag=zeros(1,2,2,2,2,2,2,2,2);
    Cdag[1,:,:,:,:,:,:,:,:]=cdag;
    Cdag=TensorMap(Cdag, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V);
    Cdag_set[2]=Cdag;

    @tensor cdag[:]:=sz[-1,-5]*sz[-2,-6]*sp[-3,-7]*Id[-4,-8];
    Cdag=zeros(1,2,2,2,2,2,2,2,2);
    Cdag[1,:,:,:,:,:,:,:,:]=cdag;
    Cdag=TensorMap(Cdag, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V);
    Cdag_set[3]=Cdag;

    @tensor cdag[:]:=sz[-1,-5]*sz[-2,-6]*sz[-3,-7]*sp[-4,-8];
    Cdag=zeros(1,2,2,2,2,2,2,2,2);
    Cdag[1,:,:,:,:,:,:,:,:]=cdag;
    Cdag=TensorMap(Cdag, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V);
    Cdag_set[4]=Cdag;

    hop_set=Matrix(undef,length(Cdag_set),length(Cdag_set));

    for cc=1:length(Cdag_set)
        for dd=1:length(Cdag_set)
            Cdag=deepcopy(Cdag_set[cc]);
            C=deepcopy(Cdag_set[dd]);
            C=permute(C,(1,2,3,4,5,6,7,8,9,));
            C=C';
            @tensor hop[:]:=Cdag[1,-1,-2,-3,-4,2,3,4,5]*C[1,-5,-6,-7,-8,2,3,4,5];

            @tensor hop[:]:=hop[1,2,6,7,4,5,9,10]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-3]*U_phy1[-2,8,6,7]*U_phy1'[8,9,10,-4];
            @tensor hop[:]:=hop[3,7,4,8]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-3]*U_phy2[-2,7,5,6]*U_phy2'[8,5,6,-4];

            hop_set[cc,dd]=hop;

        end
    end






    return hop_set
end


hop_set=hopping_2sites(U_phy1,U_phy2);

#method="y_then_x";
method="x_then_y";
y_anti_pbc=true;
boundary_phase_y=0;
if method=="x_then_y"


    A=permute_neighbour_ind(A_origin,3,4,5);#L,U,R,P,D
    A=permute_neighbour_ind(A,2,3,5);#L,R,U,P,D
    ###########
    A=permute_neighbour_ind(A,1,2,5);#R,L,U,P,D
    ###########
    @tensor A[:]:=A[1,1,-1,-2,-3];#U,P,D

    A=permute_neighbour_ind(A,1,2,3);#P,U,D
    A1=deepcopy(A);#P1,U1,D1
    A2=deepcopy(A);#P2,U2,D2

    if y_anti_pbc
        ##################
        gauge_gate1=gauge_gate(A1,2,pi*boundary_phase_y);
        @tensor A1[:]:=A1[-1,1,-3]*gauge_gate1[-2,1];

        gauge_gate2=gauge_gate(A2,2,pi*boundary_phase_y);
        @tensor A2[:]:=A2[-1,1,-3]*gauge_gate2[-2,1];
        ##################

    end
    

    A2=permute_neighbour_ind(A2,1,2,3);#U2,P2,D2
    @tensor A1A2[:]:=A1[-1,-2,1]*A2[1,-3,-4];#P1,U1,P2,D2

    A1A2=permute_neighbour_ind(A1A2,2,3,4);#P1,P2,U1,D2
    A1A2=permute_neighbour_ind(A1A2,3,4,4);#P1,P2,D2,U1
    @tensor A1A2[:]:=A1A2[-1,-2,1,1];#P1,P2

    A1A2_1=deepcopy(A1A2);




    Ident, NA, NB, NANB, CAdag, CA, CBdag, CB=Hamiltonians(U_phy1,U_phy2);



    @tensor normA[:]:=A1A2'[1,2]*A1A2[1,2];
    Norm=blocks(normA)[U1Irrep(0)][1];

    @tensor nA[:]:=A1A2'[1,3]*NA[1,2]*A1A2[2,3];
    nA=blocks(nA)[U1Irrep(0)][1];

    @tensor nB[:]:=A1A2'[1,3]*NB[1,2]*A1A2[2,3];
    nB=blocks(nB)[U1Irrep(0)][1];

    @tensor CAdagCB[:]:=A1A2'[1,5]*CAdag[4,1,2]*CB[4,2,3]*A1A2[3,5];
    CAdagCB=blocks(CAdagCB)[U1Irrep(0)][1];

    println(nA/Norm)
    println(nB/Norm)
    println(CAdagCB/Norm)

    println("all hopping terms:")
    hop_matrix=zeros(size(hop_set,1),size(hop_set,2))*im
    for cc=1:size(hop_set,1)
        for dd=1:size(hop_set,2)

            @tensor hop[:]:=A1A2'[1,2]*hop_set[cc,dd][1,2,3,4]*A1A2[3,4];
            hop=blocks(hop)[U1Irrep(0)][1];
            hop_matrix[cc,dd]=hop/Norm;

        end
    end
    println(hop_matrix)

elseif method=="y_then_x"
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
    @tensor A1A2[:]:=A1A2[-1,-2,-3,-4,-5,-6,1,1];

    A1A2=permute_neighbour_ind(A1A2,1,2,6);#P1,L1,R1,L2,P2,R2
    A1A2=permute_neighbour_ind(A1A2,2,3,6);#P1,R1,L1,L2,P2,R2
    @tensor A1A2[:]:=A1A2[-1,1,1,-2,-3,-4];#P1,L2,P2,R2

    A1A2=permute_neighbour_ind(A1A2,2,3,4);#P1,P2,L2,R2
    A1A2=permute_neighbour_ind(A1A2,3,4,4);#P1,P2,R2,L2
    @tensor A1A2[:]:=A1A2[-1,-2,1,1];#P1,P2

    A1A2_2=deepcopy(A1A2);


end




