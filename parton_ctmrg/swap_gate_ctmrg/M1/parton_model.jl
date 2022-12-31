using LinearAlgebra
using TensorKit

function create_H_term(O1,O2,direction,is_odd)
    if direction=="x"
        if is_odd
            #c1dag c2
            sign1=[1,1,1,1,0];
            sign2=[0,0,0,1,0];
            ind1=3;#index p
            ind2=1;#index p
            p1=1;
            p2=1;
        else
            # n1 n2 
            sign1=[0,0,0,0,0];
            sign2=[0,0,0,0,0];
            ind1=3;#index p
            ind2=1;#index p
            p1=0;
            p2=0;
        end

        H_term=Dict([("direction","x"),("O1", O1), ("O2", O2), ("sign1",sign1), ("sign2",sign2), ("ind1",ind1), ("ind2",ind2), ("p1",p1), ("p2",p2)]);
    end
    return H_term

end



function evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
    
    H_term=create_H_term(O1,O2,direction,is_odd);
    AA1p,AA2p=build_double_layer_swap_op(A_fused,A_fused,H_term);

    if direction=="x"
        norm=ob_2sites_x(CTM,AA_fused,AA_fused);
        ob=ob_2sites_x(CTM,AA1p,AA2p);
        norm=blocks(norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
        ob=blocks(ob)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];


    end
    
    return ob/norm
    
end






function Hamiltonians(U_phy1,U_phy2)

    Vdummy=ℂ[U1Irrep](-1=>1);
    V=ℂ[U1Irrep](0=>1,1=>1);

    Id=[1 0;0 1];
    sm=[0 1;0 0]; sp=[0 0;1 0]; sz=[1 0; 0 -1]; occu=[0 0; 0 1];
    
    @tensor Ident[:]:=Id[-1,-3]*Id[-2,-4];
    Ident=TensorMap(Ident,  V ⊗ V ← V ⊗ V);

    @tensor NA[:]:=occu[-1,-3]*Id[-2,-4];
    NA=TensorMap(NA,  V ⊗ V ← V ⊗ V);
    
    @tensor NB[:]:=Id[-1,-3]*occu[-2,-4];
    NB=TensorMap(NB,  V ⊗ V ← V ⊗ V);

    @tensor NANB[:]:=occu[-1,-3]*occu[-2,-4];
    NANB=TensorMap(NANB,  V ⊗ V ← V ⊗ V);

    @tensor cAdag[:]:=sp[-1,-3]*Id[-2,-4];
    CAdag=zeros(1,2,2,2,2);
    CAdag[1,:,:,:,:]=cAdag;
    CAdag=TensorMap(CAdag, Vdummy ⊗ V ⊗ V ← V ⊗ V);

    @tensor cBdag[:]:=sz[-1,-3]*sp[-2,-4];
    CBdag=zeros(1,2,2,2,2);
    CBdag[1,:,:,:,:]=cBdag;
    CBdag=TensorMap(CBdag, Vdummy ⊗ V ⊗ V ← V ⊗ V);

    @tensor cA[:]:=sm[-1,-3]*Id[-2,-4];
    CA=zeros(1,2,2,2,2);
    CA[1,:,:,:,:]=cA;
    CA=TensorMap(CA, Vdummy' ⊗ V ⊗ V ← V ⊗ V);

    @tensor cB[:]:=sz[-1,-3]*sm[-2,-4];
    CB=zeros(1,2,2,2,2);
    CB[1,:,:,:,:]=cB;
    CB=TensorMap(CB, Vdummy' ⊗ V ⊗ V ← V ⊗ V);



    @tensor Ident[:]:=Ident[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
    @tensor Ident[:]:=Ident[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

    @tensor NA[:]:=NA[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
    @tensor NA[:]:=NA[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

    @tensor NB[:]:=NB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
    @tensor NB[:]:=NB[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

    @tensor NANB[:]:=NANB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
    @tensor NANB[:]:=NANB[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

    @tensor CAdag[:]:=CAdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
    @tensor CAdag[:]:=CAdag[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

    @tensor CBdag[:]:=CBdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
    @tensor CBdag[:]:=CBdag[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

    @tensor CA[:]:=CA[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
    @tensor CA[:]:=CA[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

    @tensor CB[:]:=CB[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
    @tensor CB[:]:=CB[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];


    return Ident, NA, NB, NANB, CAdag, CA, CBdag, CB 
end


function build_double_layer_swap_op(A1,A2,H_term)
    A1=deepcopy(A1)
    A2=deepcopy(A2)
    A1_origin=deepcopy(A1)
    A2_origin=deepcopy(A2)



    if H_term["p1"]%2==1
        #the first index of O is dummy
        @tensor A1[:]:= A1[-1,-2,-3,-4,1]*H_term["O1"][-6,-5,1]
        @tensor A2[:]:= A2[-1,-2,-3,-4,1]*H_term["O2"][-6,-5,1]

        if H_term["direction"]=="x"
            @assert H_term["sign1"]==[1,1,1,1,0];
            @assert H_term["sign2"]==[0,0,0,1,0];
            
            gate=parity_gate(A1,1); @tensor A1[:]:=A1[1,-2,-3,-4,-5,-6]*gate[-1,1];
            gate=parity_gate(A1,2); @tensor A1[:]:=A1[-1,1,-3,-4,-5,-6]*gate[-2,1];
            gate=parity_gate(A1,3); @tensor A1[:]:=A1[-1,-2,1,-4,-5,-6]*gate[-3,1];
            gate=parity_gate(A1,4); @tensor A1[:]:=A1[-1,-2,-3,1,-5,-6]*gate[-4,1];

            gate=parity_gate(A2,4); @tensor A2[:]:=A2[-1,-2,-3,1,-5,-6]*gate[-4,1];
        end

        @assert H_term["ind1"]==3
        @assert H_term["ind2"]==1

        U=unitary(fuse(space(A1,3)⊗space(A1,6)), space(A1,3)⊗space(A1,6)); 
        @tensor A1_new[:]:=A1[-1,-2,1,-4,-5,2]*U[-3,1,2];
        @tensor A2_new[:]:=A2[1,-2,-3,-4,-5,2]*U'[1,2,-1];
    else
        @tensor A1[:]:= A1[-1,-2,-3,-4,1]*H_term["O1"][-5,1]
        @tensor A2[:]:= A2[-1,-2,-3,-4,1]*H_term["O2"][-5,1]
        A1_new=A1
        A2_new=A2
    end

    A1_double,_,_,_,_=build_double_layer_swap(A1_origin',A1_new)
    A2_double,_,_,_,_=build_double_layer_swap(A2_origin',A2_new)

    return A1_double,A2_double
end


function ob_1site_closed(CTM,AA_fused)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    @tensor envL[:]:=Cset[1][1,-1]*Tset[4][2,-2,1]*Cset[4][-3,2];
    @tensor envR[:]:=Cset[2][-1,1]*Tset[2][1,-2,2]*Cset[3][2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset[1][1,3,-1]*AA_fused[2,5,-2,3]*Tset[3][-3,5,4];
    @tensor Norm[:]:=envL[1,2,3]*envR[1,2,3];
    Norm=blocks(Norm)[Irrep[SU₂](0)];
    return Norm;
end


function ob_2sites_x(CTM,AA1,AA2)

    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    @tensor envL[:]:=Cset[1][1,-1]*Tset[4][2,-2,1]*Cset[4][-3,2];
    @tensor envR[:]:=Cset[2][-1,1]*Tset[2][1,-2,2]*Cset[3][2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset[1][1,3,-1]*AA1[2,5,-2,3]*Tset[3][-3,5,4];
    @tensor envR[:]:=Tset[1][-1,3,1]*AA2[-2,5,2,3]*Tset[3][4,5,-3]*envR[1,2,4];
    @tensor rho[:]:=envL[1,2,3]*envR[1,2,3];
    return rho;
end


function ob_2sites_y(CTM,AA1,AA2)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    @tensor envU[:]:=Cset[2][1,-1]*Tset[1][2,-2,1]*Cset[1][-3,2];
    @tensor envD[:]:=Cset[3][-1,1]*Tset[3][1,-2,2]*Cset[4][2,-3];
    @tensor envU[:]:=envU[1,2,4]*Tset[2][1,3,-1]*AA1[5,-2,3,2,-4]*Tset[4][-3,5,4];
    @tensor envD[:]:=Tset[2][-1,3,1]*AA2[5,2,3,-2,-4]*Tset[4][4,5,-3]*envD[1,2,4];
    @tensor rho[:]:=envU[1,2,3,-1]*envD[1,2,3,-2];
    return rho;
end

function ob_LU(CTM,AA_LU,AA_fused)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-3]*Tset[4][-1,4,1]*AA_LU[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_fused[-2,-4,4,3]* Tset[2][2,4,-3];

    @tensor MM_LD[:]:=Tset[4][1,3,-2]*AA_fused[3,4,-5,-3]*Cset[4][2,1]*Tset[3][-4,4,2]; 
    @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:=up[1,2,3,4,]*down[1,2,3,4];
end

function ob_RD(CTM,AA_RD,AA_fused)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-3]*Tset[4][-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_fused[-2,-4,4,3]* Tset[2][2,4,-3];

    @tensor MM_LD[:]:=Tset[4][1,3,-2]*AA_fused[3,4,-5,-3]*Cset[4][2,1]*Tset[3][-4,4,2]; 
    @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:=up[1,2,3,4,]*down[1,2,3,4];
end


function ob_LD_RU(CTM,AA_fused,AA_LD,AA_RU)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-3]*Tset[4][-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_RU[-2,-4,4,3,-5]* Tset[2][2,4,-3];

    @tensor MM_LD[:]:=Tset[4][1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset[4][2,1]*Tset[3][-4,4,2]; 
    @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:=up[1,2,3,4,-1]*down[-2,1,2,3,4];
end

function ob_LU_RD(CTM,AA_fused,AA_LU,AA_RD)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-4]*Tset[4][-2,4,1]*AA_LU[4,-3,-5,3,-1]; 
    @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_fused[-2,-4,4,3]* Tset[2][2,4,-3];

    @tensor MM_LD[:]:=Tset[4][1,3,-1]*AA_fused[3,4,-4,-2]*Cset[4][2,1]*Tset[3][-3,4,2]; 
    @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 

    MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:=up[-1,1,2,3,4]*down[1,2,3,4,-2];
end


function ob_LU_RU_LD(CTM,AA_fused,AA_LU,AA_RU,AA_LD)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-4]*Tset[4][-2,4,1]*AA_LU[4,-3,-5,3,-1]; 
    @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_RU[-2,-4,4,3,-5]* Tset[2][2,4,-3];

    @tensor MM_LD[:]:=Tset[4][1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset[4][2,1]*Tset[3][-4,4,2]; 
    @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:=up[-1,1,2,3,4,-2]*down[-3,1,2,3,4];
end

function ob_LD_RU_RD(CTM,AA_fused,AA_LD,AA_RU,AA_RD)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-3]*Tset[4][-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_RU[-2,-4,4,3,-5]* Tset[2][2,4,-3];

    @tensor MM_LD[:]:=Tset[4][1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset[4][2,1]*Tset[3][-4,4,2]; 
    @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:=up[1,2,3,4,-2]*down[-1,1,2,3,4,-3];
end




function ob_LD_RD(CTM,AA_fused,AA_LD,AA_RD)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-3]*Tset[4][-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_fused[-2,-4,4,3]* Tset[2][2,4,-3];

    @tensor MM_LD[:]:=Tset[4][1,3,-1]*AA_LD[3,4,-4,-2,-5]*Cset[4][2,1]*Tset[3][-3,4,2]; 
    @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 

    MM_LU=permute(MM_LU,(3,4,),(1,2,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,5,));
    MM_RD=permute(MM_RD,(3,4,),(1,2,5,));

    left=MM_LU*MM_LD;
    right=MM_RU*MM_RD;
    @tensor rho[:]:=left[1,2,3,4,-1]*right[1,2,3,4,-2];
end

function ob_RU_RD(CTM,AA_fused,AA_RU,AA_RD)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-3]*Tset[4][-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_RU[-2,-4,4,3,-5]* Tset[2][2,4,-3];

    @tensor MM_LD[:]:=Tset[4][1,3,-1]*AA_fused[3,4,-4,-2]*Cset[4][2,1]*Tset[3][-3,4,2]; 
    @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:=up[1,2,3,4,-1]*down[1,2,3,4,-2];
end