using LinearAlgebra
using TensorKit


function cal_fidelity(theta1,theta2,Gutzwiller,M,chi,tol,CTM_ite_nums,CTM_trun_tol,forced_steps)
    println("chi= "*string(chi));

    filenm1="/users/p1231/niu/Code/Julia_codes/Tensor_network/GfPEPS_parton/test_M2_projected_decoupled_rotated/theta_"*string(theta1)*"/swap_gate_Tensor_M2.jld2";
    
    
    filenm2="/users/p1231/niu/Code/Julia_codes/Tensor_network/GfPEPS_parton/test_M2_projected_decoupled_rotated/theta_"*string(theta2)*"/swap_gate_Tensor_M2.jld2";

    A1=load_state(filenm1,M,Gutzwiller);
    A2=load_state(filenm2,M,Gutzwiller);

    #overlap between A1 and A2
    conv_check="singular_value";
    CTM, A1A2, U_L,U_D,U_R,U_U=init_CTM(chi,A1,A2,"PBC",true);
    @time CTM12=CTMRG(A1A2,chi,conv_check,tol,CTM,CTM_ite_nums,CTM_trun_tol,forced_steps);

    #overlap between A1 and A1
    conv_check="singular_value";
    CTM, A1A1, U_L,U_D,U_R,U_U=init_CTM(chi,A1,A1,"PBC",true);
    @time CTM11=CTMRG(A1A1,chi,conv_check,tol,CTM,CTM_ite_nums,CTM_trun_tol,forced_steps);

    #overlap between A2 and A2
    conv_check="singular_value";
    CTM, A2A2, U_L,U_D,U_R,U_U=init_CTM(chi,A2,A2,"PBC",true);
    @time CTM22=CTMRG(A2A2,chi,conv_check,tol,CTM,CTM_ite_nums,CTM_trun_tol,forced_steps);


    ov_12=overlap_CTM(CTM12,A1A2);
    ov_11=overlap_CTM(CTM11,A1A1);
    ov_22=overlap_CTM(CTM22,A2A2);

    ov_total=ov_12/sqrt(ov_11*ov_22);


    println("Normalized overlap: "*string(ov_total));flush(stdout);

    mat_filenm="fidelity_M"*string(M)*"_chi"*string(chi)*"_theta_"*string(theta1)*"_"*string(theta2)*".mat";
    matwrite(mat_filenm, Dict(
        "ov_11" => ov_11,
        "ov_22" => ov_22,
        "ov_12" => ov_12,
        "ov_total" => ov_total
    ); compress = false)
end



function overlap_CTM(CTM,AA)
    #arXiv:1711.04798    figure7
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    #3x3 term
    @tensor envL[:]:=Cset[1][1,-1]*Tset[4][2,-2,1]*Cset[4][-3,2];
    @tensor envR[:]:=Cset[2][-1,1]*Tset[2][1,-2,2]*Cset[3][2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset[1][1,3,-1]*AA[2,5,-2,3]*Tset[3][-3,5,4];
    @tensor Norm[:]:=envL[1,2,3]*envR[1,2,3];
    ov_3x3=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];

    #2x3 term
    @tensor Norm[:]:=Cset[1][2,1]*Cset[2][5,7]*Cset[3][7,6]*Cset[4][3,2]*Tset[1][1,4,5]*Tset[3][6,4,3];
    ov_2x3=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];

    #3x2 term
    @tensor Norm[:]:=Cset[1][1,2]*Cset[2][2,3]*Cset[3][6,7]*Cset[4][7,5]*Tset[2][3,4,6]*Tset[4][5,4,1];
    ov_3x2=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];

    #2x2 term
    @tensor Norm[:]:=Cset[1][1,2]*Cset[2][2,3]*Cset[3][3,4]*Cset[4][4,1];
    ov_2x2=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];

    ov_total=abs(ov_3x3*ov_2x2)/abs(ov_2x3*ov_3x2)
    #println("overlap is:"*string(ov_total))
    return ov_total
end



function load_state(filenm,M,Gutzwiller)
    data=load(filenm)
    if M==1
        P_G=data["P_G"];

        psi_G=data["psi_G"];   #P1,P2,L,R,D,U
        M1=psi_G[1];
        M2=psi_G[2];
        M3=psi_G[3];
        M4=psi_G[4];
        M5=psi_G[5];
        M6=psi_G[6];

        if Gutzwiller
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


        #convert to the order of PEPS code
        A=permute(A,(1,5,4,2,3,));

    elseif M==2
        P_G=data["P_G"];
        
        psi_G=data["psi_G"];   #P1,P2,L,R,D,U
        M1=psi_G[1];
        M2=psi_G[2];
        M3=psi_G[3];
        M4=psi_G[4];
        M5=psi_G[5];
        M6=psi_G[6];
        M7=psi_G[7];
        M8=psi_G[8];
        M9=psi_G[9];
        M10=psi_G[10];
        
        if Gutzwiller
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
        @tensor A[:]:=A[-1,-2,-3,-4,-5,1]*M7[1,-7,-6];
        @tensor A[:]:=A[-1,-2,-3,-4,-5,-6,1]*M8[1,-8,-7];
        @tensor A[:]:=A[-1,-2,-3,-4,-5,-6,-7,1]*M9[1,-9,-8];
        @tensor A[:]:=A[-1,-2,-3,-4,-5,-6,-7,-8,1]*M10[1,-10,-9];
        
        U_phy_dummy=unitary(fuse(space(A,1)⊗space(A,10)), space(A,1)⊗space(A,10));#this doesn't do anything
        @tensor A[:]:=A[1,-2,-3,-4,-5,-6,-7,-8,-9,2]*U_phy_dummy[-1,1,2];
        # P,L,R,D,U
        
        
        bond=data["bond_gate"];#dummy, D1, D2 
        
        #Add bond:both parity gate and bond operator
        @tensor A[:]:=A[-1,-6,-7,1,2,3,4,-12,-13]*bond[-2,-8,1]*bond[-3,-9,2]*bond[-4,-10,3]*bond[-5,-11,4];
        
        U_phy2=unitary(fuse(space(A,1)⊗space(A,2)⊗space(A,3)⊗space(A,4)⊗space(A,5)), space(A,1)⊗space(A,2)⊗space(A,3)⊗space(A,4)⊗space(A,5));
        @tensor A[:]:=A[1,2,3,4,5,-2,-3,-4,-5,-6,-7,-8,-9]*U_phy2[-1,1,2,3,4,5];
        #P,L,R,D,U
        
        
        
        #swap between spin up and spin down modes, since |L,U,P><D,R|====L,U,P|><|R,D
        special_gate=special_parity_gate(A,4);
        @tensor A[:]:=A[-1,-2,-3,1,-5,-6,-7,-8,-9]*special_gate[-4,1];
        @tensor A[:]:=A[-1,-2,-3,-4,1,-6,-7,-8,-9]*special_gate[-5,1];
        @tensor A[:]:=A[-1,-2,-3,-4,-5,1,-7,-8,-9]*special_gate[-6,1];
        @tensor A[:]:=A[-1,-2,-3,-4,-5,-6,1,-8,-9]*special_gate[-7,1];
        
        gate=swap_gate(A,4,5);@tensor A[:]:=A[-1,-2,-3,1,2,-6,-7,-8,-9]*gate[-4,-5,1,2];  
        gate=swap_gate(A,6,7);@tensor A[:]:=A[-1,-2,-3,-4,-5,1,2,-8,-9]*gate[-6,-7,1,2];  
        
        
        
        #group virtual legs on the same legs
        U1=unitary(fuse(space(A,2)⊗space(A,3)),space(A,2)⊗space(A,3)); 
        U2=unitary(fuse(space(A,8)⊗space(A,9)),space(A,8)⊗space(A,9));
        @tensor A[:]:=A[-1,1,2,-3,-4,-5,-6,-7,-8]*U1[-2,1,2];
        @tensor A[:]:=A[-1,-2,1,2,-4,-5,-6,-7]*U2'[1,2,-3];
        @tensor A[:]:=A[-1,-2,-3,1,2,-5,-6]*U1'[1,2,-4];
        @tensor A[:]:=A[-1,-2,-3,-4,1,2]*U2[-5,1,2];
        
        
        
        A1=deepcopy(A);
        
        
        gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2]*gate[-4,-5,1,2];           
        A=permute(A,(1,2,3,5,4,));#P,L,R,U,D
        
        gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5]*gate[-3,-4,1,2]; 
        A=permute(A,(1,2,4,3,5,));#P,L,U,R,D
        
        gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5]*gate[-1,-2,1,2]; 
        A=permute(A,(2,1,3,4,5,));#L,P,U,R,D
        
        gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5]*gate[-2,-3,1,2]; 
        A=permute(A,(1,3,2,4,5,));#L,U,P,R,D
        
        #convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D
        
        
        #convert to the order of PEPS code
        A=permute(A,(1,5,4,2,3,));
        
    end

    return A
end

function build_double_layer_swap(Ap,A)
    #display(space(A))


    gate=swap_gate(Ap,1,4); @tensor Ap[:]:=Ap[1,-2,-3,2,-5]*gate[-1,-4,1,2];  
    gate=swap_gate(Ap,2,3); @tensor Ap[:]:=Ap[-1,1,2,-4,-5]*gate[-2,-3,1,2];  
    gate=parity_gate(Ap,4); @tensor Ap[:]:=Ap[-1,-2,-3,1,-5]*gate[-4,1];
    gate=parity_gate(Ap,2); @tensor Ap[:]:=Ap[-1,1,-3,-4,-5]*gate[-2,1];
    
    




    Ap=permute(Ap,(1,2,),(3,4,5))
    A=permute(A,(1,2,),(3,4,5));
    
    # U_L=unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
    # U_D=unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));
    # U_R=inv(U_L);
    # U_U=inv(U_D);

    # U_Lp=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    # U_Dp=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    # U_Rp=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    # U_Up=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # println(norm(U_R-U_Rp)/norm(U_R))
    # println(norm(U_L-U_Lp)/norm(U_L))
    # println(norm(U_D-U_Dp)/norm(U_D))
    # println(norm(U_U-U_Up)/norm(U_U))

    U_L=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    U_D=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    U_R=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    U_U=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # display(space(U_L))
    # display(space(U_D))
    # display(space(U_R))
    # display(space(U_D))

    uMp,sMp,vMp=tsvd(Ap);
    uMp=uMp*sMp;
    uM,sM,vM=tsvd(A);
    uM=uM*sM;

    uMp=permute(uMp,(1,2,3,),())
    uM=permute(uM,(1,2,3,),())
    Vp=space(uMp,3);
    V=space(vM,1);
    U=unitary(fuse(Vp' ⊗ V), Vp' ⊗ V);

    @tensor double_LD[:]:=uMp[-1,-2,1]*U'[1,-3,-4];
    @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

    vMp=permute(vMp,(1,2,3,4,),());
    vM=permute(vM,(1,2,3,4,),());

    @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
    @tensor double_RU[:]:=vMp[1,-2,-4,2]*double_RU[-1,1,-3,-5,2];

    #display(space(double_RU))

    double_LD=permute(double_LD,(1,2,),(3,4,5,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,),(3,));
    #display(space(double_LD))
    double_RU=permute(double_RU,(1,4,5,),(2,3,));
    double_RU=double_RU*U_R;
    double_RU=permute(double_RU,(1,4,),(2,3,));
    double_RU=double_RU*U_U;
    double_LD=permute(double_LD,(1,2,),(3,));
    double_RU=permute(double_RU,(1,),(2,3,));
    AA_fused=double_LD*double_RU;


    ##########################
    @tensor U_LU[:]:=U_L'[-1,-2,-5]*U_U'[-6,-3,-4];
    gate1=swap_gate(U_LU,1,4);
    gate2=swap_gate(U_LU,3,4);
    @tensor U_LU[:]:=U_LU[1,-2,-3,2,-5,-6]*gate1[-1,-4,1,2];
    @tensor U_LU[:]:=U_LU[-1,-2,1,2,-5,-6]*gate2[-3,-4,1,2];
    @tensor U_LU[:]:=U_LU[1,2,3,4,-3,-4]*U_L[-1,1,2]*U_U[3,4,-2];
    @tensor AA_fused[:]:=AA_fused[1,-2,-3,2]*U_LU[-1,-4,1,2];


    @tensor U_DR[:]:=U_D'[-1,-2,-5]*U_R'[-6,-3,-4];
    gate1=swap_gate(U_DR,1,2);
    gate2=swap_gate(U_DR,1,4);
    @tensor U_DR[:]:=U_DR[1,2,-3,-4,-5,-6]*gate1[-1,-2,1,2];
    @tensor U_DR[:]:=U_DR[1,-2,-3,2,-5,-6]*gate2[-1,-4,1,2];

    @tensor U_DR[:]:=U_DR[1,2,3,4,-3,-4]*U_D[-1,1,2]*U_R[3,4,-2];
    @tensor AA_fused[:]:=AA_fused[-1,1,2,-4]*U_DR[-2,-3,1,2];

    return AA_fused, U_L,U_D,U_R,U_U
end

function build_double_layer_NoSwap(Ap,A)
    #display(space(A))

    Ap=permute(Ap,(1,2,),(3,4,5))
    A=permute(A,(1,2,),(3,4,5));
    
    # U_L=unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
    # U_D=unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));
    # U_R=inv(U_L);
    # U_U=inv(U_D);

    # U_Lp=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    # U_Dp=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    # U_Rp=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    # U_Up=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # println(norm(U_R-U_Rp)/norm(U_R))
    # println(norm(U_L-U_Lp)/norm(U_L))
    # println(norm(U_D-U_Dp)/norm(U_D))
    # println(norm(U_U-U_Up)/norm(U_U))

    U_L=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    U_D=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    U_R=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    U_U=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # display(space(U_L))
    # display(space(U_D))
    # display(space(U_R))
    # display(space(U_D))

    uMp,sMp,vMp=tsvd(Ap);
    uMp=uMp*sMp;
    uM,sM,vM=tsvd(A);
    uM=uM*sM;

    uMp=permute(uMp,(1,2,3,),())
    uM=permute(uM,(1,2,3,),())
    Vp=space(uMp,3);
    V=space(vM,1);
    U=unitary(fuse(Vp' ⊗ V), Vp' ⊗ V);

    @tensor double_LD[:]:=uMp[-1,-2,1]*U'[1,-3,-4];
    @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

    vMp=permute(vMp,(1,2,3,4,),());
    vM=permute(vM,(1,2,3,4,),());

    @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
    @tensor double_RU[:]:=vMp[1,-2,-4,2]*double_RU[-1,1,-3,-5,2];

    #display(space(double_RU))

    double_LD=permute(double_LD,(1,2,),(3,4,5,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,),(3,));
    #display(space(double_LD))
    double_RU=permute(double_RU,(1,4,5,),(2,3,));
    double_RU=double_RU*U_R;
    double_RU=permute(double_RU,(1,4,),(2,3,));
    double_RU=double_RU*U_U;
    double_LD=permute(double_LD,(1,2,),(3,));
    double_RU=permute(double_RU,(1,),(2,3,));
    AA_fused=double_LD*double_RU;


    ##########################

    AA_fused=permute(AA_fused,(1,2,3,4,));


    return AA_fused, U_L,U_D,U_R,U_U
end


function fuse_CTM_legs(CTM,U_L,U_D,U_R,U_U)
    #fuse CTM legs
    Tset=CTM["Tset"];

    #T4
    T4=permute(Tset[4],(1,4,),(2,3,));
    T4=T4*U_R;
    T4=permute(T4,(1,3,2,),());
    Tset[4]=T4
    #display(space(T4))

    #T3
    T3=permute(Tset[3],(1,4,),(2,3,));
    T3=T3*U_U;
    T3=permute(T3,(1,3,2,),());
    Tset[3]=T3
    #display(space(T3))

    #T2
    T2=permute(Tset[2],(2,3,),(1,4,));
    T2=U_L*T2;
    T2=permute(T2,(2,1,3,),());
    Tset[2]=T2
    #display(space(T2))

    #T1
    T1=permute(Tset[1],(2,3,),(1,4,));
    T1=U_D*T1;
    T1=permute(T1,(2,1,3,),());
    Tset[1]=T1
    #display(space(T1))

    CTM["Tset"]=Tset;
    return CTM
end

function spectrum_conv_check(ss_old,C_new)
    _,ss_new,_=svd(permute(C_new,(1,),(2,)));
    ss_new=to_array(ss_new);
    ss_new=sort(ss_new, rev=true);
    ss_old=ss_old/ss_old[1];
    ss_new=ss_new/ss_new[1];
    #display(ss_new)
    if length(ss_old)>length(ss_new)
        dss=copy(ss_old);
        siz=length(ss_new)
    elseif length(ss_old)<=length(ss_new)
        dss=copy(ss_new);
        siz=length(ss_old)
    end
    dss[1:siz]=ss_old[1:siz]-ss_new[1:siz]
    # println("spectra diff:")
    # println(ss_old);
    # println(ss_new)
    er=norm(dss);
    return er,ss_new
end

function CTMRG(AA_fused,chi,conv_check,tol,CTM,CTM_ite_nums, CTM_trun_tol,forced_steps,CTM_ite_info=true,CTM_conv_info=false)

    #Ref: PHYSICAL REVIEW B 98, 235148 (2018)
    #initial corner transfer matrix

    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    conv_check="singular_value"

    ss_old1=ones(chi)*2;
    ss_old2=ones(chi)*2;
    ss_old3=ones(chi)*2;
    ss_old4=ones(chi)*2;
    d=2;
    rho_old=Matrix(I,d^3,d^3);

    #Iteration

    print_corner=false;
    if print_corner
        println("corner 4:")
        C4_spec=to_array(tsvd(Cset[4],(1,),(2,))[2]);
        C4_spec=C4_spec/C4_spec[1];
        println(C4_spec);
        println("corner 1:")
        C1_spec=to_array(tsvd(Cset[1],(1,),(2,))[2]);
        C1_spec=C1_spec/C1_spec[1];
        println(C1_spec);
        println("corner 3:")
        C3_spec=to_array(tsvd(Cset[3],(1,),(2,))[2]);
        C3_spec=C3_spec/C3_spec[1];
        println(C3_spec);
        println("corner 2:")
        C2_spec=to_array(tsvd(Cset[2],(1,),(2,))[2]);
        C2_spec=C2_spec/C2_spec[1];
        println(C2_spec);
        println("CTM init finished")
    end
    


    if CTM_ite_info
        println("start CTM iterations:")
    end
    ite_num=0;
    ite_err=1;

    #define number of steps
    if forced_steps==nothing
        total_steps=CTM_ite_nums;
    else
        total_steps=forced_steps
    end

    for ci=1:total_steps
        ite_num=ci;
        #direction_order=[1,2,3,4];
        #direction_order=[4,1,2,3];
        direction_order=[3,4,1,2];
        for direction in direction_order
            Cset,Tset=CTM_ite(Cset, Tset, AA_fused, chi, direction,CTM_trun_tol,CTM_ite_info);
        end

        print_corner=false;
        if print_corner
            println("corner 4:")
            C4_spec=to_array(tsvd(Cset[4],(1,),(2,))[2]);
            C4_spec=C4_spec/C4_spec[1];
            println(C4_spec);
            println("corner 1:")
            C1_spec=to_array(tsvd(Cset[1],(1,),(2,))[2]);
            C1_spec=C1_spec/C1_spec[1];
            println(C1_spec);
            println("corner 3:")
            C3_spec=to_array(tsvd(Cset[3],(1,),(2,))[2]);
            C3_spec=C3_spec/C3_spec[1];
            println(C3_spec);
            println("corner 2:")
            C2_spec=to_array(tsvd(Cset[2],(1,),(2,))[2]);
            C2_spec=C2_spec/C2_spec[1];
            println(C2_spec);
            println("next iteration:")
        end
        


        if conv_check=="singular_value" #check convergence of singular value
            er1,ss_new1=spectrum_conv_check(ss_old1,Cset[1]);
            er2,ss_new2=spectrum_conv_check(ss_old2,Cset[2]);
            er3,ss_new3=spectrum_conv_check(ss_old3,Cset[3]);
            er4,ss_new4=spectrum_conv_check(ss_old4,Cset[4]);

            er=maximum([er1,er2,er3,er4]);
            ite_err=er;

            CTM_temp=deepcopy(CTM);
            CTM_temp["Cset"]=Cset;
            CTM_temp["Tset"]=Tset;
            Ov=overlap_CTM(CTM_temp,AA_fused);

            if CTM_ite_info
                println("CTMRG iteration: "*string(ci)*", CTMRG err: "*string(er)*",  overlap= "*string(Ov));flush(stdout);
            end
            
            if (er<tol)&(forced_steps==nothing)
                break;
            end
            ss_old1=ss_new1;
            ss_old2=ss_new2;
            ss_old3=ss_new3;
            ss_old4=ss_new4;
        elseif conv_check=="density_matrix" #check reduced density matrix

            # ob_opts.SiteNumber=1;
            # CTM_tem.Cset=Cset;
            # CTM_tem.Tset=Tset;
            # rho_new=ob_CTMRG(CTM_tem,A,ob_opts).A;
            # er=sum(sum((abs(rho_old-rho_new))));
            # disp(['CTMRG iteration: ',num2str(ci),' CTMRG err: ',num2str(er)]);
            # if er<tol
            #     break;
            # end
            # rho_old=rho_new;
        end

        # if ci==CTM_ite_nums
        #     display(er)
        #     warn("CTMRG does not converge: " * string(er));
        # end
    end

    CTM["Cset"]=Cset;
    CTM["Tset"]=Tset;

    return CTM


end

function CTM_ite(Cset, Tset, AA_fused, chi, direction, trun_tol,CTM_ite_info)

    AA=permute(AA_fused, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());

    @tensor MMup[:]:=Cset[mod1(direction,4)][1,2]*Tset[mod1(direction,4)][2,3,-3]*Tset[mod1(direction-1,4)][-1,4,1]*AA[4,-2,-4,3];
    @tensor MMlow[:]:=Tset[mod1(direction-1,4)][1,3,-1]*AA[3,4,-4,-2]*Cset[mod1(direction-1,4)][2,1]*Tset[mod1(direction-2,4)][-3,4,2];


    @tensor MMup_reflect[:]:=Tset[mod1(direction,4)][-1,3,1]* Cset[mod1(direction+1,4)][1,2]* AA[-2,-4,4,3]* Tset[mod1(direction+1,4)][2,4,-3];
    #@tensor MMlow_reflect[:]:=AA[-2,4,3,-4]*Tset[mod1(direction+1,4)][-3,3,1]*Tset[mod1(direction-2,4)][2,4,-1]*Cset[mod1(direction-2,4)][1,2]; #this is slow compared to other coners, I don't know why
    @tensor MMlow_reflect[:]:=Tset[mod1(direction+1,4)][-4,-3,2]*Tset[mod1(direction-2,4)][1,-2,-1]*Cset[mod1(direction-2,4)][2,1];
    @tensor MMlow_reflect[:]:=MMlow_reflect[-1,1,2,-3]*AA[-2,1,2,-4];

    MMup=permute(MMup,(1,2,),(3,4,))

    # _,ss,_=tsvd(MMup)
    # display(convert(Array,ss))

    MMlow=permute(MMlow,(1,2,),(3,4,))
    MMup_reflect=permute(MMup_reflect,(1,2,),(3,4,))
    MMlow_reflect=permute(MMlow_reflect,(1,2,),(3,4,))

    

    RMup=permute(MMup*MMup_reflect,(3,4,),(1,2,));
    RMlow=MMlow*MMlow_reflect;


    #println(norm(MMlow))

    M=RMup*RMlow;


    uM,sM,vM = tsvd(M; trunc=truncdim(chi));



    ###################
    # sM_dense=sort(convert(Array,diag(convert(Array,sM))),rev=true);
    # sM_dense=sM_dense/sM_dense[1];
    # println(sM_dense)
    sM=truncate_multiplet(sM,1e-5);

    # sM_dense=sort(convert(Array,diag(convert(Array,sM))),rev=true);
    # sM_dense=sM_dense/sM_dense[1];
    # println(sM_dense)
    #################


    
    sM=sM/norm(sM)
    sM_inv=pinv(sM);
    sM_dense=convert(Array,sM)

    # println("svd:")
    # sm_=sort(diag(sM_dense),rev=true)
    # println(sm_/sm_[1])

    # _,sM_test,_ = tsvd(M; trunc=truncdim(chi+1));
    # sm_=sort(diag(convert(Array,sM_test)),rev=true)
    # println(sm_/sm_[1])

    for c1=1:size(sM_dense,1)
        if sM_dense[c1,c1]<trun_tol
            sM_dense[c1,c1]=0;
        end
    end

    # sM_dense_diag=sort(diag(sM_dense),rev=true);
    # sM_dense_diag=sM_dense_diag/sM_dense_diag[1];
    # println(sM_dense_diag)

    #display(sM_dense)
    #display(pinv.(sM_dense))

    #display(sM_inv)
    #display(convert(Array,sM_inv))
    #sM_inv_sqrt=sqrt.(convert(Array,sM_inv))
    #display(space(sM_inv))
    #display(sM_inv_sqrt)
    sM_inv_sqrt=TensorMap(pinv.(sqrt.(sM_dense)),codomain(sM_inv)←domain(sM_inv))


    PM_inv=RMlow*vM'*sM_inv_sqrt;
    PM=sM_inv_sqrt*uM'*RMup;
    PM=permute(PM,(2,3,),(1,));

    @tensor M5tem[:]:=Tset[mod1(direction-1,4)][4,3,1]*AA[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
    @tensor M1tem[:]:=Cset[mod1(direction,4)][1,2]*Tset[mod1(direction,4)][2,3,-2]*PM_inv[1,3,-1];
    @tensor M7tem[:]:=Cset[mod1(direction-1,4)][1,2]*Tset[mod1(direction-2,4)][-1,3,1]* PM[2,3,-2];

    # println(norm(M5tem))
    # println(norm(M1tem))
    # println(norm(M7tem))

    Cset[mod1(direction,4)]=M1tem/norm(M1tem);
    Tset[mod1(direction-1,4)]=M5tem/norm(M5tem);
    Cset[mod1(direction-1,4)]=M7tem/norm(M7tem);
    return Cset,Tset
end


function init_CTM(chi,Aa,Ab,type,CTM_ite_info)
    if CTM_ite_info
        display("initialize CTM")
    end
    #numind(A)
    #numin(A)
    #numout(A)

    CTM=[];
    Cset=Vector(undef,4);
    Tset=Vector(undef,4);

    if Gutzwiller
        AA_fused, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(deepcopy(Aa'),deepcopy(Ab));
    
        if type=="PBC"
            for direction=1:4
                inds=(mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),5);
                Ab_rotate=permute(Ab,inds);
                Aa_rotate=permute(Aa,inds);
                Aap_rotate=Aa_rotate';

                @tensor M[:]:=Aap_rotate[1,-1,-3,2,3]*Ab_rotate[1,-2,-4,2,3];
                Cset[direction]=M;
                @tensor M[:]:=Aap_rotate[-1,-3,-5,1,2]*Ab_rotate[-2,-4,-6,1,2];
                Tset[direction]=M;
            end

            #fuse legs
            ul_set=Vector(undef,4);
            ur_set=Vector(undef,4);
            for direction=1:2
                ul_set[direction]=unitary(fuse(space(Cset[direction], 3) ⊗ space(Cset[direction], 4)), space(Cset[direction], 3) ⊗ space(Cset[direction], 4));
                ur_set[direction]=unitary(fuse(space(Tset[direction], 5) ⊗ space(Tset[direction], 6)), space(Tset[direction], 5) ⊗ space(Tset[direction], 6));
            end
            for direction=3:4
                ul_set[direction]=unitary(fuse(space(Cset[direction], 3) ⊗ space(Cset[direction], 4))', space(Cset[direction], 3) ⊗ space(Cset[direction], 4));
                ur_set[direction]=unitary(fuse(space(Tset[direction], 5) ⊗ space(Tset[direction], 6))', space(Tset[direction], 5) ⊗ space(Tset[direction], 6));
            end
            for direction=1:4
                C=Cset[direction];
                ul=ur_set[mod1(direction-1,4)];
                ur=ul_set[direction];
                ulp=permute(ul',(3,),(1,2,));
                urp=permute(ur',(3,),(1,2,));
                #@tensor Cnew[(-1);(-2)]:=ulp[-1,1,2]*C[1,2,3,4]*ur[-2,3,4]
                @tensor Cnew[:]:=ulp[-1,1,2]*C[1,2,3,4]*ur[-2,3,4];#put all indices in tone side so that its adjoint has the same index order
                Cset[direction]=Cnew;

                T=Tset[direction];
                ul=ul_set[direction];
                ur=ur_set[direction];
                ulp=permute(ul',(3,),(1,2,));
                urp=permute(ur',(3,),(1,2,));
                #@tensor Tnew[(-1);(-2,-3,-4)]:=ulp[-1,1,2]*T[1,2,-2,-3,3,4]*ur[-4,3,4]
                @tensor Tnew[:]:=ulp[-1,1,2]*T[1,2,-2,-3,3,4]*ur[-4,3,4];#put all indices in tone side so that its adjoint has the same index order
                Tset[direction]=Tnew;
            end
        elseif type=="random"
        end
        CTM=Dict([("Cset", Cset), ("Tset", Tset)]);
        CTM=fuse_CTM_legs(CTM,U_L,U_D,U_R,U_U);
    else
        AA_fused, U_L,U_D,U_R,U_U=build_double_layer_swap(deepcopy(Aa'),deepcopy(Ab));

        if type=="PBC"
            @tensor C1[:]:=AA_fused[1,-1,-2,3]*U_R[2,2,1]*U_D[3,4,4];
            @tensor C2[:]:=AA_fused[-1,-2,3,1]*U_D[1,2,2]*U_L[3,4,4];
            @tensor C3[:]:=AA_fused[-2,3,1,-1]*U_L[1,2,2]*U_U[4,4,3];
            @tensor C4[:]:=AA_fused[1,3,-1,-2]*U_R[2,2,1]*U_U[4,4,3];

            @tensor T4[:]:=AA_fused[1,-1,-2,-3]*U_R[2,2,1];
            @tensor T1[:]:=AA_fused[-1,-2,-3,1]*U_D[1,2,2];
            @tensor T2[:]:=AA_fused[-2,-3,1,-1]*U_L[1,2,2];
            @tensor T3[:]:=AA_fused[-3,1,-1,-2]*U_U[2,2,1];

            Cset[1]=C1;
            Cset[2]=C2;
            Cset[3]=C3;
            Cset[4]=C4;

            Tset[1]=T1;
            Tset[2]=T2;
            Tset[3]=T3;
            Tset[4]=T4;
            

        elseif type=="random"
        end
        CTM=Dict([("Cset", Cset), ("Tset", Tset)]);



    end

    return CTM, AA_fused, U_L,U_D,U_R,U_U

end

function to_array_multiplet(S)
    ss=[];
    for (k,b) in blocks(S)
        #println(typeof(b))
        #println(diag(b))
        ss=vcat(ss,diag(b));
    end
    ss=sort(ss);
    ss=ss[end:-1:1];


    # toret = similar(S);

    # for (k,b) in blocks(S)
    #     copyto!(blocks(toret)[k],LinearAlgebra.diagm(LinearAlgebra.diag(b).^(-1/2)));
    # end
    
    return ss

end
function to_array(S)
    ss=diag(convert(Array,S));
    return ss
end
# function to_array(S)
#     ss=[];
#     for (k,b) in blocks(S)
#         # println(typeof(b))
#          #println(diag(b))
#          ss=vcat(ss,diag(b));
#     end
#     ss=sort(ss);
#     ss=ss[end:-1:1];


#     # toret = similar(S);

#     # for (k,b) in blocks(S)
#     #     copyto!(blocks(toret)[k],LinearAlgebra.diagm(LinearAlgebra.diag(b).^(-1/2)));
#     # end
    
#     return ss

# end

function truncate_multiplet(s,multiplet_tol)
    s=deepcopy(s);
    #the multiplet is not due to su(2) symmetry

    s_mini=to_array_multiplet(s)[end];
    for (k,b) in blocks(s)
        #println(typeof(b))
        #println(diag(b))
        L=size(b)[1];
        for cc=1:L
            if abs(b[cc,cc]-s_mini)/s_mini<multiplet_tol
                b[cc,cc]=0;
            end
        end
        copyto!(blocks(s)[k],b);
    end


    return s
end


# function truncate_multiplet(s,chi,multiplet_tol,trun_tol)
#     #the multiplet is not due to su(2) symmetry
#     s_dense=sort(to_array(s),rev=true);

#     println(s_dense/s_dense[1])

#     if length(s_dense)>chi
#         value_trun=s_dense[chi+1];
#     else
#         value_trun=0;
#     end
#     value_max=maximum(s_dense);

#     s_Dict=convert(Dict,s);
    
#     space_full=space(s,1);
#     for sp in sectors(space_full)

#         diag_elem=diag(s_Dict[:data][string(sp)]);
#         for cd=1:length(diag_elem)
#             if ((diag_elem[cd]/value_max)<trun_tol) | (diag_elem[cd]<=value_trun) |(abs((diag_elem[cd]-value_trun)/value_trun)<multiplet_tol)
#                 diag_elem[cd]=0;
#             end
#         end
#         s_Dict[:data][string(sp)]=diagm(diag_elem);
#     end
#     s=convert(TensorMap,s_Dict);

#     s_=sort(diag(convert(Array,s)),rev=true);
#     s_=s_/s_[1];
#     print(s_)
#     # @assert 1+1==3
#     return s
# end





