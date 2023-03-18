using KrylovKit
function build_double_layer_swap_op(A1,O1,has_extra_leg)
    A1=deepcopy(A1)
    A1_origin=deepcopy(A1)



    if has_extra_leg
        @tensor A1[:]:= A1[-1,-2,-3,-4,1]*O1[-5,1,-6]#the last index is extra
        A1_new=A1
        A1_double,_,_,_,_=build_double_layer_swap_extra_leg(A1_origin',A1_new)
    else
        @tensor A1[:]:= A1[-1,-2,-3,-4,1]*O1[-5,1]
        A1_new=A1
        A1_double,_,_,_,_=build_double_layer_swap(A1_origin',A1_new)
    end

    return A1_double
end

function build_double_layer_NoSwap_op(A1,O1,has_extra_leg)
    A1=deepcopy(A1)
    A1_origin=deepcopy(A1)



    if has_extra_leg
        @tensor A1[:]:= A1[-1,-2,-3,-4,1]*O1[-5,1,-6]#the last index is extra
        A1_new=A1
        A1_double,_,_,_,_=build_double_layer_NoSwap_extra_leg(A1_origin',A1_new)
    else
        @tensor A1[:]:= A1[-1,-2,-3,-4,1]*O1[-5,1]
        A1_new=A1
        A1_double,_,_,_,_=build_double_layer_NoSwap(A1_origin',A1_new)
    end

    return A1_double
end


# function build_double_layer_swap_op(A1,A2,O1,O2,has_extra_leg)
#     A1=deepcopy(A1)
#     A2=deepcopy(A2)
#     A1_origin=deepcopy(A1)
#     A2_origin=deepcopy(A2)


#     if has_extra_leg
#         @tensor A1[:]:= A1[-1,-2,-3,-4,1]*O1[-5,1,-6]#the last index is extra
#         @tensor A2[:]:= A2[-1,-2,-3,-4,1]*O2[-5,1,-6]#the last index is extra
#         A1_new=A1
#         A2_new=A2

#         A1_double,_,_,_,_=build_double_layer_swap_extra_leg(A1_origin',A1_new)
#         A2_double,_,_,_,_=build_double_layer_swap_extra_leg(A2_origin',A2_new)
#     else
#         @tensor A1[:]:= A1[-1,-2,-3,-4,1]*O1[-5,1]
#         @tensor A2[:]:= A2[-1,-2,-3,-4,1]*O2[-5,1]
#         A1_new=A1
#         A2_new=A2

#         A1_double,_,_,_,_=build_double_layer_swap(A1_origin',A1_new)
#         A2_double,_,_,_,_=build_double_layer_swap(A2_origin',A2_new)
#     end

#     return A1_double,A2_double
# end


function build_double_layer_swap_extra_leg(Ap,A)
    #The last index of A tensor is an extra virtual index, such as that comes from decomposition of Heisenberg interaction


    gate=swap_gate(Ap,1,4); @tensor Ap[:]:=Ap[1,-2,-3,2,-5]*gate[-1,-4,1,2];  
    gate=swap_gate(Ap,2,3); @tensor Ap[:]:=Ap[-1,1,2,-4,-5]*gate[-2,-3,1,2];  
    gate=parity_gate(Ap,4); @tensor Ap[:]:=Ap[-1,-2,-3,1,-5]*gate[-4,1];
    gate=parity_gate(Ap,2); @tensor Ap[:]:=Ap[-1,1,-3,-4,-5]*gate[-2,1];


    Ap=permute(Ap,(1,2,),(3,4,5))
    A=permute(A,(1,2,),(3,4,5,6));
    
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
    vM=permute(vM,(1,2,3,4,5,),());

    @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5,-6];
    @tensor double_RU[:]:=vMp[1,-2,-4,2]*double_RU[-1,1,-3,-5,2,-6];

    #display(space(double_RU))

    double_LD=permute(double_LD,(1,2,),(3,4,5,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,),(3,));
    #display(space(double_LD))


    double_RU=permute(double_RU,(1,4,5,6,),(2,3,));
    double_RU=double_RU*U_R;
    double_RU=permute(double_RU,(1,5,4,),(2,3,));
    double_RU=double_RU*U_U;
    double_LD=permute(double_LD,(1,2,),(3,));
    double_RU=permute(double_RU,(1,),(2,4,3,));
    AA_fused=double_LD*double_RU;


    ##########################
    @tensor U_LU[:]:=U_L'[-1,-2,-5]*U_U'[-6,-3,-4];
    gate1=swap_gate(U_LU,1,4);
    gate2=swap_gate(U_LU,3,4);
    @tensor U_LU[:]:=U_LU[1,-2,-3,2,-5,-6]*gate1[-1,-4,1,2];
    @tensor U_LU[:]:=U_LU[-1,-2,1,2,-5,-6]*gate2[-3,-4,1,2];
    @tensor U_LU[:]:=U_LU[1,2,3,4,-3,-4]*U_L[-1,1,2]*U_U[3,4,-2];
    @tensor AA_fused[:]:=AA_fused[1,-2,-3,2,-5]*U_LU[-1,-4,1,2];


    @tensor U_DR[:]:=U_D'[-1,-2,-5]*U_R'[-6,-3,-4];
    gate1=swap_gate(U_DR,1,2);
    gate2=swap_gate(U_DR,1,4);
    @tensor U_DR[:]:=U_DR[1,2,-3,-4,-5,-6]*gate1[-1,-2,1,2];
    @tensor U_DR[:]:=U_DR[1,-2,-3,2,-5,-6]*gate2[-1,-4,1,2];

    @tensor U_DR[:]:=U_DR[1,2,3,4,-3,-4]*U_D[-1,1,2]*U_R[3,4,-2];
    @tensor AA_fused[:]:=AA_fused[-1,1,2,-4,-5]*U_DR[-2,-3,1,2];

    return AA_fused, U_L,U_D,U_R,U_U
end

function build_double_layer_NoSwap_extra_leg(Ap,A)
    #The last index of A tensor is an extra virtual index, such as that comes from decomposition of Heisenberg interaction

    Ap=permute(Ap,(1,2,),(3,4,5))
    A=permute(A,(1,2,),(3,4,5,6));
    
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
    vM=permute(vM,(1,2,3,4,5,),());

    @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5,-6];
    @tensor double_RU[:]:=vMp[1,-2,-4,2]*double_RU[-1,1,-3,-5,2,-6];

    #display(space(double_RU))

    double_LD=permute(double_LD,(1,2,),(3,4,5,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,),(3,));
    #display(space(double_LD))


    double_RU=permute(double_RU,(1,4,5,6,),(2,3,));
    double_RU=double_RU*U_R;
    double_RU=permute(double_RU,(1,5,4,),(2,3,));
    double_RU=double_RU*U_U;
    double_LD=permute(double_LD,(1,2,),(3,));
    double_RU=permute(double_RU,(1,),(2,4,3,));
    AA_fused=double_LD*double_RU;


    ##########################


    return AA_fused, U_L,U_D,U_R,U_U
end

function evaluate_correl_spinspin(direction, AA_fused, AA_op1, AA_op2, CTM, method, distance)
    correl_funs=Vector(undef,distance);

    C1=CTM["Cset"][1];
    C2=CTM["Cset"][2];
    C3=CTM["Cset"][3];
    C4=CTM["Cset"][4];
    T1=CTM["Tset"][1];
    T2=CTM["Tset"][2];
    T3=CTM["Tset"][3];
    T4=CTM["Tset"][4];
    if method=="dimerdimer"#operator on a single site conserves su2 symmetry
        if direction=="x"
            @tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[7,2]*T1[3,4,-1]*AA_op1[5,6,-2,4]*T3[-3,6,7];
            @tensor vb[:]:=T1[-1,4,3]*AA_op2[-2,6,5,4]*T3[7,6,-3]*C2[3,1]*T2[1,5,2]*C3[2,7];
            @tensor ov[:]:=va[1,2,3]*vb[1,2,3]
            correl_funs[1]=blocks(ov)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
            
            for dis=2:distance
                @tensor va[:]:=va[1,3,5]*T1[1,2,-1]*AA_fused[3,4,-2,2]*T3[-3,4,5];
                @tensor ov[:]:=va[1,2,3]*vb[1,2,3]
                correl_funs[dis]=blocks(ov)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
            end
            return correl_funs
        end
    elseif method=="spinspin" #operator on a single site breaks su2 symmetry, so there is an extra index obtained from svd of two-site operator
        if direction=="x"
            @tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[7,2]*T1[3,4,-1]*AA_op1[5,6,-2,4,-4]*T3[-3,6,7];
            @tensor vb[:]:=T1[-1,4,3]*AA_op2[-2,6,5,4,-4]*T3[7,6,-3]*C2[3,1]*T2[1,5,2]*C3[2,7];
            @tensor ov[:]:=va[1,2,3,4]*vb[1,2,3,4]
            correl_funs[1]=blocks(ov)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
            
            for dis=2:distance
                @tensor va[:]:=va[1,3,5,-4]*T1[1,2,-1]*AA_fused[3,4,-2,2]*T3[-3,4,5];
                @tensor ov[:]:=va[1,2,3,4]*vb[1,2,3,4]
                correl_funs[dis]=blocks(ov)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
            end
            return correl_funs
        end
    end
end


function correl_TransOp(vl,Tup,Tdown,AAfused)
    if AAfused==[]
        
        @tensor vl[:]:=vl[-1,1,3]*Tup[1,2,-2]*Tdown[-3,2,3];
        
    else
        
        @tensor vl[:]:=vl[-1,1,3,5]*Tup[1,2,-2]*AAfused[3,4,-3,2]*Tdown[-4,4,5];
        
    end
    return vl
end
function solve_correl_length(n_values,AA_fused,CTM,direction)
    T1=CTM["Tset"][1];
    T2=CTM["Tset"][2];
    T3=CTM["Tset"][3];
    T4=CTM["Tset"][4];
    println(fuse(space(T1,1)'⊗space(AA_fused,1)', space(T3,3)))
    if direction=="x"
        correl_TransOp_fx(x)=correl_TransOp(x,T1,T3,AA_fused)

        Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        eu,ev=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
        eus=eu;
        Qspin=eu*0;
        QN=eu*0;

        Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((2,0)=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            Qspin=vcat(Qspin,0*eu.+0);
            QN=vcat(QN,0*eu.+2);
        end

        # Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2,0)=>1);
        # vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        # if norm(vl_init)>0
        #     eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
        #     eus=vcat(eus,eu);
        #     Qspin=vcat(Qspin,0*eu.+0);
        #     QN=vcat(QN,0*eu.-2);
        # end

        Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,1)=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            Qspin=vcat(Qspin,0*eu.+1);
            QN=vcat(QN,0*eu.+0);
        end

        Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((2,1)=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            Qspin=vcat(Qspin,0*eu.+1);
            QN=vcat(QN,0*eu.+2);
        end

        # Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2,1)=>1);
        # vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        # if norm(vl_init)>0
        #     eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
        #     eus=vcat(eus,eu);
        #     Qspin=vcat(Qspin,0*eu.+1);
        #     QN=vcat(QN,0*eu.-2);
        # end

        Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            Qspin=vcat(Qspin,0*eu.+1/2);
            QN=vcat(QN,0*eu.+1);
        end

        # Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-1,1/2)=>1);
        # vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        # if norm(vl_init)>0
        #     eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
        #     eus=vcat(eus,eu);
        #     Qspin=vcat(Qspin,0*eu.+1/2);
        #     QN=vcat(QN,0*eu.-1);
        # end

        Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((3,1/2)=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            Qspin=vcat(Qspin,0*eu.+1/2);
            QN=vcat(QN,0*eu.+3);
        end

        # Vl=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-3,1/2)=>1);
        # vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        # if norm(vl_init)>0
        #     eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
        #     eus=vcat(eus,eu);
        #     Qspin=vcat(Qspin,0*eu.+1/2);
        #     QN=vcat(QN,0*eu.-3);
        # end

        eus_abs=abs.(eus);
        @assert maximum(eus_abs)==eus_abs[1]

        eus_abs_sorted=sort(eus_abs,rev=true);
        eus_abs_sorted=eus_abs_sorted/eus_abs_sorted[1];
        Qspin=Qspin[sortperm(eus_abs,rev=true)];
        QN=QN[sortperm(eus_abs,rev=true)];

        
        return eus_abs_sorted, Qspin, QN
    end
  
end


function cal_correl(M, AA_fused,AA_SS,AA_SAL,AA_SBL,AA_SAR,AA_SBR, chi,CTM, distance)
    #M: number of virtual modes 
    


    #single-unitcell correlations
    norm=ob_1site_closed(CTM,AA_fused);
    
    SS_cell_ob=ob_1site_closed(CTM,AA_SS);
    SS_cell_ob=SS_cell_ob/norm;

    
    norms=evaluate_correl_spinspin("x", AA_fused, AA_fused, AA_fused, CTM, "dimerdimer", 10);
    norm_coe=norms[5]/norms[4] #get a rough normalization coefficient to avoid that the number becomes two small
    norms=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_fused, AA_fused, CTM, "dimerdimer", distance);
    dimer_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SS, AA_SS, CTM, "dimerdimer", distance);

    SASA_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SAL, AA_SAR, CTM, "spinspin", distance);
    SASB_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SAL, AA_SBR, CTM, "spinspin", distance);
    SBSA_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SBL, AA_SAR, CTM, "spinspin", distance);
    SBSB_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SBL, AA_SBR, CTM, "spinspin", distance);

    dimer_ob=dimer_ob./norms;
    SASA_ob=SASA_ob./norms;
    SASB_ob=SASB_ob./norms;
    SBSA_ob=SBSA_ob./norms;
    SBSB_ob=SBSB_ob./norms;

    println(norms)

    eus_x, Qspin_x, QN_x=solve_correl_length(5,AA_fused/norm_coe,CTM,"x");


    _,corner_spec=svd(convert(Array,CTM["Cset"][1]))

    mat_filenm="correl_M"*string(M)*"_chi"*string(chi)*".mat";
    matwrite(mat_filenm, Dict(
        "corner_spec" => corner_spec,
        "SS_cell_ob" => SS_cell_ob,
        "dimer_ob" => dimer_ob,
        "SASA_ob" => SASA_ob,
        "SASB_ob" => SASB_ob,
        "SBSA_ob" => SBSA_ob,
        "SBSB_ob" => SBSB_ob,
        "eus_x" => eus_x,
        "Qspin_x"=> Qspin_x,
        "QN_x"=> QN_x,
        "CTM_space"=> string(space(CTM["Cset"][1]))
    ); compress = false)
end

function ob_1site_closed(CTM,AA_fused)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    @tensor envL[:]:=Cset[1][1,-1]*Tset[4][2,-2,1]*Cset[4][-3,2];
    @tensor envR[:]:=Cset[2][-1,1]*Tset[2][1,-2,2]*Cset[3][2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset[1][1,3,-1]*AA_fused[2,5,-2,3]*Tset[3][-3,5,4];
    @tensor Norm[:]:=envL[1,2,3]*envR[1,2,3];
    Norm=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
    return Norm;
end