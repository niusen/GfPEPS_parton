


function Entropy_finite_size(CTM,U_L,U_D,U_R,U_U,M,chi,N)

    println("M="*string(M));
    println("chi="*string(chi));
    println("N="*string(N));flush(stdout);




    Tleft=CTM["Tset"][4];
    Tright=CTM["Tset"][2];
    @tensor O1[:]:=Tleft[-3,1,-1]*U_L[1,-2,-4];
    @tensor O2[:]:=Tright[-1,1,-3]*U_R[-4,-2,1];

    @tensor OO[:]:=O1[-2,-3,-5,1]*O2[-1,1,-4,-6];
    U_fuse_chichi=unitary(fuse(space(OO,1)⊗ space(OO,2)),space(OO,1)⊗ space(OO,2));
    @tensor OO[:]:=U_fuse_chichi[-1,1,2]*OO[1,2,-2,3,4,-4]*U_fuse_chichi'[3,4,-3];




    gate=parity_gate(O1,2);
    gate_dense=convert(Array,gate);
    Id=Matrix(I, size(gate_dense)[1],size(gate_dense)[1]);
    P_even=zeros(2,size(gate_dense)[1],2,size(gate_dense)[1]);
    P_even[1,:,1,:]=Id;
    P_even[2,:,2,:]=gate_dense;
    P_odd=zeros(2,size(gate_dense)[1],2,size(gate_dense)[1]);
    P_odd[1,:,1,:]=Id;
    P_odd[2,:,2,:]=-gate_dense;

    V=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>2);
    P_even=TensorMap(P_even, V⊗space(gate,1), V⊗space(gate,1));
    P_odd=TensorMap(P_odd, V⊗space(gate,1), V⊗space(gate,1));



    println("calculate entropy for N="*string(N));

    ###########################################
    Projector=P_even;

    @tensor OO_P[:]:=OO[-1,1,-3,2]*Projector[-2,2,-4,1];

    @tensor OO_OO_P[:]:=OO[-1,2,-4,1]*OO[-2,3,-5,2]*Projector[-3,1,-6,3];

    OO_P=permute(OO_P,(1,2,),(3,4,));
    OO_OO_P=permute(OO_OO_P,(1,2,3,),(4,5,6,));

    Norm=deepcopy(OO_P);
    Renyi2=deepcopy(OO_OO_P);
    for cc=1:N-2
        Norm=Norm*OO_P;
        Renyi2=Renyi2*OO_OO_P;
    end
    @tensor Norm[:]:=Norm[1,2,3,4]*OO_P[3,4,1,2];
    @tensor Renyi2[:]:=Renyi2[1,2,3,4,5,6]*OO_OO_P[4,5,6,1,2,3];

    Norm=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1]/2;
    Renyi2=blocks(Renyi2)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1]/2;
    Renyi2_even=-log(Renyi2/Norm^2);

    ###########################################
    println("Alert: the odd sector acquires extra gate which has not been implemented")
    Projector=P_odd;

    @tensor OO_P[:]:=OO[-1,1,-3,2]*Projector[-2,2,-4,1];

    @tensor OO_OO_P[:]:=OO[-1,2,-4,1]*OO[-2,3,-5,2]*Projector[-3,1,-6,3];

    OO_P=permute(OO_P,(1,2,),(3,4,));
    OO_OO_P=permute(OO_OO_P,(1,2,3,),(4,5,6,));

    Norm=deepcopy(OO_P);
    Renyi2=deepcopy(OO_OO_P);
    for cc=1:N-2
        Norm=Norm*OO_P;
        Renyi2=Renyi2*OO_OO_P;
    end
    @tensor Norm[:]:=Norm[1,2,3,4]*OO_P[3,4,1,2];
    @tensor Renyi2[:]:=Renyi2[1,2,3,4,5,6]*OO_OO_P[4,5,6,1,2,3];

    Norm=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1]/2;
    Renyi2=blocks(Renyi2)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1]/2;
    Renyi2_odd=-log(Renyi2/Norm^2);


    ES_filenm="Entropy_finite_size"*"_M"*string(M)*"_chi"*string(chi)*"_N"*string(N)*".mat";
    matwrite(ES_filenm, Dict(
        "Renyi2_odd" => Renyi2_odd,
        "Renyi2_even" => Renyi2_even
    ); compress = false)


end




function Topo_entropy_Renyi2(CTM,U_L,U_D,U_R,U_U,M,chi,N_eu)

    println("M="*string(M));
    println("chi="*string(chi));


    Tleft=CTM["Tset"][4];
    Tright=CTM["Tset"][2];
    @tensor O1[:]:=Tleft[-3,1,-1]*U_L[1,-2,-4];
    @tensor O2[:]:=Tright[-1,1,-3]*U_R[-4,-2,1];

    @tensor OO[:]:=O1[-2,-3,-5,1]*O2[-1,1,-4,-6];
    U_fuse_chichi=unitary(fuse(space(OO,1)⊗ space(OO,2)),space(OO,1)⊗ space(OO,2));
    @tensor OO[:]:=U_fuse_chichi[-1,1,2]*OO[1,2,-2,3,4,-4]*U_fuse_chichi'[3,4,-3];




    gate=parity_gate(O1,2);
    gate_dense=convert(Array,gate);
    Id=Matrix(I, size(gate_dense)[1],size(gate_dense)[1]);
    P_even=zeros(2,size(gate_dense)[1],2,size(gate_dense)[1]);
    P_even[1,:,1,:]=Id;
    P_even[2,:,2,:]=gate_dense;
    P_odd=zeros(2,size(gate_dense)[1],2,size(gate_dense)[1]);
    P_odd[1,:,1,:]=Id;
    P_odd[2,:,2,:]=-gate_dense;

    V=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>2);
    P_even=TensorMap(P_even, V⊗space(gate,1), V⊗space(gate,1));
    P_odd=TensorMap(P_odd, V⊗space(gate,1), V⊗space(gate,1));


    println("Alert: the odd sector acquires extra gate which has not been implemented")

    ###########################################
    println("calculate topo entropy for even sector");
    Projector=P_even;

    Renyi2=trace_boundary_H(N_eu,OO,Projector,"OO_OO_P")/2;
    Norm=trace_boundary_H(N_eu,OO,Projector,"OO_P")/2;
    Renyi2_even=-log(Renyi2/Norm^2);

    println("calculate topo entropy for odd sector");
    Projector=P_odd;

    Renyi2=trace_boundary_H(N_eu,OO,Projector,"OO_OO_P")/2;
    Norm=trace_boundary_H(N_eu,OO,Projector,"OO_P")/2;
    Renyi2_odd=-log(Renyi2/Norm^2);

    ES_filenm="Topo_entropy_Renyi2"*"_M"*string(M)*"_chi"*string(chi)*".mat";
    matwrite(ES_filenm, Dict(
        "Renyi2_odd" => Renyi2_odd,
        "Renyi2_even" => Renyi2_even
    ); compress = false)


end
  

function trace_boundary_H(N_eu,OO,Projector,type)
    if type=="OO_OO_P"
        println("Trace H^2")
    elseif type=="OO_P"
        println("Trace H")
    end
    Spins=[0,0,0,1/2,1,3/2,2,5/2];
    Qns=[0,2,-2,1,2,3,4,5];
    euL_set=Vector(undef,length(Spins));
    evL_set=Vector(undef,length(Spins));
    euR_set=Vector(undef,length(Spins));
    evR_set=Vector(undef,length(Spins));
    for sps=1:length(Spins)
        if type=="OO_OO_P"
            V=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Qns[sps],Spins[sps])=>1);
            vr_init=TensorMap(randn, space(OO,3)'*space(OO,3)'*space(Projector,3)',V);
            vr_init=permute(vr_init,(1,2,3,4,),());
            Rcontraction_fun1(x)=R_action_OO_OO_P(OO,Projector,x);

            vl_init=TensorMap(randn, space(OO,3)*space(OO,3)*space(Projector,3),V');
            vl_init=permute(vl_init,(4,1,2,3,),());
            Lcontraction_fun1(x)=L_action_OO_OO_P(OO,Projector,x);
        elseif type=="OO_P"
            V=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Qns[sps],Spins[sps])=>1);
            vr_init=TensorMap(randn, space(OO,3)'*space(Projector,3)',V);
            vr_init=permute(vr_init,(1,2,3,),());
            Rcontraction_fun2(x)=R_action_OO_P(OO,Projector,x);

            vl_init=TensorMap(randn, space(OO,3)*space(Projector,3),V');
            vl_init=permute(vl_init,(3,1,2,),());
            Lcontraction_fun2(x)=L_action_OO_P(OO,Projector,x);
        end
            if norm(vl_init)<1e-12
                euR_set[sps]=[];
                euR_set[sps]=[];
                euL_set[sps]=[];
                euL_set[sps]=[];
                continue;
            end
            
        if type=="OO_OO_P"
            @time eur,evr=eigsolve(Rcontraction_fun1, vr_init, N_eu,:LM,Arnoldi(krylovdim=N_eu*5));
            @time eul,evl=eigsolve(Lcontraction_fun1, vl_init, N_eu,:LM,Arnoldi(krylovdim=N_eu*5));
        elseif type=="OO_P"
            @time eur,evr=eigsolve(Rcontraction_fun2, vr_init, N_eu,:LM,Arnoldi(krylovdim=N_eu*5));
            @time eul,evl=eigsolve(Lcontraction_fun2, vl_init, N_eu,:LM,Arnoldi(krylovdim=N_eu*5));
        end


            if length(eul)<length(eur)
                eur=eur[1:length(eul)];
                evr=evr[1:length(evl)];
            elseif length(eul)>length(eur)
                eul=eul[1:length(eur)];
                evl=evl[1:length(evr)];
            end

            @assert norm(abs.(eul)-abs.(eur))<1e-12
            euR_set[sps]=eur;
            evR_set[sps]=evr;
            euL_set[sps]=eul;
            evL_set[sps]=evl;
    
            println("Spin="*string(Spins[sps])*", Qn="*string(Qns[sps]));flush(stdout);
            println("Eigenvalues:"*string(eur));flush(stdout);

    end


    #check that the leading eigenvalue is in S=0 sector
    for cc=2:length(euL_set)
        if length(euR_set[cc])>0
            @assert maximum(abs.(euR_set[cc]))/maximum(abs.(euR_set[1]))<(1-1e-6)
        end
    end

    #take only S=0 sector
    euL=euL_set[1];
    evL=evL_set[1];
    euR=euR_set[1];
    evR=evR_set[1];
    #truncate 
    N_keep=1;
    for cc=2:length(euL)
        if abs(euL[cc])/abs(euL[1])>(1-1e-6)
            N_keep=N_keep+1;
        end
    end
    @assert length(euL)>N_keep; #ensure that all degenerate largest eigenvalues are obtained

    euL=euL[1:N_keep];
    evL=evL[1:N_keep];
    euR=euR[1:N_keep];
    evR=evR[1:N_keep];

    println("largest eigenvalues:"*string(euL));flush(stdout);

    #choose correct gauge of left and right eigenvectors;
    M=zeros(length(evL),length(evR))*(1+0*im);
    for ca=1:length(evL)
        for cb=1:length(evR)
            if type=="OO_OO_P"
                @tensor ov[:]:=evL[ca][4,1,2,3]*evR[cb][1,2,3,4];
            elseif type=="OO_P"
                @tensor ov[:]:=evL[ca][3,1,2]*evR[cb][1,2,3];
            end
            ov=blocks(ov)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
            M[ca,cb]=ov;
            
        end
    end
    M_inv=pinv(M);
    #H=evR*euR*(M_inv*evL);

    #compute total sum 
    tot=0;
    for ca=1:length(evR)
        for cb=1:length(evL)
            if type=="OO_OO_P"
                @tensor ov[:]:=evR[ca][1,2,3,4]*evL[cb][4,1,2,3];
            elseif type=="OO_P"
                @tensor ov[:]:=evR[ca][1,2,3]*evL[cb][3,1,2];
            end
            ov=blocks(ov)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
            #tot=tot+ov*euR[ca]*M_inv[ca,cb];
            tot=tot+ov*M_inv[ca,cb];
        end
    end
    return tot
end




function R_action_OO_OO_P(OO,Projector,v0)
    @tensor v_new[:]:=OO[-1,4,5,6]*OO[-2,2,3,4]*Projector[-3,6,1,2]*v0[5,3,1,-4];
    return v_new
end

function L_action_OO_OO_P(OO,Projector,v0)
    @tensor v_new[:]:=OO[5,4,-2,6]*OO[3,2,-3,4]*Projector[1,6,-4,2]*v0[-1,5,3,1];
    return v_new
end

function R_action_OO_P(OO,Projector,v0)
    @tensor v_new[:]:=OO[-1,2,3,4]*Projector[-2,4,1,2]*v0[3,1,-3];
    return v_new
end

function L_action_OO_P(OO,Projector,v0)
    @tensor v_new[:]:=OO[3,2,-2,4]*Projector[1,4,-3,2]*v0[-1,3,1];
    return v_new
end

