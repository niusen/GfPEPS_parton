using TensorKit


function create_vaccum(Length)
#      ^
#      |3
#  --> A -->
#   1     2

    
    Vf0=ℂ[FermionNumber](0=>1);
    Vf=ℂ[FermionNumber](0=>1,1=>1);
    A = TensorMap(randn, Vf0 ← Vf0 ⊗ Vf);

    A_dict=convert(Dict,A)
    m=A_dict[:data]["FermionNumber(0)"];
    m[1]=1;
    A_dict[:data]["FermionNumber(0)"]=m;
    A=convert(TensorMap, A_dict);

    A_set=Vector(undef,Length);
    for cc=1:Length
        A_set[cc]=A;
    end
    return A_set
end

function create_left_mpo()
    Vf0=ℂ[FermionNumber](1=>1);
    Vf=ℂ[FermionNumber](0=>1, 1=>1);
    mpo = TensorMap(randn, Vf0 ← Vf);

    mpo_dict=convert(Dict,mpo)
    m=mpo_dict[:data]["FermionNumber(1)"];
    m[1]=1;
    mpo_dict[:data]["FermionNumber(1)"]=m;
    mpo=convert(TensorMap, mpo_dict);
    return mpo
end

function create_right_mpo()
    
    Vf=ℂ[FermionNumber](0=>1, 1=>1);
    Vf0=ℂ[FermionNumber](0=>1);
    mpo = TensorMap(randn, Vf ← Vf0);

    mpo_dict=convert(Dict,mpo)
    m=mpo_dict[:data]["FermionNumber(0)"];
    m[1]=1;
    mpo_dict[:data]["FermionNumber(0)"]=m;
    mpo=convert(TensorMap, mpo_dict);
    return mpo
end

function create_mpo(coe)
#     ^
#     |4
# --> mpo -->  3
#  1  ^
#     |2


    Vf=ℂ[FermionNumber](0=>1, 1=>1);
    mpo = TensorMap(randn, Vf ⊗ Vf ← Vf ⊗ Vf);

    mpo_dict=convert(Dict,mpo)
    m=mpo_dict[:data]["FermionNumber(0)"];
    m[1]=1;
    mpo_dict[:data]["FermionNumber(0)"]=m;

    m=mpo_dict[:data]["FermionNumber(1)"];
    m[1,1]=0;
    m[1,2]=coe;
    m[2,1]=0;
    m[2,2]=0;
    mpo_dict[:data]["FermionNumber(1)"]=m;

    m=mpo_dict[:data]["FermionNumber(2)"];
    m[1]=1;
    mpo_dict[:data]["FermionNumber(2)"]=m;
    mpo=convert(TensorMap, mpo_dict);
    return mpo
end

function build_mpo_set(Length,coes)
    @assert length(coes)==Length;
    mpo_set=Vector(undef,Length);
    mpo_left=create_left_mpo();
    mpo_right=create_right_mpo();

    for cc=1:Length
        mpo_set[cc]=create_mpo(coes[cc]);
    end

    @tensor mpo1[:]:=mpo_left[-1,1]*mpo_set[1][1,-2,-3,-4];
    mpo1=permute(mpo1,(1,2,),(3,4,));
    mpo_set[1]=mpo1;

    @tensor mpo_end[:]:=mpo_right[1,-3]*mpo_set[Length][-1,-2,1,-4];
    mpo_end=permute(mpo_end,(1,2,),(3,4,));
    mpo_set[Length]=mpo_end;

    return mpo_set
end

function mpo_mps(mpo_set,mps_set)
    mps_set=deepcopy(mps_set);
    @assert length(mpo_set)==length(mps_set)
    for cc=1:length(mps_set)
        # println(space(mpo_set[cc]))
        # println(space(mps_set[cc]))
        @tensor mps[:]:=mpo_set[cc][-1,1,-3,-5]*mps_set[cc][-2,-4,1];
        mps_set[cc]=mps;
    end
    #fuse legs
    UL=unitary(fuse(space(mps_set[1],1)⊗space(mps_set[1],2)),space(mps_set[1],1)⊗space(mps_set[1],2));
    @tensor mps[:]:=mps_set[1][1,2,-3,-4,-5]*UL[-1,1,2];
    mps_set[1]=mps;

    for cc=1:length(mps_set)-1
        UL=unitary(fuse(space(mps_set[cc+1],1)⊗space(mps_set[cc+1],2)),space(mps_set[cc+1],1)⊗space(mps_set[cc+1],2));
        @tensor mps[:]:=mps_set[cc][-1,1,2,-3]*UL'[1,2,-2];
        mps_set[cc]=mps;
        @tensor mps[:]:=mps_set[cc+1][1,2,-2,-3,-4]*UL[-1,1,2];
        mps_set[cc+1]=mps;
    end

    UR=unitary(fuse(space(mps_set[end],2)⊗space(mps_set[end],3)),space(mps_set[end],2)⊗space(mps_set[end],3));
    @tensor mps[:]:=mps_set[end][-1,1,2,-3]*UR[-2,1,2];
    mps_set[end]=mps;

    for cc=1:length(mps_set)
        mps_set[cc]=permute(mps_set[cc],(1,),(2,3,));
    end


    return mps_set
end