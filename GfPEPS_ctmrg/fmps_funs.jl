using TensorKit

function hop_mpo(x1,x2,L)

    #     ^
    #     |4
    # --> mpo -->  3
    #  1  ^
    #     |2

    ###################
    Vf1=ℂ[FermionNumber](0=>1);
    Vf2=ℂ[FermionNumber](0=>1,1=>1);
    Vf3=ℂ[FermionNumber](0=>1);
    Vf4=ℂ[FermionNumber](0=>1,1=>1);
    mpo = TensorMap(randn, Vf1 ⊗ Vf2 ← Vf3 ⊗ Vf4);

    mpo_dict=convert(Dict,mpo)

    m=mpo_dict[:data]["FermionNumber(0)"];
    m[1]=1;
    mpo_dict[:data]["FermionNumber(0)"]=m;

    m=mpo_dict[:data]["FermionNumber(1)"];
    m[1]=1;
    mpo_dict[:data]["FermionNumber(1)"]=m;

    mpo=convert(TensorMap, mpo_dict);
    mpo_left=mpo;

    ###################
    Vf1=ℂ[FermionNumber](0=>1);
    Vf2=ℂ[FermionNumber](0=>1,1=>1);
    Vf3=ℂ[FermionNumber](-1=>1);
    Vf4=ℂ[FermionNumber](0=>1,1=>1);
    mpo = TensorMap(randn, Vf1 ⊗ Vf2 ← Vf3 ⊗ Vf4);

    mpo_dict=convert(Dict,mpo)
    m=mpo_dict[:data]["FermionNumber(0)"];
    m[1]=1;
    mpo_dict[:data]["FermionNumber(0)"]=m;
    mpo=convert(TensorMap, mpo_dict);
    mpo_x1=mpo;

    ###################
    Vf1=ℂ[FermionNumber](-1=>1);
    Vf2=ℂ[FermionNumber](0=>1,1=>1);
    Vf3=ℂ[FermionNumber](-1=>1);
    Vf4=ℂ[FermionNumber](0=>1,1=>1);
    mpo = TensorMap(randn, Vf1 ⊗ Vf2 ← Vf3 ⊗ Vf4);

    mpo_dict=convert(Dict,mpo)

    m=mpo_dict[:data]["FermionNumber(0)"];
    m[1]=1;
    mpo_dict[:data]["FermionNumber(0)"]=m;

    m=mpo_dict[:data]["FermionNumber(-1)"];
    m[1]=1;
    mpo_dict[:data]["FermionNumber(-1)"]=m;

    mpo=convert(TensorMap, mpo_dict);
    mpo_middle=mpo;

    ###################
    Vf1=ℂ[FermionNumber](-1=>1);
    Vf2=ℂ[FermionNumber](0=>1,1=>1);
    Vf3=ℂ[FermionNumber](0=>1);
    Vf4=ℂ[FermionNumber](0=>1,1=>1);
    mpo = TensorMap(randn, Vf1 ⊗ Vf2 ← Vf3 ⊗ Vf4);

    mpo_dict=convert(Dict,mpo)
    m=mpo_dict[:data]["FermionNumber(0)"];
    m[1]=1;
    mpo_dict[:data]["FermionNumber(0)"]=m;
    mpo=convert(TensorMap, mpo_dict);
    mpo_x2=mpo;

    ###################
    mpo_right=mpo_left;
    

    mpo_set=Vector(undef,L);
    for cc=1:x1-1
        mpo_set[cc]=mpo_left;
    end
    mpo_set[x1]=mpo_x1;
    for cc=x1+1:x2-1
        mpo_set[cc]=mpo_middle;
    end
    mpo_set[x2]=mpo_x2;
    for cc=x2+1:L
        mpo_set[cc]=mpo_right;
    end

    #from looking at swap gate diagram one can see an extra -1 is needed
    mpo_set[x1]=-mpo_set[x1];

    return mpo_set

end

function overlap_1D(mps1,mps2)
    mps1=deepcopy(mps1);
    mps2=deepcopy(mps2);
    @tensor Left[:]:=mps1[1]'[-1,1,2]*mps2[1][2,-2,1];
    for cc=2:length(mps1)-1
        @tensor Left[:]:=Left[1,2]*mps1[cc]'[-1,3,1]*mps2[cc][2,-2,3];
    end
    @tensor Left[:]:=Left[1,2]*mps1[end]'[4,3,1]*mps2[end][2,4,3];
    return blocks(Left)[FermionNumber(0)][1]
end