# for (a,b) in blocks(W_set[1,1])
#     println(a)
#     println(b)
# end

function create_vaccum_mps(L)
    #create vaccum
    @assert mod(L,2)==0

    V=ℂ[U1Irrep](0=>1,1=>2,2=>1);
    V_R=ℂ[U1Irrep](0=>1);

    M=zeros(1,1,4)*im;M[1,1,1]=1;
    M=TensorMap(M, V_R ← V_R ⊗ V);
    mps_set=Array{Any}(undef, Int(L/2))
    for cc=1:Int(L/2)
        mps_set[cc]=M;
    end
    return mps_set
end

function create_mpo(W)
    size1=size(W)[1]#number of fermionic modes
    size2=size(W)[2]#number of particle
    @assert mod(size1,2)==0


    #V=ℂ[U1Irrep](0=>1,1=>2,2=>1);
    V=ℂ[U1Irrep](0=>1,1=>1);

    W_L=zeros(1,2)*im;W_L[1,2]=1;
    V_L=ℂ[U1Irrep](1=>1);
    W_L=TensorMap(W_L, V_L ←  V);

    W_R=zeros(2,1)*im;W_R[1,1]=1;
    V_R=ℂ[U1Irrep](0=>1);
    W_R=TensorMap(W_R, V ← V_R);

    
    W_set=Array{Any}(undef, size2,Int(size1/2))
    #element order after reshape: a[1,1]=<0,0>, a[2,1]=<up,0>, a[1,2]=<0,down>, a[2,2]=<up,down>
    for bb=1:size2
        for cc=1:size1/2
            W1=zeros(2,2,2,2)*im;
            W1[1,:,1,:]=Matrix(I,2,2);
            W1[2,:,1,:]=[0 0;1 0]'*W[Int(2*cc-1),bb];
            W1[2,:,2,:]=[1 0;0 -1];
            W1=TensorMap(W1,V ⊗ V ← V ⊗ V);


            W2=zeros(2,2,2,2)*im;
            W2[1,:,1,:]=Matrix(I,2,2);
            W2[2,:,1,:]=[0 0;1 0]'*W[Int(2*cc),bb];
            W2[2,:,2,:]=[1 0;0 -1];
            W2=TensorMap(W2,V ⊗ V ← V ⊗ V);
            
            U=unitary(fuse(space(W1,2)⊗space(W1,2)),space(W1,2)⊗space(W1,2));
            @tensor T[:]:=W1[-1,1,5,3]*W2[5,2,-3,4]*U[-2,1,2]*U'[3,4,-4];            

            if cc==1
                @tensor T[:]:=W_L[-1,1]*T[1,-2,-3,-4];
            elseif cc==Int(size1/2)
                @tensor T[:]:=T[-1,-2,1,-4]*W_R[1,-3];
            end
            W_set[bb,Int(cc)]=T;

        end
    end
    return W_set

end


function mpo_mps(mpo_set,mps_set)
    mpo_set=deepcopy(mpo_set);
    mps_set=deepcopy(mps_set);
    L=length(mps_set);
    @assert length(mpo_set)==length(mps_set)
    for cc=1:L
        @tensor mps[:]:=mpo_set[cc][-1,1,-3,-5]*mps_set[cc][-2,-4,1];
        mps_set[cc]=mps;
    end
    #fuse legs
    UL=unitary(fuse(space(mps_set[1],1)⊗space(mps_set[1],2)),space(mps_set[1],1)⊗space(mps_set[1],2));
    @tensor mps[:]:=mps_set[1][1,2,-3,-4,-5]*UL[-1,1,2];
    mps_set[1]=mps;

    for cc=1:L-1
        UL=unitary(fuse(space(mps_set[cc+1],1)⊗space(mps_set[cc+1],2)),space(mps_set[cc+1],1)⊗space(mps_set[cc+1],2));
        # println(UL)
        # println(UL')
        # println(UL*UL')
        @tensor mps[:]:=mps_set[cc][-1,1,2,-3]*UL'[1,2,-2];
        mps_set[cc]=mps;
        @tensor mps[:]:=mps_set[cc+1][1,2,-2,-3,-4]*UL[-1,1,2];
        mps_set[cc+1]=mps;
    end

    UR=unitary(fuse(space(mps_set[end],2)⊗space(mps_set[end],3)),space(mps_set[end],2)⊗space(mps_set[end],3));
    @tensor mps[:]:=mps_set[end][-1,1,2,-3]*UR[-2,1,2];
    mps_set[end]=mps;

    for cc=1:L
        mps_set[cc]=permute(mps_set[cc],(1,),(2,3,));
    end

    return mps_set
end

function overlap_1D(mps1,mps2)
    mps1=deepcopy(mps1);
    mps2=deepcopy(mps2);
    @tensor Left[:]:=mps1[1]'[-1,1,2]*mps2[1][2,-2,1];
    for cc=2:length(mps1)-1
        @tensor Left[:]:=Left[1,2]*mps1[cc]'[-1,3,1]*mps2[cc][2,-2,3];
    end
    @tensor Left[:]:=Left[1,2]*mps1[end]'[4,3,1]*mps2[end][2,4,3];

    return blocks(Left)[Irrep[U₁](0)][1]
end