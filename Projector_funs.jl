function projector_virtual(V)
    VV1=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)'
    VV2=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (-2, 0)=>3, (-4, 0)=>1, (-1, 1/2)=>2, (-3, 1/2)=>2, (-2, 1)=>1)
    
    if V==Rep[U₁](0=>1, 1=>1)
        P_odd=Vector(undef,1);
        P_even=Vector(undef,1);

        M=zeros(1,2)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁](0=>1),V);
        P_even[1]=T;

        M=zeros(1,2)*im;
        M[1,2]=1;
        T=TensorMap(M,Rep[U₁](1=>1),V);
        P_odd[1]=T;
    elseif V==Rep[U₁](0=>1, -1=>2, -2=>1)
        P_odd=Vector(undef,1);
        P_even=Vector(undef,2);

        M=zeros(1,4)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁](0=>1),V);
        P_even[1]=T;

        M=zeros(1,4)*im;
        M[1,4]=1;
        T=TensorMap(M,Rep[U₁](-2=>1),V);
        P_even[2]=T;

        M=zeros(2,4)*im;
        M[1,2]=1;
        M[2,3]=1;
        T=TensorMap(M,Rep[U₁](-1=>2),V);
        P_odd[1]=T;
    elseif V==VV1
        P_odd=Vector(undef,1);
        P_even=Vector(undef,2);

        M=zeros(1,4)*im;
        M[1,1]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1)',V);
        P_even[1]=T;

        M=zeros(1,4)*im;
        M[1,2]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((2, 0)=>1)',V);
        P_even[2]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1, 1/2)=>1)',V);
        P_odd[1]=T;
    elseif V==VV2
        P_odd=Vector(undef,2);
        P_even=Vector(undef,4);

        M=zeros(1,16)*im;
        M[1,1]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1),V);
        P_even[1]=T;

        M=zeros(3,16)*im;
        M[1,2]=1;
        M[2,3]=1;
        M[3,4]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2, 0)=>3),V);
        P_even[2]=T;

        M=zeros(1,16)*im;
        M[1,5]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-4, 0)=>1),V);
        P_even[3]=T;

        M=zeros(3,16)*im;
        M[1,14]=1;
        M[2,15]=1;
        M[3,16]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2, 1)=>1),V);
        P_even[4]=T;

        M=zeros(4,16)*im;
        M[1,6]=1;
        M[2,7]=1;
        M[3,8]=1;
        M[4,9]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-1, 1/2)=>2),V);
        P_odd[1]=T;

        M=zeros(4,16)*im;
        M[1,10]=1;
        M[2,11]=1;
        M[3,12]=1;
        M[4,13]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-3, 1/2)=>2),V);
        P_odd[2]=T;

    end


    return P_odd,P_even
end

function projector_virtual_devided(V)
    VV1=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)'
    VV2=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (-2, 0)=>3, (-4, 0)=>1, (-1, 1/2)=>2, (-3, 1/2)=>2, (-2, 1)=>1)
    
    if V==VV1
        Ps=Vector(undef,3);

        M=zeros(1,4)*im;
        M[1,1]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1)',V);
        Ps[1]=T;

        M=zeros(1,4)*im;
        M[1,2]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((2, 0)=>1)',V);
        Ps[2]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1, 1/2)=>1)',V);
        Ps[3]=T;
    elseif V==VV2
        Ps=Vector(undef,6);

        M=zeros(1,16)*im;
        M[1,1]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1),V);
        Ps[1]=T;

        M=zeros(3,16)*im;
        M[1,2]=1;
        M[2,3]=1;
        M[3,4]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2, 0)=>3),V);
        Ps[2]=T;

        M=zeros(1,16)*im;
        M[1,5]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-4, 0)=>1),V);
        Ps[3]=T;

        M=zeros(3,16)*im;
        M[1,14]=1;
        M[2,15]=1;
        M[3,16]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2, 1)=>1),V);
        Ps[4]=T;

        M=zeros(4,16)*im;
        M[1,6]=1;
        M[2,7]=1;
        M[3,8]=1;
        M[4,9]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-1, 1/2)=>2),V);
        Ps[5]=T;

        M=zeros(4,16)*im;
        M[1,10]=1;
        M[2,11]=1;
        M[3,12]=1;
        M[4,13]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-3, 1/2)=>2),V);
        Ps[6]=T;

    end


    return Ps
end


function projector_physical(V)
    VV1=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>3, (2, 0)=>1, (-2, 0)=>1, (1, 1/2)=>2, (-1, 1/2)=>2, (0, 1)=>1)
    if V==Rep[U₁](0=>2, 1=>1, -1=>1)

        M=zeros(2,4)*im;
        M[1,1]=1;
        M[2,2]=1;
        T=TensorMap(M,Rep[U₁](0=>2),V);
        P_even=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,Rep[U₁](1=>1,-1=>1),V);
        P_odd=T;
    elseif V==VV1
        M=zeros(8,16)*im;
        M[1,1]=1;
        M[2,2]=1;
        M[3,3]=1;
        M[4,4]=1;
        M[5,5]=1;
        M[6,14]=1;
        M[7,15]=1;
        M[8,16]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>3, (2, 0)=>1, (-2, 0)=>1, (0, 1)=>1),V);
        P_even=T;

        M=zeros(8,16)*im;
        M[1,6]=1;
        M[2,7]=1;
        M[3,8]=1;
        M[4,9]=1;
        M[5,10]=1;
        M[6,11]=1;
        M[7,12]=1;
        M[8,13]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1, 1/2)=>2, (-1, 1/2)=>2),V);
        P_odd=T;

    end

    

    return P_odd,P_even

end

