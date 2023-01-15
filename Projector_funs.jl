function projector_virtual(V)
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

    end


    return P_odd,P_even
end


function projector_physical(V)
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


    end

    return P_odd,P_even

end

