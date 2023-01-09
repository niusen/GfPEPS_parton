function QN_str_search(Str)
    Leftb=Str[2];
    Rightb=Str[end];
    left_pos=[];
    right_pos=[];
    L=length(Str);
    for cc=1:L
        if Str[cc]==Leftb
            # println(cc)
            left_pos=vcat(left_pos,cc)
        end
    end

    for cc=1:L
        if Str[cc]==Rightb
            # println(cc)
            right_pos=vcat(right_pos,cc)
        end
    end

    xx=string(Irrep[U₁](1) ⊠ Irrep[SU₂](1/2));
    Slash=xx[end-3];
    slash_pos=[];
    for cc=1:L
        if Str[cc]==Slash
            # println(cc)
            slash_pos=vcat(slash_pos,cc)
        end
    end

    return left_pos,right_pos,slash_pos
end

function Get_Vspace_Qn(V1)
    Qnlist1=[];
    
    for s in sectors(V1)
        #println(string(s))
        st=replace(string(s), "Irrep[U₁]" => "a");
        st=replace(st, "⊠ Irrep[SU₂]" => "a");
        #println(st)
        left_pos,right_pos,slash_pos=QN_str_search(string(st));

        Qn=parse(Int64, st[left_pos[1]+1:right_pos[1]-1])

        #println(Spin)
        Dim=dim(V1, s)

        Qnlist1=vcat(Qnlist1,Int.(ones(Dim))*Qn);
        
    end
    return Qnlist1
end




function gauge_gate(A,p1,phase)
    V1=space(A,p1);
    S=unitary( V1, V1);


    S_dense=convert(Array,S)*(1+0*im);
    oddlist1=Get_Vspace_Qn(V1);
    for c1=1:length(oddlist1)

        S_dense[c1,c1]=exp(oddlist1[c1]*im*phase);
 
    end
    S=TensorMap(S_dense,V1 ← V1);
    return S
end
