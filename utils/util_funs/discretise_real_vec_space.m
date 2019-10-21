function multi_grid = discretise_real_vec_space(N,lowers,uppers,dims)
%a = linspace(-4,4,N);
%a(1) = -realmax;
%a(end) = realmax;
if dims ~= 3
    error('implementation done only for dims=3')
end
%multi_grid = zeros(N*ones(1,dims));
multi_grid = cell((N+1)^dims,1);
tts = zeros(N+2,dims);
for ii = 1:dims
    tts(:,ii) = [-realmax,linspace(lowers(ii),uppers(ii),N),realmax]; 
end

ii = 1;
for i1 = 1:(N+1)
    for i2 = 1:(N+1)
        for i3 = 1:(N+1)
            multi_grid{ii} = [tts(i1,1),tts(i2,2),tts(i3,3);tts(i1+1,1),tts(i2+1,2),tts(i3+1,3)];
            ii = ii + 1;
        end
    end
end


end

