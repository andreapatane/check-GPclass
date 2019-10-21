function open_region = fathoming(c_lb,ubstar)
if c_lb > ubstar
    open_region = false;
    %         if debug
    %             disp('killed in fathoming')
    %         end
else
    open_region = true;
end

end