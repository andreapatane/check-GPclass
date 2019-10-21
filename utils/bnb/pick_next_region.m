function [flagBreak,c_r_i_cell,ubs,lbs,region,x_overs,curr_x_over,lb_b4_split,ub_b4_split,mu_sigma_bounds_array,mu_sigma_bounds_curr,min_lbs_init] = pick_next_region(ubstar,region,lbs,ubs,x_overs,mu_sigma_bounds_array,tollerance)

%best-bound-first
count = 0;

min_lbs_init =  min(cell2mat(lbs));
aux = inf;
while aux > (ubstar - tollerance)
    if isempty(region)
        flagBreak = true;
        break;
    else
        flagBreak = false;
    end
    [~,idx] = min(cell2mat(lbs));
    lb_b4_split = lbs{idx};
    ub_b4_split = ubs{idx};
    %[~,idx] = max(cell2mat(lbs));
    %[~,idx] = min(cell2mat(ubs));
    %disp(['lbs: ', num2str((cell2mat(lbs)))])
    %disp(['ubs: ', num2str((cell2mat(ubs)))])
    c_r_i_cell = splitregion(region{idx});
    curr_x_over = x_overs{idx};
    mu_sigma_bounds_curr = mu_sigma_bounds_array{idx};
    aux = lbs{idx};
    region(idx) = [];
    ubs(idx) = [];
    lbs(idx) = [];
    %disp('After picking')
    %disp(['lbs: ', num2str((cell2mat(lbs)))])
    %disp(['ubs: ', num2str((cell2mat(ubs)))])
    x_overs(idx) = [];
    mu_sigma_bounds_array(idx) = [];
    if false && (count > 1)
        disp('killed in pick')
    end
    count = count + 1;
end










end


function c_r_i_cell = splitregion(reg)

[~,split_idx] =  max(reg(2,:) - reg(1,:));

%split_idx = randi(size(reg,2));
mb = 0.5*(reg(1,split_idx) + reg(2,split_idx));

c_r_i_cell{1} = reg;
c_r_i_cell{2} = reg;
c_r_i_cell{1}(2,split_idx) = mb;
c_r_i_cell{2}(1,split_idx) = mb;


end