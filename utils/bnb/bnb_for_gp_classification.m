function [out_L, out_U, count, exitFlag] = bnb_for_gp_classification(testIdx,x_L,x_U,params_for_gp_toolbox,max_or_min,mode,bound_comp_opts,varargin)

% --exitFlags--
% 1 - terminating condition met by comparing lb_b4_star to ubstar
% 11 - terminating condition met while picking next region
% 12 - terminating condition met because no more regions to check
% 2 - region minimum size undercut
% 3 - max iterations reached
% 4 - branch and bound stagnated


%parsing varargin for the input I need in the specific mode required. Those
%are GP specific vectors/matrixes that are needed for the bound computation
%switch mode
%    case 'binarypi'
%        trainedSystem = varargin{1}; %I can get the outputIdx of this
%        prior to entering here!!!
%        S = varargin{2};
%    case 'binarymu'
%        trainedSystem = varargin{1};
%end

%the pixels I want to modify are found by checking which hyper-rectangle
%side is not trivial
aux_stats = tic;
numOfIterB4Display = 1000;


pixels2modify = bound_comp_opts.pix_2_mod;
%and this is used to introduce compact representation for the regions (as trivial sides are easily reconstructed from x_L)
region = { [x_L(pixels2modify);x_U(pixels2modify)] };

%The initial region of branch and bound is trivially the whole region [x_L,x_U]
[c_x_l,c_x_u] = expand_domain_variables(region{1},pixels2modify,x_L);

%I compute lower and upper bounds on the current region, as well as a point
%solution of the over-approximated problem (this can be iteratively used to start off search again in the same region...)

%Something of the form:
if (isfield(params_for_gp_toolbox,'covfunc') && isequal(params_for_gp_toolbox.covfunc,@covSEard)) || (isfield(params_for_gp_toolbox,'kernel') &&  isequal(params_for_gp_toolbox.kernel,'sqe'))
    kernel_name = 'sqe';
end
%[lbstar,ubstar,x_over_star] = compute_curr_lb_ub(max_or_min,mode,...
%    kernel_name,c_x_l,c_x_u,training_data,training_labels,params_for_gp_toolbox,bound_comp_opts,varargin{:});

bound_comp_opts.iteration_count = 0;
[lbstar,ubstar,x_over_star,mu_sigma_bounds,bound_comp_opts] = compute_curr_lb_ub(testIdx,max_or_min,mode,...
    kernel_name,c_x_l,c_x_u,params_for_gp_toolbox,bound_comp_opts,[],struct(),varargin{:});
bound_comp_opts.iteration_count = bound_comp_opts.iteration_count + 1;


if strcmp(max_or_min,'max')
    %lbstar = - lbstar;
    %ubstar = - ubstar;
    temp = - lbstar;
    lbstar = - ubstar;
    ubstar = temp;
end


%I save current lower annd upper bounds
ubs = {ubstar};
lbs = {lbstar};
x_overs = {x_over_star};
mu_sigma_bounds_array = {mu_sigma_bounds};
%if lower and upper bounds are epsilon-close, then the search is over!
if check_terminating_condition(lbstar,ubstar,bound_comp_opts.tollerance,bound_comp_opts.min_region_size,max(c_x_u - c_x_l))
    region = {};
    exitFlag = 1;
end



%here I start the while loop exploring the tree space associated to branch
%and bound...

str_old = ' ';
disp(str_old)
flagBreak = false;
avg_improv = 0;
while ~isempty(region) && bound_comp_opts.iteration_count < bound_comp_opts.max_iterations && ~flagBreak
    
    %I pick the next next region to split, and I split it in two sub-regions
    [flagBreak,c_r_i_cell,ubs,lbs,region,x_overs,curr_x_over,lb_b4_split,ub_b4_split,mu_sigma_bounds_array,mu_sigma_bounds_curr,min_lbs_init] = pick_next_region(ubstar,region,lbs,ubs,x_overs,mu_sigma_bounds_array,bound_comp_opts.tollerance);
    if flagBreak
        lbstar = max(lbstar,min_lbs_init);
        disp(' ')
        disp('--------------------------------------------------------------------------------------------------------------')
        disp('Terminating condition met while picking next region!')
        disp('--------------------------------------------------------------------------------------------------------------')
        disp(' ')
        exitFlag = 11;
        break;
    end
    
    
    %performing lower and upper bounding on the freshly generated two
    %subregions in c_r_i_cell
    for ii = 1:length(c_r_i_cell) %this counter only ever reaches 2 as a consequence
        
        [c_x_l,c_x_u] = expand_domain_variables(c_r_i_cell{ii},pixels2modify,x_L);  
        %computing region diameter for stats purposes...
        rs = max(c_r_i_cell{ii}(2,:) - c_r_i_cell{ii}(1,:));
        %bounding
        [c_lb,c_ub,x_over,mu_sigma_bounds] = compute_curr_lb_ub(testIdx,max_or_min,mode,...
            kernel_name,c_x_l,c_x_u,params_for_gp_toolbox,bound_comp_opts,curr_x_over,mu_sigma_bounds_curr,varargin{:});
        
        if strcmp(max_or_min,'max')
            temp = - c_lb;
            c_lb = - c_ub;
            c_ub = temp;
        end 
        if c_ub <= ubstar
            ubstar = c_ub;
            lbstar = c_lb;
        end
        
        %fathoming...
        %closing region if lower and upper bound are within tolerance
        if ~check_terminating_condition(c_lb,ubstar,bound_comp_opts.tollerance,bound_comp_opts.min_region_size,rs)
            %if region is still worth looking at, I stack it.
            region{end+1} = c_r_i_cell{ii};
            ubs{end+1} = c_ub;
            lbs{end+1} = c_lb;
            x_overs{end+1} = x_over;
            mu_sigma_bounds_array{end+1} = mu_sigma_bounds;
        end
        avg_improv = avg_improv + abs(c_lb - lb_b4_split);

        
        
        %end
        
        
        if  check_terminating_condition(lb_b4_split,ubstar,bound_comp_opts.tollerance,bound_comp_opts.min_region_size,rs) || (isempty(region) && (ii ==2))
            lbstar = max(lbstar,lb_b4_split);
            flagBreak = true;
            disp(' ')
            disp('--------------------------------------------------------------------------------------------------------------')
            disp('Terminating condition met!')
            disp('--------------------------------------------------------------------------------------------------------------')
            disp(' ')
            if isempty(region)
                exitFlag = 12;
            elseif (rs < bound_comp_opts.min_region_size)
                exitFlag = 2;
            else
                exitFlag = 1;
            end
            break
        else
            if true
                %bound_comp_opts.iteration_count
                
                if ~mod(bound_comp_opts.iteration_count,numOfIterB4Display) || (bound_comp_opts.iteration_count >= (bound_comp_opts.max_iterations - 1))
                    strig_c_lb = sprintf('%.4f',c_lb);
                    strig_c_ub = sprintf('%.4f',c_ub);
                    strig_ubstar = sprintf('%.4f',ubstar);
                    string_lb_b4_split = sprintf('%.4f',lb_b4_split);
                    string_rs = sprintf('%.4f',rs);
                    mar = sprintf('%.4f',ubstar-lb_b4_split);
                    elapsTime = toc(aux_stats);
                    string_time = sprintf('%.4f',elapsTime);
                    aux_stats = tic;

                    str_old = ['c_lb: ', strig_c_lb, '; c_ub: ', strig_c_ub, ...
                        '; rs: ', string_rs ...
                        '; lb_b4: ',string_lb_b4_split,'; ubstar: ', strig_ubstar,'; stack: ', int2str(length(region))...
                        ,'; iter: ', int2str(bound_comp_opts.iteration_count) , '; elapsed: ', string_time ,' -- Current margin: ', mar, ' --'  ];
                    disp(str_old)     
                    if ii == 2
                        avg_improv = avg_improv/(2*numOfIterB4Display);
                        %disp(['Current average improvement: ', num2str(avg_improv) ])
                        if avg_improv < 1.0e-08
                            if bound_comp_opts.var_ub_every_NN_iter == realmax
                                bound_comp_opts.var_ub_start_at_iter = bound_comp_opts.iteration_count;
                                bound_comp_opts.var_ub_every_NN_iter = 1;
                                %disp('The search currently stagnated - I will start updating the variance upper bound now and hope for the best...')
                            else
                                disp(' ')
                                disp(['--------------------------------------------------------------------------------------------------------------',...
                                    '--------------------------------------------------------------------------------------------------------------'])
                                disp(['Branch and Bound search has stagnated, probably because bad bounds on the variance. If the bound obtained ',...
                                    'is not good enough, Consider re-running the optimization, updating the variance bound more often, or modifying the discretisation'])
                                disp(['--------------------------------------------------------------------------------------------------------------',...
                                    '--------------------------------------------------------------------------------------------------------------'])
                                disp(' ')
                                exitFlag = 4;
                                flagBreak = true;
                                break
                            end
                        else
                            avg_improv = 0;
                        end
                    end
                    
                end
            end
            
        end
    end
    bound_comp_opts.iteration_count = bound_comp_opts.iteration_count + 1;
end

if  bound_comp_opts.iteration_count >= bound_comp_opts.max_iterations
    disp(' ')
    disp('--------------------------------------------------------------------------------------------------------------')
    disp('Maximum Number of Iterations Exceeded');
    disp('--------------------------------------------------------------------------------------------------------------')
    disp(' ')
    exitFlag = 3;
else 
    disp(['Done. Final number of iterations: ',int2str(bound_comp_opts.iteration_count)])
end

if strcmp(max_or_min,'max')
    out_U = - lbstar;
    out_L = - ubstar;
else
    out_U =  ubstar;
    out_L =  lbstar;
end
count = bound_comp_opts.iteration_count;
end

function boolValue = check_terminating_condition(lb,ub,tollerance,min_region_size,rs)
boolValue = (lb > (ub - tollerance))  || (rs < min_region_size)  ;
end