function [X_train,y_train,X_test,y_test] = generate_2d_synthetic_datasets(num_trs,num_tes)

% Generate toy data

nn = 600;

X1 = randn(nn,2);
y1 = ones(nn,1);
X2 = randn(nn,2);
y2 = zeros(nn,1);
X1(:,1) = X1(:,1) + 5;
X2(:,2) = X2(:,2) + 5;
X = [X1;X2];
y = [y1;y2];


%%
X = standardise(X);


%Split into training and testing data
%test_split = 0.2;
perm_idxs = randperm(length(y));
perm_idxs = perm_idxs(1:(num_trs + num_tes));
%train_idxs = perm_idxs(1: floor((1-0.2)*length(y)) );
%test_idxs = perm_idxs(floor((1-0.2)*length(y) + 1) : end );
train_idxs = perm_idxs(1:num_trs);
test_idxs = perm_idxs((num_trs + 1):end);

X_train = X(train_idxs,:);
y_train = y(train_idxs,:);
y_train(y_train == 0 ) = -1;

X_test = X(test_idxs,:);
y_test = y(test_idxs,:);
y_test(y_test == 0 ) = -1;


end