function [X_train,y_train,X_test,y_test ] = load_spam_dataset(num_trs,num_tes)


data_folder = 'data/'; %directory where mnist data are stored
file = 'Spam_Data.txt'; % file name for domain values of training images. This is expected to be a Nx784 matrix, where N is the number of samples.

data = csvread([data_folder, file]);

X = data(:,[5 7 16 17 19 21 23 25 52 53 57]);
y = data(:,58);

X = standardise(X);


%Split into training and testing data
perm_idxs = randperm(length(y));
perm_idxs = perm_idxs(1:(num_trs + num_tes));
train_idxs = perm_idxs(1:num_trs);
test_idxs = perm_idxs((num_trs + 1):end);

X_train = X(train_idxs,:);
y_train = y(train_idxs,:);
y_train(y_train == 0 ) = -1;

X_test = X(test_idxs,:);
y_test = y(test_idxs,:);
y_test(y_test == 0 ) = -1;


end