function [X_train,y_train,X_test,y_test ] = load_mnist358(num_trs,num_tes)




%gp_training_opts.data_folder = get_data_folder();
data_folder = 'data/'; %directory where mnist data are stored
train_file = 'mnist358_train.csv'; % file name for domain values of training images. This is expected to be a Nx784 matrix, where N is the number of samples.
test_file = 'mnist358_test.csv'; % file name for labels of testing images.

X = csvread([data_folder, train_file]);
training_inputs = X(1:num_trs,2:785)./255;
training_labels = X(1:num_trs,1);
[training_inputs,~]  = averagePooling(training_inputs); %subsampling pixels by a factor of 2.

test_data = csvread([data_folder, test_file]);
test_inputs = test_data(1:num_tes,2:785)./255;
test_labels = test_data(:,1);
[test_inputs,~]  = averagePooling(test_inputs); %subsampling pixels by a factor of 2.
 
X_train = training_inputs;


y_train = full(ind2vec((training_labels+1)'))';

X_test = test_inputs;
y_test = full(ind2vec((test_labels+1)'))';
y_test = y_test(1:num_tes,:);




end
