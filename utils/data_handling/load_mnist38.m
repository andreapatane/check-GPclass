function [X_train,y_train,X_test,y_test ] = load_mnist38(num_trs,num_tes,downsamplingMode)



%gp_training_opts.data_folder = get_data_folder();
data_folder = 'data/'; %directory where mnist data are stored
train_file = 'mnist38_train.csv'; % file name for domain values of training images. This is expected to be a Nx784 matrix, where N is the number of samples.
test_file = 'mnist38_test.csv'; % file name for labels of testing images.

X = csvread([data_folder, train_file]);
training_inputs = X(1:num_trs,2:785)./255;
training_labels = X(1:num_trs,1);
training_labels = (((training_labels.*2)-6)./5)-1;
if downsamplingMode
    [training_inputs,~]  = averagePooling(training_inputs); %subsampling pixels by a factor of 2.
end
test_data = csvread([data_folder, test_file]);
test_inputs = test_data(1:num_tes,2:785)./255;
test_labels = test_data(1:num_tes,1);
test_labels = (((test_labels.*2)-6)./5)-1;
if downsamplingMode
    [test_inputs,~]  = averagePooling(test_inputs); %subsampling pixels by a factor of 2.
end
X_train = training_inputs;

%size(X_train,2)

y_train = training_labels;
X_test = test_inputs;
y_test = test_labels;

y_train(y_train == 0 ) = -1;

y_test(y_test == 0 ) = -1;



end
