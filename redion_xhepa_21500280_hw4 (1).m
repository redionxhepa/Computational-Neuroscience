
%Computational Neuroscience Homework 4  by Redion Xhepa
function redion_xhepa_21500280_hw4(question)
clc
close all

switch question
    case '1'    
%Question 1 Hw4
disp('Question 1 ');
disp('Note: If the order of the plots seems meaningless rerun the code.Matlab sometimes acts weird in my Linux OS.');
disp('All the plot are labelled,check the code for consistency');
load hw4_data1.mat;
disp('*********************');
disp('*********************');
disp('Question 1 part a');
disp('Plot of variance explained and plot of the first 25 PCs');
[rows,columns] = size(faces);
%calculate PCA 
[coeff,score,latent,tsquared,explained,mu]=pca(faces); %directly taken from Matlab documentation
%Plot the porpotion of the variance explained
 figure;
 plot(explained(1:100,:));
 xlabel(' ');
 ylabel('Proportion of variance ');
 title('Proportion of variance explained by each individual PC');
%Display the first 25 PC
 figure;
 dispImArray(coeff(:,1:25)');
  title('The first 25 PCs');

disp('*********************');
disp('*********************');
disp('Question 1 part b');

% %reconstruction made with  10 PCs
% 
% %The idea is simple  T=XW  where T is scores,X is our original data,W are
% %the loadings. As result if we want to get X again we have X=T*transpose(W)
% 
 resconst_10 = score(:,1:10)*coeff(:,1:10)'+mean(faces); %means should be added since PCA demeans the data
% %reconstruction made with  25 PCs
resconst_25 = score(:,1:25)*coeff(:,1:25)'+mean(faces);
%reconstruction made with  50 PCs
resconst_50 = score(:,1:50)*coeff(:,1:50)'+mean(faces);

%Display the first 36 real images 
figure;
dispImArray(faces(1:36,:));
title('The first 36 faces (real ones)' );

%Display the first 36  reconstructed images using 10 PCs
figure;
dispImArray(resconst_10(1:36,:));
title('The first 36 faces recostructed using 10 PC' );

%Display the first 36  reconstructed images using 25 PCs
figure;
dispImArray(resconst_25(1:36,:));
title('The first 36 faces recostructed using 25 PC' );

%Display the first 36  reconstructed images using 50 PCs
figure;
dispImArray(resconst_50(1:36,:));
title('The first 36 faces recostructed using 50 PC' );
%calculate error square
error_square_10=(faces-resconst_10).^2;
error_square_25=(faces-resconst_25).^2;
error_square_50=(faces-resconst_50).^2;

% %Mean squared error
mserror_10= mean(error_square_10,2);
mserror_25= mean(error_square_25,2);
mserror_50=mean(error_square_50,2);

%Calculate the means and the std-s
  mean_10=mean(mserror_10);
  std_10=std(mserror_10);
%  
 mean_25=mean(mserror_25);
 std_25=std(mserror_25);
 %
  mean_50=mean(mserror_50);
  std_50=std(mserror_50);
  
 print1=["Mean of MSE for 10 PC is ",num2str(mean_10)];
 print2= ["Std of MSE for 10 PC  is  ",num2str(std_10)];
  print3=["Mean of MSE for 25 PC  is  ",num2str(mean_25)];
 print4= ["Std of MSE for 25 PC  is  ",num2str(std_25)];
  print5=["Mean of MSE for 50 PC is   ",num2str(mean_50)];
 print6= ["Std of MSE for 50 PC is  ",num2str(std_50)];
 disp(print1);
 disp(print2);
 disp(print3);
 disp(print4);
 disp(print5);
 disp(print6);
  
 disp('*********************');
disp('*********************');
disp('Question 1 part c');

%For IC=10 and reconstruct it
[icasig_10, mixing_matrix_10, seperating_matrix_10] = fastica(faces, 'lastEig', 50, 'numOfIC', 10, 'verbose', 'off');
reconstruction_ic_10 = mixing_matrix_10 * icasig_10;

%For IC=25 and reconstruct it
[icasig_25, mixing_matrix_25, seperating_matrix_25] = fastica(faces, 'lastEig', 50, 'numOfIC', 25, 'verbose', 'off');
reconstruction_ic_25 = mixing_matrix_25 * icasig_25;

%For IC=50 and reconstruct it
[icasig_50, mixing_matrix_50, seperating_matrix_50] = fastica(faces, 'lastEig', 50, 'numOfIC', 50, 'verbose', 'off');
reconstruction_ic_50 = mixing_matrix_50 * icasig_50;


%calculate error square
error_square_10_ic=(faces-reconstruction_ic_10).^2;
error_square_25_ic=(faces-reconstruction_ic_25).^2;
error_square_50_ic=(faces-reconstruction_ic_50).^2;


%Mean squared error
mserror_10_ic= mean(error_square_10_ic,2);
mserror_25_ic= mean(error_square_25_ic,2);
mserror_50_ic=mean(error_square_50_ic,2);

%Calculate the means and the std-s
  mean_10_ic=mean(mserror_10_ic);
  std_10_ic=std(mserror_10_ic);
 
 mean_25_ic=mean(mserror_25_ic);
 std_25_ic=std(mserror_25_ic);
 
  mean_50_ic=mean(mserror_50_ic);
  std_50_ic=std(mserror_50_ic);
  
 print1_ic=["Mean of MSE for 10 IC is ",num2str(mean_10_ic)];
 print2_ic= ["Std of MSE for 10 IC  is  ",num2str(std_10_ic)];
  print3_ic=["Mean of MSE for 25 IC  is  ",num2str(mean_25_ic)];
 print4_ic= ["Std of MSE for 25 IC is  ",num2str(std_25_ic)];
  print5_ic=["Mean of MSE for 50 IC is   ",num2str(mean_50_ic)];
 print6_ic= ["Std of MSE for 50 IC is  ",num2str(std_50_ic)];
 disp(print1_ic);
 disp(print2_ic);
 disp(print3_ic);
 disp(print4_ic);
 disp(print5_ic);
 disp(print6_ic);
 display the plots 
 
%Display the first 36 original photos again for part 1.c
figure;
dispImArray(faces(1:36,:));
title('The first 36 original faces' );
 

 %Display the first 36  reconstructed images using 10 ICs
figure;
dispImArray(reconstruction_ic_10(1:36,:));
title('The first 36 faces recostructed using 10 IC' );
 

%Display the first 36  reconstructed images using 10 ICs
figure;
dispImArray(reconstruction_ic_25(1:36,:));
title('The first 36 faces recostructed using 25 IC' );


%Display the first 36  reconstructed images using 10 ICs
figure;
dispImArray(reconstruction_ic_50(1:36,:));
title('The first 36 faces recostructed using 50 IC' );

 disp('*********************');
disp('*********************');
disp('Question 1 part d');


%calculate the scalar term so to have strictly positive matrix
scalar_constant=min(min(faces));
strictly_nnegativ_data=faces-scalar_constant;

%For NNMF=10 and reconstruct
[factor1_10, factor2_10] = nnmf(strictly_nnegativ_data,10);
reconstruction_nnmf_10= factor1_10 * factor2_10 + scalar_constant;

%For NNMF=25 and reconstruct
[factor1_25, factor2_25] = nnmf(strictly_nnegativ_data,25);
reconstruction_nnmf_25= factor1_25 * factor2_25 + scalar_constant;

%For NNMF=50 and reconstruct
[factor1_50, factor2_50] = nnmf(strictly_nnegativ_data,50);
reconstruction_nnmf_50= factor1_50 * factor2_50 + scalar_constant;

%calculate error square
error_square_10_nnmf=(faces-reconstruction_nnmf_10).^2;
error_square_25_nnmf=(faces-reconstruction_nnmf_25).^2;
error_square_50_nnmf=(faces-reconstruction_nnmf_50).^2;


%Mean squared error
mserror_10_nnmf= mean(error_square_10_nnmf,2);
mserror_25_nnmf= mean(error_square_25_nnmf,2);
mserror_50_nnmf=mean(error_square_50_nnmf,2);

%Calculate the means and the std-s
  mean_10_nnmf=mean(mserror_10_nnmf);
  std_10_nnmf=std(mserror_10_nnmf);
 
 mean_25_nnmf=mean(mserror_25_nnmf);
 std_25_nnmf=std(mserror_25_nnmf);
 
  mean_50_nnmf=mean(mserror_50_nnmf);
  std_50_nnmf=std(mserror_50_nnmf);
  
 print1_ic=["Mean of MSE for 10 nnmf is ",num2str(mean_10_nnmf)];
 print2_ic= ["Std of MSE for 10 nnmf  is  ",num2str(std_10_nnmf)];
  print3_ic=["Mean of MSE for 25 nnmf  is  ",num2str(mean_25_nnmf)];
 print4_ic= ["Std of MSE for 25 nnmf is  ",num2str(std_25_nnmf)];
  print5_ic=["Mean of MSE for 50 nnmf is   ",num2str(mean_50_nnmf)];
 print6_ic= ["Std of MSE for 50 nnmf is  ",num2str(std_50_nnmf)];
 disp(print1_ic);
 disp(print2_ic);
 disp(print3_ic);
 disp(print4_ic);
 disp(print5_ic);
 disp(print6_ic);
 %display the plots 
 
%Display the first 36 original photos again for part 1.d
figure;
dispImArray(faces(1:36,:));
title('The first 36 original faces' );
 

 %Display the first 36  reconstructed images using 10 nnmf
figure;
dispImArray(reconstruction_nnmf_10(1:36,:));
title('The first 36 faces recostructed using 10 nnmf' );
 

%Display the first 36  reconstructed images using 25 nnmf
figure;
dispImArray(reconstruction_nnmf_25(1:36,:));
title('The first 36 faces recostructed using 25 nnmf' );

%Display the first 36  reconstructed images using 50 nnmf
figure;
dispImArray(reconstruction_nnmf_50(1:36,:));
title('The first 36 faces recostructed using 50 nnmf' )
     
    case '2'
     disp('Question 2 ');
     %Computational Neuroscience H4 question 2

disp('Question 2');
load hw4_data2.mat;

disp('*********************');
disp('*********************');
disp('Question 2 part a');



%with Euclidean metric
confusion_matrix_eucld=zeros(181); %allocate the array for euclidean metric
for i=1:181
    for j=1:181
     confusion_matrix_eucld(i,j) = pdist2(vresp(i,:),vresp(j,:),'euclidean');
    end
end
% %with Cosine metric
confusion_matrix_cosine=zeros(181); %allocate the array for euclidean metric
for i=1:181
    for j=1:181
     confusion_matrix_cosine(i,j) = pdist2(vresp(i,:),vresp(j,:),'cosine');
    end
end
%%with correlation metric
confusion_matrix_correlation=zeros(181); %allocate the array for euclidean metric
for i=1:181
    for j=1:181
     confusion_matrix_correlation(i,j) = pdist2(vresp(i,:),vresp(j,:),'correlation');
    end
end

%Plot the reslting matrices with imagesc

imagesc(confusion_matrix_eucld);
title('similarity estimates based on euclidean distance metric');
xlabel('vresp_1' );
ylabel('vresp_2');
figure;
imagesc(confusion_matrix_cosine);
title('similarity estimates based on cosine distance metric');
xlabel('vresp_1' );
ylabel('vresp_2');
figure;
imagesc(confusion_matrix_correlation);
title('similarity estimates based on correlation distance metric');
xlabel('vresp_1' );
ylabel('vresp_2');

% disp('*********************');
% disp('*********************');
% disp('Question 2 part b');
confusion_matrix_eucld = pdist(vresp, 'euclidean');
confusion_matrix_cosine = pdist(vresp, 'cosine');
confusion_matrix_correlation = pdist(vresp, 'correlation');

%perform multidimensiona scaling analysis on the data with Euclidean
%distance
 [twoD_data_euclid, ~]=cmdscale(confusion_matrix_eucld,2);  %in two dimensions 

%perform multidimensiona scaling analysis on the data with cosine
%distance
 [twoD_data_cosine, ~]=cmdscale(confusion_matrix_cosine,2);  %in two dimensions 

 %perform multidimensiona scaling analysis on the data with correlation
%distance
 [twoD_data_correlation, ~]=cmdscale(confusion_matrix_correlation,2);  %in two dimensions 


 %Scatter MDS  using Euclidean distance
 figure;
 h1 = gscatter(twoD_data_euclid(:,1),twoD_data_euclid(:,2),stype,'krb','ov^',[],'off');
grid on ;
legend('Group1','Group2','Location','best')
title('Scatter MDS  using Euclidean distance');
figure;

 %Scatter MDS  using cosine distance
 h1 = gscatter(twoD_data_cosine(:,1),twoD_data_cosine(:,2),stype,'krb','ov^',[],'off');
grid on ;
legend('Group1','Group2','Location','best')
title('Scatter MDS  using cosine distance');
figure;
 %Scatter MDS  using correlation distance
 h1 = gscatter(twoD_data_correlation(:,1),twoD_data_correlation(:,2),stype,'krb','ov^',[],'off');
grid on ;
legend('Group1','Group2','Location','best')
title('Scatter MDS  using correlation distance');

disp('*********************');
disp('*********************');
disp('Question 2 part c');

%perform k-means using euclidean distance
kmean_euclid = kmeans(twoD_data_euclid, 2);
%perform k-means using cosine distance
kmean_cosine = kmeans(twoD_data_cosine, 2);
%perform k-means using correlation distance
kmean_correlation = kmeans(twoD_data_correlation,2);

%make plots
 figure;
 h1 = gscatter(twoD_data_euclid(:,1),twoD_data_euclid(:,2),kmean_euclid,'krb','ov^',[],'off');
 grid on ;
 title('k-means using euclidean distance');
 
  figure;
 h1 = gscatter(twoD_data_cosine(:,1),twoD_data_cosine(:,2),kmean_cosine,'krb','ov^',[],'off');
 grid on ;
 title('k-means using cosine distance');
 
  figure;
 h1 = gscatter(twoD_data_correlation(:,1),twoD_data_correlation(:,2),kmean_correlation,'krb','ov^',[],'off');
 grid on ;
 title('k-means using correlation distance');
     case '3'
       disp('Question 3 ');
     case '4'
         
         disp('Note:Sometimes for nothing Matlab gives stupid errors.Please run the code for the second time');
disp('Question 4');
disp('*********************');
disp('*********************');
disp('Question 4 part a');
%Define the parameters
A=1;  %amplitude
neurons=21; %21 neurons in total
standart_dev=1;
mu_s=linspace(-10,10,21); %create evenly space means between -10 and 10
f_i=[];
x=-10:0.1:10;
maxes=[];
indeces_max=[];
%Superimposed plots of the tuning curves
for i=1:21
    f_i(i,:)=A.*exp(-(x-mu_s(i)).^2./(2*standart_dev.^2));
    [~ ,m ]=max(f_i(i,:));
    maxes(i)=x(m);
    plot(x,f_i);
    hold on; 
end
title('Tuning curves');
xlabel('Stimulus');
ylabel('Response');

figure;
%Population response to stimulus x=-1
for i=1:21
    f_1(i,:)=A.*exp(-(-1-mu_s(i)).^2./(2*standart_dev.^2));
end
 plot(maxes,f_1,'o-');
title('Response for x=-1');
xlabel('Stimulus');
ylabel('Response');

disp('*********************');
disp('*********************');
disp('Question 4 part b');
trials=200;
response=zeros(trials,21); %allocate the response matrix
stimulus_matrix=zeros(1,trials); %store the stimuli
%Find the responses and corrupt responses
for j=1:trials
     stimulus = -5 + 10.*rand();
     stimulus_matrix(1,j)=stimulus;
     for i=1:21
      response(j,i)= A.*exp(-(stimulus-mu_s(i)).^2./(2*standart_dev.^2));
     end
end
corrupted_response=response+(1/20).*randn(trials,21) + 0; %add mean 0 and standart_dev=1/20

%winner-take-all decoder
neuron_indeces_highest=[];
for i=1:trials
    [~ , index]=max(corrupted_response(i,:));
    neuron_indeces_highest=[neuron_indeces_highest index];
end

%find the optimal responses for the x-s in range -5 and 5 
f_i_new=[];
x_new=-5:0.1:5;
maxes_new=[];
%Superimposed plots of the tuning curves
for i=1:21
    f_i_new(i,:)=A.*exp(-(x_new-mu_s(i)).^2./(2*standart_dev.^2));
    [~ ,m ]=max(f_i_new(i,:));
    maxes_new(i)=x_new(m);
end

stimulus_estimate=maxes_new(neuron_indeces_highest);
 plot(stimulus_estimate);
 hold on;
plot(stimulus_matrix);
legend('Estimated Stimulus','Real Stimulus');
grid on ;

%calculate mean and standart_dev of the error estimation
error=stimulus_matrix-stimulus_estimate;
mean_b=mean(error);
standart_dev_b=std(error);
output_b=['mean for Winner-take-all error is ',num2str(mean_b),' std  for Winner-take-all error is ',num2str(standart_dev_b)];
disp(output_b);
title('part b');

disp('*********************');
disp('*********************');
disp('Question 4 part c');

%to be checked again
ML=[]; 
for i=1:trials
    ML(i) = MLE_est(corrupted_response(i,:),mu_s);
end
figure;
 plot(ML);
 hold on;
plot(stimulus_matrix);
legend('Estimated Stimulus','Real Stimulus');
grid on ;
disp('mean is ');
mean(stimulus_matrix-ML)
std(stimulus_matrix-ML)
title('Part c');



disp('*********************');
disp('*********************');
disp('Question 4 part d');
print_e=['sigma  0.1  Mean 0.42,Std 4.2'];
print_f=['sigma  0.2  Mean 0.26,Std 3.39'];
print_g=['sigma 0.5   Mean 0.11,Std 1.68'];
print_h=['sigma  1   Mean -0.05,Std 0.82'];
print_i=['sigma  2  Mean 0.00026,Std 0.079'];
print_j=['sigma  5   Mean 0.0007,Std 0.13'];
print_k=['sigma  0.1  Mean 0.42,Std 4.2'];
MAP=[];
for i=1:200
     MAP(i) = map(corrupted_response(i,:),mu_s);
   
end


figure;
 plot(MAP);
 hold on;
plot(stimulus_matrix);
legend('Estimated Stimulus','Real Stimulus');
grid on ;
disp('mean is ');
mean(stimulus_matrix-MAP)
disp('std is ')
std(stimulus_matrix-MAP)
title('Part d');

disp('*********************');
disp('*********************');
disp('Question 4 part e');

std_1 = [0.1,0.2,0.5, 1, 2, 5];
linear_sample=linspace(-5,5,200);
for k=1:6
   %Find the responses and corrupt responses
   trials=200;
   response=zeros(trials,21); %allocate the response matrix
   stimulus_matrix=zeros(1,trials); %store the stimuli
   standart_dev=std_1(k);
  for j=1:trials
     stimulus = linear_sample(j);
     stimulus_matrix(1,j)=stimulus;
     for i=1:21
      response(j,i)= A.*exp(-(stimulus-mu_s(i)).^2./(2*standart_dev.^2));
     end
  end
  
  corrupted_response=response+(1/20).*randn(trials,21) + 0; %add mean 0 and standart_dev=1/20
   ML=[];
  for i=1:trials
    ML(i) = MLE_est(corrupted_response(i,:),mu_s);
  end

means(k)=mean(stimulus_matrix-ML);
%stds(k)=std(stimulus_matrix-ML); 
end

disp(print_e);
disp(print_f)
disp(print_g)
disp(print_h)
disp(print_i)
disp(print_j)
disp(print_k)

       
     case '5'
       
%Computational Neuroscience Hw4 question 5 

disp('*********************');
disp('*********************');
disp('Question 5 part a');
 load hw4_data4.mat;
%perform multidimensiona scaling analysis on the data with Euclidean
%distance
 confusion_matrix_eucld=zeros(181); %allocate the array for euclidean metric
 for i=1:181
     for j=1:181
      confusion_matrix_eucld(i,j) = pdist2(vresp(i,:),vresp(j,:),'euclidean');
    end
 end
 % confusion_matrix_eucld = pdist(vresp, 'euclidean');
 [twoD_data, stress]=cmdscale(confusion_matrix_eucld,2);  %in two dimensions 

%This is just a line by line folowing of the Matlab  documentation 
%https://www.mathworks.com/help/stats/create-and-visualize-discriminant-analysis-classifier.html
%Plot 
h1 = gscatter(twoD_data(:,1),twoD_data(:,2),stype,'krb','ov^',[],'off');
grid on ;
legend('Group1','Group2','Location','best')
hold on

%Create the linear classifier
X = [twoD_data(:,1),twoD_data(:,2)];
MdlLinear = fitcdiscr(X,stype);

%retrieve the coefficients for the linear boundary line
MdlLinear.ClassNames([1 2]);
K = MdlLinear.Coeffs(1,2).Const;
L = MdlLinear.Coeffs(1,2).Linear;
%Plot the curve that separates the first and second classes.
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h3 = ezplot(f,[-20 20 -20 20]);
h3.Color = 'k';
h3.LineWidth = 2;
axis([-15 15 -5 5])
xlabel('data1')
ylabel('data2')
title(' Linear Classification for euclidean metric ');

%1 leave out cross validation
%Cross validation leave 1 out
%backup 2d data and stype
temp_stype=stype;
temp_twoD_data=twoD_data;
 accurrate = zeros(1,181);
 for i=1:181
    linear_model= fitcdiscr(temp_twoD_data(2:end,:),temp_stype(2:length(temp_stype)));
    accurrate(i)= predict(linear_model,temp_twoD_data(1,:))==temp_stype(1);
    temp_twoD_data=circshift(temp_twoD_data,1); temp_stype=circshift(temp_stype,1);
 end
   disp('Accuracy for part a is ');
  accurracy_a=sum(accurrate)/181

disp('*********************');
disp('*********************');
disp('Question 5 part b');
load hw4_data4.mat;

confusion_matrix_corr_b = pdist(vresp, 'correlation');
twoD_data_b = cmdscale(confusion_matrix_corr_b, 2);

%This is just a line by line folowing of the Matlab  documentation 
%https://www.mathworks.com/help/stats/create-and-visualize-discriminant-analysis-classifier.html
%Plot 
figure;
h1_b = gscatter(twoD_data_b(:,1),twoD_data_b(:,2),stype,'krb','ov^',[],'off');
grid on ;
legend('Group1','Group2','Location','best')
hold on

%Create the linear classifier for correlation metric one
X_b = [twoD_data_b(:,1),twoD_data_b(:,2)];
MdlLinear_b = fitcdiscr(X_b,stype);

%retrieve the coefficients for the linear boundary line
MdlLinear_b.ClassNames([1 2]);
K_b = MdlLinear_b.Coeffs(1,2).Const;
L_b = MdlLinear_b.Coeffs(1,2).Linear;
%Plot the curve that separates the first and second classes.
f = @(x1,x2) K_b + L_b(1)*x1 + L_b(2)*x2;
h3 = ezplot(f,[-20 20 -20 20]);
h3.Color = 'k';
h3.LineWidth = 2;
axis([-0.2 0.7 -0.3 0.4])
xlabel('data1')
ylabel('data2')
title(' Linear Classification for correlation metric ');

% 1 leave out cross validation
% Cross validation leave 1 out
% backup 2d data and stype
temp_stype=stype;
temp_twoD_data=twoD_data_b;
 accurrate = zeros(1,181);
 %this code is based on this documentation of Mathworks
 %https://www.mathworks.com/help/stats/discriminant-analysis.html
 %
 for i=1:181
    linear_model= fitcdiscr(temp_twoD_data(2:end,:),temp_stype(2:length(temp_stype)));
    accurrate(i)= predict(linear_model,temp_twoD_data(1,:))==temp_stype(1);
    temp_twoD_data=circshift(temp_twoD_data,1); temp_stype=circshift(temp_stype,1);
 end
  
 disp('Accuracy for part b is ');
 accurracy_b=sum(accurrate)/181

  disp('*********************');
disp('*********************');
disp('Question 5 part c');

%shikoje prap se ka gabim me shume mundesi 
accuracy_matrix=zeros(1,5);
for i=1:5
    %store the temp data and temp stype
    confusion_matrix_corr_b = pdist(vresp, 'correlation');
    temp_twoD_data = cmdscale(confusion_matrix_corr_b, i);
   validation=fitcdiscr(temp_twoD_data,stype','leaveout','on');
   accuracy_matrix(1,i)=(1-kfoldLoss(validation));
end
figure;
bar(accuracy_matrix);

title('classification accuracy using leave-one-out cross validation');
xlabel('dimensions');
ylim([0.9 1]);
ylabel('accuracy(as ratio)');
end
end

function est = MLE_est(response, mus)
    x =[-5:0.01:5];
    likeL = [];
    for i = 1:length(x)
        l = 0;
        for j = 1:length(response)
            l = l - ((response(j) - exp(-((x(i) - mus(j))^2) / 2)) ^ 2);
        end
        likeL(i) = l;
    end
    [~,index] = max(likeL);
    est = x(index);
end

function est = map(response, mu)
    x = [-5:0.01:5];
    likelihood = [];
    for i = 1:length(x)
        l = 0;
        for j = 1:length(response)
            l = l - ((response(j) - exp(-((x(i) - mu(j))^2) / 2)) ^ 2);
        end
        likelihood(i) = l - (x(i)^2)/12.5;
    end
    [~,index] = max(likelihood);
    est = x(index);
end
       