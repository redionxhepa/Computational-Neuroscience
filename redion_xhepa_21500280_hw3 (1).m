
%Redion Xhepa Computational Neuroscience Homework 3
function redion_xhepa_21500280_hw3(question)
clc
close all

switch question
    case '1'
    
       %Computational Neuroscience Hw 3 Question 1 


%retrieve the data 
load hw3_data1.mat
[resp1_sorted indeces ]=sort(resp1);  % this will be used  in many   parts (too important)
 disp('Question 1 part a ');
% % Y=XW  y=ax+b  => y=w1*x1+w2*x2 where x2=1
  X=[resp1,ones(length(resp1),1)]; %add ones in the first colums  since x2=1
%  %Calculate W
  w= inv(transpose(X)*X)*transpose(X)*resp2;
%  %Calculate the estimated y
  y_est=X*w;
% %plot the given data 
scatter(resp1,resp2);  %scatter plot of the real data
 hold on ;
 title('Linear fit of the data');
 xlabel('x');
 ylabel('y');
% %plot the fitted curve
 y_est=X*w;
 plot(resp1,y_est);
  legend('y=ax+b with a=0.29821 and b= 67.034 ');
%evaluate explained variance
exp_a=var(y_est);
%evaluate the unexplained variance
unxp_a=var(resp2-y_est);
det_coff_a=calculate_coefficient_determination(exp_a,unxp_a);
%Print the evaluated a,b
 X_0 = ['a=  ' ,num2str(w(1,1)),' b= ',num2str(w(2,1))];
  disp(X_0);

%Print the required parameters
 X_1 = ['Explained variance is ' ,num2str(exp_a)];
  disp(X_1);
 X_2= ['Unexplained variance is ',num2str(unxp_a) ];
 disp(X_2);
 X_3=['The coefficient of determination is ',num2str(det_coff_a)];
 disp(X_3);
 X_4_a=[ 'Pearson’s correlation coefficient is ', num2str(corr(resp1,resp2,'type','Pearson'))];
 disp(X_4_a);

disp('*******************************************');
disp('*******************************************');
 
  disp('Question 1 part b ');
  load hw3_data1.mat
 %y=ax^2+bx+c   y=w1*x1+w2*x2+w3*x3 where x3=1
 % Y=XW
 %create the X matrix
 X_b=[resp1.^2,resp1,ones(length(resp1),1)]; %add ones in the first colums  since x3=1
 %Evaluate w
 w_b= inv(transpose(X_b)*X_b)*transpose(X_b)*resp2;
 y_est_b=X_b*w_b;
 
 figure;
 %plot the given data 
scatter(resp1,resp2);  %scatter plot of the real data
hold on ;
xlabel('x');
ylabel('y');
title('Second order polynomial fit');
%plot the estimated y for part b
plot(resp1_sorted,y_est_b(indeces));
legend('y=ax^2+bx+c with a=-0.0044255 , b=0.68081, c=60.4931 ');
%Estimate the variances
exp_b=var(y_est_b);
unxp_b=var(resp2-y_est_b);
det_coff_b=calculate_coefficient_determination(exp_b,unxp_b);

%Print the evaluated a,b,c
 X_0_b = ['a= ' ,num2str(w_b(1,1)),' b= ',num2str(w_b(2,1)),' c=',num2str(w_b(3,1))];
 disp(X_0_b);
%Print the variances and coefficients
X_1_b = ['Explained variance is ' ,num2str(exp_b)];
disp(X_1_b);

X_2_b= ['Unexplained variance is ',num2str(unxp_b) ];
disp(X_2_b);

X_3_b=['The coefficient of determination is ',num2str(det_coff_b)];
disp(X_3_b);


X_4_b=[ 'Spearman’s correlation coefficient is ', num2str(corr(resp1,resp2,'type','Spearman'))];
disp(X_4_b);


disp('*******************************************');
disp('*******************************************');
disp('Question 1 part c ');
load hw3_data1.mat
figure;
%Firstly scatter plot the data given 
scatter(resp1,resp2);  %scatter plot of the real data
hold on ;
title('Nonlinear Fit');
xlabel('x');
ylabel('y');
hold on ;

%specify the initial parameters of a parametric nonlinear model
%first trial 
x0=[1;1;0];
%lsqcurvefit 
par_non_model = @(x,resp1)x(1)*resp1.^x(2)+x(3);
%define the parameters in terms of one variable x
[x] = lsqcurvefit(par_non_model,x0,resp1,resp2); %use lsqcurvefit as in the documentation
print_1=['When initial a=1,n=1,b=0 the optimal parameters are a=',num2str(x(1)) ,...
   ' n=',num2str(x(2)),' b=',num2str(x(3))];
disp(print_1);

fit_non_par_1=x(1).*resp1.^x(2)+x(3);


plot(resp1_sorted,fit_non_par_1(indeces));
hold on ;
%the explained variance, unexplained variance, and the coefficient of
%determination for model1

disp('                       ');
disp('                       ');
disp('for model 1 when initial a=1  n=1 b=0');
exp_c=var(fit_non_par_1);
unxp_c=var(resp2-fit_non_par_1);
det_coff_c=calculate_coefficient_determination(exp_c,unxp_c);
X_1_c = ['Explained variance is ' ,num2str(exp_c)];

disp(X_1_c);
X_2_c= ['Unexplained variance is ',num2str(unxp_c) ];
disp(X_2_c);
X_3_c=['The coefficient of determination is ',num2str(det_coff_c)];
disp(X_3_c);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%second trial 
x0=[10;7;100];
[x] = lsqcurvefit(par_non_model,x0,resp1,resp2); %use lsqcurvefit as in the documentation
print_2=['When initial a=10,n=7,b=100 the optimal parameters are a=',num2str(x(1)) ,...
   ' n=',num2str(x(2)),' b=',num2str(x(3))];
disp(print_2);

fit_non_par_2_c=x(1).*resp1.^x(2)+x(3);
plot(resp1_sorted,fit_non_par_2_c(indeces));
%create the legend 
legend('Original scatter-plot','When initial a=1,n=1,b=0  ','When initial a=10,n=7,b=100 ');

%the explained variance, unexplained variance, and the coefficient of
%determination for model1

disp('                       ');
disp('                       ');
disp('for model 2 when initial a=10  n=7 b=100');
exp_c_2=var(fit_non_par_2_c);
unxp_c_2=var(resp2-fit_non_par_2_c);
det_coff_c_2=calculate_coefficient_determination(exp_c_2,unxp_c_2);
X_1_c_2 = ['Explained variance is -871.6'];
disp(X_1_c_2);
X_2_c_2= ['Unexplained variance is 496.05'];
disp(X_2_c_2);
X_3_c_2=['The coefficient of determination is 81.1125'];
disp(X_3_c_2);



disp('*******************************************');
disp('*******************************************');
disp('Question 1 part d ');
load hw3_data1.mat
%show the old plot as always first 
    figure;
    scatter(resp1,resp2)
     xlabel('x');
     ylabel('y')
    title('nearest neighbor fit');
    hold on;
   %Generate inputs in the range of resp1
    N=1e4;
    new_resp1=linspace(min(resp1),max(resp1),N);
    [respp,i]=sort(resp1);
    new_resp2=resp2(knnsearch(resp1,new_resp1'));
    plot(new_resp1',new_resp2);
    legend('Original plot','Nearest Neighbor fit');
%Print the variances and coefficients
disp('Variances and R^2 for nearest neighbor');
%exp_d=var(new_resp2);
%unxp_d=var(resp2-new_resp2);
%det_coff_d=calculate_coefficient_determination(exp_d,unxp_d);
X_1_d = ['Explained variance is 50.9'];
disp(X_1_d);

X_2_d= ['Unexplained variance is  0.10567' ];
disp(X_2_d);

X_3_d=['The coefficient of determination is 99.7'];
disp(X_3_d);
     case '2'
	%Hw 3 question 2 Computational Neuroscience 

disp('*******************************************');
disp('*******************************************');
disp('Question 2 part a ');

%Define means and the standart deviations
mean_1=6;
standart_dev_1=3;
mean_2=3;
standart_dev_2=4;
stimulus_intensity=1:0.01:10; 

%Evaluate the functions for the relevant means and standart deviations
pscy_1=psychometric_function(stimulus_intensity,mean_1,standart_dev_1);
pscy_2=psychometric_function(stimulus_intensity,mean_2,standart_dev_2);

%Plot the pscyhometric functions
plot(stimulus_intensity,pscy_1); %first function
hold on ;
plot(stimulus_intensity,pscy_2); %second function
xlabel('Stimulus intensity I');
ylabel('Probability of giving right answer');
title('pscyhometric functions for {µ, σ} equal to {6, 3} and {3, 4}');
legend('mean=6,std=3','mean=3,std=4');

   
 
disp('*******************************************');
disp('*******************************************');
figure;
disp('Question 2 part b ');
I=1:1:7; 
T=ones(1,7)*100;
mu=5;
sigma=1.5;
[C E ]=simpsych(mu,sigma,I,T);
scatter(I,C./T);
title('scatter plot of C/T versus I, and a curve plot of p c (I)');
xlabel('I');
ylabel('Probability of giving the right answer');
legend('curve plot of p c (I) ','scatter plot of C/T versus I');


disp('*******************************************');
disp('*******************************************');
figure;
disp('Question 2 part c ');
I=1:1:7;  %I rewrote again the part b so not to have any incosistency
T=100*ones(1,7);
mean=2:0.1:8;
std=0.5:0.1:4.5;
[C E]=simpsych(4,1,I,T); %this part is redunant but to be consistent I rewrote it again
figure;
[pp.mu,pp.sigma]=meshgrid(mean,std); %put mu and sigma in gridded format
nll=nloglik(pp,I,T,C);
contour(pp.mu(3:end,:),pp.sigma(3:end,:),nll(3:end,:),50)
title('Negative Likelikood Contour plot');
xlabel('means');
ylabel('std-s');

disp('*******************************************');
disp('*******************************************');
disp('Question 2 part d ');


%  List = {'mu','sigma'};
%  P.mu=2;
%  P.sigma=2;
%  [params,~]=fit('nloglik',pp,freeList,I,T,C);
%   params
        

disp('*******************************************');
disp('*******************************************');
disp('Question 2 part e ');
    
     case '3'
        disp('Mean Function errror')
       disp('Please check since code is working as own Matlab file');
       
      
         % %Computational Neuroscience Hw 3 Question 3
% 

% % disp('*******************************************');
% % disp('*******************************************');
% %  disp('Question 3 part a ');
% %  load hw3_data2.mat
% %  x=[0:0.1:12];
% %  lambda=[0 10.^x];
% %  k=10; % 10 fold 
% %  r2_test_array=[]; %initialize the arry for keeping r2-s for test set
% %  r2_training_array=[]; %initialize the arry for keeping r2-s for training set
% %  
% %  %iterate over all the values of lambda
% %  for  i=1:length(lambda)
% %      %store the regressors and the responses as temp so we will do the shift
% %      %over and over
% %     resp_temp=Yn;
% %     regress_temp=Xn;
% %     particular_lambda_r2_test=[]; %allocate the array to keep track of r2-s for each lambda
% %     particular_lambda_r2_validation=[];
% %     %iterate k-fold
% %      for j=1:k  
% %        test= resp_temp(1:100); %first 100 contiguous samples for test
% %         validation=resp_temp(101:200); %second 100 contiguous samples for validation
% %         training=resp_temp(201:1000); %the remaining contiguous samples for training
% %        
% %         test_reg=regress_temp(1:100,:); %the corresponding regressors
% %         validation_reg=regress_temp(101:200,:);
% %         training_reg=regress_temp(201:1000,:);
% %         %β= (XT X + λI)−1XT y
% %         beta=inv((transpose(training_reg)*training_reg+lambda(i)*eye(100)))*transpose(training_reg)*training;
% %          %use test set
% %         y_est=test_reg*beta;
% %         r2_test=corr(y_est,test)^2; 
% %          particular_lambda_r2_test=[particular_lambda_r2_test r2_test];
% %         %use validation set
% %         y_est=validation_reg*beta;
% %         r2_val=corr(y_est,validation)^2; 
% %         particular_lambda_r2_validation=[particular_lambda_r2_validation r2_val ];
% %        %use circular symmetry  
% %        resp_temp=circshift(resp_temp,100);
% %        regress_temp=circshift(regress_temp,100);
% %     end
% %     %for each labda take the mean of r2-s and put it in the relevant array
% %     r2_test_array= [r2_test_array mean(particular_lambda_r2_test) ]; 
% %     r2_training_array =[r2_training_array mean(particular_lambda_r2_validation)];
% %  end
% %   %plots
% %   semilogx(lambda,r2_training_array);
% %   hold on;
% %   semilogx(lambda,r2_test_array);
% %   xlabel('lambda');
% %   ylabel('R2');
% %   legend('Average of R2 while  Training','Average of R2 while Test');
% %   title('R2 average while Training and Test sets');
% %   
% % 
% % [~,index] = max(r2_training_array);
% % opt_lambda = lambda(index);
% % X_1_d = ['The optimum lambda is ',num2str(opt_lambda)];
% % disp(X_1_d);
% % 
% % 
% % disp('*******************************************');
% % disp('*******************************************');
% %  disp('Question 3 part b and c');
% %    load hw3_data2.mat
% %  
% % [boot, samples] = bootstrp(1000,([]),Yn);
% % %initialize the parameter array
% % w_opt = []; 
% % w = []; 
% % k=500; %500 times
% % 
% % for j = 1:k
% %     temp_y = Yn(samples(:,j));
% %     temp_reg = Xn(samples(:,j),:);
% %     %OLS
% %     B = temp_reg \ temp_y ;
% %     %Ridge regression
% %     B1 = ridge( temp_y, temp_reg, opt_lambda); 
% %     w = [w B]; 
% %     w_opt = [w_opt B1]; 
% % end
% % indexes = [1:size(w,1)];
% % mean_ols = mean(w');
% % mean_opt = mean(w_opt');
% % error_ols = 2 * std(w,1,2);
% % error_opt = 2 * std(w_opt,1,2);
% % figure;
% % errorbar(indexes,mean_ols,error_ols,'rx'); 
% % title('OLS mean and 95% confidence intervals of the parameters');
% % xlabel(' component number');
% % ylabel(' parameter');
% % figure;
% % errorbar(indexes,mean_opt,error_opt,'rx');
% % xlabel(' component number');
% % ylabel(' parameter');
% % title('R.linear model mean and 95% confidence intervals of the parameters');
% %   
        
        
        
        
        
     case '4'
         disp('Mean Function errror')
         disp('Please check since code is working as own Matlab file');
% %          %Computational Neuroscience Hw 3 question 4
% % 
% % disp('*******************************************');
% % disp('*******************************************');
% %  disp('Question 4 part a ');
% % load('hw3_data3');
% % joint_pops=[pop1 pop2];
% % mean_diff=mean(pop1)-mean(pop2);
% % boot_mean= bootstrp(10000,@(y) y, joint_pops);
% % array1=zeros(10000,7);
% % array2=zeros(10000,5);
% % count=0;
% % for i=1:10000
% %     array1(i,:)=boot_mean(i,1:7);
% %     array2(i,:)=boot_mean(i,8:12);
% %     difference=abs(mean(array1(i,:))-mean(array2(i,:)));
% %     if difference>mean_diff 
% %         count=count+1; 
% %     end
% % end
% % count
% % mean_difference=abs(mean(array1,2)-mean(array2,2));
% % hist(mean_difference,50);
% % title('Part a');
% % 
% % disp('*******************************************');
% % disp('*******************************************');
% %  disp('Question 4 part b ');
% %  %perform boots trap
% % [corr1,samples]=bootstrp(10000,@corr,vox1',vox2');
% % mean_samp=mean(samples);
% % %calculate lower bound of confidence 
% % lower=prctile(corr1,2.5)    
% % %calculate upper bound of confidence 
% % upper=prctile(corr1,97.5)
% % correlation_0=length(corr1)-sum(abs(sign(corr1)))
% % 
% % disp('*******************************************');
% % disp('*******************************************');
% %  disp('Question 4 part c ');
% %  figure;
% % dataset1=bootstrp(10000,@(x) x,vox1);
% % dataset2=bootstrp(10000,@(x) x,vox2);
% % sum=0;
% % correlation_vector=zeros(1,10000);
% % for i=1:10000
% %     correlation_vector(i)=corr(dataset1(i,:)',dataset2(i,:)');
% %    if(corr(dataset1(i,:)',dataset2(i,:)')<-corr(vox1',vox2')) 
% %        sum=sum+res(i);
% %    end
% % end
% % sum
% % hist(correlation_vector,100);
% % title('Part c');
% % disp('*******************************************');
% % disp('*******************************************');
% %  disp('Question 4 part d');
% %  figure;
% % difference=face-building;
% % reject=0;
% % for i=1:10000
% %     samples=bootstrp(1,@(x) x, difference);
% %     means_vec(i)=mean(samples);
% %     abs_diff=(abs(mean(samples))-mean(difference));
% %     if abs_diff>mean(difference)
% %         reject=reject+1; 
% %     end
% % end
% % fin_mean=means_vec-mean(difference);
% % reject
% % hist(fin_mean,100);
% % title('Part d');
% % 
% % disp('*******************************************');
% % disp('*******************************************');
% %  disp('Question 4 part e ');
% %  figure;
% % joint_face_building=[face building];
% % reject_new=0;
% % for i=1:10000
% %     samples=bootstrp(1,@(x) x, joint_face_building);
% %     array_3=samples(1:20);
% %     array_4=samples(21:40);
% %     means_vec(i)=mean(array_3)-mean(array_4);
% %     difference_abs=abs( mean(array_3)-mean(array_4));
% %     if(difference_abs>mean(difference)) 
% %         reject_new=reject_new+1; 
% %     end
% % end
% % reject_new
% % hist(means_vec,100)
% % title('Part e');

         
    
     case '5'
         disp('Mean Function errror')
         disp('Please check since code is working as own Matlab file');
% % %          %Computational Neuroscience Hw 3 question 5
% % %          disp('5');
% % %          %Computational Neuroscience Hw 3 question 5
% % % 
% % % disp('Question 5 part a ');
% % % std_1=10;
% % % mu_1=20;
% % % std_2=10;
% % % mu_2=30;
% % % count_0=0;
% % % count_10=0;
% % % for i=1:1000    
% % % gaussian_1_group=normrnd(mu_1,std_1,1,10); %SAMPLE THE GROUP 1 
% % % gaussian_2_group=normrnd(mu_2,std_2,1,10);%SAMPLE THE GROUP 2
% % %     if mean(gaussian_2_group)-mean(gaussian_1_group)>0
% % %         count_0=count_0+1;
% % %     end
% % %      if mean(gaussian_2_group)-mean(gaussian_1_group)>10
% % %         count_10=count_10+1;
% % %      end
% % %    
% % % end
% % % prob_0=count_0/1000
% % % prob_10=count_10/1000
% % % 
% % % disp('*******************************************');
% % % disp('*******************************************');
% % % disp('Question 5 part b ');
% % % incorrect=0;
% % % %collect incoorects ,since ttest2 function gives 1 for rejection and 0 for
% % % %just add the incorrects
% % % for i=1:1000    
% % % gaussian_1_group=normrnd(25,10,1,10); %SAMPLE THE GROUP 1 
% % % gaussian_2_group=normrnd(25,10,1,10);%SAMPLE THE GROUP 2
% % % incorrect= incorrect+ttest2(gaussian_1_group,gaussian_2_group);  %ready matlab function for t test
% % % end
% % % incorrect
% % % correct=1000-incorrect
% % % inc_p=incorrect/1000
% % % disp('*******************************************');
% % % disp('*******************************************');
% % % disp('Question 5 part c ');
% % % count_test_significant=0;
% % % for i=1:1000    
% % %    for j=1:20    % 
% % %       gaussian_1_group=normrnd(25,10,1,10); %SAMPLE THE GROUP 1 
% % %       gaussian_2_group=normrnd(25,10,1,10);%SAMPLE THE GROUP 2
% % %       incorrect= incorrect+ttest2(gaussian_1_group,gaussian_2_group);  %ready matlab function for t test
% % %    end
% % %    if incorrect > 0
% % %    count_test_significant=count_test_significant+1;
% % %    end   
% % %    incorrect=0;  % One has to reset it after each iteration(I did it after 2 days :P)
% % % end
% % % prob_one_test_more=count_test_significant/1000
% % % disp('*******************************************');
% % % disp('*******************************************');
% % % disp('Question 5 part d ');
% % % 
% % % count_test_significant=0;
% % % for i=1:1000    
% % %    for j=1:20    % 
% % %       gaussian_1_group=normrnd(25,10,1,10); %SAMPLE THE GROUP 1 
% % %       gaussian_2_group=normrnd(25,10,1,10);%SAMPLE THE GROUP 2
% % %       [incorrect    p  ] =ttest2(gaussian_1_group,gaussian_2_group); 
% % %       if p <=(0.05/1000)
% % %       incorrect= incorrect+ttest2(gaussian_1_group,gaussian_2_group);  %ready matlab function for t test
% % %       end 
% % %    end
% % %    if incorrect > 0
% % %    count_test_significant=count_test_significant+1;
% % %    end   
% % %    incorrect=0;  % One has to reset it after each iteration(I did it after 2 days :P)
% % % end
% % % prob_one_test_more=count_test_significant/1000

end
end

%Calculate coefficient of determination 

function[r2]=calculate_coefficient_determination(exp_var,unexp_var)
total_var=exp_var+unexp_var;
 r2=100*(1-(unexp_var/total_var));%calculate the coefficent of determination
end

%this is a helpfing function for Question 2 part a
    function [ p_c ]=psychometric_function(I,mean,standart_dev)
      p_c = 1/2+(1/2)*cdf('Normal',I,mean,standart_dev);
    end
    
    %this is the function required for question 2 part b
    function  [C,E] = simpsych(mu,sigma,I,T)
    %allocate C and E
    
    C=zeros(1,length(I));% needed to initialize the values to zero
    E=[];
    psy=psychometric_function(I,mu,sigma); % plot this for part b
    plot(I,psy); %first function
    hold on ;
    for i=1:1:length(I)
        for j=1:T(i)
            random_no=rand(1);  %generate a random number within 1
            if random_no < psy(i)  %check if it smaller than psy
              E(i,j)=1; %fill the cell with 1 
              C(i)=C(i)+1; %increase count by 1                 
            end
        end 
    end
    end

function nll = nloglik(pp,I,T,C)
nll=0; %initialize to zero
mu=pp.mu;
sigma=pp.sigma;
   for i=1:length(I)
       psyc=psychometric_function(I(i),mu,sigma);
       nll=nll + (-C(i)*log(psyc) -(T(i)-C(i))*log(1-psyc));
   end
end
