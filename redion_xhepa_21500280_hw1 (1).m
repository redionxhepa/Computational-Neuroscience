%Redion Xhepa
%21500280



function redion_xhepa_21500280_hw1(question)
clc
close all

switch question
    case '1'
	disp('1')
        %% question 1 code goes here
 %Computational NeuroScience HW1 Question 1
%Define the matrix A given
A=[1 0 -1 2;2 1 -1 5;3 3 0 9];

%Part A 
%Find null space solution
disp("Question 1 Part A " )
 null_space_solutions=null(A,'r');
 disp('The null space solution of A is composed of :(as shown below)');
 disp(null_space_solutions);

%Part B
%Test the particular solution found analytically
disp("Question 1 Part B " )
 x=[2 1 1 0]';  % a particular solution
 disp(' [2 1 1 0] is a particular solution');
 result=A*x
 disp('A*x^T is as expected [1 4 9 ]');

%Part C
%There is no matlab validation(it comes from out knowledge of Lin.Algeb.

%Part D
disp("Question 1 Part D " )
pseudo_inverse=pinv(A);
disp('Pseudo-inverse of A is');
pseudo_inverse
%Part E
b = [1;4;9];disp("Sparsest solution is ");disp(A\b);
%Part F(least norm solution
disp("Question 1 Part F " )
disp("Least norm solution is");
disp(pinv(A)*b);

  case '2'
	disp('2')
    
%Part a 
disp(" Question 2: Part A");
 A=[1.2969,0.8648;0.2161,0.1441];
 disp('Determinant of matrix A');
 det(A) %find the determinant of  matrix A;
 disp('Inverse of matrix A');
 inv(A) %find the inverse of matrix A

%Part b
disp(" Question 2: Part B");
A_rounded=roundUpToThree(A); % create the matrix which its elements are 
                             %the rounded up to three decimal digits
  disp('Determinant of matrix A rounded');                            
det(A_rounded) %find the determinant of  matrix A;
disp('Inverse of matrix A rounded');
inv(A_rounded) %find the inverse of matrix A

%Part c
%At first we type this sequence of commands
disp(" Question 2: Part C");
 format long
 A = hilb(5);
 m = eig(A);
v = ones(5,1);
w = v./norm(v);
% 
% %Variables needed to keep track of the number of iterations
 count=0;  %initialize a counter
 epsilon=0.0000000000000000000001; % to serve as  an indicator when value will not change
 old_ma=0;  %store old ma to be compared with the new value,initialized randomly with 0
 ma=0; %store ma,initialized randomly with 0
difference=1;   %start it as 1 just to skip first loop
% 
 while abs(difference)>epsilon
   old_ma=ma; %pass the calculated ma  and stored as an old value
  count=count+1; %increase the count
 
  v = A*w; %perform the operations required(as told in the question)
  w = v./norm(v);
 ma = w'*A*w;
 difference=ma-old_ma; % difference is taken between the old value and the new one
 end
 X = ['Iterations done are ',num2str(count),' and final value of ma is  ',num2str(ma)];
 disp(X); %show the value of the counter and the final value of ma
 disp('Below will be printed the array which contains the eigenvalues of matrix A');
 disp(m);
 disp(' ma=1.56 is the greatest eigenvalue');


%Part d(we base upon the code of part c)

%At first we type this sequence of commands
disp(" Question 2: Part D");
format long
A = pascal(5);
m = eig(A);
m = flipud(m);


% %Variables needed to keep track of the number of iterations
 count_e=0;  %initialize a counter
 epsilon=0.0000000000000000000001; % to serve as  an indicator when value will not change
 old_e=0;  %store old ma to be compared with the new value,initialized randomly with 0
 e=0; %store e,initialized randomly with 0
 difference_e=1;   %start it as 1 just to skip first loop

while abs(difference_e)>epsilon
old_e=e;  %store the old value of e 
count_e=count_e+1; %increase the counter by 1

[Q,R] = qr(A);
A = R*Q;
ma = diag(A);
e = norm(m-ma);
disp(e); %for each iteration display the value of e
difference_e=e-old_e;
end
disp('Number of iterations needed for part D');
disp(count_e);  %display the numver of iterations neeeded.


case '3'
	disp('3')
    %question 3 in computational neuroscience hw 1

load('XData.mat'); %at first we load the data 
% %Part A
disp('Question 3 Part A');
 sample_mean=mean(x);   %calculate  mean
 sample_median=median(x);  %calculate median
 todisplay=['Sample mean is ',num2str(sample_mean),' and sample median is ',num2str(sample_median)];
disp(todisplay); 
% %Part B
disp('Question 3 Part B');
 standart_deviation=std(x); %calculate the standart deviation
 inter_quartile = iqr(x);% calculate the interquartile range
 todisplay_2=['Standart deviation is ',num2str(standart_deviation),' and Interquartile range  is ',num2str(inter_quartile)];
 disp(todisplay_2);

%Part C
disp('Question 3 Part C');
%for n=3 bins
histogram_c=histogram(x,3,'BinLimits',[70,130]); %computes the histogram for   70<=X<=130 for n=3 bins
title('For n=3');
%for n=6 bins
figure;
histogram_c=histogram(x,6,'BinLimits',[70,130]); %computes the histogram for   70<=X<=130 for n=6 bins
title('For n=6');
%for n=12 bins
figure;
histogram_c=histogram(x,12,'BinLimits',[70,130]); %computes the histogram for   70<=X<=130 for n=12 bins
title('For n=12');
%Part D
disp('Question 3 Part D');
y = sort(x);
n = length(x);
f = ((1:n)-3/8)/(n+1/4);
q = 4.91*(f.^0.14 - (1-f).^0.14);
figure(1);clf;plot(q,y,'*-');grid;
title('Normal Quantile Plot');
%Part E
disp('Question 3 Part E');
%Calculate the sample mean using bootstrapping with N=1000
N=1000; %number of samplings(iterations)
n=50; %number of bins
bootsTrapped = bootstrp(N,@mean,x); %x is our data,mean is the function to be computed
mean_New=mean(bootsTrapped);%mean of our new data set
stand_error= std(bootsTrapped)/sqrt(1000); %standart error is std/sqrt(number of samples)
boots_trapDisp=['The mean of the bootstrap sampling : ',num2str(mean_New),' Standart error is ',num2str(stand_error)];
disp(boots_trapDisp);
% 
% %Calculate the 95 % confidence interval
[confidence_interval dis]=bootci(1000,@mean,x);
disp_confidence_intervals=['Lower bound is ',num2str(confidence_interval(1,1)) ,' Upper bound is ',num2str(confidence_interval(2,1))];
disp(disp_confidence_intervals);
%Histogram
 histogram(bootsTrapped,n);
title('Bootstrap mean distribution for n=50 bins');
 xlabel('x');
 ylabel('mean' );

% %Part F
disp('Question 3 Part F');
% %Calculate the standart deviation using bootstrapping with N=1000
N=1000;
n=50;
bootsTrapped_std = bootstrp(N,@std,x); %x is our data,standart deviation is the function to be computed
std_New=mean(bootsTrapped_std);%mean of our new data set
stand_error_2= std(bootsTrapped_std)/sqrt(1000); %standart error is std/sqrt(number of samples)
boots_trapDisp_std=['The std of the bootstrap sampling : ',num2str(std_New),' Standart error is ',num2str(stand_error_2)];
disp(boots_trapDisp_std);
%95 confidence intervals
[confidence_interval dis]=bootci(1000,@std,x);
disp_confidence_intervals=['Lower bound is ',num2str(confidence_interval(1,1)) ,' Upper bound is ',num2str(confidence_interval(2,1))];
disp(disp_confidence_intervals);
%Histogram part
histogram(bootsTrapped_std,n);
title('Bootstrap standart distribution for n=50 bins');
xlabel('x');
ylabel('std' );


% %Part G.E
disp('Question 3 Part G(repear part E)');
% %Calculate the mean using 	jackknife resampling with N=30
N=30;
n=5;
jacked_means = jackknife(@mean,x); %compute mean of the "jackkniffed" dataset
mean_New_jack=mean(jacked_means);%mean of our new data set
stand_error_3= std(jacked_means)/sqrt(N); %standart error is std/sqrt(number of samples)
boots_jackDisp_mean=['The mean of the jackknife sampling : ',num2str(mean_New_jack),' Standart error is ',num2str(stand_error_3)];
disp(boots_jackDisp_mean);

%95 confidence intervals
[confidence_interval]=paramci(fitdist(jacked_means,'Normal'));
disp_confidence_intervals_2=['Lower bound is ',num2str(confidence_interval(1,1)) ,' Upper bound is ',num2str(confidence_interval(2,1))];
disp(disp_confidence_intervals_2);
histogram(jacked_means,n);
title('Jackknife mean distribution for n=5 bins');
xlabel('x');
ylabel('mean' );

% %Part G.F

disp('Question 3 Part G(repear part F)');
% %Calculate the std using 	jackknife resampling with N=30
N=30;
n=5;
jacked_stds = jackknife(@std,x); %compute mean of the "jackkniffed" dataset
std_New_jack=std(jacked_stds);%std of our new data set
stand_error_4= std(jacked_stds)/sqrt(N); %standart error is std/sqrt(number of samples)
boots_jackDisp_std=['The std of the jackknife sampling : ',num2str(std_New_jack),' Standart error is ',num2str(stand_error_4)];
disp(boots_jackDisp_std);

% %95 confidence intervals
[confidence_interval]=paramci(fitdist(jacked_stds,'Normal'));
 disp_confidence_intervals_3=['Lower bound is ',num2str(confidence_interval(1,1)) ,' Upper bound is ',num2str(confidence_interval(2,1))];
 disp(disp_confidence_intervals_3);
histogram(jacked_stds,n);
title('Jackknife std distribution for n=5 bins');
xlabel('x');
ylabel('std' );

 case '4'
	disp('4')
    
%Question 4 Computational Neuroscience HW1
%Part A
disp('Question 4 part A');
x=[0:0.001:1];%Define x
L_x_languange= nchoosek(869,103)*(x.^(103)).*((1-x).^766); %exppression of the likelihood for language 
L_x_no_languange= nchoosek(2353,199)*(x.^(199)).*((1-x).^2154); %exppression of the likelihood for no-language 
bar(x,L_x_languange);
title('Language Activation Likelihood');
xlabel('x');
ylabel('L(x)');
figure;
bar(x,L_x_no_languange);
title('No Language Activation Likelihood');
xlabel('x');
ylabel('L(x)');
disp('Question 4 part B');
%Part B , finding corresponig_x for maximum values
max_languagemax=max(L_x_languange);
index_1=find(L_x_languange==max_languagemax);%find the index in the array
                              %normalize it so we can find the corresp. x
x_corresponding=(index_1-1)/(length(L_x_languange)-1);
out=['Maximum xl is  ', num2str(x_corresponding)];
disp(out);

max_no_languagemax=max(L_x_no_languange);
index_2=find(L_x_no_languange==max_no_languagemax);%find the index in the array
                              %normalize it so we can find the corresp. x
x_corresponding_2=(index_2-1)/(length(L_x_no_languange)-1);
out_2=['Maximum xnl is  ', num2str(x_corresponding_2)];
disp(out_2);

%Part C
%for languanage case
disp('Question 4 part C');
total_language=sum(L_x_languange);
probability_x_lang=(L_x_languange)/(total_language);
%for non-language case
total_no_language=sum(L_x_no_languange);
probability_x_no_lang=(L_x_no_languange)/(total_no_language);
%plot discrete distributions
plot(x,probability_x_lang);
title('Discrete Posterior Prob. when Language');
figure;
plot(x,probability_x_no_lang);
title('Discrete Posterior Prob. when no Language');

%CDF part
 cum_language=cumsum(probability_x_lang); % cumulative distrib. given language
 cum_no_language=cumsum(probability_x_no_lang);%cumulative distrib.,given no language
 %Plot CDF-s
 plot(x,cum_language);
 title('Cumulative distribution when language');
 %axis equal;
 figure;
 plot(x,cum_no_language);
 title('Cumulative distribution when no language');
 %axis equal;
%Calculate the confidence interval
min_boundary_CI_language=x(min(find(cum_language>0.025)));
max_boundary_CI_language=x(min(find(cum_language>0.975)));
min_boundary_CI_no_language=x(min(find(cum_no_language>0.025)));
max_boundary_CI_no_language=x(min(find(cum_no_language>0.975)));
info_1=[' CI for language is ',num2str(min_boundary_CI_language),' to ', num2str(max_boundary_CI_language)];
info_2=[' CI for nolanguage is ',num2str(min_boundary_CI_no_language),' to ', num2str(max_boundary_CI_no_language)];
disp(info_1);
disp(info_2);
%Note at first I wrote a special function to do the job of
%"min(find(cum_language>" but I found this short command I deleted it.

%Part D  
disp('Question 4 part D');
%performing the product over x and y(2 loops needed)
for x=1:1001
    for y=1:1001 
      joint_prob(x,y)=probability_x_lang(x)*probability_x_no_lang(y);
    end
end
imagesc(joint_prob);
title('Joint distribution of languange and non-language');

%performing the posteriot probabilities(greater than etc.)
 % the border(checking) will be line y=x
greater_language=0;
greater_nonlanguage=0;
for x=1:length(joint_prob)-1
    for y=x+1:length(joint_prob)
        greater_language= joint_prob(y,x) +greater_language; 
    end
  end
greater_nonlanguage=1-greater_language;
greater_nonlanguage
greater_language

case '5'
	disp('5')
    %Part A- look at the end of the script,function definition


%Part B-Testing
disp("Question 5 part B");
mean2=[1,5];
cov2=[4,3;3,9];  
samples2=ndRandn(mean2,cov2,1000);
disp('Sample mean'); 
sampled_mean=mean(samples2)
disp('Sample  covariance'); 
sampled_cov=cov(samples2)

%Note please note that the code is based on internet codes and the relevant
%citation is given.
[eig_vec,eig_val] = eig(sampled_cov);

% Index of the largest eig_vector
[largest_eig_vec_ind_c, r] = find(eig_val == max(max(eig_val)));
largest_eig_vec = eig_vec(:, largest_eig_vec_ind_c);
% Largest eig_value
largest_eig_val = max(max(eig_val));
% Smallest eig_vector and eig_value
if(largest_eig_vec_ind_c == 1)
    smallest_eig_val = max(eig_val(:,2))
    smallest_eig_vec = eig_vec(:,2);
else
    smallest_eig_val = max(eig_val(:,1));
    smallest_eig_vec = eig_vec(1,:);
end
% Calculate the angle between the x-axis and the largest eig_vector
angle = atan2(largest_eig_vec(2), largest_eig_vec(1));
% Make the angle is between 0 and 2pi
if(angle < 0)
    angle = angle + 2*pi;
end
%  data mean coordinates
avg = mean(samples2);
%  95% confidence interval error ellipse
chisquare_val = 2.4477;
theta_grid = linspace(0,2*pi);
phi = angle;
X0=avg(1);
Y0=avg(2);
a=chisquare_val*sqrt(largest_eig_val);
b=chisquare_val*sqrt(smallest_eig_val);
% the ellipse in x and y coordinates 
ellipse_x_r  = a*cos( theta_grid );
ellipse_y_r  = b*sin( theta_grid );
%Define a rotation matrix
R = [ cos(phi) sin(phi); -sin(phi) cos(phi) ];
%let's rotate the ellipse to some angle phi
r_ellipse = [ellipse_x_r;ellipse_y_r]' * R;
% Draw the error ellipse
plot(r_ellipse(:,1) + X0,r_ellipse(:,2) + Y0,'-')
hold on;
% Plot the original data
plot(samples2(:,1), samples2(:,2), '.');
mindata = min(min(samples2));
maxdata = max(max(samples2));
xlim([mindata-3, maxdata+3]);
ylim([mindata-3, maxdata+3]);
hold on;
% Set the axis labels
hXLabel = xlabel('x values');
hYLabel = ylabel('y values');
title('Scatter plot of random bivariate sample with 2sigma ellipse');

end
end
  
function [roundedThree] = roundUpToThree(A)
     roundedThree=round((10^3.*A))./(10^3);
end

function [samples] = ndRandn(mean,cov,num)
  if nargin < 3 %if num not specified it makes it 1 as default
    num=1; 
  end
  X=randn([num length(cov)]);;%create the samples matrix (num by N)
  [U,S,V]=svd(cov); %
  Y=sqrt(S)*U';
  samples=X*Y;
  samples=samples+mean;
end
