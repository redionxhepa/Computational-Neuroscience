

%Redion Xhepa   ID:21500280

function redion_xhepa_21500280_hw2(question)
clc
close all
%parameters

%There might be too many inconsistencies with the report photos in the way
%that labelling might hold since at the last day I had issues with the
%Runge Kutta code and I had to change it.
switch question
    case '1'
disp('Question 1  part b ');
tau_1 = 1e-2;  % 10ms 
resistance_1 = 1e1; % resistance is 1kOhm
current_initial_1 = 2e-3; %initial current 2mA
t0=0; %time starts at 0
t_final=0.1; %time ends at 0.1s
v_0=0;  %initial conditions given
h=0.00001; % step size
t=t0:h:t_final;   %time parameters
time_length=length(t)-1;
%Solution analytically
analytical_sol = resistance_1*current_initial_1-resistance_1*current_initial_1*exp(-1/tau_1*t);
value = 0;
for i=1:length(t)-1
    next = (1-h/tau_1)*value(1,i)+h/tau_1*resistance_1*current_initial_1; %discretizazion with time step h
    value = [value next];
end

plot(t,analytical_sol*100,'*');
hold on ;
plot(t,value*100);
legend('Numerical Solution','analytical solution with stars');
xlabel('Time t in seconds');
ylabel('Voltage in Volts');


% %Computational Neuroscience Hw2_Question_1
% %Part b of question 1
 disp('Question 1 part c ');
 thressholf=1/100;  %considering our units normally is one but in our code should be dicided by 100
 value = 0;
for i=1:length(t)-1
    next = (1-h/tau_1)*value(1,i)+h/tau_1*resistance_1*current_initial_1; %discretizazion with time step h
    if  next >=thressholf
        next=0;  %reset
    end
    value = [value next];
end
figure; 
plot(t,value*100);
title('Spike emission thressholded ');
xlabel('time in ms');
ylabel('Voltage in V');



%part d in the last minutes started not to work .I changed sth but I could
%not find the error.
% disp('Question 1 part d ');
% firing_rate=[];
% current_initial_1=2e-3:1e-4:10e-3;
% thressholf=1/100;  %considering our units normally is one but in our code should be dicided by 100
%  value = 0;
% for k=1:length(current_initial_1)
% for i=1:length(t)-1
%     next = (1-h/tau_1)*value(1,i)+h/tau_1*resistance_1*current_initial_1(k); %discretizazion with time step h
%     if  next >=thressholf
%         next=0;  %reset
%     end
%     value = [value next];
% end
%   peak=numel(findpeaks(value));
%   firing_rate=[firing_rate peak];
% end
% %divide by max t
% figure
% firing_rate= firing_rate/max(t);
% plot(current_initial_1,firing_rate)
% xlabel('current)')

%Question 2

case '2'

%Computational Neuroscience HW2_Question 2
disp('Question 2 having some errors so I commented the code');
% disp('Question 2 Part A ');
% repeat=200; %repeat for 200 times
% time_wanted=1;  %1 sec long poisson process
% mean_rate_wanted=15; % mean rate 15 Hz
% spike_count_hspikes1=[]; %allocate the array which will hold spike count for hspikes1 function 
% spike_count_hspikes2=[]; %allocate the array which will hold spike count for hspikes2 function 
% inter_spike_intr_hspikes1=[];%allocate the array which will hold  inter-spike interval for hspikes1 function 
% inter_spike_intr_hspikes2=[];%allocate the array which will hold  inter-spike interval for hspikes2 function 
% 
% 
% 
% 
% for i=1:repeat
%     [spike1  ints1 ]=hspikes1(time_wanted,mean_rate_wanted);
%     [spike2  ints2 ]=hspikes2(time_wanted,mean_rate_wanted);
%     spike_count_hspikes1(i)= spike1 ;
%     inter_spike_intr_hspikes1(i)=ints1 ;
%     spike_count_hspikes2(i)= spike2;
%     inter_spike_intr_hspikes2(i)=ints2;
% end
% 
% %Construct the histograms for each part
% histogram(spike_count_hspikes1);
% title('Spike count Histogram for hspikes1 function');
% histogram(spike_count_hspikes2);
% title('Spike count Histogram for hspikes2 function')
% histogram(inter_spike_intr_hspikes1);
% title('Inter-spike interval Histogram for hspikes1 function')
% histogram(inter_spike_intr_hspikes2);
% title('Inter-spike interval Histogram for hspikes2 function')
% 
% 
% disp('Question 2 Part B ');
% %Define the requirements for spike times and time length
% repeat=50;
% time_stimulus=4;
% time_step=0.001;
% lambda=20*sin(20.*t)+30;  %firing rate is time varying
% 
% %allocate the spike 
% spike_count_inh=[];
% inter_spike_intr_inh=[]; %this will be redunant 
% 
% %raster plot showing the spike times from 50 repeats of a 4-second stimulus
% %Calculate the spike count for a time stimulus of 4 sec 50 times
% for i=1:repeat
%    [spike_count_inh(i)  inter_spike_intr_inh(i) ] =ihspikes(time_stimulus,lambda); 
% end
% 
% 
% 
% 
% function [spike_count,inter_spike_intr] = hspikes1(length_sec,mean_rate)
% time_step=0.001;
% t=0:time_step:length_sec;
% bins_no=(length_sec/time_step)+1; %number of bins
% sample=rand(1,bins_no); %generate random samples
% spike_array=[];  %allocate the spike array
% lambda=mean_rate/length_sec;
% probability_bernoulli=lambda*time_step;
% times=[]; % allocate the array which will store the times where the spikes occur
% 
% for i=1:bins_no
%     if sample(1,i)<=probability_bernoulli
%         spike_array=[spike_array 1];   %add spike   
%         times=[times t(i)];         %store the time when it happened
%     else               
%         spike_array=[spike_array 0];%no spike
%     end
% end
%  %spike count vector is added elementwise and the total n.o of spikes is
%  %calculated
%  spike_count=sum(spike_array);   
% % calculate the inter-spike intervals
% inter_spike_intr=diff(times);
% end
% %Poisson process created by exponential distribution
% function [spike_count,inter_spike_intr] = hspikes2(length_sec,mean_rate)
% %calculate lambda
% inter_spike_intr=[];
% lambda=mean_rate/length_sec; %calculate lambda
% mean_exp=1/lambda; %calculate the mean which will be as an input to exprnd function
% interval_sum=0;  %initialize the total interval time to 0
%  while  interval_sum<length_sec
%      inter_spike_intr= [inter_spike_intr  exprnd(mean_exp)]; %numbers from the exponential distribution with mean parameter 1/lambda                          
%      interval_sum=sum(inter_spike_intr); %keep track of the sum              
%  end
% spike_count=1+length(inter_spike_intr);  
% end
% 
% %to be checked later on 
% function [spike_count,inter_spike_intr]=ihspikes(length_sec,mean_rate)
% time_step=0.001;
% t=0:time_step:length_sec;
% bins_no=round(length_sec/time_step)+1; %number of bins
% sample=rand(1,bins_no); %generate random samples
% spike_array=[];  %allocate the spike array
%     
% lambda=mean_rate/length_sec;
% probability_bernoulli=lambda*time_step;
% times=[]; % allocate the array which will store the times where the spikes occur
% 
% for i=1:bins_no
%     if sample(1,i)<probability_bernoulli
%         spike_array=[spike_array 1];   %add spike   
%         times=[times t(i)];         %store the time when it happened
%     else               
%         spike_array=[spike_array 0];%no spike
%     end
% end
%  %spike count vector is added elementwise and the total n.o of spikes is
%  %calculated
%  spike_count=sum(spike_array);   
% % calculate the inter-spike intervals
% inter_spike_intr=diff(times);
% end

%Question 3  

% 
case '3'
% %Computational Neuroscince Hw question 3
load 'c2p3.mat';
disp('Question 3 part a ');
repeat=10;  %10 times
%Firstly we convert stim to double as shown in the hint
stim_double=double(stim);  
%Find the number of spikes
indeces_pos=find(counts>0);
spikes_no=counts(indeces_pos);
 %Initialize STA-s
 size_1=size(stim_double(:,:,1));
STA_images = zeros([size(stim_double(:,:,1)) repeat]);
%length of count
length_count=length(counts);  %store the length of counts (will be needed)



for j = 1:repeat
    %do it with for loop,longer but safe-check(check errors again-do not forget )
    index_sim=[];  %initialize it to add the terms later on 
    for k=1:length_count
        if counts(k,1)>0
            newterm=k-j;
            index_sim= [index_sim newterm];
        end
    end
    %part bbbbbbbb

    index_sim_length=length(index_sim);
    
    array_2=[];  %allocate the array to store the terms later on 
	for m=1:index_sim_length
        if  index_sim(1,m)>0
            newterm2=index_sim(1,m);  %check later on for consistency
            array_2=[array_2  newterm2];
        end
    end
    %calculate new stimulus index
    stim_new = stim_double(:,:,array_2);
   array_3=[];  %allocate the array to store the terms later on 
	for p=1:index_sim_length
        if  index_sim(1,p)>0
            newterm3=spikes_no(p);  %check later on for consistency
            array_3=[array_3  newterm3];
        end
    end
    STA_images(:,:,j)=sum(stim_new.*reshape(array_3/sum(array_3),1,1,length(array_3)),3);
end
        

%Plot the 10 figures
 for k = 1:repeat
 	figure; 
     imagesc(STA_images(:,:,k),[min(min(min(STA_images))) max(max(max(STA_images)))]);
  end

%%Part B Question 3 
 disp('Question 3 part b');
% %Sum the STA 
  total_STA=sum(STA_images);
  imagesc(squeeze(total_STA));
 ylabel('spatial');
  xlabel(' time '); 
  title('STA-s'' sum');
  
% 
% 
% 
% 
 disp('Question  3 part c');
%STA 1 taken 
STA_init = STA_images(:,:,1); 
bin=50; %number of bins
frob=[]; %allocate the array to store  frobenius inner products
pos_dot=[];
  %Calculate Frobenius Inner Products
   for i=1:size(stim,3)
      frob(i)=sum(sum(double(stim(:,:,i)).*(STA_init.')));
   end
 
   for i=1:length(counts)
        if counts(i,1)>0
           pos_dot = [pos_dot  frob(i)];
        end
   end

 %histrograms
 figure
 hist_1=hist(frob,-3:0.1:3);
   hist_2=hist(pos_dot,-3:0.1:3);
  new_hist=[hist_1/max(hist_1); hist_2/max(hist_2)].';  %recheck this, result convincing ? 
  bar(-3:0.1:3,new_hist,'stacked');
 legend('Cumulative Projections','Spike preceding projections');
  title('Stack Histogram');
 
  %%Question 4 
  % %impulse input
  case '4'
disp('Question 4 part a '),
 N=49;
 impuls_input=[1 ; zeros(N,1)]; 
 response_1=unknownNeuron1(impuls_input');
 response_2=unknownNeuron2(impuls_input');
 disp('check time invariance');
 %Check time invariance
 shift_1=[];
 shift_2=[];
 for k=1:N
    difference_1=circshift(response_1,k)-unknownNeuron1(circshift(impuls_input,k)');
     if norm(difference_1)==0
         shift_1(k)=1;
     end
 end
 
 for k=1:N
    difference_2=circshift(response_1,k)-unknownNeuron2(circshift(impuls_input,k)');
    if norm(difference_2)==0
         shift_2(k)=1;
    end
 end
 
 %collect the sums and check if they are equal to 49
 sum_1=sum(shift_1);
 sum_2=sum(shift_2);
 
 if sum_1==49
     disp('Neuron 1 is time invariant');
 else 
      disp('Neuron 1  is not  timeinvariant');
   
 end
 
 if sum_2==49
     disp('Neuron 2 time invariant');
 else 
      disp('Neuron 2 time is not invariant');
 end
 
 
  disp('check linearity');
  
  %by counterexample 
  test=2*impuls_input';
if test ==unknownNeuron1(2*impuls_input')
    disp('Neuron 1 linear');
else 
     disp('Neuron 1 not  linear');
end

if test ==unknownNeuron2(2*impuls_input')
    disp('Neuron 2 linear');
else 
     disp('Neuron 2 not  linear');
end
    

%%% part b
 disp('Question 4 part b ');

 
 
 disp('Question 4 part c '),

  
 
  
  %Question 5 
  case '5'
% %Question 5 part A (finished)
disp('Question 5 Part A');
figure;
x_co =-10:10;
y_co = -10:10;
gaussian_center=2;
gaussian_sorrund=4;
repeat=21;
for i = 1:repeat
    for k = 1:repeat
        DOG(i,k) = (1/(2*pi*(gaussian_center^2)))*exp(-1*(x_co(i)^2+y_co(k)^2)/(2*(gaussian_center^2)))-...
        (1/(2*pi*(gaussian_sorrund^2)))*exp(-1*(x_co(i)^2+y_co(k)^2)/(2*(gaussian_sorrund^2))); 
    end
end 
imagesc(x_co,y_co,DOG);
xlabel('x axis');
ylabel('y axis ');
title('Generated receptive field');
% 
% %%%%%%%% Question 5 part b finished
  disp('Question 5 part b ');
  figure;
 input_image=imread('hw2_image.bmp');
conv_response = conv2(input_image(:,:,1),DOG);
 imagesc(x_co,y_co,conv_response);
 colormap(gray)
 xlabel('x axis');
 ylabel('y axis');
 title('responses of each neuron to the image given ');
 
%  %%%%%%%% Question 5 part c finished
%  disp('Question 5 part C ');
  threshhold=1e-3;
  conv_response(conv_response>threshhold)=1;
  conv_response(conv_response<=threshhold)=0;
  imagesc(conv_response)
  title('Edge detector' ); 
   
   %%%%Question 5 part d finished 
   disp('Question 5 part d');  %code is renewed after for loops stuff is deleted
      figure;
        x_co=-10:10;
        y_co=-10:10;
        [x_co  y_co] = meshgrid(x_co,y_co);
        obtain_gabor=generate_gabor(x_co,y_co,pi/2,3,3,6,0); %theta=pi/2, sigma_l=3 _ sigma_w=3,phi,0,lambda=6
        meshz(x_co,y_co,obtain_gabor,obtain_gabor);  %previosuly this was done redundantly in two for loops     
        title('Generated receptive field');
 %%%%Question 5 part e finished 
   disp('Question 5 part e');    
   figure;
   response_to_gabor = conv2(input_image(:,:,1),obtain_gabor);
   imagesc(response_to_gabor);
   title('Neural activity as an image');
    colormap(gray) ;   %one might put it in grayscale to see the efects
   % better
   
   disp('Question 5 part f'); 
   figure;
   
   
   disp('Question 5 part g');
   figure;
   %First response 
   obtain_gabor_1=generate_gabor(x_co,y_co,0,3,3,6,0); %theta=0, sigma_l=3 _ sigma_w=3,phi,0,lambda=6
   gabor_response_1= conv2(input_image(:,:,1),obtain_gabor_1); %first response
   %Second response
   obtain_gabor_2=generate_gabor(x_co,y_co,pi/6,3,3,6,0); %theta=pi/6, sigma_l=3 _ sigma_w=3,phi,0,lambda=6
   gabor_response_2= conv2(input_image(:,:,1),obtain_gabor_2); %first response
   %Thirs response
   obtain_gabor_3=generate_gabor(x_co,y_co,pi/3,3,3,6,0); %theta=pi/3, sigma_l=3 _ sigma_w=3,phi,0,lambda=6
   gabor_response_3= conv2(input_image(:,:,1),obtain_gabor_3); %third response
   %Fourth response
   obtain_gabor_4=generate_gabor(x_co,y_co,pi/2,3,3,6,0); %theta=pi/2, sigma_l=3 _ sigma_w=3,phi,0,lambda=6
   gabor_response_4= conv2(input_image(:,:,1),obtain_gabor_4); %fourth response
   
   %calculate the total response
   total_response=gabor_response_1+gabor_response_2+gabor_response_3+gabor_response_4;
   imagesc(total_response);
   title('Total response of four gabor filters');
   colormap(gray) 
   
   
end
end   
  
   
 %Generate Gabor 
 function[gabor]=generate_gabor(x,y,theta,sigma_l,sigma_w,lambda,phi)
    gabor=exp(-((x*cos(theta)+y*sin(theta)).^2)/(2*sigma_l.^2) -((-x*sin(theta)+y*cos(theta)).^2)/(2*sigma_w.^2)).*cos(phi+ 2*pi*(x*cos(theta)+y*sin(theta))/lambda);
end



