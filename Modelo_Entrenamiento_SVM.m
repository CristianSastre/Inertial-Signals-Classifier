clc;clear;close all;
load('traindata.mat');
% DIMENSION
    A=9;

    data1=zeros(7352,128*A); % all signals in one matrix
    
    data1(:,(1:128))=Acc_X;
    data1(:,(129:128*2))=Acc_Y;
    data1(:,((128*2)+1:128*3))=Acc_Z;
    
    data1(:,((128*3)+1:128*4))=AccBody_X;
    data1(:,((128*4)+1:128*5))=AccBody_Y;
    data1(:,((128*5)+1:128*6))=AccBody_Z;
    
    data1(:,((128*6)+1:128*7))=Gyr_X;
    data1(:,((128*7)+1:128*8))=Gyr_Y;
    data1(:,((128*8)+1:128*9))=Gyr_Z;   


fourier=fft(data1');
% fourier=fft(data1);
data=abs(fourier); %fourier
 data=data';
 
angulo=angle(fourier);
angulo=angulo';

% activity string to number
activity_num = grp2idx(activity);
P=length(activity_num);
 
y = activity_num(1:7352);

% Parameters
desviacion=zeros(7352,3);
frecuenciamed=zeros(7352,3);
ness=zeros(7352,3);
kur=zeros(7352,3);
entro=zeros(7352,3);
desviacion2=zeros(7352,3);
ness2=zeros(7352,3);
kur2=zeros(7352,3);
entro2=zeros(7352,3);
rango=zeros(7352,3);
modelo=zeros(7352,3);
trime=zeros(7352,3);
rango2=zeros(7352,3);
modelo2=zeros(7352,3);
trime2=zeros(7352,3);
frecuenciangulo=zeros(7352,3);

for k=1:A
    for n=1:7352
            desviacion(n,k)=(std(data(n,(128*(k-1))+1:128*k)));
            desviacion2(n,k)=(std(angulo(n,(128*(k-1))+1:128*k)));
            ness(n,k)=skewness(data(n,(128*(k-1))+1:128*k));
            ness2(n,k)=skewness(angulo(n,(128*(k-1))+1:128*k));
            kur(n,k)=kurtosis(data(n,(128*(k-1))+1:128*k));
            kur2(n,k)=kurtosis(angulo(n,(128*(k-1))+1:128*k));
            entro(n,k)=entropy(data(n,(128*(k-1))+1:128*k));
            entro2(n,k)=entropy(angulo(n,(128*(k-1))+1:128*k));
            frecuenciamed(n,k)=meanfreq(data(n,(128*(k-1))+1:128*k));
            rango(n,k)=range(data(n,(128*(k-1))+1:128*k));
            modelo(n,k)=mode(data(n,(128*(k-1))+1:128*k));
            trime(n,k)=trimmean(data(n,(128*(k-1))+1:128*k),25);        
            frecuenciangulo(n,k)=meanfreq(angulo(n,(128*(k-1))+1:128*k));
    end
end

X=[desviacion,desviacion2,ness,ness2,kur,kur2,entro,entro2,frecuenciamed,frecuenciangulo,rango,modelo,trime];
rand_num = randperm(size(X,1));
X_train = X(rand_num(1:round(0.8*length(rand_num))),:);
y_train = y(rand_num(1:round(0.8*length(rand_num))),:);

X_test = X(rand_num(round(0.8*length(rand_num))+1:end),:);
y_test = y(rand_num(round(0.8*length(rand_num))+1:end),:);

template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);

classificationSVM = fitcecoc(...
    X_train, ...
    y_train, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', [1; 2; 3; 4; 5; 6]);

test_accuracy_for_iter = sum((predict(classificationSVM,X_test) == y_test))/length(y_test)*100