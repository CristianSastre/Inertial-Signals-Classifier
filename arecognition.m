function [y]= arecognition(Acc,AccBody,Gyr)
    
data1=zeros(1,128*9);
 
    data1(1,(1:128))=Acc(1,1:128);      %Acc X
    data1(1,(129:128*2))=Acc(2,1:128);  %Acc Y
    data1(1,((128*2)+1:128*3))=Acc(3,1:128);    %Acc Z
    
    data1(1,((128*3)+1:128*4))=AccBody(1,1:128); %AccBody X
    data1(1,((128*4)+1:128*5))=AccBody(2,1:128);    %AccBody Y
    data1(1,((128*5)+1:128*6))=AccBody(3,1:128);    %AccBody Z
    
    data1(1,((128*6)+1:128*7))=Gyr(1,1:128);    %Gyr X
    data1(1,((128*7)+1:128*8))=Gyr(2,1:128);    %Gyr Y
    data1(1,((128*8)+1:128*9))=Gyr(3,1:128);    %Gyr Z

    fourier=fft(data1');
% fourier=fft(data1);
data=abs(fourier); %fourier
 data=data';
 
angulo=angle(fourier);
angulo=angulo';

desviacion=zeros(7352,3);
frecuenciamed=zeros(7352,3);
frecuenciangulo=zeros(7352,3);
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

for k=1:9
    for n=1:1
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

load('ModeloSVM.mat');
prediccion=predict(classificationSVM,X);

 if prediccion(1) ==1
    y='standing';
 elseif prediccion(1) ==2
     y='sitting';
 elseif prediccion(1) ==3
     y='laying';
 elseif prediccion(1) ==4
     y='walking';
 elseif prediccion(1) ==5
     y='downstairs';
 elseif prediccion(1) ==6
     y='upstairs';
 end

end
