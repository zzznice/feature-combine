clear all
clc
%%%%%�ҳ����ݼ�������
%��str=
%%%%% 
name='G_n'
%fid=fopen(strcat(name,'.txt'));
fid=fopen(strcat('D:\MATLAB\data\G_n\G_n.txt'));
string=fscanf(fid,'%s'); %�ļ�����
%ƥ����ַ���
firstmatches=findstr(string,'>')+7;%��ʼλ��
endmatches=findstr(string,'>')-1;
firstnum=length(firstmatches); %firstnum=endnum���е�����
endnum=length(endmatches);
totalnum=45; 

for k=1:firstnum-1
    j=1;
    lensec(k)=endmatches(k+1)-firstmatches(k)+1;%ÿ�����еĳ���
   for mm=firstmatches(k):endmatches(k+1)
        sequence(k,j)=string(mm); %�ַ�����
        j=j+1;
   end
   
end
clear paac
for lambda=1:45
    clear paac
for i=1:firstnum-1
    paac(i,:)= PAAC(sequence(i,1:lensec(i)),lambda);
    xlswrite(strcat(name,'PseAAC_',num2str(lambda),'.xlsx'),paac,'Sheet1','A1');
    
end
end


