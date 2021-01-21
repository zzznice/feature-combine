clear all
clc
g=1;
%��������
name='G_n'
%fid=fopen(strcat(name,'.txt'));
fid=fopen('D:\MATLAB\data\G_n\G_n.txt')
string=fscanf(fid,'%s'); %�ļ�����
%ƥ����ַ���
firstmatches=findstr(string,'>')-1;%��ʼλ��
endmatches=findstr(string,'>')+7;%����λ��
firstnum=length(firstmatches); %firstnum=endnum���е�����
endnum=length(endmatches);
 for k=2:firstnum-1
    j=1;
    lensec(k)=firstmatches(k+1)-endmatches(k)+1;%ÿ�����еĳ���
   for mm=endmatches(k):firstmatches(k+1)
  % for mm=firstmatches(k):endmatches(k)
        sequence(k,j)=string(mm); %�ַ�����
        j=j+1;
   end 
 end
input=sequence;
output1=[];

for i=1:firstnum-1
    protein=input(i,:);
    output =Dipeptide(protein,lensec(i),g);%����ÿһ�����е�g-gap����ֵ�ź�
    output1=[output1;output];
    %output1=output1*100; 
end

%save output1
xlswrite(strcat(name,'g_gap_',num2str(g),'.xlsx'),output1,'Sheet1','A1');

