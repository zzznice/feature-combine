clear all
clc
%%%%%找出数据集的序列
%求str=
%%%%% 
name='G_n'
%fid=fopen(strcat(name,'.txt'));
fid=fopen(strcat('D:\MATLAB\data\G_n\G_n.txt'));
string=fscanf(fid,'%s'); %文件输入
%匹配的字符串
firstmatches=findstr(string,'>')+7;%开始位置
endmatches=findstr(string,'>')-1;
firstnum=length(firstmatches); %firstnum=endnum序列的条数
endnum=length(endmatches);
totalnum=45; 

for k=1:firstnum-1
    j=1;
    lensec(k)=endmatches(k+1)-firstmatches(k)+1;%每条序列的长度
   for mm=firstmatches(k):endmatches(k+1)
        sequence(k,j)=string(mm); %字符序列
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


