clear all
clc
g=1;
%导入数据
name='G_n'
%fid=fopen(strcat(name,'.txt'));
fid=fopen('D:\MATLAB\data\G_n\G_n.txt')
string=fscanf(fid,'%s'); %文件输入
%匹配的字符串
firstmatches=findstr(string,'>')-1;%开始位置
endmatches=findstr(string,'>')+7;%结束位置
firstnum=length(firstmatches); %firstnum=endnum序列的条数
endnum=length(endmatches);
 for k=2:firstnum-1
    j=1;
    lensec(k)=firstmatches(k+1)-endmatches(k)+1;%每条序列的长度
   for mm=endmatches(k):firstmatches(k+1)
  % for mm=firstmatches(k):endmatches(k)
        sequence(k,j)=string(mm); %字符序列
        j=j+1;
   end 
 end
input=sequence;
output1=[];

for i=1:firstnum-1
    protein=input(i,:);
    output =Dipeptide(protein,lensec(i),g);%计算每一条序列的g-gap的数值信号
    output1=[output1;output];
    %output1=output1*100; 
end

%save output1
xlswrite(strcat(name,'g_gap_',num2str(g),'.xlsx'),output1,'Sheet1','A1');

