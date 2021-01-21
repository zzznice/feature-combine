clear all
clc
%导入数据(控制变量:name)
name='G_n';
% name='zong317';
%fid=fopen(strcat(name,'.txt'))
fid=fopen('D:\MATLAB\data\G_n\G_n.txt')
string=fscanf(fid,'%s'); %文件输入
%匹配的字符串
% firstmatches=findstr(string,'M');%开始位置
firstmatches=findstr(string,'>')-1;%开始位置
endmatches=findstr(string,'>')+7;%结束位置
firstnum=length(firstmatches); %firstnum=endnum序列的条数
endnum=length(endmatches);
for k=1:firstnum-1
    j=1;
    lensec(k)=firstmatches(k+1)-endmatches(k)+1;%每条序列的长度
   for mm=endmatches(k):firstmatches(k+1)
  % for mm=firstmatches(k):endmatches(k) 
        sequence(k,j)=string(mm); %字符序列
        j=j+1;
    end
end
%控制变量:lamdashu
totalnumber=45;
for lamdashu=1:totalnumber
%   strrep(str1,str2,str3) 
%   它把str1中所有的str2字串用str3来替换
% % %%%%%%物理化学性质1_疏水性
for i=1:firstnum-1
c= strrep(sequence(i,:),'A','-1.63 ');
c1= strrep(c,'C','-0.63 ');
c2= strrep(c1,'D','-0.12 ');
c3= strrep(c2,'E','0.27 ');
c4= strrep(c3,'F','0.87 ');
c5= strrep(c4,'G','-2.19 ');
c6= strrep(c5,'H','0.77 ');
c7=strrep(c6,'I','-0.28 ');
c8=strrep(c7,'K','0.47 ');
c9=strrep(c8,'L','0.0 ');
c10=strrep(c9,'M','0.02 ');
c11=strrep(c10,'N','-0.16 ');
c12=strrep(c11,'P','-0.64 ');
c13=strrep(c12,'Q','0.46 ');
c14=strrep(c13,'R','1.42 ');
c15=strrep(c14,'S','-0.75 ');
c16= strrep(c15,'T','-0.55 ');
c17= strrep(c16,'V','-0.77 ');
c18= strrep(c17,'W','2.11 ');
c19= strrep(c18,'Y','1.34 ');
b{i}=c19;
clear c c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19
end
 xx=[];
acf1=[];
for ii=1:firstnum-1
     shuzhi=str2num(b{ii});
     [hang,changdu]=size(shuzhi);
    for lamda=1:lamdashu
     for j=1:changdu-lamda
       xx(ii,j)=shuzhi(1,j)*shuzhi(1,j+lamda);
     end
      acf1(ii,lamda)=sum(xx(ii,1:changdu-lamda))/(changdu-lamda);
   end
   clear shuzhi 
end
% % % %%%%%%%%物理化学性质2亲水性
for i=1:firstnum-1
c= strrep(sequence(i,:),'A','0.11 ');
c1= strrep(c,'C','0.45 ');
c2= strrep(c1,'D','-1.58 ');
c3= strrep(c2,'E','-1.53 ');
c4= strrep(c3,'F','1.21 ');
c5= strrep(c4,'G','-0.22 ');
c6= strrep(c5,'H','0.19 ');
c7=strrep(c6,'I','0.79 ');
c8=strrep(c7,'K','-1.71 ');
 c9=strrep(c8,'L','1.08 ');
 c10=strrep(c9,'M','0.61 ');
 c11=strrep(c10,'N','-0.46 ');
 c12=strrep(c11,'P','0.35 ');
 c13=strrep(c12,'Q','-0.44 ');
 c14=strrep(c13,'R','-2.15 ');
 c15=strrep(c14,'S','0.15 ');
c16= strrep(c15,'T','0.33 ');
c17= strrep(c16,'V','0.53 ');
c18= strrep(c17,'W','1.43 ');
c19= strrep(c18,'Y','0.87 ');
b{i}=c19;
clear c c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19
end
 xx=[];
acf2=[];
for ii=1:firstnum-1
     shuzhi=str2num(b{ii});
     [hang,changdu]=size(shuzhi);
    for lamda=1:lamdashu
       for j=1:changdu-lamda
       xx(ii,j)=shuzhi(1,j)*shuzhi(1,j+lamda);
     end
      acf2(ii,lamda)=sum(xx(ii,1:changdu-lamda))/(changdu-lamda);
   end
   clear shuzhi 
end
% % %%%%%%%%物理化学性质3侧链分子质量
for i=1:firstnum-1
c= strrep(sequence(i,:),'A','1.18 ');
c1= strrep(c,'C','-2.98 ');
c2= strrep(c1,'D','-0.33 ');
c3= strrep(c2,'E','-0.37 ');
c4= strrep(c3,'F','0.25 ');
c5= strrep(c4,'G','0.14 ');
c6= strrep(c5,'H','-0.03 ');
c7=strrep(c6,'I','0.75 ');
c8=strrep(c7,'K','1.5 ');
 c9=strrep(c8,'L','0.67 ');
 c10=strrep(c9,'M','1.22 ');
 c11=strrep(c10,'N','-1.12 ');
 c12=strrep(c11,'P','0.39 ');
 c13=strrep(c12,'Q','-0.04 ');
c14=strrep(c13,'R','-0.01 ');
c15= strrep(c14,'S','-0.98 ');
c16= strrep(c15,'T','-0.74 ');
c17= strrep(c16,'V','0.99 ');
c18= strrep(c17,'W','-0.1 ');
c19= strrep(c18,'Y','-0.4 ');
b{i}=c19;
%li=[li;c19];
%name = strcat('','c19','');
clear c c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19
end
 xx=[];
acf3=[];
for ii=1:firstnum-1
     shuzhi=str2num(b{ii});
     [hang,changdu]=size(shuzhi);
    for lamda=1:lamdashu;
       for j=1:changdu-lamda
       xx(ii,j)=shuzhi(1,j)*shuzhi(1,j+lamda);
     end
      acf3(ii,lamda)=sum(xx(ii,1:changdu-lamda))/(changdu-lamda);
   end
   clear shuzhi 
end
% % %%%%%%%%物理化学性质4极化率
for i=1:firstnum-1
c= strrep(sequence(i,:),'A','1.03 ');
c1= strrep(c,'C','-2.21 ');
c2= strrep(c1,'D','0.82 ');
c3= strrep(c2,'E','0.78 ');
c4= strrep(c3,'F','0.11 ');
c5= strrep(c4,'G','0.58 ');
c6= strrep(c5,'H','0.25 ');
c7=strrep(c6,'I','-0.5 ');
c8=strrep(c7,'K','-2.21 ');
 c9=strrep(c8,'L','-0.38 ');
 c10=strrep(c9,'M','-1.49 ');
 c11=strrep(c10,'N','-0.41 ');
 c12=strrep(c11,'P','0.46 ');
 c13=strrep(c12,'Q','-0.28 ');
c14=strrep(c13,'R','0.75 ');
c15= strrep(c14,'S','0.53 ');
c16= strrep(c15,'T','0.91 ');
c17= strrep(c16,'V','0.22 ');
c18= strrep(c17,'W','-0.74 ');
c19= strrep(c18,'Y','-0.32 ');
b{i}=c19;
%li=[li;c19];
%name = strcat('','c19','');
clear c c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19
end
 xx=[];
 acf4=[];
for ii=1:firstnum-1
    shuzhi=str2num(b{ii});
    [hang,changdu]=size(shuzhi);
    for lamda=1:lamdashu
    for j=1:changdu-lamda
        xx(ii,j)=shuzhi(1,j)*shuzhi(1,j+lamda);
    end
      acf4(ii,lamda)=sum(xx(ii,1:changdu-lamda))/(changdu-lamda);
    end
   clear shuzhi 
end
% % % %%%%%%%%物理化学性质5极性
for i=1:firstnum-1
c= strrep(sequence(i,:),'A','0.15 ');
c1= strrep(c,'C','-0.06 ');
c2= strrep(c1,'D','-1.16 ');
c3= strrep(c2,'E','1.06 ');
c4= strrep(c3,'F','0.29 ');
c5= strrep(c4,'G','1.13 ');
c6= strrep(c5,'H','-1.01 ');
c7=strrep(c6,'I','0.73 ');
c8=strrep(c7,'K','-0.28 ');
 c9=strrep(c8,'L','-0.99 ');
 c10=strrep(c9,'M','0.5 ');
 c11=strrep(c10,'N','0.59 ');
 c12=strrep(c11,'P','-0.67 ');
 c13=strrep(c12,'Q','-1.11 ');
c14=strrep(c13,'R','0.76 ');
c15= strrep(c14,'S','-2.12 ');
c16= strrep(c15,'T','0.57 ');
c17= strrep(c16,'V','0.45 ');
c18= strrep(c17,'W','-0.06 ');
c19= strrep(c18,'Y','1.23 ');
b{i}=c19;
clear c c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19
end
  xx=[];
acf5=[];
for ii=1:firstnum-1
     shuzhi=str2num(b{ii});
     [hang,changdu]=size(shuzhi);
    for lamda=1:lamdashu;
       for j=1:changdu-lamda
       xx(ii,j)=shuzhi(1,j)*shuzhi(1,j+lamda);
     end
      acf5(ii,lamda)=sum(xx(ii,1:changdu-lamda))/(changdu-lamda);
   end
   clear shuzhi 
end
% % % %%%%%%%物理化学性质6溶剂化自由能
for i=1:firstnum-1
c= strrep(sequence(i,:),'A','-0.45 ');
c1= strrep(c,'C','-0.14 ');
c2= strrep(c1,'D','-1.43 ');
c3= strrep(c2,'E','-0.49 ');
c4= strrep(c3,'F','0.38 ');
c5= strrep(c4,'G','-1.29 ');
c6= strrep(c5,'H','-0.99 ');
c7=strrep(c6,'I','0.9 ');
c8=strrep(c7,'K','0.81 ');
 c9=strrep(c8,'L','2.34 ');
 c10=strrep(c9,'M','-1.63 ');
 c11=strrep(c10,'N','0.29 ');
 c12=strrep(c11,'P','-0.41 ');
 c13=strrep(c12,'Q','-0.06 ');
c14=strrep(c13,'R','1.08 ');
c15= strrep(c14,'S','0.92 ');
c16= strrep(c15,'T','1.31 ');
c17= strrep(c16,'V','0.28 ');
c18= strrep(c17,'W','-1.35 ');
c19= strrep(c18,'Y','-0.07 ');
b{i}=c19;
clear c c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19
end
xx=[];
acf6=[];
for ii=1:firstnum-1
     shuzhi=str2num(b{ii});
     [hang,changdu]=size(shuzhi);
    for lamda=1:lamdashu;
       for j=1:changdu-lamda
       xx(ii,j)=shuzhi(1,j)*shuzhi(1,j+lamda);
     end
      acf6(ii,lamda)=sum(xx(ii,1:changdu-lamda))/(changdu-lamda);
   end
   clear shuzhi 
end
% % % %%%%%%%物理化学性质7残基
for i=1:firstnum-1
c= strrep(sequence(i,:),'A','-1.4 ');
c1= strrep(c,'C','-0.13 ');
c2= strrep(c1,'D','-0.25 ');
c3= strrep(c2,'E','1.84 ');
c4= strrep(c3,'F','0.45 ');
c5= strrep(c4,'G','-1.45 ');
c6= strrep(c5,'H','0.42 ');
c7=strrep(c6,'I','-0.28 ');
c8=strrep(c7,'K','1.02 ');
 c9=strrep(c8,'L','1.05 ');
 c10=strrep(c9,'M','0.74 ');
 c11=strrep(c10,'N','-0.93 ');
 c12=strrep(c11,'P','2.22 ');
 c13=strrep(c12,'Q','-1.56 ');
c14=strrep(c13,'R','-1.4 ');
c15= strrep(c14,'S','0.86 ');
c16= strrep(c15,'T','1.31 ');
c17= strrep(c16,'V','-1.58 ');
c18= strrep(c17,'W','-1.11 ');
c19= strrep(c18,'Y','0.16 ');
b{i}=c19;
clear c c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19
end
xx=[];
acf7=[];
for ii=1:firstnum-1
     shuzhi=str2num(b{ii});
     [hang,changdu]=size(shuzhi);
    for lamda=1:lamdashu;
       for j=1:changdu-lamda
       xx(ii,j)=shuzhi(1,j)*shuzhi(1,j+lamda);
     end
      acf7(ii,lamda)=sum(xx(ii,1:changdu-lamda))/(changdu-lamda);
   end
   clear shuzhi 
end

A=[acf1,acf2,acf3,acf4,acf5,acf6,acf7];

xlswrite(strcat(name,'ACF7_',num2str(lamdashu),'.xlsx'),A,'Sheet1','A1');
end
% % % % % % %%%%%%%物理化学性质10
% for i=1:firstnum-1
% c= strrep(sequence(i,:),'A','1.181 ');
% c1= strrep(c,'C','1.461 ');
% c2= strrep(c1,'D','1.587 ');
% c3= strrep(c2,'E','1.862 ');
% c4= strrep(c3,'F','2.228 ');
% c5= strrep(c4,'G','0.881 ');
% c6= strrep(c5,'H','2.025 ');
% c7=strrep(c6,'I','1.810 ');
% c8=strrep(c7,'K','2.258 ');
%  c9=strrep(c8,'L','1.931 ');
%  c10=strrep(c9,'M','2.034 ');
%  c11=strrep(c10,'N','1.655 ');
%  c12=strrep(c11,'P','1.468 ');
%  c13=strrep(c12,'Q','1.932 ');
% c14=strrep(c13,'R','2.560 ');
% c15= strrep(c14,'S','1.298 ');
% c16= strrep(c15,'T','1.525 ');
% c17= strrep(c16,'V','1.645 ');
% c18= strrep(c17,'W','2.663 ');
% c19= strrep(c18,'Y','2.368 ');
% b{i}=c19;
% clear c c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19
% end
%  xx=[];
% acf=[];
% for ii=1:firstnum-1
%      shuzhi=str2num(b{ii});
%      [hang,changdu]=size(shuzhi);
%     for lamda=1:lamdashu;
%        for j=1:changdu-lamda
%        xx(ii,j)=shuzhi(1,j)*shuzhi(1,j+lamda);
%      end
%       acf(ii,lamda)=sum(xx(ii,1:changdu-lamda))/(changdu-lamda);
%    end
%    clear shuzhi 
% end
% save acf  
% %导入数据
% A=xlsread('317phychen9.xlsx');
% A=[A,acf];
% xlswrite('317phychen10.xlsx',A,'Sheet1','A1');



