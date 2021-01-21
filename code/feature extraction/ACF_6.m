clear all
clc
%导入数据(控制变量:name)
name='G_p';
% name='zong317';
%fid=fopen(strcat(name,'.txt'))
fid=fopen('D:\MATLAB\data\G_p\G_p-4.txt')
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
c= strrep(sequence(i,:),'A','-0.4 ');
c1= strrep(c,'C','0.17 ');
c2= strrep(c1,'D','-1.31 ');
c3= strrep(c2,'E','-1.22 ');
c4= strrep(c3,'F','1.92 ');
c5= strrep(c4,'G','-0.67 ');
c6= strrep(c5,'H','-0.64 ');
c7=strrep(c6,'I','1.25 ');
c8=strrep(c7,'K','-0.67 ');
c9=strrep(c8,'L','1.22 ');
c10=strrep(c9,'M','1.02 ');
c11=strrep(c10,'N','-0.92  ');
c12=strrep(c11,'P','-0.49 ');
c13=strrep(c12,'Q','-0.91 ');
c14=strrep(c13,'R','-0.59 ');
c15=strrep(c14,'S','-0.55 ');
c16= strrep(c15,'T','-0.28 ');
c17= strrep(c16,'V','0.91 ');
c18= strrep(c17,'W','0.50 ');
c19= strrep(c18,'Y','1.67  ');
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
c= strrep(sequence(i,:),'A','-0.5  ');
c1= strrep(c,'C','-1.0  ');
c2= strrep(c1,'D','3 ');
c3= strrep(c2,'E','3 ');
c4= strrep(c3,'F','-2.5 ');
c5= strrep(c4,'G','0 ');
c6= strrep(c5,'H','-0.5 ');
c7=strrep(c6,'I','-1.8 ');
c8=strrep(c7,'K','3.0 ');
 c9=strrep(c8,'L','-1.8 ');
 c10=strrep(c9,'M','-1.3 ');
 c11=strrep(c10,'N','0.2 ');
 c12=strrep(c11,'P','0 ');
 c13=strrep(c12,'Q','0.2 ');
 c14=strrep(c13,'R','3.0 ');
 c15=strrep(c14,'S','0.3 ');
c16= strrep(c15,'T','-0.4 ');
c17= strrep(c16,'V','-1.5 ');
c18= strrep(c17,'W','-3.4 ');
c19= strrep(c18,'Y','-2.3  ');
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
c= strrep(sequence(i,:),'A','15 ');
c1= strrep(c,'C','47 ');
c2= strrep(c1,'D','59 ');
c3= strrep(c2,'E','73  ');
c4= strrep(c3,'F','91  ');
c5= strrep(c4,'G','1 ');
c6= strrep(c5,'H','82  ');
c7=strrep(c6,'I','57  ');
c8=strrep(c7,'K','73  ');
 c9=strrep(c8,'L','57  ');
 c10=strrep(c9,'M','75 ');
 c11=strrep(c10,'N','58 ');
 c12=strrep(c11,'P','42 ');
 c13=strrep(c12,'Q','72 ');
c14=strrep(c13,'R','101 ');
c15= strrep(c14,'S','31 ');
c16= strrep(c15,'T','45 ');
c17= strrep(c16,'V','43 ');
c18= strrep(c17,'W','130 ');
c19= strrep(c18,'Y','107 ');
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
% % %%%%%%%%物理化学性质4极性
for i=1:firstnum-1
c= strrep(sequence(i,:),'A','8.1 ');
c1= strrep(c,'C','5.5 ');
c2= strrep(c1,'D','13.0 ');
c3= strrep(c2,'E','12.3 ');
c4= strrep(c3,'F','5.2 ');
c5= strrep(c4,'G','9.0 ');
c6= strrep(c5,'H','10.4 ');
c7=strrep(c6,'I','5.2 ');
c8=strrep(c7,'K','11.3 ');
 c9=strrep(c8,'L','4.9 ');
 c10=strrep(c9,'M','5.7 ');
 c11=strrep(c10,'N','11.6 ');
 c12=strrep(c11,'P','8.0 ');
 c13=strrep(c12,'Q','10.5 ');
c14=strrep(c13,'R','10.5 ');
c15= strrep(c14,'S','9.2 ');
c16= strrep(c15,'T','8.6 ');
c17= strrep(c16,'V','5.9 ');
c18= strrep(c17,'W','5.4 ');
c19= strrep(c18,'Y','6.2 ');
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
% % % %%%%%%%%物理化学性质5极化率
for i=1:firstnum-1
c= strrep(sequence(i,:),'A','-0.046 ');
c1= strrep(c,'C','0.128 ');
c2= strrep(c1,'D','0.105 ');
c3= strrep(c2,'E','0.151 ');
c4= strrep(c3,'F','0.29 ');
c5= strrep(c4,'G','0.0 ');
c6= strrep(c5,'H','0.23 ');
c7=strrep(c6,'I','0.186 ');
c8=strrep(c7,'K','0.219 ');
 c9=strrep(c8,'L','0.186 ');
 c10=strrep(c9,'M','0.221 ');
 c11=strrep(c10,'N','0.134 ');
 c12=strrep(c11,'P','0.131 ');
 c13=strrep(c12,'Q','0.18 ');
c14=strrep(c13,'R','0.291 ');
c15= strrep(c14,'S','0.062 ');
c16= strrep(c15,'T','0.108 ');
c17= strrep(c16,'V','0.14 ');
c18= strrep(c17,'W','0.409 ');
c19= strrep(c18,'Y','0.298 ');
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
c= strrep(sequence(i,:),'A','0.67 ');
c1= strrep(c,'C','0.38 ');
c2= strrep(c1,'D','-1.2 ');
c3= strrep(c2,'E','-0.76 ');
c4= strrep(c3,'F','2.3 ');
c5= strrep(c4,'G','0 ');
c6= strrep(c5,'H','0.64 ');
c7=strrep(c6,'I','1.90 ');
c8=strrep(c7,'K','-0.57 ');
 c9=strrep(c8,'L','1.90 ');
 c10=strrep(c9,'M','2.4 ');
 c11=strrep(c10,'N','-0.61 ');
 c12=strrep(c11,'P','1.2 ');
 c13=strrep(c12,'Q','-0.22 ');
c14=strrep(c13,'R','-2.1 ');
c15= strrep(c14,'S','0.01 ');
c16= strrep(c15,'T','0.52 ');
c17= strrep(c16,'V','1.50 ');
c18= strrep(c17,'W','2.60 ');
c19= strrep(c18,'Y','1.60 ');
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
   A=[acf1,acf2,acf3,acf4,acf5,acf6];

xlswrite(strcat(name,'ACF6_',num2str(lamdashu),'.xlsx'),A,'Sheet1','A1');
end
% % % %%%%%%%物理化学性质7残基
% for i=1:firstnum-1
% c= strrep(sequence(i,:),'A','-1.36 ');
% c1= strrep(c,'C','-0.88 ');
% c2= strrep(c1,'D','-0.52 ');
% c3= strrep(c2,'E','0.43 ');
% c4= strrep(c3,'F','0.9 ');
% c5= strrep(c4,'G','-1.83 ');
% c6= strrep(c5,'H','0.55 ');
% c7=strrep(c6,'I','0.07 ');
% c8=strrep(c7,'K','0.67 ');
%  c9=strrep(c8,'L','-0.05 ');
%  c10=strrep(c9,'M','0.31 ');
%  c11=strrep(c10,'N','-0.29 ');
%  c12=strrep(c11,'P','-0.64 ');
%  c13=strrep(c12,'Q','0.26 ');
% c14=strrep(c13,'R','1.26 ');
% c15= strrep(c14,'S','-1.33 ');
% c16= strrep(c15,'T','-0.71 ');
% c17= strrep(c16,'V','-0.36 ');
% c18= strrep(c17,'W','2.05 ');
% c19= strrep(c18,'Y','1.47 ');
% b{i}=c19;
% clear c c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19
% end
% xx=[];
% acf7=[];
% for ii=1:firstnum-1
%      shuzhi=str2num(b{ii});
%      [hang,changdu]=size(shuzhi);
%     for lamda=1:lamdashu;
%        for j=1:changdu-lamda
%        xx(ii,j)=shuzhi(1,j)*shuzhi(1,j+lamda);
%      end
%       acf7(ii,lamda)=sum(xx(ii,1:changdu-lamda))/(changdu-lamda);
%    end
%    clear shuzhi 
% end
% % %%%%%%%物理化学性质8侧链体积
% for i=1:firstnum-1
% c= strrep(sequence(i,:),'A','-1.24 ');
% c1= strrep(c,'C','-0.77 ');
% c2= strrep(c1,'D','-0.89 ');
% c3= strrep(c2,'E','-0.29 ');
% c4= strrep(c3,'F','1.18 ');
% c5= strrep(c4,'G','-1.99 ');
% c6= strrep(c5,'H','0.18 ');
% c7=strrep(c6,'I','0.58 ');
% c8=strrep(c7,'K','0.75 ');
%  c9=strrep(c8,'L','0.58 ');
%  c10=strrep(c9,'M','0.59 ');
%  c11=strrep(c10,'N','-0.38 ');
%  c12=strrep(c11,'P','-0.84 ');
%  c13=strrep(c12,'Q','0.22 ');
% c14=strrep(c13,'R','0.89 ');
% c15= strrep(c14,'S','-1.19 ');
% c16= strrep(c15,'T','-0.58 ');
% c17= strrep(c16,'V','-0.03 ');
% c18= strrep(c17,'W','2.01 ');
% c19= strrep(c18,'Y','1.23 ');
% b{i}=c19;
% clear c c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19
% end
% xx=[];
% acf8=[];
% for ii=1:firstnum-1
%      shuzhi=str2num(b{ii});
%      [hang,changdu]=size(shuzhi);
%     for lamda=1:lamdashu;
%        for j=1:changdu-lamda
%        xx(ii,j)=shuzhi(1,j)*shuzhi(1,j+lamda);
%      end
%       acf8(ii,lamda)=sum(xx(ii,1:changdu-lamda))/(changdu-lamda);
%    end
%    clear shuzhi 
% end
% % % %%%%%%%物理化学性质9质量
% for i=1:firstnum-1
% c= strrep(sequence(i,:),'A','-1.55 ');
% c1= strrep(c,'C','-0.51 ');
% c2= strrep(c1,'D','-0.12 ');
% c3= strrep(c2,'E','0.33 ');
% c4= strrep(c3,'F','0.92 ');
% c5= strrep(c4,'G','-2 ');
% c6= strrep(c5,'H','0.59 ');
% c7=strrep(c6,'I','-0.19 ');
% c8=strrep(c7,'K','0.3 ');
%  c9=strrep(c8,'L','-0.19 ');
%  c10=strrep(c9,'M','0.4 ');
%  c11=strrep(c10,'N','-0.15 ');
%  c12=strrep(c11,'P','-0.71 ');
%  c13=strrep(c12,'Q','0.3 ');
% c14=strrep(c13,'R','1.21 ');
% c15= strrep(c14,'S','-1.03 ');
% c16= strrep(c15,'T','-0.58 ');
% c17= strrep(c16,'V','-0.64 ');
% c18= strrep(c17,'W','2.18 ');
% c19= strrep(c18,'Y','1.43 ');
% b{i}=c19;
% clear c c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19
% end
% xx=[];
% acf9=[];
% for ii=1:firstnum-1
%      shuzhi=str2num(b{ii});
%      [hang,changdu]=size(shuzhi);
%     for lamda=1:lamdashu;
%        for j=1:changdu-lamda
%        xx(ii,j)=shuzhi(1,j)*shuzhi(1,j+lamda);
%      end
%       acf9(ii,lamda)=sum(xx(ii,1:changdu-lamda))/(changdu-lamda);
%    end
%    clear shuzhi 
% end
% % % %%%%%%%物理化学性质10分子体积
% for i=1:firstnum-1
% c= strrep(sequence(i,:),'A','-0.733 ');
% c1= strrep(c,'C','-0.862 ');
% c2= strrep(c1,'D','-3.656 ');
% c3= strrep(c2,'E','1.477 ');
% c4= strrep(c3,'F','1.891 ');
% c5= strrep(c4,'G','1.33 ');
% c6= strrep(c5,'H','-1.673 ');
% c7=strrep(c6,'I','2.131 ');
% c8=strrep(c7,'K','0.533 ');
%  c9=strrep(c8,'L','-1.505 ');
%  c10=strrep(c9,'M','2.219 ');
%  c11=strrep(c10,'N','1.299 ');
%  c12=strrep(c11,'P','-1.628 ');
%  c13=strrep(c12,'Q','-3.005 ');
% c14=strrep(c13,'R','1.502 ');
% c15= strrep(c14,'S','-4.76 ');
% c16= strrep(c15,'T','2.213 ');
% c17= strrep(c16,'V','-0.544 ');
% c18= strrep(c17,'W','0.672 ');
% c19= strrep(c18,'Y','3.097 ');
% b{i}=c19;
% clear c c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19
% end
% xx=[];
% acf10=[];
% for ii=1:firstnum-1
%      shuzhi=str2num(b{ii});
%      [hang,changdu]=size(shuzhi);
%     for lamda=1:lamdashu;
%        for j=1:changdu-lamda
%        xx(ii,j)=shuzhi(1,j)*shuzhi(1,j+lamda);
%      end
%       acf10(ii,lamda)=sum(xx(ii,1:changdu-lamda))/(changdu-lamda);
%    end
%    clear shuzhi 
% end
% 
% % % %%%%%%%物理化学性质11氨基酸组分
% for i=1:firstnum-1
% c= strrep(sequence(i,:),'A','0 ');
% c1= strrep(c,'C','2.75 ');
% c2= strrep(c1,'D','1.38 ');
% c3= strrep(c2,'E','0.92 ');
% c4= strrep(c3,'F','0 ');
% c5= strrep(c4,'G','0.74 ');
% c6= strrep(c5,'H','0.58 ');
% c7=strrep(c6,'I','0 ');
% c8=strrep(c7,'K','0.33 ');
%  c9=strrep(c8,'L','0 ');
%  c10=strrep(c9,'M','0 ');
%  c11=strrep(c10,'N','1.33 ');
%  c12=strrep(c11,'P','0.39 ');
%  c13=strrep(c12,'Q','0.9 ');
% c14=strrep(c13,'R','0.64 ');
% c15= strrep(c14,'S','1.41 ');
% c16= strrep(c15,'T','0.71 ');
% c17= strrep(c16,'V','0 ');
% c18= strrep(c17,'W','0.12 ');
% c19= strrep(c18,'Y','0.21 ');
% b{i}=c19;
% clear c c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19
% end
% xx=[];
% acf11=[];
% for ii=1:firstnum-1
%      shuzhi=str2num(b{ii});
%      [hang,changdu]=size(shuzhi);
%     for lamda=1:lamdashu;
%        for j=1:changdu-lamda
%        xx(ii,j)=shuzhi(1,j)*shuzhi(1,j+lamda);
%      end
%       acf11(ii,lamda)=sum(xx(ii,1:changdu-lamda))/(changdu-lamda);
%    end
%    clear shuzhi 
% end
% 
% A=[acf1,acf2,acf3,acf4,acf5,acf6,acf7,acf8,acf9,acf10,acf11];
% 
% xlswrite(strcat(name,'ACF11_',num2str(lamdashu),'.xlsx'),A,'Sheet1','A1');
% end
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
