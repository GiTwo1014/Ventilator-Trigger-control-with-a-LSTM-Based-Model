% x=xlsread('data/Adolescent_Normal.xlsx',1,'C:D');
fileFolder=fullfile('/Users/tanyatan/Desktop/Laboratory/Alldata/data-ASL/');
dirOutput=dir(fullfile(fileFolder,'*.xlsx'));
fileNames={dirOutput.name};
fullseg=[]
test = zeros(1,3);m=1;
for i=1:length(fileNames)
    name=fileNames(i);
    name=string(strcat('/Users/tanyatan/Desktop/Laboratory/Alldata/data-ASL/',name));%读取struct变量的格式
%     disp(strcat('data/',name));
    data=xlsread(name,1,'C:D');
    IE=data(:,2);
    FlowLmin=data(:,1);
    

%     首先我们约定  2是呼气，1是吸气
%     开始滑窗
    [in temp]=find(IE==1);    %in代表吸气触发点对应的采样点序号集合（1个间隔代表20ms）
    [ex temp]=find(IE==2);    %ex代表呼气触发点对应的采样点序号集合 
    
    x0=1;xn=length(IE);       %xn代表标签总数
    
    hw_Width=[12];            %滑窗长度设定
    %seg代表切割数据储存矩阵，tag代表标记储存变量，I代表状态1，E代表状态2，N代表状态0
    seg_I=zeros(1,2*hw_Width+1);tag_I=[];   %产生1*（2*hw_Width+1）的0矩阵 1*25 
    % seg_PI=zeros(1,2*hw_Width+1);tag_PI=[];
    seg_E=zeros(1,2*hw_Width+1);tag_E=[];
    % seg_PE=zeros(1,2*hw_Width+1);tag_PE=[];
    seg_N=zeros(1,2*hw_Width+1);tag_N=[];
    % seg_PN=zeros(1,2*hw_Width+1);tag_PN=[];
    T_in=1;T_ex=1;T_non=1; %
    
    % for in point data 吸气触发-1 采样方式：针对每一个IE=1点，进行两次采样，范围为【21，IE，3】【22，IE，2】
%     subplot(3,1,1)
    hold on
    
    for i=1:length(in) % 根据数据集中吸气点的个数进行循环
%         a=randi([hw_Width*2,hw_Width*3],1,24); %产生一个1*24的矩阵，这个矩阵元素都是小于等于hw_Width*2=24的
%         a=randi([1,hw_Width],1,8)
%         hw_Width_=[1]
%         a=randi([1,hw_Width_],1,1)
%         a=[18,19]
          a=[21,22]%吸气要早给，一出现就代表 吸气 改前原代码
%           a=[19,20,21,22,19,18,20,22]

        for j=1:length(a)   
            if in(i)-a(j)>=1 & in(i)-a(j)+2*hw_Width <= xn    % in the range
                i_start=in(i)-a(j);i_end=in(i)-a(j)+2*hw_Width;
                seg_I(T_in,:)= FlowLmin(i_start:i_end);
    %             seg_PI(T_in,:)= PressurecmH2O(i_start:i_end);
                tag_I(T_in)=1;
    %             tag_PI(T_in)=1;
%                 plot(seg_I(T_in,:))
                T_in=T_in+1
            end
        end
    end
   test(m,1:1)= T_in-1;
    
    % for ex point data 呼气触发-2  采样方式：针对每一个IE=2点，进行八次采样，范围为【12，IE，24】内25的窗口随机8次
%     subplot(3,1,2)
    hold on
    for i=1:length(ex)
%         a=randi([hw_Width*2,hw_Width*3],1,24);
          a=randi([1,hw_Width],1,8) %改前源代码
%           a=[3,4,2,5,1,3,4,6]
%         a=[1,2,3,4]%呼气要晚些出现， 已经过去20多个滑窗位置，才判定为呼气
        for j=1:length(a)
            if ex(i)-a(j)>=1 & ex(i)-a(j)+2*hw_Width <= xn% in the range
                i_start=ex(i)-a(j);i_end=ex(i)-a(j)+2*hw_Width;
                seg_E(T_ex,:)= FlowLmin(i_start:i_end);
    %             seg_PE(T_ex,:)= PressurecmH2O(i_start:i_end);
                tag_E(T_ex)=-1;
    %             tag_PE(T_ex)=-1;
%                 plot(seg_E(T_ex,:))
                T_ex=T_ex+1;
            end
        end
    end
    test(m,2:2)= T_ex-1;
    % for non-dataset 中间态-0
%     subplot(3,1,3)
    hold on
    for i=1:4*T_ex
        a=randi(xn,1,1);    
        if a>=1 & a+2*hw_Width <= xn% in the range
%             t1=in>=a;t2=in<a+2*hw_Width; f1=find(t1.*t2==1);
%             t1=ex>=a;t2=ex<a+2*hw_Width; f2=find(t1.*t2==1);
            t1=in>=a;t2=in<=a+2*hw_Width; f1=find(t1.*t2==1);
            t1=ex>=a;t2=ex<=a+2*hw_Width; f2=find(t1.*t2==1);
            if length(f1)==0 & length(f2)==0        
                seg_N(T_non,:)= FlowLmin(a:a+2*hw_Width);
    %         seg_PN(T_non,:)= PressurecmH2O(a:a+2*hw_Width);
                tag_N(T_non)=0;
    %         tag_PN(T_non)=0;
%             plot(seg_N(T_non,:))
                T_non=T_non+1;
            end
        end    
    end
    test(m,3:3)= T_non-1;
    m=m+1
    tag_E=tag_E';tag_I=tag_I';tag_N=tag_N';
    tag=[tag_I;tag_E;tag_N];
    seg=[seg_I;seg_E;seg_N];
    % seg_P=[seg_PI;seg_PE;seg_PN];
    seg=[seg tag];
    fullseg=[fullseg;seg]
%     segsample=seg;
% 7:3 0.15  8:2 0.09 9:1 0.04
%     num=round(length(seg)*0.7);
%     segsample=datasample(seg,num);
%     fullseg=[fullseg;segsample]
    
    
    
    %输出seg I 吸气触发序列（窗口25） tag I:seg I的对应标记（1）
    %输出seg E 呼气触发序列（窗口25） tag E:seg E的对应标记（-1），-1在输入模型时，需要改为2
    %输出seg N 中间状态序列（窗口25） tag N:seg N的对应标记（0）
    %将输出结果：seg I+tag I、seg E+tag E、seg N+tag N,存为xlsx文件即可，参考 step25-flow.xlsx    
end
i_n1=find(fullseg(:,26)==-1)
i_1=find(fullseg(:,26)==1)
i_0=find(fullseg(:,26)==0)
plot(mean(fullseg(i_0,1:25)))
hold on;
plot(std(fullseg(i_0,1:25)))
plot(mean(fullseg(i_1,1:25)))
plot(mean(fullseg(i_n1,1:25)))
legend({'正常';'正常标准差';'吸气';'呼气'}) 
%导出到xlsx
pathout='1128 新标注数据 吸气21_22 呼气1_12随机8个位置.xlsx'
title=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","I/E"];
xlswrite(pathout,title,1,'A1');
xlswrite(pathout,fullseg,1,'A2');