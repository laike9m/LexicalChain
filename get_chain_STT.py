# -*- coding:gbk -*-
'''
该程序从语音识别结果stt文件中把词链找出来,并且统计每个伪句子分界处的词链强度
在得到section的真实长度时,用的是手工标注的sgm文件
在获取词链时,用的是识别结果stt文件
变量定义：
max_chain_length: 最大词链长度,初始值设定为section中最长的那个的长度(字的个数)
w: Pseudosentence size,伪句子长度,初步设定为40字
charsum: 计算这篇文字的总字数
pseudoseqsum: 伪句子的个数
one_word: 一元子词
two_word: 二元子词
one_syllable: 一元音节
two_syllable: 二元音节
'''
import os
import re
import math
import numpy as np
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt

def time_cmp(sgm_time,stt_time):
    '''
        比较sgm和stt中的time的大小，返回他们的差。若sgm大,返回正值；stt大返回负值.两个时间都是str类型
    '''
    stt_time_list = stt_time.split(':')
    stt = 0
    if len(stt_time_list) == 2:
        stt = float(stt_time_list[0])*60 + float(stt_time_list[1])
    if len(stt_time_list) == 3:
        stt = float(stt_time_list[0])*3600 + float(stt_time_list[1])*60 \
                   + float(stt_time_list[2])
    sgm = float(sgm_time)
    return sgm - stt

def read_sgm(sgm_lines):
    '''
        从sgm文件中读取信息最后一个有意义的段落的结束时间:
    '''
    section_lines = [l for l in sgm_lines if l.startswith('<section')]
    section_lines.reverse()
    for l in section_lines:
        if re.search('type=report',l):#最后一个有意义的段
            endt = re.search(r"(?<=endTime=)[\d.]*",l).group()
            return endt #返回最后一个有意义段落的结束时间

class Chain:
    def __init__(self,sgmfile,sttfile,pinyinfile,w):
        self.w = w
        self.stt = open(sttfile,'r')
        self.stt_lines = self.stt.readlines()
        self.pinyin = open(pinyinfile,'r')
        self.get_avg_section_length(sgmfile)
        self.make_oneword_list()
        self.make_twoword_list()
        self.make_onesyllable_list(self.pinyin)
        self.make_twosyllable_list()
        self.make_oneword_chain()
        self.make_twoword_chain()
        self.make_onesyl_chain()
        self.make_twosyl_chain()
        self.stt.close()
        self.pinyin.close()
        
        
    def get_avg_section_length(self,sgmfile):
        '''
                得到某个sgm文件中section的平均长度.现在写的是错的,因为完全没有考虑到sgm中的字和实际识别的是不一样的,长度也
                有差别,所以文字的还是应该以stt为准,而sgm只是提供starttime来和stt里的时间对应上,以此把stt分段。这段程序可能
                可以参考cmp2methods.不用那么复杂,开始时间的差异基本确定在一秒之内,可以写如果一秒之内没有,则报错+结束
                已经换成了已stt的长度计，现在的代码用于训练是可以的。用于测试的时候,没有sgm,还得思考一下应该怎么做。
        '''
        sgm = open(sgmfile,'r')
        sgm_lines = sgm.readlines()
        self.section_start = []  #存储标准的section开始的时间,从sgm文件提取
        self.section_end = []  #存储标准的section结束的时间,从sgm文件提取
        for i in range(len(sgm_lines)):
            if sgm_lines[i].startswith('<section '):
                start_time = re.search(r'(?<=startTime=)[^ ]*',sgm_lines[i])
                self.section_start.append(start_time.group())
                end_time = re.search(r'(?<=endTime=)[^>]*',sgm_lines[i])
                self.section_end.append(end_time.group())
        
        '''
        seg_info存储stt中的分段信息,是一个字典构成的列表,形式如下:
        [
         {begt:'1:30',length:39,zi:'江泽明...'},
         {begt:'2:00',length:50,zi:'下面播放...'},
         ...
        ]                '''
        sgm_meaninful_end = read_sgm(sgm_lines)   #最后一个report段落的结束时间
        
        #形成stt的[(begt1,endt1),(begt2,endt2)]
        stt_timeline = []
        for line in self.stt_lines:
            obj = re.search(r'(?<=begt=)([^ ]+).*(?<=endt=)([^ ]+)',line)
            if obj:
                if time_cmp(sgm_meaninful_end, obj.group(1)) > 0:
                    t = (obj.group(1),obj.group(2))
                    stt_timeline.append(t)
        
        #形成stt_roi_list:[(begt1,endt1),(begt2,endt2)],这已经是每个roi了
        stt_roi_list = []
        for i in range(len(stt_timeline)-1):
            stt_roi_list.append((stt_timeline[i][1],stt_timeline[i+1][0]))
            d = {'roi_begt':stt_timeline[i][1],'roi_endt':stt_timeline[i+1][0], \
                 'isboundary':0}
            d = OrderedDict(sorted(list(d.items())))
                
        sgm_timeline = []      #保存sgm段落起始/结束时间  
        can_start = 0
        for line in sgm_lines:
            if line.startswith('<section'):
                if not can_start:   #can_start=0,还未找到起始点
                    if 'nontrans' in line:
                        continue    #跳过开头部分的nontrans块
                    else:
                        can_start = 1
                if can_start:
                    obj = re.search(r'(?<=startTime=)([^ ]+).*(?<=endTime=)([\d.]+)',line)
                    if obj:
                        t = (obj.group(1),obj.group(2))
                        sgm_timeline.append(t)
        
        stt_block_start = []    #保存stt每个块的起始时间,为后面分段提供依据
        stt_block_start.append(stt_timeline[0][0])     #插入第一个stt句子的起始时间,肯定是第一块的开始   
        for t_sgm in sgm_timeline[1:]:  #把sgm boundary对应到stt_roi
            smallest = 1000000
            for t_stt in stt_roi_list:
                delta1 = abs(time_cmp(t_sgm[0],t_stt[0]))
                delta2 = abs(time_cmp(t_sgm[0],t_stt[1]))
                delta = delta1 if delta1<delta2 else delta2
                if smallest > delta:
                    smallest = delta
                else:
                    #roi的tuple的[1]就是起始时间
                    stt_block_start.append(stt_roi_list[stt_roi_list.index(t_stt)-1][1])
                    break   #delta开始增大,说明上一个就是最近时间点,这个roi就是边界          
                
        self.seg_info = []
        self.zi_list = []    #存储字的列表,从第一个字到最后一个字
        
        for line in self.stt_lines:#构成seg_info
            match = re.search(r'(?<=zi=)[^ ]*',line)
            obj = re.search(r'(?<=begt=)([^ ]+).*(?<=endt=)([^ ]+)',line)
            if match and obj:
                self.zi_list += match.group()   #这步一定会做,因为要把所有字放入zi_list
                begt = obj.group(1)
                if begt in stt_block_start:#是段落开始
                    new_segment = {}
                    new_segment['begt'] = begt
                    new_segment['zi'] = ''
                    new_segment['zi'] += match.group()
                    new_segment['length'] = len(new_segment['zi'])
                    self.seg_info.append(new_segment)
                else:
                    self.seg_info[-1]['zi'] += match.group()
                    self.seg_info[-1]['length'] = len(self.seg_info[-1]['zi'])

        self.charsum = len(self.zi_list) #计算这篇文字的总字数
        self.pseudoseqsum = math.ceil(self.charsum/self.w)#计算伪句子的个数
        self.section_length = [item['length'] for item in self.seg_info]
        self.segmentation = [sum(self.section_length[0:i+1])/self.w-1   #这是实际分界对应的伪句子分界编号
                             for i in range(len(self.section_length))]
        '''
        approximately_seg是把segmentation:[11.5,23.975,...]扩展成
        [11,12,23,24,...]这样的,如果检测出的边界在这个list里面,就认为识别到了正确的分界
        '''
        self.approximately_seg = []
        for i in self.segmentation:
            if int(i)<i:
                self.approximately_seg.extend([int(i),int(i)+1])
            else:#i是整数的情况,则三个位置都是正确的
                self.approximately_seg.extend([i-1,i,i+1])
        sgm.close()
        
        
    def make_oneword_list(self): 
        '''把之前得到的zi_list识别结果中的所有字先和伪句子编号组成一个tuple,再放入一个list'''
        '''把字按40个一段输出到文件 ps.txt,并带上伪句子号,方便之后观察'''
        '''
        ps_file = open('ps.txt','w')
        for zi_index in range(len(self.zi_list)):
            if zi_index % 40 == 0:
                ps_file.write('\n' + str(math.ceil(zi_index/40)) + ':')
            ps_file.write(self.zi_list[zi_index])
        ps_file.close()
        '''
        zi_index = 0  #数了多少个字
        ps_index = 0    #伪句子的编号
        for zi_index in range(len(self.zi_list)):
            ps_index = int(zi_index/self.w)
            self.zi_list[zi_index] = (self.zi_list[zi_index],ps_index)
    
    def make_twoword_list(self): 
        '''创建二维子词构成的list,根据之前得到的一维子词list,直接合并即可.如果之前没有创建一维词list需要添加上'''
        self.two_zi_list = []
        for i in range(len(self.zi_list)-1):
            add = [self.zi_list[i],self.zi_list[i+1]]
            self.two_zi_list.append(add)
            
    def make_onesyllable_list(self,pinyin): 
        '''创建一维音节构成的list,根据stt中的sylid,在拼音表中找到对应拼音,替换掉原来的字'''
        zi_index = 0  #数了多少个字
        ps_index = 0    #伪句子的编号
        self.syl_list = []    #存储(syl,ps_index)的列表
        syl_lines = pinyin.readlines()
        for block in self.stt_lines:
            match = re.search(r'(?<=sylid=)[^\n]*',block)
            if match:
                self.syl_list += match.group().split()
        for zi_index in range(len(self.syl_list)):
            ps_index = int(zi_index/self.w)
            add = syl_lines[int(self.syl_list[zi_index])].rstrip('\n')
            if add != 'sil':
                add = add[:-1]
            self.syl_list[zi_index] = (add,ps_index)
            
    def make_twosyllable_list(self): 
        '''创建二维子词构成的list,根据之前得到的一维子词list,直接合并即可'''
        #make_onesyllable_list(stt)
        self.two_syl_list = []
        for i in range(len(self.syl_list)-1):
            add = [self.syl_list[i],self.syl_list[i+1]]
            self.two_syl_list.append(add)
    
    def make_oneword_chain(self):
        '''构造一维子词链,形成{chainword1:[chainobj1_1,chainobj1_2],chainword2:[chainobj2_1,chainobj2_2]}这样的dict'''
        self.oneword_chain = {}   #dict
        i = 0
        for zi in self.zi_list: #一定记得zi是tuple
            key = zi[0]
            if key in list(self.oneword_chain.keys()):
                if self.oneword_chain[key][-1].add(i,zi):
                    newchain = chain(i)
                    newchain.add(i, zi)
                    self.oneword_chain[key].append(newchain)
            else:
                newchain = chain(i)
                newchain.add(i, zi)
                self.oneword_chain[key] = Mylist()  #没有key的时候新增了这一句话,创建一个list作为该key的value
                self.oneword_chain[key].append(newchain)
            i = i + 1
    
    def make_twoword_chain(self):
        '''构造二维子词链,形式和一维相同,只是key略有差别'''
        self.twoword_chain = {}   #dict
        i = 0
        for zi in self.two_zi_list: #一定记得zi是tuple
            key = zi[0][0]+zi[1][0] #把'江'+'泽'='江泽'作为key
            if key in list(self.twoword_chain.keys()):
                if self.twoword_chain[key][-1].add(i,zi):
                    newchain = chain(i)
                    newchain.add(i, zi)
                    self.twoword_chain[key].append(newchain)
            else:
                newchain = chain(i)
                newchain.add(i, zi)
                self.twoword_chain[key] = Mylist()  #没有key的时候新增了这一句话,创建一个list作为该key的value
                self.twoword_chain[key].append(newchain)
            i = i + 1

    def make_onesyl_chain(self):
        '''构造一维音节链,形成{chainword1:[chainobj1_1,chainobj1_2],chainword2:[chainobj2_1,chainobj2_2]}这样的dict'''
        self.onesyl_chain = {}   #dict
        i = 0
        for syl in self.syl_list: #一定记得zi是tuple
            key = syl[0]
            if key in list(self.onesyl_chain.keys()):
                if self.onesyl_chain[key][-1].add(i,syl):
                    newchain = chain(i)
                    newchain.add(i, syl)
                    self.onesyl_chain[key].append(newchain)
            else:
                newchain = chain(i)
                newchain.add(i, syl)
                self.onesyl_chain[key] = Mylist()  
                self.onesyl_chain[key].append(newchain)
            i = i + 1
    
    def make_twosyl_chain(self):
        '''构造二维音节链'''
        self.twosyl_chain = {}   #dict
        i = 0
        for syl in self.two_syl_list: 
            key = syl[0][0]+syl[1][0] 
            if key in list(self.twosyl_chain.keys()):
                if self.twosyl_chain[key][-1].add(i,syl):
                    newchain = chain(i)
                    newchain.add(i, syl)
                    self.twosyl_chain[key].append(newchain)
            else:
                newchain = chain(i)
                newchain.add(i, syl)
                self.twosyl_chain[key] = Mylist()  #没有key的时候新增了这一句话,创建一个list作为该key的value
                self.twosyl_chain[key].append(newchain)
            i = i + 1
            #if len(self.twosyl_chain[key]) > 1:
                #print(self.twosyl_chain[key])
    
    def calc_chain_strength(self,whichkind):
        #计算词链强度,根据whichkind的不同决定是计算哪一种词链强度
        if whichkind == 'oneword':
            self.chain_strength_oneword = [0]*(self.pseudoseqsum - 1) #每个间隔处的词链强度列表,oneword
            for key,value in self.oneword_chain.items():
                for chainobj in value:
                    start_psnumber = chainobj.chain[0][1]  #这个词链的开始所在的伪句子序号
                    end_psnumber = chainobj.chain[-1][1]   #这个词链的结尾所在的伪句子序号
                    try:
                        if start_psnumber > 0 and start_psnumber<self.pseudoseqsum:
                            self.chain_strength_oneword[start_psnumber-1] += 1 #相应位置的词链强度+1
                        if end_psnumber < self.pseudoseqsum - 1:
                            self.chain_strength_oneword[end_psnumber] += 1
                    except IndexError:
                        print('indexerror in computing onewordchain strength')
        elif whichkind == 'twoword':
            self.chain_strength_twoword = [0]*(self.pseudoseqsum - 1) #每个间隔处的词链强度列表,oneword
            for key,value in self.twoword_chain.items():
                for chainobj in value:
                    start_psnumber = chainobj.chain[0][0][1]  #这个词链的开始所在的伪句子序号
                    end_psnumber = chainobj.chain[-1][1][1]   #这个词链的结尾所在的伪句子序号
                    try:
                        if start_psnumber > 0 and start_psnumber<self.pseudoseqsum:
                            self.chain_strength_twoword[start_psnumber-1] += 1 #相应位置的词链强度+1
                        if end_psnumber < self.pseudoseqsum - 1:
                            self.chain_strength_twoword[end_psnumber] += 1
                    except IndexError:
                        print('indexerror in computing twowordchain strength')
        elif whichkind == 'onesyl':
            self.chain_strength_onesyl = [0]*(self.pseudoseqsum - 1) #每个间隔处的词链强度列表,oneword
            for key,value in self.onesyl_chain.items():
                for chainobj in value:
                    start_psnumber = chainobj.chain[0][1]  #这个词链的开始所在的伪句子序号
                    end_psnumber = chainobj.chain[-1][1]   #这个词链的结尾所在的伪句子序号
                    try:
                        if start_psnumber > 0 and start_psnumber<self.pseudoseqsum:
                            self.chain_strength_onesyl[start_psnumber-1] += 1 #相应位置的词链强度+1
                        if end_psnumber < self.pseudoseqsum - 1:
                            self.chain_strength_onesyl[end_psnumber] += 1
                    except IndexError:
                        print('indexerror in computing onesylchain strength')
        elif whichkind == 'twosyl':
            self.chain_strength_twosyl = [0]*(self.pseudoseqsum - 1) #每个间隔处的词链强度列表,oneword
            for key,value in self.twosyl_chain.items():
                for chainobj in value:
                    start_psnumber = chainobj.chain[0][0][1]  #这个词链的开始所在的伪句子序号
                    end_psnumber = chainobj.chain[-1][1][1]   #这个词链的结尾所在的伪句子序号
                    try:
                        if start_psnumber > 0 and start_psnumber<self.pseudoseqsum:
                            self.chain_strength_twosyl[start_psnumber-1] += 1 #相应位置的词链强度+1
                        if end_psnumber < self.pseudoseqsum - 1:
                            self.chain_strength_twosyl[end_psnumber] += 1
                    except IndexError:
                        print('indexerror in computing twosylchain strength')
        else:
            print('whichkind输入错误')
            
    def calc_fm(self, whichkind, threshold, w1=None, w2=None):
        '''
                计算F-measure,threshold是权值,w1和w2是融合时的权值,满足w1+w2=1
                准确率  = 检测到的正确新闻边界数/算法返回的新闻边界数    
                召回率  = 检测到的正确新闻边界数/人工标注的新闻边界数
        F-measure = 2*准确率*召回率/(准确率+召回率)
        '''
        temp_chain_strength = []
        detect_border_num = 0 #超过阈值的边界数
        real_border_num = 0 #检测到的正确新闻边界数
                
        if whichkind == 'oneword':
            temp_chain_strength = self.chain_strength_oneword
        elif whichkind == 'twoword':
            temp_chain_strength = self.chain_strength_twoword
        elif whichkind == 'onesyl':
            temp_chain_strength = self.chain_strength_onesyl
        elif whichkind == 'twosyl':
            temp_chain_strength = self.chain_strength_twosyl
        else:
            print('whichkind error')
            return 0
        for i in range(len(temp_chain_strength)):
            if temp_chain_strength[i] > threshold:
                detect_border_num += 1
                if i in self.approximately_seg:
                    real_border_num += 1
        precision = real_border_num/detect_border_num
        recall = real_border_num/len(self.segmentation)
        F_measure = 2 * precision * recall/(precision + recall)
        return precision,recall,F_measure
        

class Mylist(list):
    '''继承list类,保证可以正常输出含有一堆chain obj的list'''
    def __str__(self):
        returnlist = ""
        for item in self:
            returnlist += str(item)
        return returnlist
        
class chain():
    def __init__(self,start_index):
        self.start_index = start_index    #词链开始的位置,整段话中的第几个字
        self.chain = []         #存储词链的list
        self.max_chain_length = 1100
    
    def add(self,index,item):
        '''向词链中添加元素'''
        if index-self.start_index > self.max_chain_length:
            '''如果跨过了最大长度,则返回1,否则返回0
                        并且这个时候字典需要在同一个key的列表里面添加新元素,不过这是在类外面处理的事情
                        一个chain类只知道自己有什么,不会管别的
            '''
            return 1
        else:
            self.chain.append(item)
            return 0
        
    def __str__(self):
        return str(self.start_index) + str(self.chain)
    
def main(C, sgmfilename, test='no'):
    '''
        根据已经产生的四种词链,计算伪句子分界处的词链强度
        先计算单独使用每一种的结果,再考虑合起来使用的问题
    '''
    C.calc_chain_strength('oneword')
    C.calc_chain_strength('twoword')
    C.calc_chain_strength('onesyl')
    C.calc_chain_strength('twosyl')
    
    if test == 'no':    #实际工作模式
        _,tail = os.path.split(sgmfilename)
        name = tail.rsplit('.sgm')[0]
        chainstrength = {}
        chainstrength['oneword'] = C.chain_strength_oneword
        chainstrength['twoword'] = C.chain_strength_twoword
        chainstrength['onesyl'] = C.chain_strength_onesyl
        chainstrength['twosyl'] = C.chain_strength_twosyl
        pickle.dump(chainstrength,open('chainstrength\\%s.p' % name,'wb'))              
        
    if test == 'yes':   #测试模式
        precision_array = []
        recall_array = []
        F_measure_array = [] 
        threshold_array = [range(20,40),range(55,75),range(10,30),range(55,75)]
        
        plt.figure()
        for i in [1,2,3,4]: 
            x = np.array(threshold_array[i-1])
            precision_array = []
            recall_array = []
            F_measure_array = []
            precision = 0
            recall = 0
            F_measure = 0
            for threshold in threshold_array[i-1]:
                if i == 1:
                    precision,recall,F_measure = C.calc_fm('oneword',threshold)
                elif i == 2:
                    precision,recall,F_measure = C.calc_fm('twoword',threshold)
                elif i == 3:
                    precision,recall,F_measure = C.calc_fm('onesyl',threshold)
                elif i == 4:
                    precision,recall,F_measure = C.calc_fm('twosyl',threshold)
                precision_array.append(precision)
                recall_array.append(recall)
                F_measure_array.append(F_measure)
            y = np.array(precision_array)
            z = np.array(recall_array)
            plt.subplot(220+i)
            plt.plot(x,y,linewidth=2,label='precision')
            plt.plot(x,z,"b--",label="recall")
            plt.xlabel("Threshold")
            plt.ylabel("Volt")
            plt.ylim(0,1)
            #plt.legend()
            if i == 1:
                plt.title("oneword performance curve")
            elif i == 2:
                plt.title("twoword performance curve")
            elif i == 3:
                plt.title("onesyl performance curve")
            elif i == 4:
                plt.title("twosyl performance curve")
            plt.legend()
        plt.show()
        
        '''
        for threshold in threshold_array:
            precision,recall,F_measure = C.calc_fm('oneword',threshold)
            precision_array.append(precision)
            recall_array.append(recall)
            F_measure_array.append(F_measure)
        plt.figure(figsize=(8,4))
        y = np.array(precision_array)
        z = np.array(recall_array)
        plt.plot(x,y,linewidth=2)
        plt.plot(x,z,"b--",label="recall")
        plt.xlabel("Threshold")
        plt.ylabel("Volt")
        plt.ylim(0,1)
        plt.legend()
        plt.title("oneword performance curve")
        plt.show()
        '''
    
    

if __name__ == '__main__':
    maindir = '..\wav_processing\output\\'
    if not os.path.exists('chainstrength'):
        os.mkdir('chainstrength')
    sgmfiles = [f for f in os.listdir(maindir) if f.endswith('sgm')]
    sttfiles = [f for f in os.listdir(maindir) if f.endswith('stt')]
    
    for (sgmfile,sttfile) in zip(sgmfiles,sttfiles):
        C = Chain(maindir+sgmfile,maindir+sttfile,'ChSylName.txt',w=40) 
        main(C, sgmfile) #画图测试,添加test='yes'参数
        print(sgmfile+' finished')
