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
import re
import time
import math
import numpy as np
import matplotlib.pyplot as plt

class Chain:
    def __init__(self,sgmfile,sttfile,pinyinfile):
        #self.__dict__.update(locals())
        self.get_avg_section_length(sgmfile)
        self.stt = open(sttfile,'r')
        self.blocks = self.stt.readlines()
        self.pinyin = open(pinyinfile,'r')
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
        pass
        
    def get_avg_section_length(self,filename):
        '''
                得到某个sgm文件中section的平均长度.现在写的是错的,因为完全没有考虑到sgm中的字和实际识别的是不一样的,长度也
                有差别,所以文字的还是应该以stt为准,而sgm只是提供starttime来和stt里的时间对应上,以此把stt分段。这段程序可能
                可以参考cmp2methods.不用那么复杂,开始时间的差异基本确定在一秒之内,可以写如果一秒之内没有,则报错+结束
        '''
        file = open(filename,'r')
        lines = file.readlines()
        self.section_start = []  #存储section开始的行
        for i in range(len(lines)):
            if lines[i].startswith('<section ') and not lines[i+1].startswith('</section'):
                self.section_start.append(i)
        
        self.section_length = []#存储每个section的长度,统计的是纯字数
        for i in range(len(self.section_start)):
            this_section_length = 0
            this_section_start = self.section_start[i]
            if i == len(self.section_start)-1:
                this_section_end = len(lines)
            else:
                this_section_end = self.section_start[i+1] 
            for linenum in range(this_section_start,this_section_end):
                 if lines[linenum].startswith(' '): #文字行都是以空格开始
                     line = lines[linenum]
                     line = re.sub(r'[ %^,.?\n]','',line)#把带文字的行变为纯文字
                     this_section_length += len(line)
            self.section_length.append(this_section_length)
    
        self.charsum = sum(self.section_length) #计算这篇文字的总字数
        self.pseudoseqsum = math.ceil(self.charsum/40)#计算伪句子的个数
        self.segmentation = [sum(self.section_length[0:i+1])/40-1   #这是实际分界对应的伪句子分界编号
                             for i in range(len(self.section_length))]
        
        
    def make_oneword_list(self): 
        '''创建一维子词构成的list,把识别结果中的所有字先和伪句子编号组成一个tuple,再放入一个list'''
        zi_index = 0  #数了多少个字
        ps_index = 0    #伪句子的编号
        self.zi_list = []    #存储(字,ps_index)的列表
        for block in self.blocks:
            match = re.search(r'(?<=zi=)[^ ]*',block)
            if match:
                self.zi_list += match.group()
        '''把字按40个一段输出到文件 ps.txt,并带上伪句子号,方便之后观察'''
        '''
        ps_file = open('ps.txt','w')
        for zi_index in range(len(self.zi_list)):
            if zi_index % 40 == 0:
                ps_file.write('\n' + str(math.ceil(zi_index/40)) + ':')
            ps_file.write(self.zi_list[zi_index])
        ps_file.close()
        '''
        for zi_index in range(len(self.zi_list)):
            ps_index = int(zi_index/40)
            self.zi_list[zi_index] = (self.zi_list[zi_index],ps_index)
    
    def make_twoword_list(self): 
        '''创建二维子词构成的list,根据之前得到的一维子词list,直接合并即可'''
        #make_oneword_list(stt)
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
        for block in self.blocks:
            match = re.search(r'(?<=sylid=)[^\n]*',block)
            if match:
                self.syl_list += match.group().split()
        for zi_index in range(len(self.syl_list)):
            ps_index = int(zi_index/40)
            add = syl_lines[int(self.syl_list[zi_index])].rstrip('\n')
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
                            self.chain_strength_sylword[start_psnumber-1] += 1 #相应位置的词链强度+1
                        if end_psnumber < self.pseudoseqsum - 1:
                            self.chain_strength_sylword[end_psnumber] += 1
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
            for i in temp_chain_strength:
                 if i > threshold:
                     detect_border_num += 1
                     if i in self.segmentation:
                         real_border_num += 1
            precision = real_border_num/detect_border_num
            recall = real_border_num/len(self.segmentation)
            F_measure = 2 * precesion * recall/(precesion + recall)
            
        elif whichkind == 'twoword':
            temp_chain_strength = self.chain_strength_twoword
        elif whichkind == 'onesyl':
            temp_chain_strength = self.chain_strength_onesyl
        elif whichkind == 'twosyl':
            temp_chain_strength = self.chain_strength_twosyl
        else:
            print('whcihkind error')
            

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
    
def main(C):
    '''
        根据已经产生的四种词链,计算伪句子分界处的词链强度
        先计算单独使用每一种的结果,再考虑合起来使用的问题
    '''
    C.calc_chain_strength('oneword')
    C.calc_fm('oneword',30)
    
    
    
    
    
    

if __name__ == '__main__':
    C = Chain('mc970114.sgm','Mc97114.stt','ChSylName.txt') 
    main(C)
    '''
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    z = np.cos(x**2)
    plt.figure(figsize=(8,4))
    plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
    plt.plot(x,z,"b--",label="$cos(x^2)$")
    plt.xlabel("Time(s)")
    plt.ylabel("Volt")
    plt.title("PyPlot First Example")
    plt.ylim(-1.2,1.2)
    plt.legend()
    plt.show()
'''