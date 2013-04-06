# -*- coding:gbk -*-
'''
�ó��������ʶ����stt�ļ��аѴ����ҳ���,����ͳ��ÿ��α���ӷֽ紦�Ĵ���ǿ��
�ڵõ�section����ʵ����ʱ,�õ����ֹ���ע��sgm�ļ�
�ڻ�ȡ����ʱ,�õ���ʶ����stt�ļ�
�������壺
max_chain_length: ����������,��ʼֵ�趨Ϊsection������Ǹ��ĳ���(�ֵĸ���)
w: Pseudosentence size,α���ӳ���,�����趨Ϊ40��
charsum: ������ƪ���ֵ�������
pseudoseqsum: α���ӵĸ���
one_word: һԪ�Ӵ�
two_word: ��Ԫ�Ӵ�
one_syllable: һԪ����
two_syllable: ��Ԫ����
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
                �õ�ĳ��sgm�ļ���section��ƽ������.����д���Ǵ��,��Ϊ��ȫû�п��ǵ�sgm�е��ֺ�ʵ��ʶ����ǲ�һ����,����Ҳ
                �в��,�������ֵĻ���Ӧ����sttΪ׼,��sgmֻ���ṩstarttime����stt���ʱ���Ӧ��,�Դ˰�stt�ֶΡ���γ������
                ���Բο�cmp2methods.������ô����,��ʼʱ��Ĳ������ȷ����һ��֮��,����д���һ��֮��û��,�򱨴�+����
        '''
        file = open(filename,'r')
        lines = file.readlines()
        self.section_start = []  #�洢section��ʼ����
        for i in range(len(lines)):
            if lines[i].startswith('<section ') and not lines[i+1].startswith('</section'):
                self.section_start.append(i)
        
        self.section_length = []#�洢ÿ��section�ĳ���,ͳ�Ƶ��Ǵ�����
        for i in range(len(self.section_start)):
            this_section_length = 0
            this_section_start = self.section_start[i]
            if i == len(self.section_start)-1:
                this_section_end = len(lines)
            else:
                this_section_end = self.section_start[i+1] 
            for linenum in range(this_section_start,this_section_end):
                 if lines[linenum].startswith(' '): #�����ж����Կո�ʼ
                     line = lines[linenum]
                     line = re.sub(r'[ %^,.?\n]','',line)#�Ѵ����ֵ��б�Ϊ������
                     this_section_length += len(line)
            self.section_length.append(this_section_length)
    
        self.charsum = sum(self.section_length) #������ƪ���ֵ�������
        self.pseudoseqsum = math.ceil(self.charsum/40)#����α���ӵĸ���
        self.segmentation = [sum(self.section_length[0:i+1])/40-1   #����ʵ�ʷֽ��Ӧ��α���ӷֽ���
                             for i in range(len(self.section_length))]
        
        
    def make_oneword_list(self): 
        '''����һά�Ӵʹ��ɵ�list,��ʶ�����е��������Ⱥ�α���ӱ�����һ��tuple,�ٷ���һ��list'''
        zi_index = 0  #���˶��ٸ���
        ps_index = 0    #α���ӵı��
        self.zi_list = []    #�洢(��,ps_index)���б�
        for block in self.blocks:
            match = re.search(r'(?<=zi=)[^ ]*',block)
            if match:
                self.zi_list += match.group()
        '''���ְ�40��һ��������ļ� ps.txt,������α���Ӻ�,����֮��۲�'''
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
        '''������ά�Ӵʹ��ɵ�list,����֮ǰ�õ���һά�Ӵ�list,ֱ�Ӻϲ�����'''
        #make_oneword_list(stt)
        self.two_zi_list = []
        for i in range(len(self.zi_list)-1):
            add = [self.zi_list[i],self.zi_list[i+1]]
            self.two_zi_list.append(add)
            
    def make_onesyllable_list(self,pinyin): 
        '''����һά���ڹ��ɵ�list,����stt�е�sylid,��ƴ�������ҵ���Ӧƴ��,�滻��ԭ������'''
        zi_index = 0  #���˶��ٸ���
        ps_index = 0    #α���ӵı��
        self.syl_list = []    #�洢(syl,ps_index)���б�
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
        '''������ά�Ӵʹ��ɵ�list,����֮ǰ�õ���һά�Ӵ�list,ֱ�Ӻϲ�����'''
        #make_onesyllable_list(stt)
        self.two_syl_list = []
        for i in range(len(self.syl_list)-1):
            add = [self.syl_list[i],self.syl_list[i+1]]
            self.two_syl_list.append(add)
    
    def make_oneword_chain(self):
        '''����һά�Ӵ���,�γ�{chainword1:[chainobj1_1,chainobj1_2],chainword2:[chainobj2_1,chainobj2_2]}������dict'''
        self.oneword_chain = {}   #dict
        i = 0
        for zi in self.zi_list: #һ���ǵ�zi��tuple
            key = zi[0]
            if key in list(self.oneword_chain.keys()):
                 if self.oneword_chain[key][-1].add(i,zi):
                     newchain = chain(i)
                     newchain.add(i, zi)
                     self.oneword_chain[key].append(newchain)
            else:
                newchain = chain(i)
                newchain.add(i, zi)
                self.oneword_chain[key] = Mylist()  #û��key��ʱ����������һ�仰,����һ��list��Ϊ��key��value
                self.oneword_chain[key].append(newchain)
            i = i + 1
    
    def make_twoword_chain(self):
        '''�����ά�Ӵ���,��ʽ��һά��ͬ,ֻ��key���в��'''
        self.twoword_chain = {}   #dict
        i = 0
        for zi in self.two_zi_list: #һ���ǵ�zi��tuple
            key = zi[0][0]+zi[1][0] #��'��'+'��'='����'��Ϊkey
            if key in list(self.twoword_chain.keys()):
                 if self.twoword_chain[key][-1].add(i,zi):
                     newchain = chain(i)
                     newchain.add(i, zi)
                     self.twoword_chain[key].append(newchain)
            else:
                newchain = chain(i)
                newchain.add(i, zi)
                self.twoword_chain[key] = Mylist()  #û��key��ʱ����������һ�仰,����һ��list��Ϊ��key��value
                self.twoword_chain[key].append(newchain)
            i = i + 1

    def make_onesyl_chain(self):
        '''����һά������,�γ�{chainword1:[chainobj1_1,chainobj1_2],chainword2:[chainobj2_1,chainobj2_2]}������dict'''
        self.onesyl_chain = {}   #dict
        i = 0
        for syl in self.syl_list: #һ���ǵ�zi��tuple
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
        '''�����ά������'''
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
                self.twosyl_chain[key] = Mylist()  #û��key��ʱ����������һ�仰,����һ��list��Ϊ��key��value
                self.twosyl_chain[key].append(newchain)
            i = i + 1
            #if len(self.twosyl_chain[key]) > 1:
                #print(self.twosyl_chain[key])
    
    def calc_chain_strength(self,whichkind):
        #�������ǿ��,����whichkind�Ĳ�ͬ�����Ǽ�����һ�ִ���ǿ��
        if whichkind == 'oneword':
            self.chain_strength_oneword = [0]*(self.pseudoseqsum - 1) #ÿ��������Ĵ���ǿ���б�,oneword
            for key,value in self.oneword_chain.items():
                for chainobj in value:
                    start_psnumber = chainobj.chain[0][1]  #��������Ŀ�ʼ���ڵ�α�������
                    end_psnumber = chainobj.chain[-1][1]   #��������Ľ�β���ڵ�α�������
                    try:
                        if start_psnumber > 0 and start_psnumber<self.pseudoseqsum:
                            self.chain_strength_oneword[start_psnumber-1] += 1 #��Ӧλ�õĴ���ǿ��+1
                        if end_psnumber < self.pseudoseqsum - 1:
                            self.chain_strength_oneword[end_psnumber] += 1
                    except IndexError:
                        print('indexerror in computing onewordchain strength')
        elif whichkind == 'twoword':
            self.chain_strength_twoword = [0]*(self.pseudoseqsum - 1) #ÿ��������Ĵ���ǿ���б�,oneword
            for key,value in self.twoword_chain.items():
                for chainobj in value:
                    start_psnumber = chainobj.chain[0][0][1]  #��������Ŀ�ʼ���ڵ�α�������
                    end_psnumber = chainobj.chain[-1][1][1]   #��������Ľ�β���ڵ�α�������
                    try:
                        if start_psnumber > 0 and start_psnumber<self.pseudoseqsum:
                            self.chain_strength_twoword[start_psnumber-1] += 1 #��Ӧλ�õĴ���ǿ��+1
                        if end_psnumber < self.pseudoseqsum - 1:
                            self.chain_strength_twoword[end_psnumber] += 1
                    except IndexError:
                        print('indexerror in computing twowordchain strength')
        elif whichkind == 'onesyl':
            self.chain_strength_onesyl = [0]*(self.pseudoseqsum - 1) #ÿ��������Ĵ���ǿ���б�,oneword
            for key,value in self.onesyl_chain.items():
                for chainobj in value:
                    start_psnumber = chainobj.chain[0][1]  #��������Ŀ�ʼ���ڵ�α�������
                    end_psnumber = chainobj.chain[-1][1]   #��������Ľ�β���ڵ�α�������
                    try:
                        if start_psnumber > 0 and start_psnumber<self.pseudoseqsum:
                            self.chain_strength_sylword[start_psnumber-1] += 1 #��Ӧλ�õĴ���ǿ��+1
                        if end_psnumber < self.pseudoseqsum - 1:
                            self.chain_strength_sylword[end_psnumber] += 1
                    except IndexError:
                        print('indexerror in computing onesylchain strength')
        elif whichkind == 'twosyl':
            self.chain_strength_twosyl = [0]*(self.pseudoseqsum - 1) #ÿ��������Ĵ���ǿ���б�,oneword
            for key,value in self.twosyl_chain.items():
                for chainobj in value:
                    start_psnumber = chainobj.chain[0][0][1]  #��������Ŀ�ʼ���ڵ�α�������
                    end_psnumber = chainobj.chain[-1][1][1]   #��������Ľ�β���ڵ�α�������
                    try:
                        if start_psnumber > 0 and start_psnumber<self.pseudoseqsum:
                            self.chain_strength_twosyl[start_psnumber-1] += 1 #��Ӧλ�õĴ���ǿ��+1
                        if end_psnumber < self.pseudoseqsum - 1:
                            self.chain_strength_twosyl[end_psnumber] += 1
                    except IndexError:
                        print('indexerror in computing twosylchain strength')
        else:
            print('whichkind�������')
            
    def calc_fm(self, whichkind, threshold, w1=None, w2=None):
        '''
                ����F-measure,threshold��Ȩֵ,w1��w2���ں�ʱ��Ȩֵ,����w1+w2=1
                ׼ȷ��  = ��⵽����ȷ���ű߽���/�㷨���ص����ű߽���    
                �ٻ���  = ��⵽����ȷ���ű߽���/�˹���ע�����ű߽���
        F-measure = 2*׼ȷ��*�ٻ���/(׼ȷ��+�ٻ���)
        '''
        temp_chain_strength = []
        detect_border_num = 0 #������ֵ�ı߽���
        real_border_num = 0 #��⵽����ȷ���ű߽���
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
    '''�̳�list��,��֤���������������һ��chain obj��list'''
    def __str__(self):
        returnlist = ""
        for item in self:
            returnlist += str(item)
        return returnlist
        
class chain():
    def __init__(self,start_index):
        self.start_index = start_index    #������ʼ��λ��,���λ��еĵڼ�����
        self.chain = []         #�洢������list
        self.max_chain_length = 1100
    
    def add(self,index,item):
        '''����������Ԫ��'''
        if index-self.start_index > self.max_chain_length:
            '''����������󳤶�,�򷵻�1,���򷵻�0
                        �������ʱ���ֵ���Ҫ��ͬһ��key���б����������Ԫ��,���������������洦�������
                        һ��chain��ֻ֪���Լ���ʲô,����ܱ��
            '''
            return 1
        else:
            self.chain.append(item)
            return 0
        
    def __str__(self):
        return str(self.start_index) + str(self.chain)
    
def main(C):
    '''
        �����Ѿ����������ִ���,����α���ӷֽ紦�Ĵ���ǿ��
        �ȼ��㵥��ʹ��ÿһ�ֵĽ��,�ٿ��Ǻ�����ʹ�õ�����
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