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

def time_cmp(sgm_time,stt_time):
    '''
        �Ƚ�sgm��stt�е�time�Ĵ�С���������ǵĲ��sgm��,������ֵ��stt�󷵻ظ�ֵ.����ʱ�䶼��str����
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


class Chain:
    def __init__(self,sgmfile,sttfile,pinyinfile):
        #self.__dict__.update(locals())
        self.stt = open(sttfile,'r')
        self.blocks = self.stt.readlines()
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
        pass
        
    def get_avg_section_length(self,filename):
        '''
                �õ�ĳ��sgm�ļ���section��ƽ������.����д���Ǵ��,��Ϊ��ȫû�п��ǵ�sgm�е��ֺ�ʵ��ʶ����ǲ�һ����,����Ҳ
                �в��,�������ֵĻ���Ӧ����sttΪ׼,��sgmֻ���ṩstarttime����stt���ʱ���Ӧ��,�Դ˰�stt�ֶΡ���γ������
                ���Բο�cmp2methods.������ô����,��ʼʱ��Ĳ������ȷ����һ��֮��,����д���һ��֮��û��,�򱨴�+����
        '''
        file = open(filename,'r')
        lines = file.readlines()
        self.section_start = []  #�洢��׼��section��ʼ��ʱ��,��sgm�ļ���ȡ
        self.section_end = []  #�洢��׼��section������ʱ��,��sgm�ļ���ȡ
        for i in range(len(lines)):
            if lines[i].startswith('<section '):
                start_time = re.search(r'(?<=startTime=)[^ ]*',lines[i])
                self.section_start.append(start_time.group())
                end_time = re.search(r'(?<=endTime=)[^>]*',lines[i])
                self.section_end.append(end_time.group())
        
        '''
            seg_info�洢stt�еķֶ���Ϣ,��һ���ֵ乹�ɵ��б�,��ʽ����:
            [
             {begt:'1:30',length:39,zi:'������...'},
             {begt:'2:00',length:50,zi:'���沥��...'},
             ...
             ]
                         ÿһ���ֵ䶼�Ǿ���ɸѡ�õ���stt�е�ĳһ���Ρ���Щ�α�֤�ܺ���stt��ʶ�������������
                         ����:
                         Ҫ��֤����stt�ļ��ܹ�����ȷ�ķֶ�,��sgm�ļ���Ҫ��ÿ��<section���ܹ���Ӧ
                         ��stt��ĳ��begt,�����section��startTime��begt����1s֮��,���begt��Ϊ�ǷֶεĿ�ʼ
        '''
        self.seg_info = []
        self.zi_list = []    #�洢�ֵ��б�,�ӵ�һ���ֵ����һ����
        
        i = 0   #stt,block�����
        for block in self.blocks:
            is_section_start = 0    #flag,������ʾ���block�Ƿ���Ϊ�ǶεĿ�ʼ
            match = re.search(r'(?<=zi=)[^ ]*',block)
            if match:
                self.zi_list += match.group()   #�ⲽһ������,��ΪҪ�������ַ���zi_list
                stt_start = re.search(r'(?<=begt=)[^ ]*',block).group()#sttÿһ�仰����ʼʱ��
                for sgm_start in self.section_start:
                    if abs(time_cmp(sgm_start,stt_start)) < 1:
                        new_dict = {}  #�������ֵ�,��ζ��һ���µķֶ�
                        new_dict['begt'] = stt_start
                        new_dict['zi'] = match.group()
                        new_dict['length'] = len(new_dict['zi'])
                        self.seg_info.append(new_dict)  
                        is_section_start = 1
                        break
                    elif i>0:
                        pre_stt_start = re.search(r'(?<=begt=)[^ ]*',self.blocks[i-1]).group()#ǰһ��begt
                        end_index = self.section_start.index(sgm_start)-1 if self.section_start.index(sgm_start)>1 else 0
                        sgm_end = self.section_end[end_index]    #��Ӧstart_time���Ǹ�����ʱ��
                        if (time_cmp(sgm_start,pre_stt_start)>0 and time_cmp(sgm_start,stt_start)<0) or\
                            (time_cmp(sgm_end,stt_start)<0 and time_cmp(sgm_start,stt_start)>0):
                            #sgm�������仰��begt֮��
                            new_dict = {}  #�������ֵ�,��ζ��һ���µķֶ�
                            new_dict['begt'] = stt_start
                            new_dict['zi'] = match.group()
                            new_dict['length'] = len(new_dict['zi'])
                            self.seg_info.append(new_dict)  
                            is_section_start = 1
                            break
                if not is_section_start and len(self.seg_info):    #������ǶεĿ�ʼ,��ô��Ҫ���µ�ǰ�Ķ�
                    current_dict = self.seg_info[-1]
                    current_dict['zi'] += match.group()
                    current_dict['length'] = len(current_dict['zi'])
            i += 1

        self.charsum = len(self.zi_list) #������ƪ���ֵ�������
        self.pseudoseqsum = math.ceil(self.charsum/40)#����α���ӵĸ���
        self.section_length = [item['length'] for item in self.seg_info]
        self.segmentation = [sum(self.section_length[0:i+1])/40-1   #����ʵ�ʷֽ��Ӧ��α���ӷֽ���
                             for i in range(len(self.section_length))]
        '''
        approximately_seg�ǰ�segmentation:[11.5,23.975,...]��չ��
        [11,12,23,24,...]������,��������ı߽������list����,����Ϊʶ������ȷ�ķֽ�
        '''
        self.approximately_seg = []
        for i in self.segmentation:
            if int(i)<i:
                self.approximately_seg.extend([int(i),int(i)+1])
            else:#i�����������,������λ�ö�����ȷ��
                self.approximately_seg.extend([i-1,i,i+1])
        
        
    def make_oneword_list(self): 
        '''��֮ǰ�õ���zi_listʶ�����е��������Ⱥ�α���ӱ�����һ��tuple,�ٷ���һ��list'''
        '''���ְ�40��һ��������ļ� ps.txt,������α���Ӻ�,����֮��۲�'''
        '''
        ps_file = open('ps.txt','w')
        for zi_index in range(len(self.zi_list)):
            if zi_index % 40 == 0:
                ps_file.write('\n' + str(math.ceil(zi_index/40)) + ':')
            ps_file.write(self.zi_list[zi_index])
        ps_file.close()
        '''
        zi_index = 0  #���˶��ٸ���
        ps_index = 0    #α���ӵı��
        for zi_index in range(len(self.zi_list)):
            ps_index = int(zi_index/40)
            self.zi_list[zi_index] = (self.zi_list[zi_index],ps_index)
    
    def make_twoword_list(self): 
        '''������ά�Ӵʹ��ɵ�list,����֮ǰ�õ���һά�Ӵ�list,ֱ�Ӻϲ�����.���֮ǰû�д���һά��list��Ҫ�����'''
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
                            self.chain_strength_onesyl[start_psnumber-1] += 1 #��Ӧλ�õĴ���ǿ��+1
                        if end_psnumber < self.pseudoseqsum - 1:
                            self.chain_strength_onesyl[end_psnumber] += 1
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
        elif whichkind == 'twoword':
            temp_chain_strength = self.chain_strength_twoword
        elif whichkind == 'onesyl':
            temp_chain_strength = self.chain_strength_onesyl
        elif whichkind == 'twosyl':
            temp_chain_strength = self.chain_strength_twosyl
        else:
            print('whcihkind error')
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
    C.calc_chain_strength('twoword')
    C.calc_chain_strength('onesyl')
    C.calc_chain_strength('twosyl')
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