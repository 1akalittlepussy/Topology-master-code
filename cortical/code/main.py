from cmath import inf
import sys
import os
from tkinter.tix import INTEGER

#from sympy import im
from utils import *
from calculate import *
import numpy as np
import shutil
from result_struct import *
import pickle as pk
import argparse


parser = argparse.ArgumentParser(description='')
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('--method', choices={'dynamic', 'static'}, default='static', help='dynamic')
parser.add_argument('--src_dir', type=str, default='E:/Topology-master-code/cortical/process',help='数据地址')
parser.add_argument('--regionSeries_name', type=str, help='')
parser.add_argument('--in_one_dir', choices={'y', 'n'},  help='y')
parser.add_argument('--tar_dir', type=str, default='E:/Topology-master-code/results',help='结果地址')
parser.add_argument('--key_name_in_mat', type=str, default='cortical',help='关键词')
parser.add_argument('--node_num', type=int, default=90, help='截取的mat中脑区个数')
parser.add_argument('--feature_dim', type=int, help='截取的mat中脑区特征维数')
parser.add_argument('--min_feature_dim', type=int, help='最小脑区特征维数，小于的不进行处理')
parser.add_argument('--copy_src_mat', choices={'y', 'n'}, default='n', help='是否拷贝原始时间序列文件到目标样本文件夹下')
parser.add_argument('--windows_size', type=int, default=50, help='滑动窗口大小')

args = parser.parse_args()

remove_file_name = [
    'ROISignals_S19-1-0002.mat',
    'ROISignals_S19-1-0004.mat',
    'ROISignals_S19-1-0006.mat',
    'ROISignals_S19-1-0007.mat',
    'ROISignals_S19-1-0008.mat',
    'ROISignals_S19-1-0009.mat',
    'ROISignals_S19-1-0011.mat',
    'ROISignals_S19-1-0013.mat',
    'ROISignals_S19-1-0014.mat',
    'ROISignals_S19-1-0016.mat',
    'ROISignals_S19-1-0017.mat',
    'ROISignals_S19-1-0018.mat',
    'ROISignals_S19-1-0020.mat',
    'ROISignals_S19-1-0022.mat',
    'ROISignals_S19-1-0024.mat',
    'ROISignals_S19-1-0026.mat',
    'ROISignals_S19-1-0027.mat',
    'ROISignals_S19-1-0028.mat',
    'ROISignals_S21-1-0032.mat',
    'ROISignals_S2-2-0018.mat',
    'ROISignals_S2-2-0019.mat',
    'ROISignals_S8-2-0008.mat',
    'ROISignals_S4-1-0001.mat',
    'ROISignals_S4-1-0002.mat',
    'ROISignals_S4-1-0003.mat',
    'ROISignals_S4-1-0004.mat',
    'ROISignals_S4-1-0005.mat',
    'ROISignals_S4-1-0006.mat',
    'ROISignals_S4-1-0007.mat',
    'ROISignals_S4-1-0008.mat',
    'ROISignals_S4-1-0009.mat',
    'ROISignals_S4-1-0010.mat',
    'ROISignals_S4-1-0011.mat',
    'ROISignals_S4-1-0012.mat',
    'ROISignals_S4-1-0013.mat',
    'ROISignals_S4-1-0014.mat',
    'ROISignals_S4-1-0015.mat',
    'ROISignals_S4-1-0016.mat',
    'ROISignals_S4-1-0017.mat',
    'ROISignals_S4-1-0018.mat',
    'ROISignals_S4-1-0019.mat',
    'ROISignals_S4-1-0020.mat',
    'ROISignals_S4-1-0021.mat',
    'ROISignals_S4-1-0022.mat',
    'ROISignals_S4-1-0023.mat',
    'ROISignals_S4-1-0024.mat',
    'ROISignals_S4-2-0001.mat',
    'ROISignals_S4-2-0002.mat',
    'ROISignals_S4-2-0003.mat',
    'ROISignals_S4-2-0004.mat',
    'ROISignals_S4-2-0005.mat',
    'ROISignals_S4-2-0006.mat',
    'ROISignals_S4-2-0007.mat',
    'ROISignals_S4-2-0008.mat',
    'ROISignals_S4-2-0009.mat',
    'ROISignals_S4-2-0010.mat',
    'ROISignals_S4-2-0011.mat',
    'ROISignals_S4-2-0012.mat',
    'ROISignals_S4-2-0013.mat',
    'ROISignals_S4-2-0014.mat',
    'ROISignals_S4-2-0015.mat',
    'ROISignals_S4-2-0016.mat',
    'ROISignals_S4-2-0017.mat',
    'ROISignals_S4-2-0018.mat',
    'ROISignals_S4-2-0019.mat',
    'ROISignals_S4-2-0020.mat',
    'ROISignals_S4-2-0021.mat',
    'ROISignals_S4-2-0022.mat',
    'ROISignals_S4-2-0023.mat',
    'ROISignals_S4-2-0024.mat'

]

remove_dir_name = [
    
]


src_dir_path = ''
target_dir_path = ''
regionSeries_name = ''
key_name_in_mat = ''
in_one_dir = ''
node_num = 0
feature_dim = 0
min_feature_dim = 0
copy_src_mat = True

def process_static_in_one_dir(src_dir_path, target_dir_path):
    '''
        src_dir_path: 源文件夹目录
        target_dir_path: 结果文件夹目录
    '''

    filelist=os.listdir(src_dir_path)

    for file_name in filelist:

        if file_name in remove_file_name:
            continue

        file_path = src_dir_path + "/" + file_name

        # 加载数据mat shape(m * n)
        source_mat = load_data(file_path, key_name_in_mat)

        m, n = np.shape(source_mat)

        # 特征数量小于170的不要
        if m < min_feature_dim:
            continue

        # 截取前90列有效数据
        source_mat = source_mat[:feature_dim,:node_num]

        # 输入source_mat，（170 * 90），90个脑区，200个时间序列
        result_dict = calculate_static_topology_by_source_mat(source_mat)

        single_dir_path = target_dir_path + "/" + file_name[:-4]
        os.makedirs(single_dir_path)

        res2save = static_result_struct(result_dict)
        res2save.source_mat = source_mat

        with open('{}/{}.pkl'.format(single_dir_path, "calStaticRes"),"wb") as file:
            pk.dump(res2save, file)

        # copy源文件到结果文件夹
        if copy_src_mat:
            shutil.copy(file_path, single_dir_path)


def process_dynamic_in_one_dir(src_dir_path, target_dir_path):
    '''
        src_dir_path: 源文件夹目录
        target_dir_path: 结果文件夹目录
    '''

    filelist=os.listdir(src_dir_path)

    for file_name in filelist:

        if file_name in remove_file_name:
            continue

        file_path = src_dir_path + "/" + file_name

        # 加载数据mat shape(m * n)
        source_mat = load_data(file_path, key_name_in_mat)

        m, n = np.shape(source_mat)

        # 特征数量小于170的不要
        if m < min_feature_dim:
            continue

        # 截取前90列有效数据
        source_mat = source_mat[:feature_dim,:node_num]

        #source_mat = source_mat[:feature_dim,229:429]
        # 输入source_mat，（200 * 90），90个脑区，200个时间序列
        res2save = calculate_dynamic_topology_by_source_mat(source_mat, sliding_window_size=args.windows_size)

        single_dir_path = target_dir_path + "/" + file_name[:-4]
        os.makedirs(single_dir_path)
        file_name_1=file_name.strip('.mat')
        filename="result_"+file_name_1

        #with open('{}/{}.pkl'.format(single_dir_path, "calDynamicRes"),"wb") as file:
        with open('{}/{}.pkl'.format(single_dir_path, filename), "wb") as file:
            pk.dump(res2save, file)

        # copy源文件到结果文件夹
        if copy_src_mat:
            shutil.copy(file_path, single_dir_path)


def process_static_not_in_one_dir(src_dir_path, target_dir_path):
    '''
        src_dir_path: 源文件夹目录
        target_dir_path: 结果文件夹目录
    '''

    case_dir_list = os.listdir(src_dir_path)

    for case_dir_name in case_dir_list:

        if case_dir_name in remove_dir_name:
            continue

        file_path = src_dir_path + "/" + case_dir_name + "/" + regionSeries_name

        # 加载数据mat shape(m * n)
        source_mat = load_data(file_path, key_name_in_mat)

        m, n = np.shape(source_mat)

        # 特征数量小于170的不要
        if m < min_feature_dim:
            continue

        # 截取前90列有效数据
        source_mat = source_mat[:feature_dim,:node_num]

        # 输入source_mat，（170 * 90），90个脑区，200个时间序列
        result_dict = calculate_static_topology_by_source_mat(source_mat)

        single_dir_path = target_dir_path + "/" + case_dir_name
        os.makedirs(single_dir_path)

        res2save = static_result_struct(result_dict)
        res2save.source_mat = source_mat

        with open('{}/{}.pkl'.format(single_dir_path, "calStaticRes"),"wb") as file:
            pk.dump(res2save, file)

        # copy源文件到结果文件夹
        if copy_src_mat:
            shutil.copy(file_path, single_dir_path)


def process_dynamic_not_in_one_dir(src_dir_path, target_dir_path):
    '''
        src_dir_path: 源文件夹目录
        target_dir_path: 结果文件夹目录
    '''

    case_dir_list = os.listdir(src_dir_path)

    for case_dir_name in case_dir_list:

        if case_dir_name in remove_dir_name:
            continue

        file_path = src_dir_path + "/" + case_dir_name + "/" + regionSeries_name

        # 加载数据mat shape(m * n)
        source_mat = load_data(file_path, key_name_in_mat)

        m, n = np.shape(source_mat)

        # 特征数量小于170的不要
        if m < min_feature_dim:
            continue

        # 截取前90列有效数据
        source_mat = source_mat[:feature_dim,:node_num]

        # 输入source_mat，（200 * 90），90个脑区，200个时间序列
        res2save = calculate_dynamic_topology_by_source_mat(source_mat, sliding_window_size=args.windows_size)

        single_dir_path = target_dir_path + "/" + case_dir_name
        os.makedirs(single_dir_path)

        with open('{}/{}.pkl'.format(single_dir_path, "calDynamicRes"),"wb") as file:
            pk.dump(res2save, file)

        # copy源文件到结果文件夹
        if copy_src_mat:
            shutil.copy(file_path, single_dir_path)


if __name__ == '__main__':
    src_dir_path = args.src_dir
    target_dir_path = args.tar_dir
    regionSeries_name = args.regionSeries_name
    in_one_dir = args.in_one_dir
    node_num = args.node_num
    feature_dim = args.feature_dim if args.feature_dim is not None else sys.maxsize
    min_feature_dim = args.min_feature_dim if args.min_feature_dim is not None else 0
    key_name_in_mat = args.key_name_in_mat
    copy_src_mat = False if args.copy_src_mat == 'n' else True

    if in_one_dir == 'y':

        if args.method == 'dynamic':
            process_dynamic_in_one_dir(src_dir_path, target_dir_path)
        
        if args.method == 'static':
            process_static_in_one_dir(src_dir_path, target_dir_path)
    
    else:
        if args.method == 'dynamic':
            process_dynamic_not_in_one_dir(src_dir_path, target_dir_path)
        
        if args.method == 'static':
            process_static_not_in_one_dir(src_dir_path, target_dir_path)
        




'''
example
python  .\main.py --method static --src_dir E:/timeSeries处理/HCP/src/fMRI/Resting2/female --regionSeries_name RegionSeries.mat --in_one_dir n --tar_dir E:/timeSeries处理/HCP/target/Resting2/female --key_name_in_mat RegionSeries --node_num 90 

python .\main.py --method dynamic --src_dir ***/***/multi-site/HC --in_one_dir y --tar_dir ***/***/multi-site/windows_size_50/HC --key_name_in_mat ROISignals_AAL --node_num 90 --feature_dim 170 --min_feature_dim 170 --copy_src_mat n --windows_size 50
'''
