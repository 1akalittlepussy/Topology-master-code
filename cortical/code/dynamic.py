import os
import scipy.io
import  numpy as np
import nibabel as nib
from scipy.stats import ttest_ind
#from statsmodels.stats import multitest
#from statsmodels.stats.multitest import fdrcorrection
import openpyxl
from calculate import calculate_static_topology_by_source_mat
from result_struct import static_result_struct
import pickle as pk
import pandas as pd

#def getFiles(dir,suffix) :
#  res=[]
#  for root ,directory, files in os.walk(dir):
#    spl = directory
#    res.append(spl)
#    for filename in files:
#        name, suf = os.path.splitext(filename)
#        if suf == suffix:
#        res.append(filename)
#      return res

def getFiles(dir,suffix) :
  res=[]
  for root ,directory, files in os.walk(dir):
        for filename in files:
          name, suf = os.path.splitext(filename)
          if suf == suffix:
            res.append(filename)
  return res

def getDirs(dir):
    res = []
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for dirs in directory:
            res.append(dirs)

    # print(res)
    return res

def getDynamic(path):
    #创建保存文件路径matpath

    # print(path)
    pa=path
    a = path.split('\\')
    folder = a[len(a) - 3]  # 原文件夹名
    # print(folder)
    a = path.split('-')
    if a[len(a) - 2] == '1':
        a = path.split('\\')
        del a[-2]
        path=('\\'.join(a))
        # print(path)
        matPath = path.replace(folder, 'Dynamic_' + folder + '_MDD')
        matPath = matPath.replace('.pkl', '.mat')
        print(matPath)
    else:
        a = path.split('\\')
        del a[-2]
        path = ('\\'.join(a))
        # print(path)
        matPath = path.replace(folder, 'Dynamic_' + folder + '_HC')
        matPath = matPath.replace('.pkl', '.mat')
        print(matPath)
    with open(pa , "rb") as f:
        result_struct = pk.load(f)
        loc_std=result_struct.info_in_graph['local_efficiency']['std']
        loc_mean=result_struct.info_in_graph['local_efficiency']['mean']
        glo_std=result_struct.info_in_graph['global_efficiency']['std']
        glo_mean = result_struct.info_in_graph['global_efficiency']['mean']
        avgPaL_std = result_struct.info_in_graph['avg_path_length']['std']
        avgPaL_mean = result_struct.info_in_graph['avg_path_length']['mean']
        mod_std = result_struct.info_in_graph['modularity_q']['std']
        mod_mean = result_struct.info_in_graph['modularity_q']['mean']
        avgCluster_std = result_struct.info_in_graph['average_clustering']['std']
        avgCluster_mean = result_struct.info_in_graph['average_clustering']['mean']
        nodEff_std=result_struct.info_in_nodes["nodal_efficiency"]["std"]
        nodEff_mean=result_struct.info_in_nodes["nodal_efficiency"]["mean"]
        degree_std = result_struct.info_in_nodes["degree"]["std"]
        degree_mean = result_struct.info_in_nodes["degree"]["mean"]
        clus_std = result_struct.info_in_nodes["clustering"]["std"]
        clus_mean = result_struct.info_in_nodes["clustering"]["mean"]


        dynamic=[]
        dynamic.append(loc_std)
        dynamic.append(loc_mean)
        dynamic.append(glo_std)
        dynamic.append(glo_mean)
        dynamic.append(avgPaL_std)
        dynamic.append(avgPaL_mean)
        dynamic.append(mod_std)
        dynamic.append(mod_mean)
        dynamic.append(avgCluster_std)
        dynamic.append(avgCluster_mean)
        dynamic = np.concatenate((dynamic, nodEff_std,nodEff_mean,degree_std,degree_mean,clus_std,clus_mean))

        # print(dynamic.shape)
        scipy.io.savemat(matPath, {'dynamic_mat': dynamic})
'''
   批处理动态指标pkl文件
'''
def batchPklFiles(path):
    # 创建保存mat文件的文件夹
    a = path.split('\\')

    folder = a[len(a) - 1]  # 原文件夹名
    a[len(a) - 1] = 'Dynamic_' + a[len(a) - 1] + '_HC'
    newPathHC = ('\\'.join(a))
    # print(newPathHC)
    isExists = os.path.exists(newPathHC)
    if not isExists:
        os.makedirs(newPathHC)
    else:
        print("dir is exist,don't need to create")

    a[len(a) - 1] = folder
    a[len(a) - 1] = 'Dynamic_' + a[len(a) - 1] + '_MDD'
    newPathMDD = ('\\'.join(a))
    # print(newPathMDD)
    isExists = os.path.exists(newPathMDD)
    if not isExists:
        os.makedirs(newPathMDD)
    else:
        print("dir is exist,don't need to create")

    # path='E:\dynamic_no_mat\ROISignals_FunImgARCWF\ROISignals_S1-1-0001\result_ROISignals_S1-1-0001.pkl'
    # path=path.replace('\r','\\r')
    for file in getFiles(path, '.pkl'):
        #字符串处理
        a = file.split('t_')
        a = a[-1].split('.')
        matPath = os.path.join(path,file,'calDynamicRes.pkl')
        # print(matPath)
        getDynamic(matPath)

def matMerge(path,matNameF,matNameT,matNameD):
  sub_dir = getDirs(path)  # 获取子文件夹名
  for dir in sub_dir:
    d = dir.split('_')  # 分割文件夹名称获取有用信息
    if (d[0] == 'Feature'):  # 获取含.mat文件的文件夹
      # 包装用来保存合并后MAT的文件夹地址 such as Mat_ALFF_FunImgARCW_MDD
      d[0] = 'Mat_Feature'
      MatPath = path + '\\' + ('_'.join(d))
      isExists = os.path.exists(MatPath)
      if not isExists:
        os.makedirs(MatPath)
      else:
        print("dir is exist,don't need to create")
      # 包装含.mat文件的文件夹地址
      newPath = path + '\\' + dir
      # print(MatPath)
      # print(newPath)
      mat = []
      # 遍历Feature文件夹下所有mat文件，读出并写入到同一个mat中
      for file in getFiles(newPath, '.mat'):
        f = newPath + '\\' + file
        data = scipy.io.loadmat(f)
        a = data[matNameF][0]  # 1*4005的矩阵
        if np.isnan(a).sum() > 0:
          print(f+'下含有NaN数组')
        else:
          mat.append(a)
        # print(f)

      # 生成合并后的.mat文件
      mat = np.array(mat)
      MatPath = MatPath + '\Feature.mat'  # 包装保存合并后MAT的文件名
      scipy.io.savemat(MatPath, {'mat': mat})

    elif (d[0] == 'Target'):  # 获取含.mat文件的文件夹
      # 包装用来保存合并后MAT的文件夹地址 such as Mat_ALFF_FunImgARCW_MDD
      d[0] = 'Mat_Target'
      MatPath = path + '\\' + ('_'.join(d))
      isExists = os.path.exists(MatPath)
      if not isExists:
        os.makedirs(MatPath)
      else:
        print("dir is exist,don't need to create")
      # 包装含.mat文件的文件夹地址
      newPath = path + '\\' + dir
      # print(MatPath)
      # print(newPath)
      mat = []
      # 遍历Feature文件夹下所有mat文件，读出并写入到同一个mat中
      for file in getFiles(newPath, '.mat'):
        f = newPath + '\\' + file
        data = scipy.io.loadmat(f)
        a = data[matNameT]  # char
        # a = data[matNameT][0]  # 1*4005的矩阵
        # if np.isnan(a).sum() > 0:
        #   print(f+'下含有NaN数组')
        # else:
        mat.append(a)
        # print(f)

      # 生成合并后的.mat文件
      mat = np.array(mat)
      MatPath = MatPath + '\Target.mat'  # 包装保存合并后MAT的文件名
      scipy.io.savemat(MatPath, {'mat': mat})

    elif (d[0] == 'Dynamic'):  # 获取含.mat文件的文件夹
      # 包装用来保存合并后MAT的文件夹地址 such as Mat_ALFF_FunImgARCW_MDD
      d[0] = 'Mat_Dynamic'
      MatPath = path + '\\' + ('_'.join(d))
      isExists = os.path.exists(MatPath)
      if not isExists:
          os.makedirs(MatPath)
      else:
          print("dir is exist,don't need to create")
      # 包装含.mat文件的文件夹地址
      newPath = path + '\\' + dir
      # print(MatPath)
      # print(newPath)
      mat = []
      # 遍历Feature文件夹下所有mat文件，读出并写入到同一个mat中
      for file in getFiles(newPath, '.mat'):
          f = newPath + '\\' + file
          data = scipy.io.loadmat(f)
          a = data[matNameD][0]  # 1*4005的矩阵
          if np.isnan(a).sum() > 0:
              print(f + '下含有NaN数组')
          else:
              mat.append(a)
          # print(f)

      # 生成合并后的.mat文件
      mat = np.array(mat)
      MatPath = MatPath + '\Dynamic.mat'  # 包装保存合并后MAT的文件名
      scipy.io.savemat(MatPath, {'mat': mat})

def t_test(hc, mdd):
  # 两类样本t检验
  J = hc.shape[1]
  pvalues = np.zeros(J)
  for j in range(J):
    var_hc = hc[:, j]
    var_mdd = mdd[:, j]
    res = ttest_ind(var_hc, var_mdd)
    pvalues[j] = res.pvalue
  # print(pvalues)
  return pvalues

def getPvalues(pathHC,pathMDD,xlsx):
    hc_data = scipy.io.loadmat(pathHC)['mat']
    mdd_data = scipy.io.loadmat(pathMDD)['mat']
    # # 随机打乱
    # np.random.seed(1)
    # # 随机种子
    # indices_hc = np.random.permutation(hc_data.shape[0])  # x.shape[0] x第一维长度 1125份样本
    # indices_mdd = np.random.permutation(mdd_data.shape[0])
    # hc_data_in = hc_data[indices_hc]  # 切片
    # mdd_data_in = mdd_data[indices_mdd]
    # calculate P
    pvalues = t_test(hc_data, mdd_data)  # 双样本t检验
    psum = np.sum(pvalues <= 0.05)
    print(psum)
    pdata = []
    for i in range(pvalues.shape[0]):
        if pvalues[i] <= 0.05:

            temp = []
            # temp.append(i)
            # print(hc_data[:,i].sum())
            # print(hc_data.shape[0])
            avgHC = hc_data[:, i].sum() / hc_data.shape[0]
            avgMDD = mdd_data[:, i].sum() / mdd_data.shape[0]

            if i == 0:
                temp.append('local_efficiency_std')
            elif i == 1:
                temp.append('local_efficiency_mean')
            elif i == 2:
                temp.append('global_efficiency_std')
            elif i == 3:
                temp.append('global_efficiency_mean')
            elif i == 4:
                temp.append('avg_path_length_std')
            elif i == 5:
                temp.append('avg_path_length_mean')
            elif i == 6:
                temp.append('modularity_q_std')
            elif i == 7:
                temp.append('modularity_q_mean')
            elif i == 8:
                temp.append('average_clustering_std')
            elif i == 9:
                temp.append('average_clustering_mean')
            elif i in range(10, 100):
                temp.append('The ' + str(i - 9) + "th brain region nodal_efficiency_std ")
            elif i in range(100, 190):
                temp.append('The ' + str(i - 99) + "th brain region nodal_efficiency_mean ")
            elif i in range(190, 280):
                temp.append('The ' + str(i - 189) + "th brain region degree_std ")
            elif i in range(280, 370):
                temp.append('The ' + str(i - 269) + "th brainc region degree_mean ")
            elif i in range(370, 460):
                temp.append('The ' + str(i - 359) + "th brain region clustering_std ")
            elif i in range(460, 550):
                temp.append('The ' + str(i - 449) + "th brain region clustering_mean ")
            temp.append(avgHC)
            temp.append(avgMDD)
            temp.append(pvalues[i])

            pdata.append(temp)
    pindex = np.where(pvalues <= 0.05)
    print(pindex[0])
    pdata = np.array(pdata, dtype=object)
    data = pd.DataFrame(pdata)
    data.columns = ['dynamic_index', 'avgHC', 'avgMDD', 'pvalue']
    writer = pd.ExcelWriter(xlsx)  # 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.10f')  # ‘page_1’是写入excel的sheet名 float_format小数点后几位
    writer.save()
    writer.close()

    # print(pdata)

if __name__ == '__main__':

    # 处理动态指标，提取显著特征
    path = 'E:\dynamic_no_mat\ROISignals_FunImgARCWF'#所有pkl所在的根文件夹
    #path = 'E:/Topology-master-code/result50'
    batchPklFiles(path)
    matMerge('E:\dynamic_no_mat', 'con_feature_mat', 'target_mat', 'dynamic_mat')#只需要改E:\dynamic_no_mat即可
    pathHC = 'E:\dynamic_no_mat\Mat_Dynamic_ROISignals_FunImgARCWF_HC\Dynamic.mat'#只需要改E:\dynamic_no_mat即可
    pathMDD = 'E:\dynamic_no_mat\Mat_Dynamic_ROISignals_FunImgARCWF_MDD\Dynamic.mat'#只需要改E:\dynamic_no_mat即可
    xlsx = 'E:\dynamic_no_mat\pValue_Dynamic_ROISignals_FunImgARCWF.xlsx'#只需要改E:\dynamic_no_mat即可
    getPvalues(pathHC, pathMDD, xlsx)

