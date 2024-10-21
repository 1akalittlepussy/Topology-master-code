import os
import xlrd
import shutil

#dirpath = r'E:\Topology-master-code\ROISignals_FunImgARCWF'   #存放图片的文件夹
dirpath =r'E:/cortical_ROI/all'
dstpath = r'E:/cortical_ROI/separate'   #保存图片的文件夹
#datapath = r'D:\Topology-master-data-MDD-dynamic\Topology-master-data-MDD-dynamic\data\MDD\dynamic\choose.xls'   #excel表路径
datapath =r'E:\Topology-master-code\hcpremove.xls'

x1 = xlrd.open_workbook(datapath)    #读取excel
sheet1 = x1.sheet_by_name("Sheet1")    #读取Sheet1

idlist = sheet1.col_values(0)    #存放第2列,图片名称(含扩展名)
file_names = os.listdir(dirpath)    #获取文件夹下所有图片名称(含扩展名)

for i in idlist:
    for j in file_names:
        if i == j:               # 从excel里找到文件夹中对应的图片
            src = os.path.join(dirpath, '%s' % i)      # 构造图片源文件的绝对路径
            #print("src=",src)
            dst = os.path.join(dstpath, '%s' % i)     # 构造图片移动的绝对路径
            #print("dst=",dst)
            shutil.move(src, dst)
