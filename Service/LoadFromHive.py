#!/usr/bin/python
# coding:utf-8

from impala.dbapi import connect
import datetime
import Utils.Consts as consts
import numpy as np
import pandas as pd

attributeList= consts.attributeList
'''
本部分用于从 10.32.16.124 /opt/tiefan/test_Hive_Connection/ 运行脚本，从hive中获取宽表得数据

'''



def getSql():
    cql='select '
    for i in range(len(attributeList)):
        cql+="a."+attributeList[i]+" as "+attributeList[i]+","
    cql+="b.label as label from rdw.pdl_wide_table  a JOIN  rtmp.pdl_del_zhima1 b ON a.member_id = b.member_id and a.mobile_de=b.mobile where b.label!=2"
    return cql



def loadFromHive():
    print('begain read')
    sql =getSql()
    begain = datetime.datetime.now()
    conn = connect(host='idc-bigdata008.blackfi.sh', port=10000, database='rdw', auth_mechanism='GSSAPI',
                   kerberos_service_name='hive')
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()

    Data=[]
    for i in range(len(results)):
        Data.append(list(results[i]))
    Data=np.array(Data)
    attributeList.append('label')
    scvfile=pd.DataFrame(Data,columns=attributeList)
    scvfile.to_csv("hivefile.csv",encoding="utf_8_sig")

    print('cost:', (datetime.datetime.now() - begain))


loadFromHive()
