# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 02:16:11 2020

@author: Dai
"""

import pandas as pd
from sqlalchemy import create_engine

from . import util_basics
from . import util_readfile

def conn_mysql_db(db_name):
    """
    """
    login = util_readfile.read_yaml(util_basics.PROJECT_CODE_PATH + "/DaiToolkit/login.yaml")["mysql"]
    user = login["user"]
    password = login["pass"]
    sqlEngine = create_engine('mysql+pymysql://' + user + ':' + password + '@127.0.0.1/' + db_name, pool_recycle=3306)
    dbConnection = sqlEngine.connect()
    return dbConnection


try:
    MYSQL_CONN = conn_mysql_db("auto_dai")
    print ("Connect to mysql database [auto_dai] Success")
except Exception as e:
    print("Connect to mysql database [auto_dai] failed: " + str(e))


def db_table_update(df, tbl_name, if_exists='fail'):
    """
    insert new table
    update curr table (replace/append)
    """
    df.to_sql(tbl_name, MYSQL_CONN, if_exists=if_exists, index=False)


def db_query(sql_query):
    """
    SELECT FROM
    """
    df = pd.read_sql(sql_query, MYSQL_CONN)
    return df


def db_excute(sql_command):
    MYSQL_CONN.execute(sql_command)


def db_table_isexist(tbl_name):
    df = db_query("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '" + tbl_name + "'")
    if len(df) > 0:
        return True
    else:
        return False
