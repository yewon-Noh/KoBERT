import pymysql
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)

host = os.getenv('IP')
 
class MyDao:
    def __init__(self):
        pass
    
    # 공지 글 보여주는 select
    def getEmps(self):
        ret = []
        db = pymysql.connect(host=host, user="noe", passwd="1234", db="noe", charset="utf8")
        curs = db.cursor()
        
        sql = "select * from noedb order by num desc";
        curs.execute(sql)
        
        rows = curs.fetchall()
        for e in rows:
            temp = {'title':e[0],'context':e[1], 'num':e[2], 'adv':e[3]}
            ret.append(temp)
        
        db.commit()
        db.close()
        return ret

    # 댓글 보여주는 select
    def getAnss(self, num):
        ret = []
        db = pymysql.connect(host=host, user="noe", passwd="1234", db="noe", charset="utf8")
        curs = db.cursor()
        
        sql = "select * from noe_ansDB where num = %s"
        curs.execute(sql, num)
        
        rows = curs.fetchall()
        for e in rows:
            temp = {'ans_num':e[0],'num':e[1], 'ans':e[2]}
            ret.append(temp)
        
        db.commit()
        db.close()
        return ret

    # 특정 공지 보여주는 select
    def getEmpss(self, num):
        ret = []
        db = pymysql.connect(host=host, user="noe", passwd="1234", db="noe", charset="utf8")
        # db = pymysql.connect(host="127.0.0.1", user="root", passwd="1234", db="noe", charset="utf8")
        curs = db.cursor()
        
        sql = "select * from noedb where num = %s";
        curs.execute(sql, num)
        
        rows = curs.fetchall()
        for e in rows:
            temp = {'title':e[0],'context':e[1], 'num':e[2]}
            ret.append(temp)
        
        db.commit()
        db.close()
        return ret

    # 공지 추가 insert
    def insEmp(self, title, context, adv):
        db = pymysql.connect(host=host, user="noe", passwd="1234", db="noe", charset="utf8")
        curs = db.cursor()
        
        sql = '''insert into noedb (title, context, adv) values(%s,%s, %s)'''
        curs.execute(sql,(title, context, adv))
        db.commit()
        db.close()

    # 댓글 추가 insert
    def insAns(self, num, ans):
        db = pymysql.connect(host=host, user="noe", passwd="1234", db="noe", charset="utf8")
        curs = db.cursor()
        
        sql = '''insert into noe_ansDB (num, ans) values(%s,%s)'''
        curs.execute(sql,(num, ans))
        db.commit()
        db.close()
    
if __name__ == '__main__':
    noelist = MyDao().getEmps();
    print("<<<<<<<<<<<<<<<<<",noelist)