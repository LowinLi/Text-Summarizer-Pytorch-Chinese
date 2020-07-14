import requests
import json

url = 'localhost:5000'


def main():
    text = 'e公司讯，龙力退（002604）7月14日晚间公告，公司股票已被深交所决定终止上市，并在退市整理期交易30个交易日，最后交易日为7月14日，将在2020年7月15日被摘牌。公司股票终止上市后，将进入股转系统进行股份转让。'
    ret = requests.post(
            'http://{}/abstract'.format(url), json={'text':text})
    tmp = json.loads(ret.content.decode('utf-8'))
    print(tmp)

if __name__ == "__main__":
    main()