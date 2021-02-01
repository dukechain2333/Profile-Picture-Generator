import requests
from bs4 import BeautifulSoup
import os
import traceback
import time
import threading


def get_proxy():
    return requests.get("http://127.0.0.1:5010/get/").json()


def delete_proxy(proxy):
    requests.get("http://127.0.0.1:5010/delete/?proxy={}".format(proxy))


def download(url, filename, proxyUse):
    if os.path.exists(filename):
        print('file exists!')
        return
    try:
        print(url)
        r = requests.get(url, stream=True, timeout=30, headers=headers, proxies=proxyUse)
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
        return filename
    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
        raise KeyboardInterrupt
    except Exception:
        traceback.print_exc()
        if os.path.exists(filename):
            os.remove(filename)


def spyder(start, end):
    proxy = get_proxy().get("proxy")
    for i in range(start, end + 1):
        retry_count = 5
        while retry_count > 0:
            try:
                print(proxy + " " + str(retry_count))
                proxyUse = {"http": "http://{}".format(proxy)}
                url = 'http://konachan.net/post?page=%d&tags=' % i
                html = requests.get(url=url, headers=headers, proxies=proxyUse).text
                soup = BeautifulSoup(html, 'html.parser')
                for img in soup.find_all('img', class_="preview"):
                    target_url = img['src']
                    filename = os.path.join('RawImages', target_url.split('/')[-1], )
                    print(filename)
                    download(target_url, filename, proxyUse)
                    time.sleep(5)
                break
            except Exception:
                retry_count -= 1
        delete_proxy(proxy)
        print('%d / %d' % (i, end))


if __name__ == '__main__':
    if os.path.exists('RawImages') is False:
        os.makedirs('RawImages')
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36 Edg/88.0.705.56'
    }
    threadingPool = []
    i = 1
    while i < 8000:
        threadingPool.append(threading.Thread(target=spyder, args=(i, i + 50,)))
        i += 50
    for thread in threadingPool:
        thread.start()
