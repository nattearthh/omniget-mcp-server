import urllib.request
import re
url = 'https://www.facebook.com/zuck/posts/10114026953826931'
req = urllib.request.Request(url, headers={'User-Agent': 'facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)'})
try:
    html = urllib.request.urlopen(req).read().decode('utf-8')
    scontent_urls = re.findall(r'https[\\/:]+scontent[^\"\'\s<>]+?\.(?:jpg|png|webp)[^\"\'\s<>]*', html)
    print(f"Found {len(scontent_urls)} scontent URLs")
    for u in scontent_urls[:5]:
        print(u)
except Exception as e:
    print(e)
