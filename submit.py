import os
import tempfile
import zipfile

import requests

##########

#test
KEY1 = '5a86dcccefe0d276f0bdce08fff7dc8a'
KEY2 = 'c214a19111258acb8247b3c38a73017df70ab124f50f0614976b1b375a6fd59ded6fe2473a41a8f87e0ad5512dab97e4fbca179a51512330863ebd79fa4f89d1'

#test2
KEY1 = '85371aedc3d08247c38ab263bb6584ac'
KEY2 = 'f95da9e6c066e581cd43a2be7128e474904a202ba500e7f4ba4b7fc875178a7ae6f3609ea5b11813b8142a4728caf6fa827a514636fb2498e9c063e556b72d6f'

predpath = './test/predict'            # path to predictions

##########

temp_name = next(tempfile._get_candidate_names())
print(temp_name, 'created')

with zipfile.ZipFile(temp_name, 'w') as zf:
    for root,dir,files in os.walk(predpath):
        for file in sorted(files):
            fullpath = os.path.join(root,file)
            print(fullpath)
            zf.write(fullpath, file)

url = 'http://ai.ntuh.net:10010/brain/upload/%s/' % KEY2

data = {'m': KEY1}
files = {'file': open(temp_name, 'rb')}

response = requests.post(url, data = data, files=files)

print( response)

os.unlink(temp_name)
print(temp_name, 'deleted')

