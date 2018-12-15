from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

import numpy as np
import os
import sys
import multipart
from pathlib import Path
from io import BytesIO
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

model_file_url = ''
model_file_name = 'model'
classes = ['Parasitized', 'Uninfected']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    print (dest)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.hdf5')
    model = load_model(str(path) + '/models/' + model_file_name + '.hdf5')
    return model

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    print (img_bytes, BytesIO(img_bytes))
    #img = PIL.Image.open(BytesIO(img_bytes)).convert('RGB')
    img = plt.imread(BytesIO(img_bytes))
    print (type(img))
    img = np.array(img)
    img = resize(img, (64, 64, 3))
    img = img[np.newaxis, :]
    predict = model.predict(img)
    return JSONResponse({'result': str(classes[np.argmax(predict[0])])})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)
