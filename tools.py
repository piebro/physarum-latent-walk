from base64 import b64encode
import random
import os
import shutil

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML

def get_figure_of_images(img_array, titles=None, height=1, width=None, size=8):
  assert width is not None or height is not None
  if width is None: width = np.ceil(len(img_array)/height).astype(int)
  if height is None: height = np.ceil(len(img_array)/width).astype(int)

  fig = plt.figure(figsize=(width*size, height*size))

  if titles is None:
    titles = [""]*len(img_array)

  for i in range(min(len(img_array), width*height)):
    ax = fig.add_subplot(height, width, i+1)
    ax.set_axis_off()
    ax.title.set_text(titles[i])
    ax.imshow(img_array[i])
  fig.tight_layout()
  return fig

def show_video_html(paths, width=800):
  if type(paths) is str:
    paths = [paths]
  data_urls = [f"data:video/mp4;base64,{b64encode(open(path,'rb').read()).decode()}" for path in paths]
  html_strs = [f"<video width={width} style='padding:10px;' controls loop><source src=\"{data_url}\" type=\"video/mp4\"></video>" for data_url in data_urls]
  return HTML("".join(html_strs))

def save_video(video_path, img_gen, framerate=25, overwrite=False):
  temp_dir = "temp-video-" + "{0:#034x}".format(random.getrandbits(128))[2:]
  os.mkdir(temp_dir)
  for i, img in enumerate(img_gen):
    image.save_img(os.path.join(temp_dir, f'{i:06d}.png'), img)

  if os.path.isfile(video_path):
    if overwrite:
      os.remove(video_path)
    else:
      raise Exception("There is already a video at video path.")
  
  os.system(f"ffmpeg -framerate {framerate} -f image2 -i {temp_dir}/%06d.png -vcodec libx264 -pix_fmt yuva420p {video_path}")
  shutil.rmtree(temp_dir)
  return video_path