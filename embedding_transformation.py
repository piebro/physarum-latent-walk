import numpy as np
import plotly.graph_objects as go
import tsp
from scipy.spatial import distance_matrix
import scipy.interpolate
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import logistic
from opensimplex import OpenSimplex

def test_embedding_transformations(embedding_transformations, inital_embedding_size=4, img_size=(50,50,3)):
  random_color = np.random.uniform(size=(inital_embedding_size, img_size[2]))
  embeddings = np.array([np.tile(c, (img_size[0], img_size[1], 1)) for c in random_color])
  init_embeddings = embeddings.copy()
  
  for embedding_transformation in embedding_transformations:
    embeddings = embedding_transformation(embeddings)
  return init_embeddings, embeddings


def plot_embeddings_2d(init_embeddings, embeddings):
  fig = go.Figure(go.Scatter(x=init_embeddings[:,0,0,0], y=init_embeddings[:,0,0,1], mode='lines+markers', name="init_embeddings"))
  fig.add_trace(go.Scatter(x=embeddings[:,0,0,0], y=embeddings[:,0,0,1], mode='lines+markers', fill="toself", name="embeddings"))
  fig.update_yaxes(scaleanchor = "x",scaleratio = 1)
  fig.update_layout(width = 900,height = 900)
  fig.show()


def get_loop_dist_mean(embeddings):
  return np.mean([np.linalg.norm(embeddings[i-1]-embeddings[i]) for i in range(len(embeddings))])


def reorder_with_tsp():
  def transformation(embeddings):
    shape = embeddings.shape[1:]
    embeddings = embeddings.reshape((embeddings.shape[0], -1))
    dist_mat = distance_matrix(embeddings, embeddings)
    dist, permutation = tsp.tsp(dist_mat)
    embeddings = embeddings[permutation]
    embeddings = embeddings.reshape((-1, *shape))
    return embeddings
  return transformation


def init_embedding_count(count):
  return count


def interpolate_smooth_equi_dist(frame_count, smoothness=0.25, num_of_iterations=3):
  # smoothness: 0 means no smoothnes, 0.25 is max smoothnes
  def transformation(embeddings):
    embeddings = np.concatenate((embeddings, embeddings[:2]))
    for _ in range(num_of_iterations):
      l = len(embeddings)
      f = scipy.interpolate.interp1d(range(l), embeddings, axis=0, kind="linear")
      embeddings = f(np.sort(np.concatenate((np.arange(l)-smoothness, np.arange(l)+smoothness)))[1:-1])
    embeddings = embeddings[2:]
    
    embeddings = np.concatenate((embeddings, embeddings[:1]))
    for i in range(5):
      distances = np.array([np.linalg.norm(e0-e1) for e0, e1 in zip(embeddings, np.roll(embeddings, 1, axis=0))])
      distances = np.cumsum(distances / np.sum(distances) * len(distances))
      f = scipy.interpolate.interp1d(distances, embeddings, axis=0, kind="linear")
      embeddings = f(np.linspace(distances[0], len(embeddings)-0.0001, num=frame_count+1, endpoint=True))
    return embeddings[:-1]
  return transformation


def cluster(cluster_size):
  def transformation(embeddings):
    shape = embeddings.shape[1:]
    embeddings = embeddings.reshape((embeddings.shape[0], -1))

    dist_mat = distance_matrix(embeddings, embeddings)
    cluster = []
    dist_mat_min = np.min(dist_mat)
    min_row, min_col = np.where(dist_mat == dist_mat_min)
    cluster.append(min_row[0])
    cluster.append(min_col[0])
    within_cluster_dist = dist_mat_min

    for _ in range(cluster_size-2):
      min_temp_withing_cluster_dist = np.inf
      min_index = None

      for i in range(len(embeddings)):
        if i in cluster:
          continue

        temp_within_cluster_dist = within_cluster_dist
        for c in cluster:
          temp_within_cluster_dist += dist_mat[i, c]

        if temp_within_cluster_dist < min_temp_withing_cluster_dist:
          min_temp_withing_cluster_dist = temp_within_cluster_dist
          min_index = i
      
      cluster.append(min_index)
      within_cluster_dist = min_temp_withing_cluster_dist
      
    embeddings = embeddings[cluster].reshape((-1, *shape))
    return embeddings
    
  return transformation


def sliding(frame_count, horizontal=False):
  def transformation(embeddings):
    if horizontal:
      embeddings = np.swapaxes(embeddings,1,2)

    embeddings = np.concatenate((embeddings, embeddings[:1]))

    s, _, d = embeddings[0].shape
    inner_size = s-2 # use the innser size, becouse the outter ones dont work right in the inside of an images
    num_of_embeddings = (len(embeddings)-1)*inner_size
    
    a = np.empty((num_of_embeddings,s,d))

    for i in range(num_of_embeddings):
      e0 = embeddings[i//inner_size]
      e1 = embeddings[(i//inner_size)+1]

      j0 = inner_size-1-(i%inner_size)
      j1 = (i%inner_size) +1

      w0 = np.round(j0/inner_size, 4)
      w1 = np.round(j1/inner_size, 4)
      
      a[i, :, :] = e0[j0, :]*w0 + e1[j1, :]*w1
    
    f = scipy.interpolate.interp1d(range(len(a)), a, axis=0, kind="linear")

    embeddings = []
    for i in np.linspace(0,num_of_embeddings-s,frame_count, endpoint=True):
      b = np.array([f(k) for k in np.linspace(i, i+s-1, s)])
      embeddings.append(b)
    embeddings = np.array(embeddings)

    if horizontal:
      embeddings = np.swapaxes(embeddings,1,2)
    return embeddings
  return transformation


def interpolation_linear(frame_count):
  def transformation(embeddings):
    embeddings = np.concatenate((embeddings, embeddings[:1]))
    f = scipy.interpolate.interp1d(range(len(embeddings)), embeddings, axis=0, kind="linear")
    embeddings = f(np.linspace(0, len(embeddings)-1, num=frame_count, endpoint=False))
    return embeddings
  return transformation


def _apply_weights(embeddings, weights):
  s, _, d = embeddings[0].shape
  weights = weights.reshape((s,s,1))
  weights = np.tile(weights, (1,1,d))
  return np.array([e0*weights+e1*(1-weights) for e0, e1 in zip(embeddings, np.roll(embeddings, 1, axis=0))])


def weight_mid(border_size):
  def transformation(embeddings):
    s, _, d = embeddings[0].shape
    weights = np.zeros((s,s))
    weights[border_size:s-border_size,border_size:s-border_size] = 1
    weights = gaussian_filter(weights, sigma=2)
    return _apply_weights(embeddings, weights)
  return transformation


def weight_0to1(horizontal=False):
  def transformation(embeddings):
    s, _, d = embeddings[0].shape
    weights = np.linspace(np.zeros(s), np.ones(s), s, endpoint=True)
    if horizontal:
      weights = weights.T
    return _apply_weights(embeddings, weights)
  return transformation


def add(const):
  return lambda embeddings: np.array([e+const for e in embeddings])


def mult(const):
  return lambda embeddings: np.array([e*const for e in embeddings])


def power(const):
  return lambda embeddings: np.array([e**const for e in embeddings])


def mod(const):
  return lambda embeddings: np.array([np.mod(e, const) for e in embeddings])


def minus_each_other():
  return lambda embeddings: np.array([e0-e1 for e0, e1 in zip(embeddings, np.roll(embeddings, 1, axis=0))])


def mult_each_other():
  return lambda embeddings: np.array([e0*e1 for e0, e1 in zip(embeddings, np.roll(embeddings, 1, axis=0))])


def roll(axis=-1):
  return lambda embeddings: np.roll(embeddings, shift=1, axis=axis)


def loopable_noise(frame_count, local_factor=1, noise_speed=2, noise_inpact=0.2):
  def transformation(embeddings):
    embeddings = np.concatenate((embeddings, embeddings[:1]))
    count, w, h, d = embeddings.shape
    
    xy_indices = [[x,y] for x in range(w) for y in range(h)]
    interpolation_space = np.linspace(0, 1, num=count, endpoint=True)

    interpolation_funcs = [[None for i in range(w)] for j in range(h)]
    for x, y in xy_indices:
      interpolation_funcs[x][y] = scipy.interpolate.interp1d(interpolation_space, embeddings[:,x,y,:], axis=0, kind="linear")

    os = OpenSimplex(seed=np.random.randint(100000))
    new_embeddings = []
    for i in range(frame_count):
      e = np.empty((w,h,d))
      for x, y in xy_indices:
        pos_on_circle = i/frame_count*np.pi*2
        radius = noise_speed
        fs = w*(1/local_factor)
        noise = os.noise4d(x / fs, y / fs, radius*np.cos(pos_on_circle), radius*np.sin(pos_on_circle))
        value = (i/frame_count + noise*noise_inpact)%1
        e[x,y] = interpolation_funcs[x][y](value)
      new_embeddings.append(e)

    return new_embeddings
  return transformation