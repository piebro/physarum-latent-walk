import random
import os
import shutil
import json
from argparse import Namespace
from contextlib import contextmanager
import zipfile
from itertools import cycle
import pickle

import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import keras
from keras import layers
import imgaug
import pandas as pd
import plotly.graph_objects as go

import tools


@contextmanager
def run_exp(exp_dir):
  if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)
  
  run_id = "{0:#034x}".format(random.getrandbits(128))[2:]
  run_dir = os.path.join(exp_dir, run_id)  
  os.mkdir(run_dir)
  yield Namespace(**{"run_dir": run_dir, "run_id": run_id})


def save_args(run_dir, args):
  with open(os.path.join(run_dir, "args.json"), "w") as f:
    json.dump(args, f, indent=4, sort_keys=True)


def save_metrics(run_dir, metrics_obj):
  metric_path = os.path.join(run_dir, "metrics.json")

  if os.path.isfile(metric_path):
    with open(metric_path) as f:
      old_metrics_obj = json.load(f)
    for key in old_metrics_obj.keys():
      if type(old_metrics_obj[key]) is list:
        metrics_obj[key] = [*old_metrics_obj[key], *metrics_obj[key]]

  with open(metric_path, "w") as f:
    json.dump(metrics_obj, f, indent=4, sort_keys=True)


def get_args(exp_dir, run_id):
  with open(os.path.join(os.path.join(exp_dir, run_id), "args.json")) as f:
    args = json.load(f)
  return args


def get_experiment_dataframes(exp_dir):
  jsons = []
  for exp_id in os.listdir(exp_dir):
    exp_path = os.path.join(exp_dir, exp_id)
    exp_attributes = {"exp_id": exp_id}
    for fn in ["args.json", "metrics.json"]:
      if os.path.isfile(os.path.join(exp_path, fn)):
        with open(os.path.join(exp_path, fn)) as f:
          exp_attributes.update(json.load(f))
    jsons.append(exp_attributes)
  
  return pd.DataFrame.from_dict(jsons)


def run(exp_dir, dataset_dir, args):
  nn_train_gen, nn_val_gen = get_train_val_gen(args, dataset_dir)
  validation_steps=int(args["steps_per_epoch"]/10)


  model = get_model(args)
  model.compile(optimizer=keras.optimizers.Adam())

  with run_exp(exp_dir) as run:
    print("run_dir:", run.run_dir)
    save_args(run.run_dir, args)
    history = model.fit(nn_train_gen, epochs=args["epochs"], steps_per_epoch=args["steps_per_epoch"],
                        validation_data=nn_val_gen, validation_steps=validation_steps,
                        callbacks=[WeightsSaver(run.run_dir)])
    
    metrics = history.history
    metrics["embedding_dim"] = model.embedding_dim
    save_metrics(run.run_dir, metrics)

    save_weights_paths = os.path.join(run.run_dir, f"weights-{args['epochs']:03d}.h5")
    if not os.path.isfile(save_weights_paths):
      model.save_weights(save_weights_paths)

    # https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state/49504376
    with open(os.path.join(run.run_dir, f"optimizer.pkl"), 'wb') as f:
        pickle.dump(tf.keras.backend.batch_get_value(getattr(model.optimizer, 'weights')), f)

    return run.run_id


def unpack_dataset(dataset_dir, dataset_name):
  if not os.path.exists(os.path.join(dataset_dir, dataset_name)):
    with zipfile.ZipFile(os.path.join(dataset_dir, dataset_name) + ".zip", 'r') as zip_ref:
      zip_ref.extractall(dataset_dir)


def get_nn_data_gen(img_paths, batch_size ,aug):
  # img_dir => img_paths [=> img => img_aug](batching) => nn_train
  img_paths_gen = get_img_path_gen(img_paths)
  while True:
    imgs = []
    for _ in range(batch_size):
      imgs.append(np.array(image.img_to_array(image.load_img(next(img_paths_gen), color_mode="rgb"))))
    yield aug(images=np.array(imgs)/255)


def get_img_path_gen(img_paths):
  while True:
    np.random.shuffle(img_paths)
    for img_path in img_paths:
      yield img_path


def get_augmentation(args):
  if args["augmentation"]=="min":
    return imgaug.augmenters.Sequential([
      imgaug.augmenters.Fliplr(0.5),
      imgaug.augmenters.Affine(rotate=imgaug.parameters.Choice([0,90,180,270]))
    ])
  elif args["augmentation"]=="zoom_translate":
    zoom = 1.1
    t = (zoom-1)/2
    return imgaug.augmenters.Sequential([
      imgaug.augmenters.Affine(
          scale={"x": zoom, "y": zoom},
          translate_percent={"x": (-t, t), "y": (-t, t)}
          ),
      imgaug.augmenters.Fliplr(0.5),
      imgaug.augmenters.Affine(rotate=imgaug.parameters.Choice([0,90,180,270])),
    ])
  else:
    raise Exception("Unknown augmentation.")


def get_model(args, verbose=False):
  input_size = [args["img_size"], args["img_size"], 3]
  filter = [[args["filter_size"], args["filter_size"], args["filter_size"]] for _ in range(args["depth"])]
  filter[-1][-1] = args["last_filter_size"]
  return ResidualAutoencoder(filter, input_size, args, verbose=verbose)


class ResidualAutoencoder(keras.Model):
    def __init__(self, filter, input_shape, args, verbose=True, **kwargs):
      super(ResidualAutoencoder, self).__init__(**kwargs)
      self.tracker_binary_crossentropy = keras.metrics.Mean(name="binary_crossentropy")
      self.tracker_val_binary_crossentropy = keras.metrics.Mean(name="val_binary_crossentropy")
      self.my_input_shape = input_shape
      self.embedding_dim = int((input_shape[0]/(2**len(filter)))**2 * filter[-1][-1])
      self.args = args

      encoder_inputs = keras.Input(shape=input_shape)
      x = encoder_inputs      
      for f in filter:
        x = self.resConv2D(x, f)

      self.encoder = keras.Model(encoder_inputs, x, name="encoder")
      
      xy_dim = int(input_shape[0]/(2**len(filter)))
      latent_inputs = keras.Input(shape=(xy_dim, xy_dim, filter[-1][-1]))
      
      x = latent_inputs
      for f in reversed(filter):
        x = self.resConv2DTransform(x, list(reversed(f)))

      decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
      self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
      
      if verbose:
        self.encoder.summary()
        self.decoder.summary()

    @property
    def metrics(self):
      return [self.tracker_binary_crossentropy, self.tracker_val_binary_crossentropy]

    def call(self, x):
      return self.decoder(self.encoder(x))

    def predict(self, x):
      return self.decoder(self.encoder(x))

    def train_step(self, data):
      with tf.GradientTape() as tape:
          reconstruction = self.decoder(self.encoder(data))
          binary_crossentropy = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
      
      grads = tape.gradient(binary_crossentropy, self.trainable_weights)
      self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
      self.tracker_binary_crossentropy.update_state(binary_crossentropy)
      return {"binary_crossentropy": self.tracker_binary_crossentropy.result()}
    
    def test_step(self, data):
      reconstruction = self.decoder(self.encoder(data, training=False), training=False)
      binary_crossentropy = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
      self.tracker_val_binary_crossentropy.update_state(binary_crossentropy)
      return {"binary_crossentropy": self.tracker_val_binary_crossentropy.result()}

    def resConv2D(self, x, filter):
      x_shortcut = x

      x = layers.Conv2D(filter[0], 1, strides=1, padding="same")(x)
      x = layers.Activation("relu")(x)
      if self.args["batch_norm"]: x = layers.BatchNormalization()(x)

      x = layers.Conv2D(filter[1], 3, strides=2, padding="same")(x)
      x = layers.Activation("relu")(x)
      if self.args["batch_norm"]: x = layers.BatchNormalization()(x)

      x = layers.Conv2D(filter[2], 1, strides=1, padding="same")(x)
      
      x_shortcut = layers.Conv2D(filter[2], 3, strides=2, padding="same")(x_shortcut)

      x = layers.Add()([x, x_shortcut])
      x = layers.Activation('relu')(x)
      return x

    def resConv2DTransform(self, x, filter):
      x_shortcut = x

      x = layers.Conv2DTranspose(filter[0], 1, strides=1, padding="same")(x)
      x = layers.Activation("relu")(x)
      x = layers.Conv2DTranspose(filter[1], 3, strides=2, padding="same")(x)
      x = layers.Activation("relu")(x)
      x = layers.Conv2DTranspose(filter[2], 1, strides=1, padding="same")(x)
  
      x_shortcut = layers.Conv2DTranspose(filter[2], 3, strides=2, padding="same")(x_shortcut)

      x = layers.Add()([x, x_shortcut])
      x = layers.Activation('relu')(x)
      return x


class WeightsSaver(keras.callbacks.Callback):
  def __init__(self, save_dir):
    self.save_dir = save_dir

  def on_epoch_end(self, epoch, logs={}):
    if (epoch+1) < 5 or (epoch+1) % 5 == 0:
      self.model.save_weights(os.path.join(self.save_dir, f"weights-{epoch+1:03d}.h5"))


def get_train_val_gen(args, dataset_dir):
  unpack_dataset(dataset_dir, args["img_dir"])
  img_paths = [os.path.join(args["img_dir"], fn) for fn in os.listdir(args["img_dir"])]
  aug = get_augmentation(args)
  nn_train_gen = get_nn_data_gen(img_paths[:-int(len(img_paths)/10)], args["batch_size"], aug)
  nn_val_gen = get_nn_data_gen(img_paths[-int(len(img_paths)/10):], args["batch_size"], aug)
  return nn_train_gen, nn_val_gen


def continue_training(exp_dir, dataset_dir, run_id, additional_epochs):
  args = get_args(exp_dir, run_id)
  
  nn_train_gen, nn_val_gen = get_train_val_gen(args, dataset_dir)
  validation_steps=int(args["steps_per_epoch"]/10)
  
  run_dir = os.path.join(exp_dir, run_id)
  model = load_model_for_training(args, run_dir)
  
  print("continue training on:", run_dir)
  history = model.fit(nn_train_gen, epochs=args["epochs"] + additional_epochs, steps_per_epoch=args["steps_per_epoch"],
                      validation_data=nn_val_gen, validation_steps=validation_steps,
                      initial_epoch=args["epochs"], callbacks=[WeightsSaver(run_dir)])
  
  args["epochs"] += additional_epochs
  save_args(run_dir, args)

  metrics = history.history
  metrics["embedding_dim"] = model.embedding_dim
  save_metrics(run_dir, metrics)

  save_weights_paths = os.path.join(run_dir, f"weights-{args['epochs']:03d}.h5")
  if not os.path.isfile(save_weights_paths):
    model.save_weights(save_weights_paths)

  with open(os.path.join(run_dir, "optimizer.pkl"), 'wb') as f:
    pickle.dump(tf.keras.backend.batch_get_value(getattr(model.optimizer, 'weights')), f)
  return run_id


def load_model_for_training(args, run_dir):
  # load model and do one traing iteration before loading weights and optimizer state
  model = get_model(args)
  model.compile(optimizer=keras.optimizers.Adam())
  input_size = [args["img_size"], args["img_size"], 3]
  model.fit(cycle([np.zeros((args["batch_size"],*input_size))]), epochs=1, steps_per_epoch=1, verbose=0)
  model.load_weights(os.path.join(run_dir, f"weights-{args['epochs']:03d}.h5"))

  # https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state/49504376
  model.make_train_function()
  with open(os.path.join(run_dir, "optimizer.pkl"), 'rb') as f:
    weight_values = pickle.load(f)
  model.optimizer.set_weights(weight_values)
  return model


def save_video(exp_dir, dataset_dir, run_id, embedding_transformations, video_name="video", video_path=None, seed=None, epoch=None, framerate=25):
  args = get_args(exp_dir, run_id)
  unpack_dataset(dataset_dir, args["img_dir"])

  if epoch is None: epoch = args["epochs"]
  model = load_model(exp_dir, run_id, epoch)
  if seed is None: seed = random.randint(0, 999999)
  np.random.seed(seed)
  imgaug.seed(seed)

  if video_path is None:
    run_dir = os.path.join(exp_dir, run_id)
    video_path = os.path.join(run_dir, f"{video_name}_seed{seed:06d}_epoch{epoch:03d}.mp4")
  
  print("video path: ", video_path)
  if os.path.exists(video_path):
    raise Exception("Video already exists")
  
  augmentation = get_augmentation(args)
  
  img_paths = np.array([os.path.join(args["img_dir"], fn) for fn in os.listdir(args["img_dir"])])
  inital_embedding_size = embedding_transformations[0]
  img_paths = np.random.choice(img_paths, size=inital_embedding_size)  
  embeddings = use_encoder(model, img_paths, args["batch_size"], augmentation)

  for embedding_transformation in embedding_transformations[1:]:
    embeddings = embedding_transformation(embeddings)

  decoded_img_gen = use_decoder(model, embeddings, args["batch_size"])  
  tools.save_video(video_path, decoded_img_gen, framerate)

  return video_path


def load_model(exp_dir, run_id, epoch):
  args = get_args(exp_dir, run_id)
  model = get_model(args)
  model.build((args["batch_size"], args["img_size"], args["img_size"], 3))
  run_dir = os.path.join(exp_dir, run_id)
  model.load_weights(os.path.join(run_dir, f"weights-{epoch:03d}.h5"))
  return model


def use_encoder(model, img_paths, batch_size, augmentation=None):
  # img_path [=> img => img_aug_det => encoder](batching) => embedding
  img_size = image.load_img(img_paths[0], color_mode="rgb").size
  
  missing_paths_for_full_batch = [None]*(batch_size-(len(img_paths)%batch_size))
  img_path_batches = np.concatenate((img_paths, missing_paths_for_full_batch)).reshape((-1, batch_size))
  
  embeddings = []
  for img_path_batch in img_path_batches:
    batch_imgs = []
    for img_path in img_path_batch:
      if img_path is None:
        batch_imgs.append(np.zeros((*img_size, 3)))
      else:
        batch_imgs.append(np.array(image.img_to_array(image.load_img(img_path, color_mode="rgb")))/255)

    if augmentation is not None:
      batch_imgs = augmentation(images=np.array(batch_imgs))

    pred_batch = model.encoder(np.array(batch_imgs))
    embeddings.extend(pred_batch)

  return np.array(embeddings[:len(img_paths)])


def use_decoder(model, embeddings, batch_size):
  # embedding [=> nn_decoder => img (=> img saved)](batching) (=> img in fig)
  embedding_shape = embeddings[0].shape
  missing_embeddings_for_full_batch = np.zeros((batch_size-(len(embeddings)%batch_size), *embedding_shape))
  embeddings_batches = np.concatenate((embeddings, missing_embeddings_for_full_batch))
  embeddings_batches = embeddings_batches.reshape((-1, batch_size, *embedding_shape))
  
  for i, embeddings_batche in enumerate(embeddings_batches):
    imgs = model.decoder(embeddings_batche)
    if i == len(embeddings_batches)-1:
      imgs = imgs[:len(embeddings)%batch_size]
    for img in imgs:
      yield img


def plot_loss(df, run_id):
  run = df[df["exp_id"]==run_id]
  train_loss = run["binary_crossentropy"].item()
  val_loss = run["val_binary_crossentropy"].item()
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=list(range(1,len(train_loss)+1)), y=train_loss, mode='lines+markers', name="train"))
  fig.add_trace(go.Scatter(x=list(range(1,len(val_loss)+1)), y=val_loss, mode='lines+markers', name="val"))
  fig.update_layout(title=f"{run['img_dir'].item()}_{run['depth'].item()}")
  fig.show()