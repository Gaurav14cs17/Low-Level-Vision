import os.path
import time
import tensorflow as tf
import tensorflow as tf

from dataloader import dataloader
from loss import losses
from model.unet_model import *
from dataloader import synthesis







@tf.function
def train_step(model, scene, flare, loss_fn, optimizer):
  with tf.GradientTape() as tape:
    loss_value, summary = synthesis.run_step(scene,flare,model,loss_fn,noise=0.01,flare_max_gain=10.0,flare_loss_weight=1.0,training_res=512)
  grads = tape.gradient(loss_value, model.trainable_weights)
  grads, _ = tf.clip_by_global_norm(grads, 5.0)
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
  return loss_value, summary


def main():
  train_dir =  "./model_dir/"
  summary_dir = os.path.join(train_dir, 'summary')
  model_dir = os.path.join(train_dir, 'model')
  dataset_object = dataloader.DataGenerator(dir_path="D:/labs/LOW_LEVEL_IMAGEING/data/Flare7K_dataset/train/")
  dataset  =  dataset_object.get_data()
  model = UNet_model()
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  loss_fn = losses.get_loss("percep")
  ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),training_finished=tf.Variable(False, dtype=tf.bool),optimizer=optimizer,model=model)
  ckpt_mgr = tf.train.CheckpointManager(ckpt, train_dir, max_to_keep=3, keep_checkpoint_every_n_hours=3)
  latest_ckpt = ckpt_mgr.latest_checkpoint
  restore_status = None
  if latest_ckpt is not None:
    restore_status = ckpt.restore(latest_ckpt).expect_partial()
  else:
      pass

  summary_writer = tf.summary.create_file_writer(summary_dir)
  step_time_metric = tf.keras.metrics.Mean('step_time')
  step_start_time = time.time()
  for flare , scene in dataset:
    # Perform one training step.
    loss_value, summary = train_step(model, scene, flare, loss_fn, optimizer)
    if restore_status is not None:
      restore_status.assert_consumed()
      restore_status = None
    ckpt.step.assign_add(1)
    if ckpt.step % 10 == 0:
      ckpt_mgr.save()
      tf.keras.models.save_model(model, model_dir, save_format='tf')
      with summary_writer.as_default():
        tf.summary.image('prediction', summary, max_outputs=1, step=ckpt.step)
        tf.summary.scalar('loss', loss_value, step=ckpt.step)
        tf.summary.scalar('step_time', step_time_metric.result(), step=ckpt.step)
        step_time_metric.reset_state()
    step_end_time = time.time()
    step_time_metric.update_state(step_end_time - step_start_time)
    step_start_time = step_end_time
  ckpt.training_finished.assign(True)
  ckpt_mgr.save()



if __name__ == '__main__':
    main()