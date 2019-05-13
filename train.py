import tensorflow as tf
import numpy as np
import argparse
from network import msi_max
from dataset.data import get_all_test_data,get_filename_list,DatasetInputs
from core.eval import per_class_acc,print_hist_summery,get_hist
import os
import time
from datetime import datetime


NUM_CLASSES=2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 990
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1




def training(FLAGS, is_finetune=False):
    max_steps = FLAGS.max_steps
    batch_size = FLAGS.batch_size
    train_dir = FLAGS.log_dir
    image_dir = FLAGS.image_dir
    val_dir = FLAGS.val_dir
    finetune_ckpt = FLAGS.finetune
    image_w = FLAGS.image_w
    image_h = FLAGS.image_h
    image_c = FLAGS.image_c
    # should be changed if your model stored by different convention
    startstep = 0 if not is_finetune else int(FLAGS.finetune.split('-')[-1])

    image_filenames, label_filenames = get_filename_list(image_dir)
    val_image_filenames, val_label_filenames = get_filename_list(val_dir)

    with tf.Graph().as_default():

        train_data_node = tf.placeholder(tf.float32, shape=[batch_size, image_h, image_w, image_c])

        train_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])

        phase_train = tf.placeholder(tf.bool, name='phase_train')

        global_step = tf.Variable(0, trainable=False)

        # For CamVid
        images, labels = DatasetInputs(image_filenames, label_filenames, batch_size)


        val_images, val_labels = DatasetInputs(val_image_filenames, val_label_filenames, batch_size)

        # Build a Graph that computes the logits predictions from the inference model.
        loss, eval_prediction = inference(train_data_node, train_labels_node, batch_size, phase_train)

        # Build a Graph that trains the model with one batch of examples and updates the model parameters.
        train_op = train(loss, global_step)

        saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge_all()

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Build an initialization operation to run below.
            if (is_finetune == True):
                saver.restore(sess, finetune_ckpt)
            else:
                init = tf.global_variables_initializer()
                sess.run(init)

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Summary placeholders
            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
            average_pl = tf.placeholder(tf.float32)
            acc_pl = tf.placeholder(tf.float32)
            iu_pl = tf.placeholder(tf.float32)
            average_summary = tf.summary.scalar("test_average_loss", average_pl)
            acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
            iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

            for step in range(startstep, startstep + max_steps):
                image_batch, label_batch = sess.run([images, labels])
                # since we still use mini-batches in validation, still set bn-layer phase_train = True
                feed_dict = {
                    train_data_node: image_batch,
                    train_labels_node: label_batch,
                    phase_train: True
                }
                start_time = time.time()

                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value,
                                        examples_per_sec, sec_per_batch))

                    # eval current training batch pre-class accuracy
                    pred = sess.run(eval_prediction, feed_dict=feed_dict)
                    per_class_acc(pred, label_batch)

                if step % 100 == 0:
                    print("start validating.....")
                    total_val_loss = 0.0
                    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
                    for test_step in range(int(TEST_ITER)):
                        val_images_batch, val_labels_batch = sess.run([val_images, val_labels])

                        _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
                            train_data_node: val_images_batch,
                            train_labels_node: val_labels_batch,
                            phase_train: True
                        })
                        total_val_loss += _val_loss
                        hist += get_hist(_val_pred, val_labels_batch)
                    print("val loss: ", total_val_loss / TEST_ITER)
                    acc_total = np.diag(hist).sum() / hist.sum()
                    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
                    test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / TEST_ITER})
                    acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
                    iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
                    print_hist_summery(hist)
                    print(" end validating.... ")

                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.add_summary(test_summary_str, step)
                    summary_writer.add_summary(acc_summary_str, step)
                    summary_writer.add_summary(iu_summary_str, step)
                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == max_steps:
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)