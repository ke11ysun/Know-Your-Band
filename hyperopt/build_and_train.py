from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as thelosses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

# sys.path.append('..')
import measure_map
from keras_frcnn.simple_parser import get_data
from keras_frcnn import resnet as nn
from hyperopt import STATUS_OK, STATUS_FAIL
import uuid
from utils import print_json, save_json_result

def build_and_train(hype_space, save_best_weights=False):
	train_path = '/home/comp/e4252392/retraindata4frcnn.txt'
	config_output_filename = '/home/comp/e4252392/hyperopt/hyperopt_config.pickle'
	num_epochs = 20
	#for retrain best model only
	diagnose_path = '/home/comp/e4252392/hyperopt/models/hyperopt_loss_ap_plt.npy'
	real_model_path = '/home/comp/e4252392/hyperopt/models/hyperopt_model_plt_'

	print("Hyperspace:")
	print(hype_space)
	C = config.Config()
	C.num_rois = int(hype_space['num_rois']) #why int?
	# C.anchor_box_scales = hype_space['anchor_box_scales']
	# C.base_net_weights = '/home/comp/e4252392/second_res_more_epoch.h5'
	C.base_net_weights = 'model_frcnn.hdf5'

	#data
	all_imgs, classes_count, class_mapping = get_data(train_path)
	if 'bg' not in classes_count:
		classes_count['bg'] = 0
		class_mapping['bg'] = len(class_mapping)
	C.class_mapping = class_mapping

	print('Training images per class:')
	pprint.pprint(classes_count)
	print('Num classes (including bg) = {}'.format(len(classes_count)))

	with open(config_output_filename, 'wb') as config_f:
		pickle.dump(C,config_f)
		print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

	random.shuffle(all_imgs)
	num_imgs = len(all_imgs)
	train_imgs = [s for s in all_imgs]
	print('Num train samples {}'.format(len(train_imgs)))

	data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
	#data


	# build_model
	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
	else:
		input_shape_img = (None, None, 3)

	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(None, 4))
	shared_layers = nn.nn_base(int(hype_space['kernel_size']), img_input, trainable=True)

	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn = nn.rpn(int(hype_space['kernel_size']), shared_layers, num_anchors)

	classifier = nn.classifier(int(hype_space['kernel_size']), shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

	model_rpn = Model(img_input, rpn[:2])
	model_classifier = Model([img_input, roi_input], classifier)
	model_all = Model([img_input, roi_input], rpn[:2] + classifier)

	try:
		print('loading weights from {}'.format(C.base_net_weights))
		model_rpn.load_weights(C.base_net_weights, by_name=True)
		model_classifier.load_weights(C.base_net_weights, by_name=True)
	except:
		print('Could not load pretrained model weights. Weights can be found in the keras application folder \
			https://github.com/fchollet/keras/tree/master/keras/applications')

	# optimizer = Adam(lr=1e-5)
	# optimizer_classifier = Adam(lr=1e-5)
	optimizer = Adam(lr=hype_space['optimizer_lr'], decay=hype_space['optimizer_decay'])
	optimizer_classifier = Adam(lr=hype_space['optimizer_lr'], decay=hype_space['optimizer_decay'])
	model_rpn.compile(optimizer=optimizer, loss=[thelosses.rpn_loss_cls(num_anchors), thelosses.rpn_loss_regr(num_anchors)])
	model_classifier.compile(optimizer=optimizer_classifier, loss=[thelosses.class_loss_cls, thelosses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
	sgd = SGD(lr=hype_space['sgd_lr'], decay=hype_space['sgd_decay'])
	model_all.compile(optimizer=sgd, loss='mae')
	# build_model


	#build_and_train
	epoch_length = 10
	iter_num = 0
	losses = np.zeros((epoch_length, 5))
	rpn_accuracy_rpn_monitor = []
	rpn_accuracy_for_epoch = []
	start_time = time.time()
	best_loss = np.Inf
	print('Starting training')

	loss_array = []
	ap_array = []
	epoch_array = []
	epoch_array.append(0)

	result = {}
	model_name = ''

	for epoch_num in range(num_epochs):
		progbar = generic_utils.Progbar(epoch_length)
		print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

		while True:
			try:

				if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
					mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
					rpn_accuracy_rpn_monitor = []
					print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
					if mean_overlapping_bboxes == 0:
						print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

				# train
				X, Y, img_data = next(data_gen_train)
				loss_rpn = model_rpn.train_on_batch(X, Y)
				P_rpn = model_rpn.predict_on_batch(X)

				R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
				X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)
				if X2 is None:
					rpn_accuracy_rpn_monitor.append(0)
					rpn_accuracy_for_epoch.append(0)
					continue
				neg_samples = np.where(Y1[0, :, -1] == 1)
				pos_samples = np.where(Y1[0, :, -1] == 0)
				if len(neg_samples) > 0:
					neg_samples = neg_samples[0]
				else:
					neg_samples = []
				if len(pos_samples) > 0:
					pos_samples = pos_samples[0]
				else:
					pos_samples = []
				rpn_accuracy_rpn_monitor.append(len(pos_samples))
				rpn_accuracy_for_epoch.append((len(pos_samples)))

				if C.num_rois > 1:
					if len(pos_samples) < C.num_rois//2:
						selected_pos_samples = pos_samples.tolist()
					else:
						selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
					try:
						selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
					except:
						selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

					sel_samples = selected_pos_samples + selected_neg_samples
				else:
					selected_pos_samples = pos_samples.tolist()
					selected_neg_samples = neg_samples.tolist()
					if np.random.randint(0, 2):
						sel_samples = random.choice(neg_samples)
					else:
						sel_samples = random.choice(pos_samples)

				loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
				# train

				losses[iter_num, 0] = loss_rpn[1]
				losses[iter_num, 1] = loss_rpn[2]
				losses[iter_num, 2] = loss_class[1]
				losses[iter_num, 3] = loss_class[2]
				losses[iter_num, 4] = loss_class[3]

				iter_num += 1
				progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
										  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

				if iter_num == epoch_length:
					loss_rpn_cls = np.mean(losses[:, 0])
					loss_rpn_regr = np.mean(losses[:, 1])
					loss_class_cls = np.mean(losses[:, 2])
					loss_class_regr = np.mean(losses[:, 3])
					class_acc = np.mean(losses[:, 4])

					mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
					rpn_accuracy_for_epoch = []

					if C.verbose:
						print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
						print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
						print('Loss RPN classifier: {}'.format(loss_rpn_cls))
						print('Loss RPN regression: {}'.format(loss_rpn_regr))
						print('Loss Detector classifier: {}'.format(loss_class_cls))
						print('Loss Detector regression: {}'.format(loss_class_regr))
						print('Elapsed time: {}'.format(time.time() - start_time))

					# result
					curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
					iter_num = 0
					start_time = time.time()

					if curr_loss < best_loss:
						if C.verbose:
							print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
						best_loss = curr_loss

						if save_best_weights:
							real_model_path = real_model_path +str(epoch_num+1)+'.hdf5'
							model_all.save_weights(real_model_path, overwrite=True)
							print("Best weights so far saved to " + real_model_path + ". best_loss = " + str(best_loss))
							epoch_array.append(epoch_num+1)
							loss_array.append([loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, best_loss])
							album_ap, logo_ap, mAP = measure_map.measure_map(config_output_filename, real_model_path)
							ap_array.append([album_ap, logo_ap, mAP])
							np.save(diagnose_path, [epoch_array, loss_array, ap_array])
						else:
							album_ap = 'not applicable'
							logo_ap = 'not applicable'
							mAP = 'not applicable'
						model_name = "model_{}_{}".format(str(best_loss), str(uuid.uuid4())[:5])
						result = {
							'loss': best_loss,
							'loss_rpn_cls': loss_rpn_cls,
							'loss_rpn_regr': loss_rpn_regr,
							'loss_class_cls': loss_class_cls,
							'loss_class_regr': loss_class_regr,
							'album_ap': album_ap,
							'logo_ap': logo_ap,
							'mAP': mAP,
							'model_name': model_name,
							'space': hype_space,
							'status': STATUS_OK
						}
						print("RESULT UPDATED.")
						print("Model name: {}".format(model_name))
					# result
					break

			except Exception as e:
				print('Exception: {}'.format(e))
				continue

	print('Training complete, exiting.')
	print("BEST MODEL: {}".format(model_name))
	print("FINAL RESULT:")
	print_json(result)
	save_json_result(model_name, result)
	try:
		K.clear_session()
		del model_all, model_rpn, model_classifier
	except Exception as err:
		try:
			K.clear_session()
		except:
			pass
		err_str = str(err)
		print(err_str)
		traceback_str = str(traceback.format_exc())
		print(traceback_str)
		return {
			'status': STATUS_FAIL,
			'err': err_str,
			'traceback': traceback_str
		}
	print("\n\n")
	return model_name, result

