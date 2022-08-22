from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from losses.triplet_loss import compute_loss
from model.advise_pytorch import ADVISE
from dataloaders.ads_dataset import AdsDataset
from torch.utils.data import DataLoader, dataloader, dataset
import argparse

import os
import json
import time
import nltk
import numpy as np

import torch
from dataloaders.utils import load_action_reason_annots
from utils import eval_utils

split = 'test'
batch_size = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = "trained_models/"
model_name = "model_kb_vit_512_0005.pth"
action_reason_annot_path = 'data/test/QA_Combined_Action_Reason_train.json'

def accuracy_on_persuasion(results):
	correct_pred = 0
	for image_id in results:
		predicted = results[image_id]['persuasion_logits']
		truths = results[image_id]['groundtruths']
		index = predicted.index(max(predicted))
		if truths[index]==1:
			correct_pred+=1
	acc = correct_pred/len(results)
	return acc 

def top_three_accuracy(results):
	correct_pred = 0
	for image_id in results:
		predicted = results[image_id]['persuasion_logits']
		truths = results[image_id]['groundtruths']
		indexes = sorted(range(len(predicted)),key=lambda x:predicted[x])[-3:]
		for index in indexes:
			if truths[index]==1:
				correct_pred+=1
				break
	top_3_acc = correct_pred/len(results)
	return top_3_acc 

def recall_persuasion(results):
	num=0
	den=0
	for image_id in results:
		predicted = results[image_id]['persuasion_logits']
		truths = results[image_id]['groundtruths']
		indexes = sorted(range(len(predicted)),key=lambda x:predicted[x])[-3:]
		for index in indexes:
			if truths[index]==1:
				num+=1
		den+=truths.count(1)
	recall = num/den 
	return recall

def _load_vocab(filename):
	"""Loads vocabulary.

	Args:
		filename: path to the vocabulary file.

	Returns:
		a list mapping from id to word.
	"""
	with open(filename, 'r') as fp:
		vocab = ['UNK'] + [x.strip('\n').split('\t')[0] for x in fp.readlines()]
	return vocab


def export_inference(results, groundtruths, filename):
	"""Exports results to a specific file.

	Args:
		results: 
		groundtruths:
		filename: the path to the output json file.
	"""
	final_results = {}
	for image_id, result in results.items():
		pred = np.array(result['distances']).argmin()
		final_results[image_id] = groundtruths[image_id]['all_examples'][pred]

	with open(filename, 'w') as fp:
		fp.write(json.dumps(final_results))

#Currently only for the non continuous eval mode
def evaluate_predictions(groundtruths, results):
	export_inference(results, groundtruths, results_path)

	metrics = eval_utils.evaluate(results, groundtruths)
	print("Evaluation results: {}".format(json.dumps(metrics, indent=2)))

	# Save results.
	return metrics['accuracy']

def main():
	with open('configs/advise_densecap_data.json') as fp:
		data_config = json.load(fp)
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu_id', help="Please give a value for gpu id")
	parser.add_argument('--model_name', help="Please give a value for model name")
	args = parser.parse_args()
	
	if args.gpu_id is not None:
		device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
	if args.model_name is not None:
		model_name = args.model_name
	PATH = PATH + model_name

	groundtruths = load_action_reason_annots(action_reason_annot_path)

	dataset = AdsDataset(data_config, split=split)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	with open('configs/advise_kb_training.json') as fp:
		train_config = json.load(fp)

	#Init model for training
	model = ADVISE(train_config, device,is_training=False)
	model.load_state_dict(torch.load(PATH))
	model.eval()
	model = model.to(device)

	acc = 0 
	top_3_acc = 0
	recall = 0

	# Loop through the evaluation dataset.
	results = {}
	results_persuasion = {}

	# Iterate over data.
	for examples in dataloader:
		for key, value in examples.items():
			if torch.is_tensor(examples[key]):
				examples[key] = examples[key].to(device)
		#examples = examples.to(device)

		running_loss = 0.0

		# forward
		# track history if only in train
		with torch.set_grad_enabled(False):
			outputs = model(examples)
			#loss_dict = compute_loss(outputs, train_config, is_training= False)
			
			loss = torch.tensor(0.0).to(device)
			l = nn.BCELoss()
			loss += l(outputs["persuasion_logits"],examples["persuasion_strategies"])
			for image_id, out in zip(outputs['image_id'],outputs['persuasion_logits']):
				results_persuasion[image_id]={
						'persuasion_logits':list( map(lambda x: round(x,5),out.tolist()))
						}
			for image_id, out in zip(examples['image_id'],examples['persuasion_strategies']):
				if image_id in results_persuasion:
					results_persuasion[image_id].update({'groundtruths':out.tolist()})
			# for loss_name, loss_tensor in loss_dict.items():

			# 	loss += loss_tensor

	# 	for image_id, distances in zip(outputs['image_id'], outputs['distance']):
	# 		results[image_id] = {
	# 		'distances': map(lambda x: round(x, 5), distances.tolist()),
	# 		}

	# 	# statistics
	# 	running_loss += loss.item() * batch_size

	# epoch_loss = running_loss / len(dataset)
	# acc = evaluate_predictions(groundtruths, results)
	acc = accuracy_on_persuasion(results_persuasion)
	top_3_acc = top_three_accuracy(results_persuasion)
	recall = recall_persuasion(results_persuasion)

	print('{} Top-1 Accuracy : {:.4f} \nTop-3 Accuracy: {:.4f}\nRecall: {:.4f}'.format("Test", acc,top_3_acc,recall))


	# print('{} Loss: {:.4f} Acc: {:.4f}'.format("Test", epoch_loss, acc))


if __name__ == '__main__':
	main()
