import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, dataloader, dataset
#from sklearn.metrics import confusion_matrix
import numpy as np
import copy
import json
import time

from model.advise_pytorch import ADVISE
from dataloaders.ads_dataset import AdsDataset
from losses.triplet_loss import compute_loss
from utils import eval_utils
from dataloaders.utils import load_action_reason_annots

num_epochs = 20
batch_size = 128
lr = 0.001
lr_decay = 1.0
lr_decay_epochs = 25

action_reason_annot_path = 'data/train/QA_Combined_Action_Reason_train.json'

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def evaluate_predictions(groundtruths, results):
	metrics = eval_utils.evaluate(results, groundtruths)
	print("Evaluation results(on_action_reason_task): {}".format(json.dumps(metrics, indent=2)))

	# Save results.
	return metrics['accuracy']

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

def precision_persuasion(results):
	num=0
	for image_id in results:
		predicted = results[image_id]['persuasion_logits']
		truths = results[image_id]['groundtruths']
		indexes = sorted(range(len(predicted)),key=lambda x:predicted[x])[-3:]
		for index in indexes:
			if truths[index]==1:
				num+=1
	den = 3*len(results)
	precision = num/den
	return precision

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


def statistics_fun(results):
	categories = ['Reciprocity','Concreteness:Details about product','Social Impact','Authority/Expert Approval','Trustworthiness','Social Identity','Others','Eager','Fashionable','Creative','Feminine','Active','Amazed','Cheerful','Emotion',"Unclear"]
	categories_true_positive = [0]*16
	categories_false_positive = [0]*16
	categories_true_negative = [0]*16
	categories_false_negative = [0]*16
	prec = []
	for image_id in results:
		predicted = results[image_id]['persuasion_logits']
		truths = results[image_id]['groundtruths']
		pred = [0]*16
		num_pred = 0
		for i in truths:
			if i==1:
				num_pred+=1
		indexes = sorted(range(len(predicted)),key=lambda x:predicted[x])[-num_pred:]
		for i in range(len(predicted)):
			if i in indexes:
				pred[i]=1
			else:
				pred[i]=0
		for i in range(16):
			if pred[i]==1:
				if truths[i]==1:
					categories_true_positive[i]+=1
				else:
					categories_false_positive[i]+=1
			else:
				if truths[i]==1:
					categories_false_negative[i]+=1
				else:
					categories_true_negative[i]+=1
	# total_acc = (sum(categories_true_positive)+sum(categories_true_negative))/(sum(categories_true_positive)+sum(categories_true_negative)+sum(categories_false_positive)+sum(categories_false_negative))

	
	for i in range(16):
		if (categories_true_positive[i]==0 and categories_true_negative[i]==0 and categories_false_positive[i]==0 and categories_false_negative[i]==0):
			continue
		# acc = (categories_true_positive[i] + categories_true_negative[i])/(categories_true_positive[i]+categories_true_negative[i]+categories_false_positive[i]+categories_false_negative[i])
		# if (categories_true_positive[i]==0 and categories_false_negative[i]==0):
		# 	recall='N/A'
		# else:
		# 	recall = categories_true_positive[i]/(categories_true_positive[i]+categories_false_negative[i])
		if (categories_true_positive[i]==0 and categories_false_positive[i]==0):
			precision = 'N/A'
			prec[i] = 0
		else:
			precision = categories_true_positive[i]/(categories_true_positive[i]+categories_false_positive[i])
			prec[i] = precision
		# print(categories[i]," : Accuracy-",acc," Recall-",recall," Precision-",precision)
	return sum(prec)/16

def main():
	with open('configs/advise_densecap_data.json') as fp:
		data_config = json.load(fp)
	
	print("Train Dataset")
	train_dataset = AdsDataset(data_config, split='train')
	print("Val Dataset")
	val_dataset = AdsDataset(data_config, split='val')

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

	datasets = {'train': train_dataset, 'val': val_dataset}
	dataloaders = {'train': train_dataloader, 'val': val_dataloader}
	dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

	print("Dataset:", dataset_sizes)

	#Load training config
	with open('configs/advise_kb_training.json') as fp:
		train_config = json.load(fp)

	#Init model for training
	model = ADVISE(train_config, device, is_training=True)
	model = model.to(device)

	print("Model Parameters:")
	print(model)

	optimizer = optim.Adam(model.parameters(), lr = lr)
	#scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_epochs, gamma=0.1)

	best_model_wts = copy.deepcopy(model.state_dict())
	best_val_loss = float("inf")
	best_val_acc = 0
	best_precision = 0
	best_recall = 0
	best_val_top_3_acc = 0

	groundtruths = load_action_reason_annots(action_reason_annot_path)
	
	#torch.autograd.set_detect_anomaly(True)
	since = time.time()
	probabilities = {}

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch+1, num_epochs))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
				model.is_training = True
			else:
				model.eval()   # Set model to evaluate mode
				model.is_training = False
			
			# Loop through the evaluation dataset.
			results = {}
			results_persuasion = {}

			running_loss = 0.0
			
			loss_img_stmt = 0.0
			loss_stmt_img = 0.0
			loss_dense_img = 0.0
			loss_dense_stmt = 0.0
			loss_ocr_stmt = 0.0
			loss_stmt_ocr = 0.0
			loss_symb_img = 0.0
			loss_symb_stmt = 0.0
			
			loss_persuasion = 0.0
			#running_corrects = 0
			

			# Iterate over data.
			for examples in dataloaders[phase]:
				for key, value in examples.items():
					if torch.is_tensor(examples[key]):
						examples[key] = examples[key].to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(examples)
					loss_dict = compute_loss(outputs, train_config, is_training= model.is_training)
					
					loss = torch.tensor(0.0).to(device)
					
					for loss_name, loss_tensor in loss_dict.items():
						loss += loss_tensor
					
					#print(examples)
					loss_per = torch.tensor(0.0).to(device)
					l = nn.BCELoss()
					outputs["persuasion_logits"] = outputs["persuasion_logits"].to(torch.float32)
					loss_per = l(outputs["persuasion_logits"],examples["persuasion_strategies"])
					loss+=loss_per
					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * batch_size
				
				
				#print("Epoch Loss : ",running_loss/dataset_sizes[phase])
				for image_id, out in zip(outputs['image_id'],outputs['persuasion_logits']):
					results_persuasion[image_id]={
							'persuasion_logits':list( map(lambda x: round(x,5),out.tolist()))
							}
				for image_id, out in zip(examples['image_id'],examples['persuasion_strategies']):
					if image_id in results_persuasion:
						results_persuasion[image_id].update({'groundtruths':out.tolist()})
				
				
				loss_img_stmt += loss_dict['triplet_img_stmt'].item() * batch_size
				loss_stmt_img += loss_dict['triplet_stmt_img'].item() * batch_size
				loss_ocr_stmt += loss_dict['triplet_ocr_stmt'].item() * batch_size
				loss_stmt_ocr += loss_dict['triplet_stmt_ocr'].item() * batch_size
				loss_dense_img += loss_dict['triplet_dense_img'].item() * batch_size
				loss_dense_stmt += loss_dict['triplet_dense_stmt'].item() * batch_size
				loss_symb_img += loss_dict['triplet_symb_img'].item() * batch_size
				loss_symb_stmt += loss_dict['triplet_symb_stmt'].item() * batch_size
				
				loss_persuasion+= loss_per.item() * batch_size

				#print('{} Total Loss: {:.4f} Img-Stmt Loss: {:.4f} Stmt-Img Loss: {:.4f} Dense-Img Loss: {:.4f} Dense-Stmt Loss: {:.4f} Symb-Img Loss: {:.4f} Symb-Stmt Loss: {:.4f}'.format(phase, loss.item(), loss_dict['triplet_img_stmt'].item(), loss_dict['triplet_stmt_img'].item(), loss_dict['triplet_dense_img'].item(), loss_dict['triplet_dense_stmt'].item(), loss_dict['triplet_symb_img'].item(), loss_dict['triplet_symb_stmt'].item()))

				
				if phase != 'test':
					for image_id, distances in zip(outputs['image_id'], outputs['distance']):
						results[image_id] = {
						'distances': list(map(lambda x: round(x, 5), distances.tolist())),
						}
			#if phase == 'train':
			#    scheduler.step()
			
			epoch_loss = running_loss / dataset_sizes[phase]
			
			loss_img_stmt = loss_img_stmt / dataset_sizes[phase]
			loss_stmt_img = loss_stmt_img / dataset_sizes[phase]
			loss_ocr_stmt = loss_ocr_stmt / dataset_sizes[phase]
			loss_stmt_ocr = loss_stmt_ocr / dataset_sizes[phase]
			loss_dense_img = loss_dense_img / dataset_sizes[phase]
			loss_dense_stmt = loss_dense_stmt / dataset_sizes[phase]
			loss_symb_img = loss_symb_img / dataset_sizes[phase]
			loss_symb_stmt = loss_symb_stmt / dataset_sizes[phase]
			
			loss_persuasion = loss_persuasion / dataset_sizes[phase]
			
			#print('\nEpoch done')
			#print('-'*10)
			#print('{} Total Loss: {:.4f} Img-Stmt Loss: {:.4f} Stmt-Img Loss: {:.4f} OCR-Stmt Loss: {:.4f} Stmt-OCR Loss: {:.4f} Dense-Img Loss: {:.4f} Dense-Stmt Loss: {:.4f} Symb-Img Loss: {:.4f} Symb-Stmt Loss: {:.4f}'.format(phase, epoch_loss, loss_img_stmt, loss_stmt_img, loss_ocr_stmt, loss_stmt_ocr, loss_dense_img, loss_dense_stmt, loss_symb_img, loss_symb_stmt))
			
			print('{} Total Loss: {:.4f} Persuasion_loss : {:.4f}'.format(phase, epoch_loss, loss_persuasion))
			if phase != 'test':
				acc = evaluate_predictions(groundtruths, results)
				print('{} Accuracy on action_reason_task): {:.4f}'.format(phase, acc))
			
			#epoch_loss = running_loss/dataset_sizes[phase]
			# deep copy the model
			epoch_acc = accuracy_on_persuasion(results_persuasion)
			epoch_top_3_acc = top_three_accuracy(results_persuasion)
			epoch_recall = recall_persuasion(results_persuasion)
			epoch_precision = precision_persuasion(results_persuasion)
			if phase == 'val' and epoch_acc > best_val_acc:
				#best_val_loss = epoch_loss
				#print(phase,results_persuasion)
				best_val_acc = epoch_acc
				best_precision = epoch_precision
				best_recall = epoch_recall
				best_val_top_3_acc = epoch_top_3_acc
				probabilities = results_persuasion
				best_model_wts = copy.deepcopy(model.state_dict())
			#print('\n Statistics on persuasion task')
			#print(phase,"\nPersuasion Loss : ",epoch_loss)
			#persuasion_acc = statistics_fun(results_persuasion)
			print('{} \nTop-1 Accuracy(persuasion): {:.4f}\nTop-3 Accuracy: {:.4f}\nRecall : {:.4f}'.format(phase,epoch_acc,epoch_top_3_acc,epoch_recall))
			print('\n\n')

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val accuracy: {:.4f}\n Best val top-3 accuracy: {:.4f}\nBest Recall : {:.4f}'.format(best_val_acc,best_val_top_3_acc,best_recall))
	#print(probabilities)
	print("\nSize : ",len(probabilities))

	# load best model weights
	model.load_state_dict(best_model_wts)
	add = "trained_models/model_kb_vit_512_0005.pth"
	torch.save(model.state_dict(), add)
	print('Model saved at '+add)

if __name__ == '__main__':
	main()
