from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch 
import torch.nn as nn

from text_encoders.bow_encoder import BOWEncoder
from text_encoders.trans_encoder import Encoder
from text_encoders.bert_encoder import BERTEncoder
from text_encoders.simple_encoder import SimEncoder

class ADVISE(nn.Module):
	def __init__(self, config, device, is_training=True):
		super(ADVISE, self).__init__()
		# args
		# config = hyperparameters to be used in the ADVISE model
		# is_training: if True, training graph would be built. 
		
		self.config = config
		self.is_training = is_training
		self.device = device
		self._stmt_encoder = BOWEncoder(config['stmt_encoder'], is_training)
		#self._stmt_encoder = Encoder(config['stmt_encoder'], device)
		#self._stmt_encoder = BERTEncoder(config['stmt_encoder'], is_training)
		#self._stmt_encoder = SimEncoder(config['stmt_encoder'], is_training)
		
		self._ocr_encoder = BOWEncoder(config['ocr_encoder'], is_training)
		
		if config["densecap_loss_weight"] > 0:
			 self._densecap_encoder = BOWEncoder(config['densecap_encoder'], is_training)
			 #self._densecap_encoder = Encoder(config['densecap_encoder'], device)
			 #self._densecap_encoder = BERTEncoder(config['densecap_encoder'], is_training)

		if config["symbol_loss_weight"] > 0:
			 self._symbol_encoder = BOWEncoder(config['symbol_encoder'], is_training)
			 #self._symbol_encoder = Encoder(config['symbol_encoder'], device)
				
		
		self.encode_feature1 = nn.Sequential(nn.Dropout(1 - config["image_encoder"]["input_dropout_keep_prob"]),
																				nn.Linear(config["image_decoder"]["num_outputs"], config["image_encoder"]["num_outputs"]),
																				nn.BatchNorm1d(config["image_encoder"]["num_outputs"]),
																				nn.Dropout(1 - config["image_encoder"]["output_dropout_keep_prob"]))
																				
		self.encode_feature2 = nn.Sequential(nn.Dropout(1 - config["image_attention_predictor"]["input_dropout_keep_prob"]),
																				nn.Linear(config["image_encoder"]["num_outputs"], config["image_attention_predictor"]["num_outputs"]),
																				nn.BatchNorm1d(config["image_attention_predictor"]["num_outputs"]),
																				nn.Dropout(1 - config["image_attention_predictor"]["output_dropout_keep_prob"]))
		
		#self.encode_feature3 = nn.Sequential(nn.Dropout(1 - config["image_decoder"]["input_dropout_keep_prob"]),
		#                                    nn.Linear(config["image_encoder"]["num_outputs"], config["image_decoder"]["num_outputs"]),
		#                                    nn.Dropout(1 - config["image_decoder"]["output_dropout_keep_prob"]))
		
		if config["use_knowledge_branch"]:
			self.symbol_classifier = nn.Sequential(nn.Dropout(1 - config["symbol_classifier"]["input_dropout_keep_prob"]),
																					nn.Linear(config["image_decoder"]["num_outputs"], config["symbol_classifier"]["hidden_units"]),
																					nn.ReLU(),
																					nn.Dropout(1 - config["symbol_classifier"]["hidden_dropout_keep_prob"]),
																					nn.Linear(config["symbol_classifier"]["hidden_units"], config["symbol_classifier"]["output_units"]))
			
			weights = torch.full([config["symbol_classifier"]["output_units"] - 1], -3.0)
			self.symbol_classifier_weights = nn.Parameter(weights)
		#change
		self.attn = nn.Linear(256, 1)
		self.fc = nn.Linear(256,16)
		self.sigmoid = nn.Sigmoid()


	
	def encode_image(self, img_features, roi_features):
		"""Encodes image into embedding vector.

		Args:
			img_features: a [batch, feature_dimensions] tf.float32 tensor.
			roi_features: a [batch, number_of_regions, feature_dimensions] tf.float32 tensor.

		Raises:
			ValueError: if the pooling method of the config is invalid.

		Returns:
			img_encoded: a [batch, common_dimentions] tf.float32 tensor, l2_normalize
				is NOT applied.
			img_attention: a [batch, number_of_regions] tf.float32 tensor, output of
				the softmax. 
		"""
		config = self.config
		is_training = self.is_training

		roi_features = torch.cat((torch.unsqueeze(img_features, 1), roi_features), 1)
		roi_features = roi_features[:, :-1, :]

		_, num_of_regions, feature_dimensions = list(roi_features.shape)

		# Encode roi regions.
		roi_features_reshaped = torch.reshape(roi_features, (-1, feature_dimensions))
		roi_encoded_reshaped = self.encode_feature1(roi_features_reshaped)
		roi_encoded = torch.reshape(roi_encoded_reshaped,
																(-1, num_of_regions, config["image_encoder"]["num_outputs"])) 
																

		img_attention = None
		
		# Average pooling.
		if config["pooling_method"] == "AVG_POOL":
			img_encoded = torch.mean(roi_encoded, 1)

		# Attention pooling.
		else:
			assert 'image_attention_predictor' in config.keys()
			assert config["image_attention_predictor"]["num_outputs"] == 1

			if config["pooling_method"] == "ATT_POOL":
				attention_inputs = roi_encoded

			elif config["pooling_method"] == "ATT_POOL_DS_SUM":
				attention_inputs = torch.subtract(roi_encoded,
																			 torch.tile(torch.sum(
																					 roi_encoded, 1, keepdim=True), (1, num_of_regions, 1)))

			elif config["pooling_method"] == "ATT_POOL_DS_MAX":
				attention_inputs = torch.subtract(roi_encoded, 
																					torch.tile(torch.max(
																							roi_encoded, 1, keepdim=True), (1, num_of_regions, 1)))

			else:
				raise ValueError('Unknown pooling method %i.' % (config["pooling_method"]))

			attention_inputs_reshaped = torch.reshape( 
					attention_inputs, (-1, list(attention_inputs.shape)[-1]))
			
			img_attention = torch.reshape(self.encode_feature2(attention_inputs_reshaped),
																 (-1, num_of_regions))
			softmax = torch.nn.Softmax(dim = -1)
			
			img_attention = softmax(img_attention)
			img_encoded = torch.squeeze(torch.matmul(torch.unsqueeze(img_attention, 1), roi_encoded), 1)

		# Autoencoder: reconstruction loss.
		# roi_decoded_reshaped = self.encode_feature3(roi_encoded_reshaped)

		#Not using autoencoder loss right now so not returning features
		return img_encoded, img_attention
	

	def encode_text(self, text_strings, text_lengths, encoder):

		"""Encodes text into embedding vector.

		Args:
			text_strings: a [batch, max_text_len] tf.int32 tensor.
			text_lengths: a [batch] tf.int32 tensor.
			encoder: an instance of TextEncoder used to encode text.

		Raises:
			ValueError: if the pooling method of the config is invalid.

		Returns:
			text_encoded: a [batch, common_dimentions] tf.float32 tensor, l2_normalize
				is NOT applied.
			text_attention: a [batch, max_text_len] tf.float32 tensor, output of
				the softmax. 
		"""
		text_encoded = encoder(text_strings, text_lengths)
		return text_encoded, None

	def forward(self, examples):

		"""Builds tensorflow graph for inference.

		Args:
			examples: a python dict involving at least following k-v fields:
				img_features: a [batch, feature_dimensions] tf.float32 tensor.
				roi_features: a [batch, number_of_regions, feature_dimensions] tf.float32 tensor.
				statement_strings: a [batch, statement_max_sent_len] tf.int64 tensor.
				statement_lengths: a [batch] tf.int64 tensor.

		Returns:
			predictions: a dict mapping from output names to output tensors.

		Raises:
			ValueError: if config is not properly configured.
		"""
		config = self.config
		is_training = self.is_training
		# Encode image features.
		(img_encoded, img_attention) = self.encode_image(examples['img_features'], examples['roi_features'])
		# Encode statement features. 
		(stmt_encoded, stmt_attention) = self.encode_text(text_strings=examples['statement_strings'], 
																												text_lengths=examples['statement_lengths'], encoder=self._stmt_encoder)
		# Encode eval metric statement features.
		statement_strings = examples['eval_statement_strings']
		statement_lengths = examples['eval_statement_lengths']
		
		(number_of_val_stmts_per_image, max_stmt_len) = list(statement_strings.shape)[1:]
		statement_strings_reshaped = torch.reshape(statement_strings, (-1, max_stmt_len))
		statement_lengths_reshaped = torch.reshape(statement_lengths, (-1,))
		
		(stmt_encoded_eval, stmt_attention_eval) = self.encode_text(text_strings=statement_strings_reshaped, 
																												text_lengths=statement_lengths_reshaped,encoder=self._stmt_encoder)
		
		# For optional constraints.
		if config["densecap_loss_weight"] > 0:  
				# For densecap constraint.
				(densecap_encoded, densecap_attention) = self.encode_text(text_strings=examples['densecap_strings'], 
																																	text_lengths=examples['densecap_lengths'],
																																	encoder=self._densecap_encoder)
		if config["symbol_loss_weight"] > 0:
				# For symbol constraint.
				(symbol_encoded, symbol_attention) = self.encode_text(text_strings=examples['symbols'], 
																															text_lengths=examples['number_of_symbols'],
																															encoder=self._symbol_encoder)
		# Encode OCR text
		examples['ocr_strings'] = torch.squeeze(examples['ocr_strings'], 1)
		examples['ocr_lengths'] = torch.squeeze(examples['ocr_lengths'], 1)
		(ocr_encoded, ocr_attention) = self.encode_text(text_strings=examples['ocr_strings'], 
																																	text_lengths=examples['ocr_lengths'],
																																	encoder=self._ocr_encoder)                                                      
																															
		# Encode knowledge if specified.
		if config["use_knowledge_branch"]:
				symbol_logits = self.symbol_classifier(examples['img_features'])
				
				symbol_proba = torch.sigmoid(symbol_logits)[:, 1:]

				# Assign weight to each symbol classifier.
				symbol_classifier_wts =  2 * torch.sigmoid(self.symbol_classifier_weights)
				weights = symbol_proba * symbol_classifier_wts

				# Add encoded symbol prediction as a residual branch.
				symbol_embedding_mat = self._symbol_encoder.embedding_layer.weight[1:, :]
				symbol_pred_encoded = torch.matmul(weights, symbol_embedding_mat)
				img_encoded += symbol_pred_encoded
		
		#change
		combined_embedding = torch.cat((ocr_encoded.unsqueeze(1), img_encoded.unsqueeze(1), densecap_encoded.unsqueeze(1), symbol_encoded.unsqueeze(1)), dim=1)
		wt = self.attn(combined_embedding)
		wt = torch.softmax(wt, dim=1)
		src = torch.matmul(wt.permute(0, 2, 1), combined_embedding)
		src = torch.squeeze(src, dim = 1)
		src = self.sigmoid(self.fc(src))
			
		# Joint embedding and cosine distance computation.
		img_encoded_norm = torch.nn.functional.normalize(img_encoded, p = 2.0, dim = 1)
		stmt_encoded_norm = torch.nn.functional.normalize(stmt_encoded, p = 2.0, dim = 1)
		predictions = {
				'image_id': examples['image_id'],
				'img_encoded': img_encoded_norm,
				'stmt_encoded': stmt_encoded_norm,
			}
		stmt_encoded_eval = torch.nn.functional.normalize(stmt_encoded_eval, p = 2.0, dim = 1)
		stmt_encoded_eval = torch.reshape(stmt_encoded_eval, (-1, number_of_val_stmts_per_image, 
																									list(stmt_encoded_eval.shape)[-1]))
		
		if config['densecap_loss_weight'] > 0:  
				densecap_encoded_norm = torch.nn.functional.normalize(densecap_encoded, p = 2.0, dim = 1)          
				predictions.update({
						'dense_encoded': densecap_encoded_norm})
		if config['symbol_loss_weight'] > 0:
				symbol_encoded_norm = torch.nn.functional.normalize(symbol_encoded, p = 2.0, dim = 1)
				predictions.update({
						'number_of_symbols': examples['number_of_symbols'],
						'symb_encoded': symbol_encoded_norm})
				
		ocr_encoded_norm = torch.nn.functional.normalize(ocr_encoded, p = 2.0, dim = 1)
		predictions.update({'ocr_encoded': ocr_encoded_norm})
		
		predictions.update({'persuasion_logits': src})
		
		embed = img_encoded_norm + 0.1*densecap_encoded_norm + 0.1*symbol_encoded_norm + ocr_encoded_norm     
		distance = 1 - torch.sum(torch.mul(torch.unsqueeze(torch.nn.functional.normalize(embed, p = 2.0, dim = 1), 1), stmt_encoded_eval), axis=2)
			#distance = (3 - torch.sum(torch.mul(torch.unsqueeze(img_encoded_norm, 1), stmt_encoded_eval), axis=2) - torch.sum(torch.mul(torch.unsqueeze(densecap_encoded_norm, 1), stmt_encoded_eval), axis=2) - torch.sum(torch.mul(torch.unsqueeze(symbol_encoded_norm, 1), stmt_encoded_eval), axis=2))/3
		predictions.update({
				'distance': distance
			})
		return predictions
				
