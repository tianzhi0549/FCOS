import torch
import math
import cupy
from torch import nn
import torch.nn.functional as F
class Stream:
	ptr = torch.cuda.current_stream().cuda_stream

IRNNForward = open('fcos_core/modeling/dsc/IRNN_Forward_cuda.cu','r').read()

IRNNBackward = open('fcos_core/modeling/dsc/IRNN_Backward_cuda.cu','r').read()

# IRNNWeightBaisBackward = open('./IRNN_Weight_Bias_Backward_cuda.cu','r').read()


@cupy.util.memoize(for_each_device=True)
def cunnex(strFunction):
	return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)
# end

class irnn(torch.autograd.Function):
	def __init__(self):
		super(irnn, self).__init__()
		

	def forward(self, input_feature, weight_up, weight_right, weight_down, weight_left, bias_up, bias_right, bias_down, bias_left):
		

		assert(input_feature.is_contiguous() == True)
		assert(weight_left.is_contiguous() == True)
		assert(weight_right.is_contiguous() == True)
		assert(weight_down.is_contiguous() == True)

		assert(weight_up.is_contiguous() == True)
		assert(bias_left.is_contiguous() ==True)
		assert(bias_right.is_contiguous() == True)
		assert(bias_up.is_contiguous() == True)
		assert(bias_down.is_contiguous() == True)

		output_left = input_feature.clone()
		output_right = input_feature.clone()
		output_up = input_feature.clone()
		output_down = input_feature.clone()

		if input_feature.is_cuda == True:
			n = input_feature.nelement()
			cuda_num_threads = 1024
			cunnex('IRNNForward')(
				grid=tuple([ int((n +  cuda_num_threads - 1) / cuda_num_threads ), 1, 1 ]),
				block=tuple([ cuda_num_threads , 1, 1 ]),
				args=[
					input_feature.data_ptr(),
					
					weight_up.data_ptr(), 
					weight_right.data_ptr(),
					weight_down.data_ptr(),
					weight_left.data_ptr(),

					bias_up.data_ptr(),
					bias_right.data_ptr(),
					bias_down.data_ptr(),
					bias_left.data_ptr(),
					
					output_up.data_ptr(), 
					output_right.data_ptr(),
					output_down.data_ptr(), 
					output_left.data_ptr(),
					
					input_feature.size(1),
					input_feature.size(2),
					input_feature.size(3),
					n],
				stream=Stream
			)
		elif input_feature.is_cuda == False:
			raise NotImplementedError()
		
		
		self.save_for_backward(input_feature,weight_up,weight_right,weight_down,weight_left,output_up,output_right,output_down,output_left)

		
		return output_up,output_right,output_down,output_left
	# end
	

	def backward(self, grad_output_up,grad_output_right,grad_output_down,grad_output_left):

		input_feature,weight_up,weight_right,weight_down,weight_left,output_up,output_right,output_down,output_left = self.saved_tensors
		# print(weight_left)
		if grad_output_up.is_contiguous() != True:
			grad_output_up = grad_output_up.contiguous()
		if grad_output_right.is_contiguous() != True:
			grad_output_right = grad_output_right.contiguous()
		if grad_output_down.is_contiguous() != True:
			grad_output_down = grad_output_down.contiguous()
		if grad_output_left.is_contiguous() != True:
			grad_output_left = grad_output_left.contiguous()

		# init gradient of input_feature
		grad_input = torch.zeros_like(input_feature)
		# init gradient map of weights
		grad_weight_up_map = torch.zeros_like(input_feature)
		grad_weight_right_map = torch.zeros_like(input_feature)
		grad_weight_down_map = torch.zeros_like(input_feature)
		grad_weight_left_map = torch.zeros_like(input_feature)
		# init gradient of weights
		grad_weight_left = torch.zeros_like(weight_left)
		grad_weight_right = torch.zeros_like(weight_left)
		grad_weight_up = torch.zeros_like(weight_left)
		grad_weight_down = torch.zeros_like(weight_left)

		grad_bias_up_map = torch.zeros_like(input_feature)
		grad_bias_right_map = torch.zeros_like(input_feature)
		grad_bias_down_map = torch.zeros_like(input_feature)
		grad_bias_left_map = torch.zeros_like(input_feature)

		if input_feature.is_cuda == True:
			
			n = grad_input.nelement()

			cuda_num_threads = 1024		
			cunnex('IRNNBackward')(
				grid=tuple([ int((n +  cuda_num_threads - 1) / cuda_num_threads), 1, 1 ]),
				block=tuple([ cuda_num_threads , 1, 1 ]),
				args=[ 
					grad_input.data_ptr(),

					grad_weight_up_map.data_ptr(),
					grad_weight_right_map.data_ptr(),
					grad_weight_down_map.data_ptr(),
					grad_weight_left_map.data_ptr(),

					grad_bias_up_map.data_ptr(),  
					grad_bias_right_map.data_ptr(),
					grad_bias_down_map.data_ptr(),
					grad_bias_left_map.data_ptr(),

					weight_up.data_ptr(),
					weight_right.data_ptr(),
					weight_down.data_ptr(),
					weight_left.data_ptr(),
					
					grad_output_up.data_ptr(),
					grad_output_right.data_ptr(),
					grad_output_down.data_ptr(),
					grad_output_left.data_ptr(),
					
					output_up.data_ptr(),					
					output_right.data_ptr(),					
					output_down.data_ptr(),
					output_left.data_ptr(),

					input_feature.size(1),
					input_feature.size(2),
					input_feature.size(3),
					n],
				stream=Stream
			)
			# print(grad_weight_left_map,"<-- grad weight map")

			grad_bias_up = torch.zeros_like(weight_left).reshape(weight_left.size(0))
			grad_bias_right = torch.zeros_like(weight_left).reshape(weight_left.size(0))
			grad_bias_down = torch.zeros_like(weight_left).reshape(weight_left.size(0))
			grad_bias_left = torch.zeros_like(weight_left).reshape(weight_left.size(0))

			grad_weight_left  = grad_weight_left_map.sum(2).sum(2).sum(0).resize_as_(grad_weight_left)
			grad_weight_right = grad_weight_right_map.sum(2).sum(2).sum(0).resize_as_(grad_weight_left)
			grad_weight_up    = grad_weight_up_map.sum(2).sum(2).sum(0).resize_as_(grad_weight_left)
			grad_weight_down  = grad_weight_down_map.sum(2).sum(2).sum(0).resize_as_(grad_weight_left)

			grad_bias_up    = grad_bias_up_map.sum(2).sum(2).sum(0).resize_as_(grad_bias_up)
			grad_bias_right = grad_bias_right_map.sum(2).sum(2).sum(0).resize_as_(grad_bias_up)
			grad_bias_down  = grad_bias_down_map.sum(2).sum(2).sum(0).resize_as_(grad_bias_up)
			grad_bias_left  = grad_bias_left_map.sum(2).sum(2).sum(0).resize_as_(grad_bias_up)

			

			# n = input_feature.size(1)
			# cuda_num_threads = n		
			# cunnex('IRNNWeightBaisBackward')(
			# 	grid=tuple([ int((n +  cuda_num_threads - 1) / cuda_num_threads), 1, 1 ]),
			# 	block=tuple([ cuda_num_threads , 1, 1 ]),
			# 	args=[ 
			# 		grad_weight_up_map.data_ptr(),
			# 		grad_weight_right_map.data_ptr(),
			# 		grad_weight_down_map.data_ptr(),
			# 		grad_weight_left_map.data_ptr(),

			# 		grad_bias_up_map.data_ptr(),
			# 		grad_bias_right_map.data_ptr(),
			# 		grad_bias_down_map.data_ptr(),
			# 		grad_bias_left_map.data_ptr(),

			# 		grad_weight_up.data_ptr(),
			# 		grad_weight_right.data_ptr(),
			# 		grad_weight_down.data_ptr(),
			# 		grad_weight_left.data_ptr(),   

			# 		grad_bias_up.data_ptr(),
			# 		grad_bias_right.data_ptr(),
			# 		grad_bias_down.data_ptr(),
			# 		grad_bias_left.data_ptr(),

			# 		input_feature.size(0),
			# 		input_feature.size(1),
			# 		input_feature.size(2),
			# 		input_feature.size(3),
			# 		n],
			# 	stream=Stream
			# )
			
		elif input_feature.is_cuda == False:
			raise NotImplementedError()

		
		return grad_input, grad_weight_up,grad_weight_right,grad_weight_down,grad_weight_left,grad_bias_up, grad_bias_right, grad_bias_down, grad_bias_left