#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
	  i < (n);                                       \
      i += blockDim.x * gridDim.x)
      
#define INDEX(b,c,h,w,channels,height,width) ((b * channels + c) * height + h) * width+ w 

extern "C" __global__ void IRNNForward(
    const float*    input_feature,

    const float*    weight_up,
    const float*    weight_right,
    const float*    weight_down,
    const float*    weight_left,

    const float*    bias_up,
    const float*    bias_right,
    const float*    bias_down,
    const float*    bias_left,
    
    float*          output_up,
    float*          output_right,
    float*          output_down,
    float*          output_left,
    
    const int       channels,
    const int       height,
    const int       width,
    const int       n){
    
    CUDA_KERNEL_LOOP(index,n){
        int w = index % width;
        int h = index / width % height;
        int c = index / width / height % channels;
        int b = index / width / height / channels;

        float temp = 0;
        
        // left
        output_left[index] = input_feature[INDEX(b, c, h, width-1, channels, height, width)] > 0 ? input_feature[INDEX(b, c, h, width-1, channels, height, width)] : 0;
        for (int i = width-2; i>=w; i--)
        {
            temp = output_left[index] * weight_left[c] + bias_left[c] + input_feature[INDEX(b, c, h, i, channels, height, width)];
            output_left[index] = (temp > 0)? temp : 0;
        }

        // right
        output_right[index] = input_feature[INDEX(b, c, h, 0, channels, height, width)] > 0 ? input_feature[INDEX(b, c, h, 0, channels, height, width)] : 0;
        for (int i = 1; i <= w; i++)
        {
            temp = output_right[index] * weight_right[c] + bias_right[c] + input_feature[INDEX(b, c, h, i, channels, height, width)];
            output_right[index] = (temp > 0)? temp : 0;
        }

        // up
        output_up[index] = input_feature[INDEX(b,c,height-1,w,channels,height,width)] > 0 ? input_feature[INDEX(b,c,height-1,w,channels,height,width)] : 0;
        for (int i = height-2; i >= h; i--)
        {
            temp = output_up[index] * weight_up[c] + bias_up[c] + input_feature[INDEX(b, c, i, w, channels, height, width)];
            output_up[index] = (temp > 0)? temp : 0;
        }

        // down
        output_down[index] = input_feature[INDEX(b, c, 0, w, channels, height, width)] > 0 ? input_feature[INDEX(b, c, 0, w, channels, height, width)] : 0;
        for (int i = 1; i <= h; i++)  
        {
            temp = output_down[index] * weight_down[c] + bias_down[c] + input_feature[INDEX(b, c, i, w, channels, height, width)];
            output_down[index] = (temp > 0)? temp : 0;
        }
    }
}   