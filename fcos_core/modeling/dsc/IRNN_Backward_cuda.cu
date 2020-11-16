#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
	  i < (n);                                       \
      i += blockDim.x * gridDim.x)
      
#define INDEX(b,c,h,w,channels,height,width) ((b * channels + c) * height + h) * width+ w


extern "C" __global__ void IRNNBackward(
    float* 		        grad_input,
    
    float*              grad_weight_up_map,
    float*              grad_weight_right_map,
    float*              grad_weight_down_map,
    float*              grad_weight_left_map,

    float*    grad_bias_up_map,
    float*    grad_bias_right_map,
    float*    grad_bias_down_map,
    float*    grad_bias_left_map,

    const float*  		weight_up,
    const float*        weight_right,
    const float*        weight_down,
    const float*        weight_left,

	const float*  		grad_output_up, 
    const float*        grad_output_right,
    const float*        grad_output_down,
    const float*        grad_output_left,

	const float*  		output_up,
    const float*        output_right,
    const float*        output_down,
    const float*        output_left,

	const int 			channels, 
	const int 			height, 
	const int			width,
    const int           n) {

    CUDA_KERNEL_LOOP(index,n){

        int w = index % width;
        int h = index / width % height;
        int c = index / width / height % channels;
        int b = index / width / height / channels;

        float diff_left  = 0;
        float diff_right = 0;
        float diff_up    = 0;
        float diff_down  = 0;

        //left 
       
        for (int i = 0; i<=w; i++)
        {   
            diff_left *= weight_left[c];
            diff_left += grad_output_left[INDEX(b, c, h, i, channels, height, width)];
            diff_left *= (output_left[INDEX(b, c, h, i, channels, height, width)]<=0)? 0 : 1;
        }
        

        float temp = grad_output_left[INDEX(b, c, h, 0, channels, height, width)];
        for (int i = 1; i < w +1 ; i++)
        {
            temp = (output_left[INDEX(b, c, h, i-1, channels, height, width)] >0?1:0) * temp * weight_left[c] + grad_output_left[INDEX(b, c, h, i, channels, height, width)];
        }

        if (w != width - 1){
            grad_weight_left_map[index] = temp * output_left[INDEX(b, c, h, w+1, channels, height, width)] * (output_left[index] > 0? 1:0);
            grad_bias_left_map[index] = diff_left;
        }

        // right 

        for (int i = width -1; i>=w; i--)
        {   
            diff_right *= weight_right[c];
            diff_right += grad_output_right[INDEX(b, c, h, i, channels, height, width)];
            diff_right *= (output_right[INDEX(b, c, h, i, channels, height, width)]<=0)? 0 : 1;
        }
        

        temp = grad_output_right[INDEX(b, c, h, width-1, channels, height, width)];
        for (int i = width -2; i > w - 1 ; i--)
        {
            temp = (output_right[INDEX(b, c, h, i+1, channels, height, width)] >0?1:0) * temp * weight_right[c] + grad_output_right[INDEX(b, c, h, i, channels, height, width)];
        }

        if (w != 0){
            grad_weight_right_map[index] = temp * output_right[INDEX(b, c, h, w-1, channels, height, width)] * (output_right[index] > 0? 1:0);
            grad_bias_right_map[index] = diff_right;
        }

        // up

        
        for (int i = 0; i<=h; i++)
        {   
            diff_up *= weight_up[c];
            diff_up += grad_output_up[INDEX(b, c, i, w, channels, height, width)];
            diff_up *= (output_up[INDEX(b, c, i, w, channels, height, width)]<=0)? 0 : 1;
        }
       

        temp = grad_output_up[INDEX(b, c, 0, w, channels, height, width)];
        for (int i = 1; i < h +1 ; i++)
        {
            temp = (output_up[INDEX(b, c, i-1, w, channels, height, width)] >0?1:0) * temp * weight_up[c] + grad_output_up[INDEX(b, c, i, w, channels, height, width)];
        }

        if (h != height - 1){
            grad_weight_up_map[index] = temp * output_up[INDEX(b, c, h+1, w, channels, height, width)] * (output_up[index] > 0? 1:0);
            grad_bias_up_map[index] = diff_up;
        }

        // down

        for (int i = height -1; i>=h; i--)
        {   
            diff_down *= weight_down[c];
            diff_down += grad_output_down[INDEX(b, c, i, w, channels, height, width)];
            diff_down *= (output_down[INDEX(b, c, i, w, channels, height, width)]<=0)? 0 : 1;
        }
        

        temp = grad_output_down[INDEX(b, c, height-1, w, channels, height, width)];
        for (int i = height -2; i > h - 1 ; i--)
        {
            temp = (output_down[INDEX(b, c, i+1, w, channels, height, width)] >0?1:0) * temp * weight_down[c] + grad_output_down[INDEX(b, c, i, w, channels, height, width)];
        }

        if (h != 0){
            grad_weight_down_map[index] = temp * output_down[INDEX(b, c, h-1, w, channels, height, width)] * (output_down[index] > 0? 1:0);
            grad_bias_down_map[index] = diff_down;
        }
        grad_input[index] = diff_down + diff_left + diff_right + diff_up;
    }
}