#include "ex1.h"

#define HIST_SIZE 256
#define HIST_SIZE_MEMORY 256 * sizeof(int)
#define IMG_SIZE (IMG_HEIGHT*IMG_WIDTH)

// Make the given histogram a CDF array
__device__ void prefix_sum(int arr[], int arr_size) { //TODO: switch to parallel
    int sum = 0;
    for(int i=0;i<arr_size;i++) {
        sum += arr[i];
        arr[i] = sum;
    }
    return;
}

// __device__ void map_calculation(int arr[], uchar* map, int arr_size) {
    
//     return;
// }

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__ 
void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

// Zeros the maps array of a given context
__device__ void reset_maps_array(uchar *maps, int *hist) {
    for (int i=0;i<TILE_COUNT * TILE_COUNT * HIST_SIZE;i++) {

                maps[i] = (unsigned char)0;
                hist[i] = (int)0;

    }


}




__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps,int *hist) {
    int tid=threadIdx.x;


    if(tid == 0) {
        reset_maps_array(maps, hist);
    }
    __syncthreads();

    // Each thread would calculate the histogram contribution
    // of a single row of length T in a specific tile.

    // Calculate offset of this thread in the image pixels array
    int tileID = tid / 8;
    int rowInTile = tid % 8;

    int numberOfTilesInRowImg = IMG_WIDTH / TILE_WIDTH;
    int gridRow = tileID / numberOfTilesInRowImg;
    int gridCol = tileID % numberOfTilesInRowImg;

    int left = TILE_WIDTH*gridCol;
    int right = TILE_WIDTH*(gridCol+1) - 1;
    int top = TILE_WIDTH*gridRow;
    int bottom = TILE_WIDTH*(gridRow+1) - 1;

    for (int y = 0; y < 8; y++)
    {
        for (int x=left; x<=right; x++) {
            uchar *row = all_in + (top + rowInTile*8 + y) * IMG_WIDTH;
            int val = row[x];
            atomicAdd((&(hist[(gridRow * numberOfTilesInRowImg + gridCol) * HIST_SIZE + val])),1);
        }
    }

    __syncthreads();

    // Make the histogram into a map by only the first thread in every tile
    if(rowInTile == 0) { //TODO: make it parallel
        // Now make the histogram a CDF, by running prefix_sum
        prefix_sum(&hist[(gridRow * numberOfTilesInRowImg + gridCol) * HIST_SIZE],HIST_SIZE);
        
        // Perform map calculation for each tile        
        // Make the given CDF array a map, using the given definition
        for(int i=0;i<HIST_SIZE;i++) {
            maps[(gridRow * numberOfTilesInRowImg + gridCol) * HIST_SIZE + i] = (float(hist[(gridRow * numberOfTilesInRowImg + gridCol) * HIST_SIZE + i]) * 255) / (TILE_WIDTH*TILE_WIDTH);
            //hist[(gridRow * numberOfTilesInRowImg + gridCol) * HIST_SIZE + i] = (float(hist[(gridRow * numberOfTilesInRowImg + gridCol) * HIST_SIZE + i]) * 255) / (TILE_WIDTH*TILE_WIDTH);
        }
    }

    // __syncthreads();
    // // Now we copy hist values to maps
    // for (int i=0;i<TILE_COUNT;i++) {
    //     for (int j=0;j<TILE_COUNT;j++) {
    //         for (int n=0;n<HIST_SIZE;n++) {
    //                 maps[i * TILE_COUNT * HIST_SIZE + j * HIST_SIZE + n] = (unsigned char)(hist[i * TILE_COUNT * HIST_SIZE + j * HIST_SIZE + n]);
    //         }
    //     }
    // }

    __syncthreads();

    
    interpolate_device(maps, all_in, all_out);
    return; 
}


/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    uchar *all_in;
    uchar *all_out;
    uchar *maps;
    int *hist;
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    CUDA_CHECK(cudaMalloc((void**) &(context->all_in), IMG_WIDTH * IMG_WIDTH));
    CUDA_CHECK(cudaMalloc((void**) &(context->all_out), IMG_WIDTH * IMG_WIDTH));
    CUDA_CHECK(cudaMalloc((void**) &(context->maps), TILE_COUNT * TILE_COUNT * HIST_SIZE));
    CUDA_CHECK(cudaMalloc((void**) &(context->hist), TILE_COUNT * TILE_COUNT * HIST_SIZE * sizeof(int)));



    return context;
}



/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out) //TODO: why one certain context for all images?
{
    //in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial


    
    // calculate the number of threads in one image
    int threads_in_block = (TILE_WIDTH * TILE_COUNT * TILE_COUNT) / 8;

    for (int i=0;i<N_IMAGES;i++) {
        CUDA_CHECK((cudaMemcpy(context->all_in, (images_in + i * IMG_SIZE), IMG_SIZE, cudaMemcpyHostToDevice)));

        process_image_kernel<<<1,threads_in_block>>>(context->all_in,context->all_out,context->maps,context->hist);

        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK((cudaMemcpy((images_out + i * IMG_SIZE), context->all_out, IMG_SIZE, cudaMemcpyDeviceToHost)));
    }

}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    CUDA_CHECK(cudaFree((void**)(context->all_in)));
    CUDA_CHECK(cudaFree((void**)(context->all_out)));
    CUDA_CHECK(cudaFree((void**)(context->maps)));
    CUDA_CHECK(cudaFree((void**)(context->hist)));

    free(context);
}


/////////////////BULK PROCESS/////////////////
__global__ void bulk_process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps,int *hist) { //TODO:
    int tid=threadIdx.x;
    int bid=blockIdx.x;


    //int hist[TILE_COUNT][TILE_COUNT][HIST_SIZE];

    // Each thread would calculate the histogram contribution
    // of a single row of length T in a specific tile.

    // Calculate offset of this thread in the image pixels array
    int imgID = bid;
    int tileID = tid / 8;
    int rowInTile = tid % 8;

    int numberOfTilesInRowImg = IMG_WIDTH / TILE_WIDTH;
    int gridRow = tileID / numberOfTilesInRowImg;
    int gridCol = tileID % numberOfTilesInRowImg;


    int left = TILE_WIDTH*gridCol;
    int right = TILE_WIDTH*(gridCol+1) - 1;
    int top = TILE_WIDTH*gridRow;
    int bottom = TILE_WIDTH*(gridRow+1) - 1;

      if(tid == 0) {
        reset_maps_array(&maps[imgID*numberOfTilesInRowImg*numberOfTilesInRowImg*HIST_SIZE], &hist[imgID*numberOfTilesInRowImg*numberOfTilesInRowImg*HIST_SIZE]);
    }
    __syncthreads();

    for (int y = 0; y < 8; y++)
    {
        for (int x=left; x<=right; x++) {
            uchar *row = all_in + imgID*IMG_SIZE + (top + rowInTile*8 + y) * IMG_WIDTH ;
            int val = row[x];
            atomicAdd((&(hist[imgID*numberOfTilesInRowImg*numberOfTilesInRowImg*HIST_SIZE + (gridRow * numberOfTilesInRowImg + gridCol) * HIST_SIZE + val])),1);
        }
    }

    __syncthreads();

    // Make the histogram into a map by only the first thread in every tile
    if(rowInTile == 0) {
        // Now make the histogram a CDF, by running prefix_sum
        prefix_sum(&hist[imgID*numberOfTilesInRowImg*numberOfTilesInRowImg*HIST_SIZE + (gridRow * numberOfTilesInRowImg + gridCol) * HIST_SIZE],HIST_SIZE);
        
        // Perform map calculation for each tile        
        // Make the given CDF array a map, using the given definition
        for(int i=0;i<HIST_SIZE;i++) {
            maps[imgID*numberOfTilesInRowImg*numberOfTilesInRowImg*HIST_SIZE + (gridRow * numberOfTilesInRowImg + gridCol) * HIST_SIZE + i] = 
            (float(hist[imgID*numberOfTilesInRowImg*numberOfTilesInRowImg*HIST_SIZE + (gridRow * numberOfTilesInRowImg + gridCol) * HIST_SIZE + i]) * 255) / (TILE_WIDTH*TILE_WIDTH);
            
        }
    }
 
    __syncthreads();

    interpolate_device(&maps[imgID*numberOfTilesInRowImg*numberOfTilesInRowImg*HIST_SIZE], (all_in + imgID*IMG_SIZE), (all_out + imgID*IMG_SIZE));

    __syncthreads();

    return; 
}


/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    uchar *all_in;
    uchar *all_out;
    uchar *maps;
    int *hist;};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    CUDA_CHECK(cudaMalloc((void**) &(context->all_in), IMG_WIDTH * IMG_WIDTH * N_IMAGES));
    CUDA_CHECK(cudaMalloc((void**) &(context->all_out), IMG_WIDTH * IMG_WIDTH * N_IMAGES));
    CUDA_CHECK(cudaMalloc((void**) &(context->maps), TILE_COUNT * TILE_COUNT * HIST_SIZE * N_IMAGES));
    // CUDA_CHECK(cudaMemset((context->maps),(unsigned char)0,TILE_COUNT * TILE_COUNT * HIST_SIZE * N_IMAGES));
    CUDA_CHECK(cudaMalloc((void**) &(context->hist), TILE_COUNT * TILE_COUNT * HIST_SIZE * sizeof(int) * N_IMAGES));
    // CUDA_CHECK(cudaMemset((context->hist),0,TILE_COUNT * TILE_COUNT * HIST_SIZE * sizeof(int) * N_IMAGES));


    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    // calculate the number of threads in one image
    int threads_in_block = (TILE_WIDTH * TILE_COUNT * TILE_COUNT) / 8;

    CUDA_CHECK((cudaMemcpy(context->all_in, images_in, (N_IMAGES * IMG_SIZE), cudaMemcpyHostToDevice)));

    bulk_process_image_kernel<<<N_IMAGES,threads_in_block>>>(context->all_in,context->all_out,context->maps,context->hist);

    CUDA_CHECK((cudaMemcpy(images_out, context->all_out, (N_IMAGES * IMG_SIZE), cudaMemcpyDeviceToHost)));
    
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    CUDA_CHECK(cudaFree((void**)(context->all_in)));
    CUDA_CHECK(cudaFree((void**)(context->all_out)));
    CUDA_CHECK(cudaFree((void**)(context->maps)));
    CUDA_CHECK(cudaFree((void**)(context->hist)));

    free(context);
}
