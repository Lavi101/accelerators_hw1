#include "ex1.h"

#define HIST_SIZE 256
#define IMG_SIZE (IMG_HEIGHT*IMG_WIDTH)

// Make the given histogram a CDF array
__device__ void prefix_sum(int arr[], int arr_size) { //TODO: make sure we understood correctly
    int sum = 0;
    for(int i=0;i<=arr_size;i++) {
        sum += arr[i];
        arr[i] = sum;
    }
    return;
}

// Make the given CDF array a map, using the given definition
__device__ void map_calculation(int arr[], int arr_size) { //TODO: make sure it is possible to add our own functions
    for(int i=0;i<=arr_size;i++) {
        int val = (arr[i] / (TILE_WIDTH * TILE_WIDTH) * 255);
        arr[i] = val;
    }
    return;
}

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

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps,int *hist) { //TODO:
    int tid=threadIdx.x;
    //int hist[TILE_COUNT][TILE_COUNT][HIST_SIZE];

    // Each thread would calculate the histogram contribution
    // of a single row of length T in a specific tile.

    // Calculate offset of this thread in the image pixels array
    int tileID = tid / TILE_WIDTH;
    int rowInTile = tid % TILE_WIDTH;

    int numberOfTilesInRowImg = IMG_WIDTH / TILE_WIDTH;
    int gridRow = tileID / numberOfTilesInRowImg;
    int gridCol = tileID % numberOfTilesInRowImg;

    int offset_in_img = IMG_WIDTH * TILE_WIDTH * gridRow +
     IMG_WIDTH * rowInTile + TILE_WIDTH * gridCol;

    // Pass on tile row and update histogram
    // We'll use maps as a histogram in the meanwhile
    for(int i=0;i<TILE_WIDTH;i++) {
        int pixelValue = all_in[offset_in_img+i];
        atomicAdd(((hist) + gridRow * numberOfTilesInRowImg * HIST_SIZE + gridCol * HIST_SIZE + pixelValue),1);
    }

    __syncthreads();

    // Make the histogram into a map by only the first thread in every tile
    // TODO: make sure. cause it won't be parallelized
    if(rowInTile == 0) {
        // Now make the histogram a CDF, by running prefix_sum
        prefix_sum(((hist) + gridRow * numberOfTilesInRowImg * HIST_SIZE + gridCol * HIST_SIZE),HIST_SIZE);

        // Perform map calculation for each tile
        map_calculation(((hist) + gridRow * numberOfTilesInRowImg * HIST_SIZE + gridCol * HIST_SIZE),HIST_SIZE);
    }

    // Now we copy hist values to maps
    for (int i=0;i<TILE_COUNT;i++) {
        for (int j=0;j<TILE_COUNT;j++) {
            for (int n=0;n<HIST_SIZE;n++) {
                    maps[i * TILE_COUNT * HIST_SIZE + j * HIST_SIZE + n] = (unsigned char)(hist[i * TILE_COUNT * HIST_SIZE + j * HIST_SIZE + n]);
            }
        }
    }

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

    cudaMalloc((void**)(context->all_in), IMG_WIDTH * IMG_WIDTH);
    CUDA_CHECK(cudaMalloc((void**)(context->all_out), IMG_WIDTH * IMG_WIDTH));
    CUDA_CHECK(cudaMalloc((void**)(context->maps), TILE_COUNT * TILE_COUNT * HIST_SIZE));
    CUDA_CHECK(cudaMalloc((void**)(context->hist), TILE_COUNT * TILE_COUNT * HIST_SIZE * sizeof(int)));



    return context;
}

// Zeros the maps array of a given context
void reset_maps_array(struct task_serial_context *context) { //TODO: is needed?
    for (int i=0;i<TILE_COUNT;i++) {
        for (int j=0;j<TILE_COUNT;j++) {
            for (int n=0;n<HIST_SIZE;n++) {
                context->maps[i * TILE_COUNT * HIST_SIZE + j * HIST_SIZE + n] = 0;
            }
        }
    }
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
    int threads_in_block = TILE_WIDTH * TILE_COUNT * TILE_COUNT;


    for (int i=0;i<N_IMAGES;i++) {
        reset_maps_array(context);
        CUDA_CHECK((cudaMemcpy(context->all_in, (images_in + i * IMG_SIZE), IMG_SIZE, cudaMemcpyHostToDevice)));

        process_image_kernel<<<1,threads_in_block>>>(context->all_in,context->all_out,context->maps,context->hist);

        CUDA_CHECK((cudaMemcpy(context->all_out, (images_out + i * IMG_SIZE), IMG_SIZE, cudaMemcpyDeviceToHost)));
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

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all the input images, output images, and maps

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init

    free(context);
}
