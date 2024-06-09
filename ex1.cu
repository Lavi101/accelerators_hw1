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
        int val = (arr[i] / (TILE_WIDTH * TILE_WIDTH)) * 255);
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

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {
    int tid=threadIdx.x

    // Each thread would calculate the histogram contribution
    // of a single row of length T in a specific tile.

    // Calculate offset of this thread in the image pixels array
    int tileID = tid / TILE_WIDTH;
    int rowInTile = tid % TILE_WIDTH;

    int numberOfTilesInRowImg = IMAGE_WIDTH / TILE_WIDTH;
    int gridRow = tileID / numberOfTilesInRowImg;
    int gridCol = tileID % numberOfTilesInRowImg;

    int offset_in_img = IMAGE_WIDTH * TILE_WIDTH * gridRow +
     IMAGE_WIDTH * rowInTile + TILE_WIDTH * gridCol;

    // Pass on tile row and update histogram
    // We'll use maps as a histogram in the meanwhile
    for(int i=0;i<TILE_WIDTH;i++) {
        int pixelValue = all_in[offset_in_img+i];
        atomicAdd(&(maps[gridRow][gridCol][pixel_value]),1);
    }

    __syncthreads();

    // Make the histogram into a map by only the first thread in every tile
    // TODO: make sure. cause it won't be parallelized
    if(rowInTile == 0) {
        // Now make the histogram a CDF, by running prefix_sum
        prefix_sum(&(maps[gridRow][gridCol]),HIST_SIZE);

        // Perform map calculation for each tile
        map_calculation(&(maps[gridRow][gridCol]),HIST_SIZE);
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
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    f=cudaMalloc((void**)&(context.all_in), IMAGE_WIDTH * IMAGE_WIDTH);
    CUDA_CHECK(f)
    f=cudaMalloc((void**)&(context.all_out), IMAGE_WIDTH * IMAGE_WIDTH);
    CUDA_CHECK(f)
    f=cudaMalloc((void**)&(context.maps), TILES_COUNT * TILES_COUNT * HIST_SIZE);
    CUDA_CHECK(f)


    return context;
}

// Zeros the maps array of a given context
void reset_maps_array(struct task_serial_context *context) { //TODO: is needed?
    for (int i=0;i<TILES_COUNT;i++) {
        for (int j=0;j<TILES_COUNT;j++) {
            for (int n=0;n<HIST_SIZE;n++) {
                (*context).maps[i][j][n] = 0;
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
    int threads_in_block = TILE_WIDTH * TILES_COUNT * TILES_COUNT;


    for (int i=0;i<NUM_IMAGES;i++) {
        reset_maps_array(context);
        f = cudaMemcpy((*context).all_in, (images_in + i * IMG_SIZE), IMG_SIZE, cudaMemcpyHostToDevice);
        CUDA_CHECK(f);

        f = process_image_kernel<<<1,threads_in_block>>>((*context).all_in,(*context).all_out,(*context).maps);
        CUDA_CHECK(f);

        f = cudaMemcpy((*context).all_out, (images_out + i * IMG_SIZE), IMG_SIZE, cudaMemcpyDeviceToHost);
        CUDA_CHECK(f);
    }

}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    f=cudaFree((void**)&(context.all_in));
    CUDA_CHECK(f);
    f=cudaFree((void**)&(context.all_out));
    CUDA_CHECK(f);
    f=cudaFree((void**)&(context.maps));
    CUDA_CHECK(f);

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
