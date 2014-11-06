#include <time.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#elif __linux
    #include <CL/cl.h>
#endif

const unsigned int SUCCESS = 0;

int show_info(cl_platform_id platform_id);
int load_file_to_memory(const char *filename, char **result);
unsigned int test_results(const float* const a,
                        const float* const b,
                        const float* const results);

int main(int argc, char** argv){
    // error code returned from api calls
    int err;                            
     
    size_t global[2];// global domain size for our calculation
    size_t local[2];// local domain size for our calculation

    cl_platform_id platform_id;         // platform id
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
   
    if (argc != 4){
        printf("%s <inputfile>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const unsigned int wgSize1D = atoi(argv[2]);
    const unsigned int wgSize2D = atoi(argv[3]);
    printf("Working Group size 1D[%u] 2D[%u] kernel[%s] \n", wgSize1D, 
                                                            wgSize2D, 
                                                            argv[1]);
    // Connect to first platform
    err = clGetPlatformIDs(1,&platform_id,NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to find an OpenCL platform!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    if(show_info(platform_id) != SUCCESS){
        printf("Error: Showing information!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    int fpga = 0;
#if defined (FPGA_DEVICE)
    fpga = 1;
#endif

    err = clGetDeviceIDs(platform_id, 
                        fpga? CL_DEVICE_TYPE_ACCELERATOR:/*CL_DEVICE_TYPE_GPU*//*CL_DEVICE_TYPE_CPU*/CL_DEVICE_TYPE_ACCELERATOR,
                        1, 
                        &device_id, 
                        NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to create a device group!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    cl_char string[10240] = {0};
    // Get device name
    err = clGetDeviceInfo(device_id, 
                        CL_DEVICE_NAME, 
                        sizeof(string), 
                        &string, 
                        NULL);
    if (err != CL_SUCCESS){   
        printf("Error: could not get device information\n");
        return EXIT_FAILURE;
    }   
    printf("Name of device: %s\n", string);
  
    // Create a compute context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context){
        printf("Error: Failed to create a compute context!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    commands = clCreateCommandQueue(context, 
                                    device_id, 
                                    CL_QUEUE_PROFILING_ENABLE, 
                                    &err);
    if (!commands){
        printf("Error: Failed to create a command commands!\n");
        printf("Error: code %i\n",err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

/****************LOADING KERNEL FROM BINARY OR SOURCE*****************/
    int status;
    if(fpga){
        unsigned char *kernelbinary;
        char *xclbin=argv[1];
        printf("loading binary [%s]\n", xclbin);
        int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
        if (n_i < 0) {
            printf("failed to load kernel from xclbin: %s\n", xclbin);
            printf("Test failed\n");
            return EXIT_FAILURE;
        }
        size_t n = n_i;
        program = clCreateProgramWithBinary(context, 
                                1, 
                                &device_id, 
                                &n,
                                (const unsigned char **) &kernelbinary, 
                                &status, 
                                &err);
    }
    else{
        unsigned char *kernelsrc;
        char *clsrc = argv[1];
        printf("loading source [%s]\n", clsrc);
        int n_i = load_file_to_memory(clsrc, (char **) &kernelsrc);
        if (n_i < 0) {
            printf("failed to load kernel from source: %s\n", clsrc);
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

        program = clCreateProgramWithSource(context, 
                                    1, 
                                    (const char **) &kernelsrc, 
                                    NULL, 
                                    &err);
    }


    if((!program) || (err!=CL_SUCCESS)){
        printf("Error: Failed to create compute program from binary %d!\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
/****************************************************************/

  // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err != CL_SUCCESS){
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, 
                            device_id, 
                            CL_PROGRAM_BUILD_LOG, 
                            sizeof(buffer), 
                            buffer, 
                            &len);
        printf("%s\n", buffer);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "hash", &err);
    if(!kernel || err != CL_SUCCESS){
        printf("Error: Failed to create compute kernel!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    global[0] = 4;
    global[1] = 4;
    local[0] = wgSize1D;
    local[1] = wgSize2D;

    err = clEnqueueNDRangeKernel(commands, 
                                kernel, 
                                2, 
                                NULL, 
                                (size_t*)&global, 
                                (size_t*)&local, 
                                0, 
                                NULL, 
                                NULL);
    if (err){
        printf("Error: Failed to execute kernel! %d\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Shutdown and cleanup
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

int show_info(cl_platform_id platform_id){
    int err;
    char cl_platform_vendor[1001];
    char cl_platform_name[1001];
    char cl_platform_version[1001];

    err = clGetPlatformInfo(platform_id,CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
    if (err != CL_SUCCESS){
        printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
  
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
    if (err != CL_SUCCESS){
        printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("CL_PLATFORM_NAME %s\n",cl_platform_name);
    
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_VERSION,1000,(void *)cl_platform_version,NULL);
    if (err != CL_SUCCESS){
        printf("Error: clGetPlatformInfo(CL_PLATFORM_VERSION) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("CL_PLATFORM_VERSION %s\n",cl_platform_version);

    return SUCCESS;
}

int load_file_to_memory(const char *filename, char **result)
{ 
  int size = 0;
  FILE *f = fopen(filename, "rb");
  if (f == NULL) 
  { 
    *result = NULL;
    return -1; // -1 means file opening fail 
  } 
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  fseek(f, 0, SEEK_SET);
  *result = (char *)malloc(size+1);
  if (size != fread(*result, sizeof(char), size, f)) 
  { 
    free(*result);
    return -2; // -2 means file reading fail 
  } 
  fclose(f);
  (*result)[size] = 0;
  return size;
}

