// NAME
//   cldevice.c
// VERSION
//    0.01
// SYNOPSIS
//   Source file for the library that manage OpenCL programs,
//   creating contexts and command queues for main plataform
//   used by the host and manage opencl source and binary files
// AUTHOR
//    Marcio Machado Pereira <mpereira@ic.unicamp.br>
// COPYLEFT
//   Copyleft (C) 2015 -- UNICAMP & Samsumg R&D

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cldevice.h"
#include <sys/stat.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 1
#define DIM_LOCAL_WORK_GROUP_Z 1

cl_device_id     *_device    = NULL;
cl_context       *_context   = NULL;
cl_command_queue *_cmd_queue = NULL;
cl_mem           *_locs      = NULL;
int               _upperid;
int               _curid;
cl_platform_id    _platform;
cl_program        _program;
cl_kernel         _kernel;
cl_uint           _npairs;
cl_uint           _clid;
cl_int            _status;
int               _spir_support;
int               _gpu_present;

//
// Initialize cldevice library
//
void _cldevice_init () {

  cl_uint nplatforms;
  cl_uint i;

  if (_device == NULL) {
    //Fetch the main Platform (the first one)
    _status = clGetPlatformIDs(1, &_platform, &nplatforms);
    if (_status != CL_SUCCESS || nplatforms <= 0) {
      fprintf(stderr, "<libmptogpu> Failed to find any OpenCL platform.\n");
      exit(1);
    }

    //Get the name of the plataform
    char platformName[100];
    memset(platformName, '\0', 100);
    _status = clGetPlatformInfo(_platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);

    //Check if the plataform supports spir
    char extension_string[1024];
    memset(extension_string, '\0', 1024);
    _status = clGetPlatformInfo(_platform, CL_PLATFORM_EXTENSIONS, sizeof(extension_string), extension_string, NULL);
    char* extStringStart = strstr(extension_string, "cl_khr_spir");
    if (extStringStart != 0) {
      _spir_support = 1;
    }
    else {
      //Platform does not support cl_khr_spir extension
      _spir_support = 0;
    }

    //Fetch the device list for this platform
    _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_ALL, 0, NULL, &_npairs);
    if (_status != CL_SUCCESS) {
      fprintf(stderr, "<libmptogpu> Failed to find any OpenCL device.\n");
      exit(1);
    }
  
    _device = (cl_device_id *) malloc(sizeof(cl_device_id)*_npairs);
    _context = (cl_context *) malloc(sizeof(cl_context)*_npairs);
    _cmd_queue = (cl_command_queue *) malloc(sizeof(cl_command_queue)*_npairs);
  
    _gpu_present = 0;
    for (i = 0; i < _npairs; i++) {
      _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, _npairs, &_device[i], NULL);
      _gpu_present = _gpu_present || (_status == CL_SUCCESS);
      _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_ALL, _npairs, &_device[i], NULL);
      //Create one OpenCL context for each device in the platform
      _context[i] = clCreateContext( NULL, _npairs, &_device[i], NULL, NULL, &_status);
      if (_status != CL_SUCCESS) {
	fprintf(stderr, "<libmptogpu> Failed to create an OpenCL GPU or CPU context.\n");
	exit(1);
      }
      //Create a command queue for each context to communicate with the device
      _cmd_queue[i] = clCreateCommandQueue(_context[i], _device[i], 0, &_status);
      if (_status != CL_SUCCESS) {
	fprintf(stderr, "<libmptogpu> Failed to create commandQueue for devices.\n");
	exit(1);
      }
    }
  }
  _clid = 0;      // initialize default device with 0 (CPU)
  _upperid = 16;  // max num of buffer memory locations
  _curid = -1;    // points to invalid location
  _locs = (cl_mem *) malloc(sizeof(cl_mem)*_upperid);
}

//
// Cleanup cldevice
//
void _cldevice_finish() {

  cl_uint i;
  
  // Clean up and wait for all the comands to complete
  for (i = 0; i< _npairs; i++) {
    _status = clFlush(_cmd_queue[i]);
    _status = clFinish(_cmd_queue[i]);
  }

  // Release OpenCL allocated objects
  _status = clReleaseKernel(_kernel);
  _status = clReleaseProgram(_program);
  for (i = 0; i < _npairs; i++) {
    _status = clReleaseCommandQueue(_cmd_queue[i]);
    _status = clReleaseContext(_context[i]);
  }
  free(_cmd_queue);
  free(_context);
  free(_device);
}

//
//  Create an OpenCL program from the kernel source file
//
cl_program _create_fromSource(cl_context context,
			      cl_device_id device,
			      const char* fileName) {
    cl_int errNum;
    cl_program program;

    FILE* file = fopen(fileName, "r");
    if (file == NULL) {
      fprintf(stderr, "<libmptogpu> Failed to open file for reading: %s\n", fileName);
      return NULL;
    }

    fseek(file, 0, SEEK_END);
    size_t fsize = ftell(file);
    rewind(file);

    char* buffer = (char*)malloc(sizeof(char)*(fsize+1));
    buffer[fsize] = '\0';
    fread(buffer, sizeof(char), fsize, file);
    fclose(file);

    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&buffer,
                                        NULL, NULL);
    if (program == NULL) {
      fprintf(stderr, "<libmptogpu> Failed to create CL program from source.\n");
      return NULL;
    }

    //errNum = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
      // Determine the reason for the error
      char buildLog[16384];
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			    sizeof(buildLog), buildLog, NULL);

      fprintf(stderr, "<libmptogpu> Error in kernel: %s\n", buildLog);
      clReleaseProgram(program);
      return NULL;
    }

    return program;
}

//
//  Attempt to create the program object from a cached binary.
//
cl_program _create_fromBinary(cl_context context,
			      cl_device_id device,
			      const char* fileName) {

  FILE *fp = fopen(fileName, "rb");
  if (fp == NULL) {
    return NULL;
  }

  // Determine the size of the binary
  size_t binarySize;
  fseek(fp, 0, SEEK_END);
  binarySize = ftell(fp);
  rewind(fp);

  unsigned char *programBinary = malloc(sizeof(unsigned char) * binarySize);
  fread(programBinary, 1, binarySize, fp);
  fclose(fp);

  cl_int errNum = 0;
  cl_program program;
  cl_int binaryStatus;

  program = clCreateProgramWithBinary(context,
				      1,
				      &device,
				      &binarySize,
				      (const unsigned char**)&programBinary,
				      &binaryStatus,
				      &errNum);

  free(programBinary);

  if (errNum != CL_SUCCESS) {
    fprintf(stderr, "<libmptogpu> Error loading program binary.\n");
    return NULL;
  }

  if (binaryStatus != CL_SUCCESS) {
    fprintf(stderr, "<libmptogpu> Invalid binary for device.\n");
    return NULL;
  }

  const char* flags = NULL;
  if (_spir_support) {
    flags = "-x spir";
  }
  errNum = clBuildProgram(program, 1, &device, flags, NULL, NULL);

  if (errNum != CL_SUCCESS) {
    // Determine the reason for the error
    char buildLog[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			  sizeof(buildLog), buildLog, NULL);

    if (!_spir_support) {
      fprintf(stderr, "<libmptogpu> %s: This platform does not support cl_khr_spir extension!\n", buildLog);
    }
    else {
      fprintf(stderr, "<libmptogpu> %s\n", buildLog);
    }
    clReleaseProgram(program);
    return NULL;
  }

  return program;
}

//
//  Retreive program binary for all of the devices attached to
//  the program and store the one for the device passed in
//
int _save_toBinary(cl_program program,
		   cl_device_id device,
		   const char* fileName)
{
    cl_uint numDevices = 0;
    cl_int errNum;

    // 1 - Query for number of devices attached to program
    errNum = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint),
                              &numDevices, NULL);
    if (errNum != CL_SUCCESS) {
      fprintf(stderr, "<libmptogpu> Error querying for number of devices.\n");
      return 0;
    }

    // 2 - Get all of the Device IDs
    cl_device_id *devices = malloc(sizeof(cl_device_id)*numDevices);
    errNum = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                              sizeof(cl_device_id) * numDevices,
                              devices, NULL);
    if (errNum != CL_SUCCESS) {
      fprintf(stderr, "<libmptogpu> Error querying for devices.\n");
      free(devices);
      return 0;
    }

    // 3 - Determine the size of each program binary
    size_t *programBinarySizes = malloc(sizeof(size_t)*numDevices);
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                              sizeof(size_t) * numDevices,
                              programBinarySizes, NULL);
    if (errNum != CL_SUCCESS) {
      fprintf(stderr, "<libmptogpu> Error querying for program binary sizes.\n");
      free(devices);
      free(programBinarySizes);
      return 0;
    }

    unsigned char **programBinaries = malloc(sizeof(unsigned char)*numDevices);

    cl_uint i;
    for (i = 0; i < numDevices; i++) {
      programBinaries[i] = malloc(sizeof(unsigned char)*programBinarySizes[i]);
    }

    // 4 - Get all of the program binaries
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
			      sizeof(unsigned char*) * numDevices,
                              programBinaries, NULL);
    if (errNum != CL_SUCCESS) {
      fprintf(stderr, "<libmptogpu> Error querying for program binaries.\n");
      free(devices);
      free(programBinarySizes);
      for (i = 0; i < numDevices; i++) {
	free(programBinaries[i]);
      }
      free(programBinaries);
      return 0;
    }

    // 5 - Finally store the binaries for the device requested
    // out to disk for future reading.
    for (i = 0; i < numDevices; i++) {
        // Store the binary just for the device requested.
        if (devices[i] == device) {
	  FILE *fp = fopen(fileName, "wb");
	  fwrite(programBinaries[i], 1, programBinarySizes[i], fp);
	  fclose(fp);
	  break;
        }
    }

    // Cleanup
    free(devices);
    free(programBinarySizes);
    for (i = 0; i < numDevices; i++) {
      free(programBinaries[i]);
    }
    free(programBinaries);
    return 1;
}

//
// Return the number of devices of the Main Plataform
//
cl_uint _get_num_devices () {
  return _npairs;
}

//
// Return the number of iterations in the loop
//
cl_uint _get_num_cores (int A, int B, int C, int T) {
  int N = abs(B-T-A+1);
  float K = (float)N/C;
  return (int)ceil(K);
}

//
// Returns the current device id
//
cl_uint _get_default_device () {
  return _clid;
}

//
// Set the device id
//
void _set_default_device (cl_uint id) {
  if ((id == 1 /* GPU */) && (!_gpu_present))
    _clid = 0; // force execution into CPU 
  else
    _clid = id;
}

//
// Create a write-only memory buffer on the selected device of a given size
//
int _cl_create_write_only (long size) {
  _curid++;
  if (_curid == _upperid) {
    // todo: we need to increment the number of buffer memory locations
    // for now, return false
    _curid--;
    return 0;
  }
  _locs[_curid] = clCreateBuffer(_context[_clid], CL_MEM_WRITE_ONLY,
				 size, NULL, &_status);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<libmptogpu> Failed to create a write-only buffer for the device.\n");
    _curid--;
    return 0;
  }
  return 1;
}

//
// Create a read-only memory buffer to offloading host locations
//
int _cl_create_read_only (long size) {
  _curid++;
  if (_curid == _upperid) {
    // todo: we need to increment the number of buffer memory locations
    // for now, return false
    _curid--;
    return 0;
  }
  _locs[_curid] = clCreateBuffer(_context[_clid], CL_MEM_READ_ONLY,
				 size, NULL, &_status);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<libmptogpu> Failed to create a read-only device buffer.\n");
    _curid--;
    return 0;
  }
  return 1;
}

//
// Create a read-only memory buffer and copy the host loc to the buffer
//
int _cl_offloading_read_only (long size, void* loc) {
  _curid++;
  if (_curid == _upperid) {
    // todo: we need to increment the number of buffer memory locations
    // for now, return false
    _curid--;
    return 0;
  }
  _locs[_curid] = clCreateBuffer(_context[_clid], CL_MEM_READ_ONLY,
				 size, NULL, &_status);
  _status = clEnqueueWriteBuffer(_cmd_queue[_clid], _locs[_curid], CL_TRUE,
  				 0, size, loc, 0, NULL, NULL);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<libmptogpu> Failed to write the host location to device buffer.\n");
    _curid--;
    return 0;
  }
  return 1;
}

//
// Create a read-write memory buffer
//
int _cl_create_read_write (long size) {
  _curid++;
  if (_curid == _upperid) {
    // todo: we need to increment the number of buffer memory locations
    // for now, return false
    _curid--;
    return 0;
  }
  _locs[_curid] = clCreateBuffer(_context[_clid], CL_MEM_READ_WRITE,
				 size, NULL, &_status);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<libmptogpu> Failed to create a read & write device buffer.\n");
    _curid--;
    return 0;
  }
  return 1;
}

//
// Create a read-write memory buffer and copy the host loc to the buffer
//
int _cl_offloading_read_write (long size, void* loc) {
  _curid++;
  if (_curid == _upperid) {
    // todo: we need to increment the number of buffer memory locations
    // for now, return false
    _curid--;
    return 0;
  }

  _locs[_curid] = clCreateBuffer(_context[_clid], CL_MEM_READ_WRITE,
				 size, NULL, &_status);
  _status = clEnqueueWriteBuffer(_cmd_queue[_clid], _locs[_curid], CL_TRUE,
  				 0, size, loc, 0, NULL, NULL);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<libmptogpu> Failed to write the host location to device buffer.\n");
    _curid--;
    return 0;
  }
  return 1;
}

//
// Read the cl_memory given by index on selected device to the host variable
//
int _cl_read_buffer (long size, int id, void* loc) {

  _status = clEnqueueReadBuffer(_cmd_queue[_clid], _locs[id],
             CL_TRUE, 0, size, loc, 0, NULL, NULL);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<libmptogpu> Failed to read to host location from the device buffer.\n");
    return 0;
  }
  return 1;
}

//
// Write (sync) the host variable to cl_memory given by index
//
int _cl_write_buffer (long size, int id, void* loc) {

  _status = clEnqueueWriteBuffer(_cmd_queue[_clid], _locs[id], CL_TRUE,
  				 0, size, loc, 0, NULL, NULL);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<libmptogpu> Failed to write the host location to the selected buffer.\n");
    return 0;
  }
  return 1;
}

//
// Auxiliary Function. Return true if file exist.
//
int _does_file_exist(const char *filename) {
    struct stat st;
    int result = stat(filename, &st);
    return result == 0;
}

//
// Create OpenCL program - first attempt to load cached binary.
// If that is not available, then create the program from source
// and store the binary for future use. Return 1, if success and
// 0, otherwise.
//
int _cl_create_program (char* str) {

  int fsize = strlen(str);
  char* cl_file = malloc(fsize + 4);
  char* bc_file = malloc(fsize + 4);
  strcpy(cl_file, str); strcat(cl_file, ".cl");
  strcpy(bc_file, str); strcat(bc_file, ".bc");

  if (_does_file_exist(bc_file)) {
    //Attempting to create program from binary
    _program = _create_fromBinary(_context[_clid],
				  _device[_clid],
				  bc_file);
    if (_program != NULL) return 1;
  }
  
  //Binary not loaded, create from source
  _program = _create_fromSource(_context[_clid],
				_device[_clid],
				cl_file);
  if (_program == NULL) {
    fprintf(stderr, "<libmptogpu> Attempting to create program failed.\n");
    return 0;
  }
  if (_save_toBinary(_program, _device[_clid], bc_file) == 0) {
    fprintf(stderr, "<libmptogpu> Failed to write program binary.\n");
    return 0;
  }
  return 1;
}

//
// Create OpenCL kernel. Return 1 (=true), if success
//
int _cl_create_kernel (char* str) {
  _kernel = clCreateKernel(_program, str, NULL);
  if (_kernel == NULL) {
    fprintf(stderr, "<libmptogpu> Failed to create kernel on device.\n");
    return 0;
  }
  return 1;
}

//
// Set the kernel arguments for cl_mem buffers
//
int _cl_set_kernel_args (int nargs) {
  _status = CL_SUCCESS;
  int i;
  for (i = 0; i<nargs; i++) {
    _status |= clSetKernelArg (_kernel, i, sizeof(cl_mem), &_locs[i]);
  }
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<libmptogpu> Error setting kernel buffers on device.\n");
    return 0;
  }
  return 1;
}

//
// Set the kernel arguments for host args
//
int _cl_set_kernel_hostArg (int pos, int size, void* loc) {
  _status = clSetKernelArg (_kernel, pos, size, loc);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<libmptogpu> Error setting host args on device.\n");
    return 0;
  }
  return 1;
}

//
// Enqueues a command to execute a kernel on a device.
//
int _cl_execute_kernel(long size1, long size2, long size3, int dim) {
  cl_uint i;
  cl_uint wd = dim;
  size_t *global_size;
  size_t *local_size;

  global_size = (size_t *)malloc(3*sizeof(size_t));
  global_size[0] = (size_t)ceil(((float)size1) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
  global_size[1] = (size_t)ceil(((float)size2) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;
  global_size[2] = (size_t)ceil(((float)size2) / ((float)DIM_LOCAL_WORK_GROUP_Z)) * DIM_LOCAL_WORK_GROUP_Z;

  local_size = (size_t *)malloc(3*sizeof(size_t));
  local_size[0] = DIM_LOCAL_WORK_GROUP_X;
  local_size[1] = DIM_LOCAL_WORK_GROUP_Y;
  local_size[2] = DIM_LOCAL_WORK_GROUP_Z;

  _status = clEnqueueNDRangeKernel(_cmd_queue[_clid],
				   _kernel,
				   wd,          // number of dimmensions
				   NULL,        // global_work_offset
				   global_size, // global_work_size
				   local_size,  // local_work_size
				   0,           // num_events_in_wait_list
				   NULL,        // event_wait_list
				   NULL         // event
				   );
  if (_status == CL_SUCCESS) {
    return 1;
  }
  else {
    if (_status == CL_INVALID_WORK_DIMENSION)
      fprintf(stderr, "<libmptogpu> Error executing kernel. Number of dimmensions is not a valid value.\n");
    else if (_status == CL_INVALID_GLOBAL_WORK_SIZE)
      fprintf(stderr, "<libmptogpu> Error executing kernel. Global work size is NULL or exceed valid range.\n");
    else if (_status == CL_INVALID_WORK_GROUP_SIZE)
      fprintf(stderr, "<libmptogpu> Error executing kernel. Local work size does not match the work-group size.\n");
    else
      fprintf(stderr, "<libmptogpu> Error executing kernel on device.\n");
  }
  return 0;
}

//
// Release all OpenCL allocated buffer inside the map region.
//
void _cl_release_buffers(int upper) {
  int i;
  for (i=0; i<upper; i++) {
    _status = clReleaseMemObject(_locs[i]);
    _locs[i] = NULL;
  }
  _curid = -1;
}

//
// Release an OpenCL allocated buffer inside the map region, given an index.
//
void _cl_release_buffer(int index) {
    _status = clReleaseMemObject(_locs[index]);
    _locs[index] = NULL;
  _curid--;
}
