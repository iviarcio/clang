// NAME
//   cldevice.c
// VERSION
//    0.01
// SYNOPSIS
//   Source file for the library that manage OpenCL programs,
//   creating contexts and command queues for main plataform
//   used by the host and manage opencl source and binary files
// AUTHOR
//    Marcio Machado Pereira
// COPYLEFT
//   Copyleft (C) 2015 -- UNICAMP & Samsumg R&D

#include <stdio.h>
#include "cldevice.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

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
      perror("Failed to find any OpenCL platform");
      exit(1);
    }

    //Fetch the device list for this platform
    _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_ALL, 0, NULL, &_npairs);
    if (_status != CL_SUCCESS) {
      perror("Failed to find any OpenCL device");
      exit(1);
    }
  
    _device = (cl_device_id *) malloc(sizeof(cl_device_id)*_npairs);
    _context = (cl_context *) malloc(sizeof(cl_context)*_npairs);
    _cmd_queue = (cl_command_queue *) malloc(sizeof(cl_command_queue)*_npairs);
  
    for (i = 0; i < _npairs; i++) {
      _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_ALL,
			       _npairs, &_device[i], NULL);
      //Create one OpenCL context for each device in the platform
      _context[i] = clCreateContext( NULL, _npairs, &_device[i], NULL, NULL, &_status);
      if (_status != CL_SUCCESS) {
	perror("Failed to create an OpenCL GPU or CPU context.");
	exit(1);
      }
      //Create a command queue for each context to communicate with the device
      _cmd_queue[i] = clCreateCommandQueue(_context[i], _device[i], 0, &_status);
      if (_status != CL_SUCCESS) {
	perror("Failed to create commandQueue for devices");
	exit(1);
      }
    }
  }
  _clid = 0;      // initialize default device with 0 (CPU?)
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
      perror("Failed to open file for reading: ");
      perror(fileName);
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
      perror("Failed to create CL program from source.");
      return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
      // Determine the reason for the error
      char buildLog[16384];
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			    sizeof(buildLog), buildLog, NULL);

      perror("Error in kernel: ");
      perror(buildLog);
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
    perror("Error loading program binary.");
    return NULL;
  }

  if (binaryStatus != CL_SUCCESS) {
    perror("Invalid binary for device");
    return NULL;
  }

  errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    // Determine the reason for the error
    char buildLog[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			  sizeof(buildLog), buildLog, NULL);

    perror("Error in program: ");
    perror(buildLog);
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
      perror("Error querying for number of devices.");
      return 0;
    }

    // 2 - Get all of the Device IDs
    cl_device_id *devices = malloc(sizeof(cl_device_id)*numDevices);
    errNum = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                              sizeof(cl_device_id) * numDevices,
                              devices, NULL);
    if (errNum != CL_SUCCESS) {
      perror("Error querying for devices.");
      free(devices);
      return 0;
    }

    // 3 - Determine the size of each program binary
    size_t *programBinarySizes = malloc(sizeof(size_t)*numDevices);
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                              sizeof(size_t) * numDevices,
                              programBinarySizes, NULL);
    if (errNum != CL_SUCCESS) {
      perror("Error querying for program binary sizes.");
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
      perror("Error querying for program binaries");
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
// Returns the default device id
//
cl_uint _get_default_device () {
  return _clid;
}

//
// Set the default device id
//
void _set_default_device (cl_uint id) {
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
    perror("Failed to create a write-only buffer for the selected device");
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
    perror("Failed to create a read-only device buffer");
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
    perror("Failed to write the host location to device buffer");
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
    perror("Failed to create a read & write device buffer");
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
    perror("Failed to write the host location to device buffer");
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
    perror("Failed to read to host location from the device buffer");
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
    perror("Failed to write the host location to the selected buffer");
    return 0;
  }
  return 1;
}


//
// Create OpenCL Program from generated cached kernel function.
// The other alternative is to create from kernel source code.
// To do this, just call the function _create_fromSource
// Return true, if success and false, otherwise.
//
int _cl_create_program (char* str) {
  //  _program = _create_fromBinary(_context[_clid],
  //				_device[_clid],
  //				str);
  _program = _create_fromSource(_context[_clid],
				_device[_clid],
				str);
  if (_program == NULL) {
    perror("Attempting to create program kernel failed");
    return 0;
  }
  return 1;
}

//
// Create OpenCL kernel. Return true, if success
//
int _cl_create_kernel (char* str) {
  _kernel = clCreateKernel(_program, str, NULL);
  if (_kernel == NULL) {
    perror("Failed to create kernel on the selected device.");
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
    perror("Error setting kernel buffers on selected device.");
    return 0;
  }
  return 1;
}

//
// Set the kernel arguments for host args
//
int _cl_set_kernel_hostArg (int pos, int size, void* loc) {
  _status = CL_SUCCESS;
  int i;
    _status = clSetKernelArg (_kernel, pos, size, loc);
  if (_status != CL_SUCCESS) {
    perror("Error setting host args on selected device.");
    return 0;
  }
  return 1;
}

//
// Enqueues a command to execute a kernel on a device.
//
int _cl_execute_kernel(long work_size) {

  cl_uint wd = 1; // number of dimmensions. assume equals 1
  size_t global_size = (size_t) work_size;
  size_t local_size  = 1;  // Fix-me

  _status = clEnqueueNDRangeKernel(_cmd_queue[_clid],
				   _kernel,
				   wd, // work_dim
				   NULL, // global_work_offset
				   &global_size, // global_work_size
				   &local_size, // local_work_size
				   0, // num_events_in_wait_list
				   NULL, // event_wait_list
				   NULL // event
				   );
  if (_status == CL_SUCCESS) {
    return 1;
  }
  else {
    if (_status == CL_INVALID_WORK_DIMENSION)
      perror("Error exec kernel. Work_dim is not a valid value");
    else if (_status == CL_INVALID_GLOBAL_WORK_SIZE)
      perror("Error exec kernel. Global_work_size is NULL or exceed valid range.");
    else if (_status == CL_INVALID_WORK_GROUP_SIZE)
      perror("Error exec kernel. Local_work_size does not match the work-group size.");
    else
      perror("Error executing kernel.");
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
