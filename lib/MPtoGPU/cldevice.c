// NAME
//   cldevice.c
// VERSION
//    2.1
// SYNOPSIS
//   Source file for the library that manage OpenCL programs,
//   creating contexts and command queues for main plataform
//   used by the host and manage opencl source and binary files
// AUTHOR
//    Marcio Machado Pereira <mpereira@ic.unicamp.br>
// COPYLEFT
//   Copyleft (C) 2015--2016, UNICAMP & Samsumg R&D

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cldevice.h"
#include <sys/stat.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#elif __ALTERA__
#include <CL/opencl.h>
#else
#include <CL/cl.h>
#endif

cl_device_id     *_device    = NULL;
cl_context       *_context   = NULL;
cl_command_queue *_cmd_queue = NULL;
cl_mem           *_locs      = NULL;

cl_platform_id    _platform;
cl_program       *_program;
cl_kernel        *_kernel;
cl_uint           _ndevices;
cl_uint           _clid;
cl_int            _status;

cl_uint           _kerid;
cl_uint           _nkernels;
cl_uint           _sentinel;
char            **_strprog;

int               _spir_support;
int               _gpu_present;
int               _cpu_present;
int               _upperid;
int               _curid;
int               _verbose;
int               _profile;
int               _work_group[9] = {128, 1, 1, 256, 1, 1, 32, 8, 1};
int               _block_sizes[11] = { 2 , 4 , 8 , 16 , 32 , 64 , 128, 256 , 512 , 1024, 2048 };

cl_event          _global_event;

enum RtlModeOptions { RTL_none, RTL_verbose, RTL_profile, RTL_all };

void _clErrorCode(cl_int status) {
  if (_verbose || _profile) {
    char* code;
    switch (status) {
    case CL_DEVICE_NOT_FOUND:                         code = "CL_DEVICE_NOT_FOUND"; break;
    case CL_DEVICE_NOT_AVAILABLE:                     code = "CL_DEVICE_NOT_AVAILABLE"; break;
    case CL_COMPILER_NOT_AVAILABLE:                   code = "CL_COMPILER_NOT_AVAILABLE"; break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:            code = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
    case CL_OUT_OF_RESOURCES:                         code = "CL_OUT_OF_RESOURCES"; break;
    case CL_OUT_OF_HOST_MEMORY:                       code = "CL_OUT_OF_HOST_MEMORY"; break;
    case CL_PROFILING_INFO_NOT_AVAILABLE:             code = "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
    case CL_MEM_COPY_OVERLAP:                         code = "CL_MEM_COPY_OVERLAP"; break;
    case CL_IMAGE_FORMAT_MISMATCH:                    code = "CL_IMAGE_FORMAT_MISMATCH"; break;
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:               code = "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
    case CL_BUILD_PROGRAM_FAILURE:                    code = "CL_BUILD_PROGRAM_FAILURE"; break;
    case CL_MAP_FAILURE:                              code = "CL_MAP_FAILURE"; break;
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:             code = "CL_MISALIGNED_SUB_BUFFER_OFFSET"; break;
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:code = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;  
    case CL_INVALID_VALUE:                            code = "CL_INVALID_VALUE"; break;
    case CL_INVALID_DEVICE_TYPE:                      code = "CL_INVALID_DEVICE_TYPE"; break;
    case CL_INVALID_PLATFORM:                         code = "CL_INVALID_PLATFORM"; break;
    case CL_INVALID_DEVICE:                           code = "CL_INVALID_DEVICE"; break;
    case CL_INVALID_CONTEXT:                          code = "CL_INVALID_CONTEXT"; break;
    case CL_INVALID_QUEUE_PROPERTIES:                 code = "CL_INVALID_QUEUE_PROPERTIES"; break;
    case CL_INVALID_COMMAND_QUEUE:                    code = "CL_INVALID_COMMAND_QUEUE"; break;
    case CL_INVALID_HOST_PTR:                         code = "CL_INVALID_HOST_PTR"; break;
    case CL_INVALID_MEM_OBJECT:                       code = "CL_INVALID_MEM_OBJECT"; break;
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:          code = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; break;
    case CL_INVALID_IMAGE_SIZE:                       code = "CL_INVALID_IMAGE_SIZE"; break;
    case CL_INVALID_SAMPLER:                          code = "CL_INVALID_SAMPLER"; break;
    case CL_INVALID_BINARY:                           code = "CL_INVALID_BINARY"; break;
    case CL_INVALID_BUILD_OPTIONS:                    code = "CL_INVALID_BUILD_OPTIONS"; break;
    case CL_INVALID_PROGRAM:                          code = "CL_INVALID_PROGRAM"; break;
    case CL_INVALID_PROGRAM_EXECUTABLE:               code = "CL_INVALID_PROGRAM_EXECUTABLE"; break;
    case CL_INVALID_KERNEL_NAME:                      code = "CL_INVALID_KERNEL_NAME"; break;
    case CL_INVALID_KERNEL_DEFINITION:                code = "CL_INVALID_KERNEL_DEFINITION"; break;
    case CL_INVALID_KERNEL:                           code = "CL_INVALID_KERNEL"; break;
    case CL_INVALID_ARG_INDEX:                        code = "CL_INVALID_ARG_INDEX"; break;
    case CL_INVALID_ARG_VALUE:                        code = "CL_INVALID_ARG_VALUE"; break;
    case CL_INVALID_ARG_SIZE:                         code = "CL_INVALID_ARG_SIZE"; break;
    case CL_INVALID_KERNEL_ARGS:                      code = "CL_INVALID_KERNEL_ARGS"; break;
    case CL_INVALID_WORK_DIMENSION:                   code = "CL_INVALID_WORK_DIMENSION"; break;
    case CL_INVALID_WORK_GROUP_SIZE:                  code = "CL_INVALID_WORK_GROUP_SIZE"; break;
    case CL_INVALID_WORK_ITEM_SIZE:                   code = "CL_INVALID_WORK_ITEM_SIZE"; break;
    case CL_INVALID_GLOBAL_OFFSET:                    code = "CL_INVALID_GLOBAL_OFFSET"; break;
    case CL_INVALID_EVENT_WAIT_LIST:                  code = "CL_INVALID_EVENT_WAIT_LIST"; break;
    case CL_INVALID_EVENT:                            code = "CL_INVALID_EVENT"; break;
    case CL_INVALID_OPERATION:                        code = "CL_INVALID_OPERATION"; break;
    case CL_INVALID_GL_OBJECT:                        code = "CL_INVALID_GL_OBJECT"; break;
    case CL_INVALID_BUFFER_SIZE:                      code = "CL_INVALID_BUFFER_SIZE"; break;
    case CL_INVALID_MIP_LEVEL:                        code = "CL_INVALID_MIP_LEVEL"; break;
    case CL_INVALID_GLOBAL_WORK_SIZE:                 code = "CL_INVALID_GLOBAL_WORK_SIZE"; break;
    case CL_INVALID_PROPERTY:                         code = "CL_INVALID_PROPERTY";      
    default:                                          code = "CL_UNKNOWN_ERROR_CODE";      
    }
    fprintf(stderr, "<rtl> Error Code: %s\n", code);
  }
}
		     

void _cldevice_details(cl_device_id   id,
                       cl_device_info param_name,
                       const char*    param_str) {
  cl_uint i;
  cl_int  status = 0;
  size_t  param_size = 0;

  status = clGetDeviceInfo( id, param_name, 0, NULL, &param_size );
  if (status != CL_SUCCESS ) {
    fprintf(stderr, "<rtl> Unable to obtain device info for %s.\n", param_str);
    _clErrorCode (status);
    return;
  }

  /* the cl_device_info are preprocessor directives defined in cl.h */
  switch (param_name) {
    case CL_DEVICE_TYPE: {
      cl_device_type* devType = (cl_device_type*) alloca(sizeof(cl_device_type) * param_size);
      status = clGetDeviceInfo( id, param_name, param_size, devType, NULL );
      if (status != CL_SUCCESS ) {
        fprintf(stderr, "<rtl> Unable to obtain device info for %s.\n", param_str);
	_clErrorCode (status);
        return;
      }
      switch (*devType) {
        case CL_DEVICE_TYPE_CPU : printf("\tDevice is CPU\n"); break;
        case CL_DEVICE_TYPE_GPU : printf("\tDevice is GPU\n"); break;
        case CL_DEVICE_TYPE_ACCELERATOR : printf("\tDevice is Accelerator\n"); break;
        case CL_DEVICE_TYPE_DEFAULT : printf("\tDevice is Unknown\n"); break;
      }
    } break;
    case CL_DEVICE_VENDOR_ID :
    case CL_DEVICE_MAX_COMPUTE_UNITS :
    case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : {
      cl_uint* ret = (cl_uint*) alloca(sizeof(cl_uint) * param_size);
      status = clGetDeviceInfo( id, param_name, param_size, ret, NULL );
      if (status != CL_SUCCESS ) {
        fprintf(stderr, "<rtl> Unable to obtain device info for %s.\n", param_str);
	_clErrorCode (status);
        return;
      }
      switch (param_name) {
        case CL_DEVICE_VENDOR_ID:
          printf("\tVendor ID: 0x%x\n", *ret); break;
        case CL_DEVICE_MAX_COMPUTE_UNITS:
          printf("\tMaximum number of parallel compute units: %u\n", *ret); break;
        case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
          printf("\tMaximum dimensions for global/local work-item IDs: %u\n", *ret); break;
      }
    } break;
    case CL_DEVICE_MAX_WORK_ITEM_SIZES : {
      cl_uint maxWIDimensions;
      size_t* ret = (size_t*) alloca(sizeof(size_t) * param_size);
      status = clGetDeviceInfo( id, param_name, param_size, ret, NULL );

      status = clGetDeviceInfo( id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWIDimensions, NULL );
      if (status != CL_SUCCESS ) {
        fprintf(stderr, "<rtl> Unable to obtain device info for %s.\n", param_str);
	_clErrorCode (status);
        return;
      }
      printf("\tMaximum number of work-items in each dimension: ( ");
      for(i =0; i < maxWIDimensions; ++i ) {
        printf("%zu ", ret[i]);
      }
      printf(" )\n");
    } break;
    case CL_DEVICE_MAX_WORK_GROUP_SIZE : {
      size_t* ret = (size_t*) alloca(sizeof(size_t) * param_size);
      status = clGetDeviceInfo( id, param_name, param_size, ret, NULL );
      if (status != CL_SUCCESS ) {
        fprintf(stderr, "<rtl> Unable to obtain device info for %s.\n", param_str);
	_clErrorCode (status);
        return;
      }
      printf("\tMaximum number of work-items in a work-group: %zu\n", *ret);
    } break;
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: {
      cl_uint* preferred_size = (cl_uint*) alloca(sizeof(cl_uint) * param_size);
      status = clGetDeviceInfo( id, param_name, param_size, preferred_size, NULL );
      if (status != CL_SUCCESS ) {
        fprintf(stderr, "<rtl> Unable to obtain device info for %s.\n", param_str);
	_clErrorCode (status);
        return;
      }
      printf("\tPreferred vector width size for float: %d\n", (*preferred_size));
    } break;
    case CL_DEVICE_NAME :
    case CL_DEVICE_VENDOR : {
      char data[48];
      status = clGetDeviceInfo( id, param_name, param_size, data, NULL );
      if (status != CL_SUCCESS ) {
        fprintf(stderr, "<rtl> Unable to obtain device info for %s.\n", param_str);
	_clErrorCode (status);
        return;
      }
      switch (param_name) {
        case CL_DEVICE_NAME :
          printf("\tDevice name is %s\n", data); break;
        case CL_DEVICE_VENDOR :
          printf("\tDevice vendor is %s\n", data); break;
      }
    } break;
    case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: {
      cl_uint* size = (cl_uint*) alloca(sizeof(cl_uint) * param_size);
      status = clGetDeviceInfo( id, param_name, param_size, size, NULL );
      if (status != CL_SUCCESS ) {
        fprintf(stderr, "<rtl> Unable to obtain device info for %s.\n", param_str);
	_clErrorCode (status);
        return;
      }
      printf("\tDevice global cacheline size: %d bytes\n", (*size)); break;
    } break;
    case CL_DEVICE_GLOBAL_MEM_SIZE:
    case CL_DEVICE_MAX_MEM_ALLOC_SIZE: {
      cl_ulong* size = (cl_ulong*) alloca(sizeof(cl_ulong) * param_size);
      status = clGetDeviceInfo( id, param_name, param_size, size, NULL );
      if (status != CL_SUCCESS ) {
        fprintf(stderr, "<rtl> Unable to obtain device info for %s.\n", param_str);
	_clErrorCode (status);
        return;
      }
      switch (param_name) {
        case CL_DEVICE_GLOBAL_MEM_SIZE:
          printf("\tDevice global mem: %llu mega-bytes\n", (*size)>>20); break;
        case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
          printf("\tDevice max memory allocation: %llu mega-bytes\n", (*size)>>20); break;
      }
    } break;
  }
}

//
// Initialize cldevice
//
void _cldevice_init (int rtlmode) {

  cl_uint nplatforms;
  cl_uint ndev;
  cl_uint i;
  cl_uint idx;

  _verbose = (rtlmode == RTL_verbose) || (rtlmode == RTL_all);
  _profile = (rtlmode == RTL_profile) || (rtlmode == RTL_all);

  if (_device == NULL) {
    //Fetch the main Platform (the first one)
    _status = clGetPlatformIDs(1, &_platform, &nplatforms);
    if (_status != CL_SUCCESS || nplatforms <= 0) {
      fprintf(stderr, "<rtl> Failed to find any OpenCL platform.\n");
      _clErrorCode (_status);
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
      if (_verbose) printf("<rtl> This platform does not support cl_khr_spir extension.\n");
      _spir_support = 0;
    }

    //Fetch the device list for this platform
    _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_ALL, 0, NULL, &_ndevices);
    if (_status != CL_SUCCESS) {
      fprintf(stderr, "<rtl> Failed to find any OpenCL device.\n");
      _clErrorCode (_status);
      exit(1);
    }

    if (_verbose) printf("<rtl> Find %u devices on platform.\n", _ndevices);

    idx = 0;
    //Fetch the CPU device list for this platform. Note that the allocation
    //of handlers is done after test because device 0 is reserved to CPU
    _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_CPU, 0, NULL, &ndev);
    if (_status != CL_SUCCESS) {
      _ndevices +=1;
      _cpu_present = 0;
    }

    _device       = (cl_device_id *) calloc(_ndevices, sizeof(cl_device_id));
    _context      = (cl_context *) calloc(_ndevices, sizeof(cl_context));
    _cmd_queue    = (cl_command_queue *) calloc(_ndevices, sizeof(cl_command_queue));
    _gpu_present  = 0;

    if (_status == CL_SUCCESS) {
      if (_verbose) {
        printf("<rtl> Find %u CPU device(s).", ndev);
        if (ndev > 1)
          printf(" Only the main CPU was handled on device id %d.\n", idx);
        else
          printf(" The CPU was handled on device id %d.\n", idx);
      }
      _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_CPU, 1, &_device[idx], NULL);
      if (_status != CL_SUCCESS) {
        fprintf(stderr, "<rtl> Failed to create CPU device id .\n");
        _clErrorCode (_status);
      }
      else {
        _cpu_present = 1;
      }
    }

    idx += 1;
    if (_ndevices > idx) {
      //Try to fetch the GPU device list for this platform
      _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, 0, NULL, &ndev);
      if (_status == CL_SUCCESS) {
        if (_verbose) {
          printf("<rtl> Find %u GPU device(s). ", ndev);
          printf("GPU(s) was handled on device(s) id(s) starting with %d.\n", idx);
        }
        _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, ndev, &_device[idx], NULL);
        if (_status != CL_SUCCESS) {
          fprintf(stderr, "<rtl> Failed to create GPU device id(s) .\n");
	  _clErrorCode (_status);
        }
        else {
          _gpu_present = 1;
        }
      }
    }

    idx += ndev;
    if (_ndevices > idx) {
      //Fetch all accelerator devices for this platform
      _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &ndev);
      if (_status == CL_SUCCESS) {
        if (_verbose) {
          printf("<rtl> Find %u ACC device(s). ", ndev);
          printf("Each accelerator device was handled on device id starting with %d.\n", idx);
        }
        _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_ACCELERATOR, ndev, &_device[idx], NULL);
        if (_status != CL_SUCCESS) {
          fprintf(stderr, "<rtl> Failed to create any accelerator device id .\n");
	  _clErrorCode (_status);
        }
      }
    }

    idx += ndev;
    if (_ndevices > idx) {
      //Fetch all default devices for this platform
      _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_DEFAULT, 0, NULL, &ndev);
      if (_status == CL_SUCCESS) {
        if (_verbose) {
          printf("<rtl> Find %u unknown device(s). ", ndev);
          printf("Each unknown device was handled on device id starting with %d.\n", idx);
        }
        _status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_DEFAULT, ndev, &_device[idx], NULL);
        if (_status != CL_SUCCESS) {
          fprintf(stderr, "<rtl> Failed to create any unknown device id .\n");
	  _clErrorCode (_status);
        }
      }
    }

    for(i = 0; i < _ndevices; ++ i ) {

      if (_verbose && _device[i] != NULL) {
        printf("<rtl> Retrieve some information about device %u:\n", i);
        _cldevice_details( _device[i], CL_DEVICE_TYPE, "CL_DEVICE_TYPE" );
        _cldevice_details( _device[i], CL_DEVICE_NAME, "CL_DEVICE_NAME" );
        _cldevice_details( _device[i], CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR" );
        _cldevice_details( _device[i], CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID" );
        _cldevice_details( _device[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, "CL_DEVICE_MAX_MEM_ALLOC_SIZE" );
        _cldevice_details( _device[i], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE" );
        _cldevice_details( _device[i], CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE" );
        _cldevice_details( _device[i], CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS" );
        _cldevice_details( _device[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS" );
        _cldevice_details( _device[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, "CL_DEVICE_MAX_WORK_ITEM_SIZES" );
        _cldevice_details( _device[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE" );
        _cldevice_details( _device[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT" );
      }

      if (_device[i] != NULL) {
        cl_command_queue_properties properties;
	// enabling profile if set (i.e. rtlmode == profile or all)
	properties = _profile ? CL_QUEUE_PROFILING_ENABLE : 0;

        //Create one OpenCL context for each device in the platform
        _context[i] = clCreateContext( NULL, 1, &_device[i], NULL, NULL, &_status);
        if (_status != CL_SUCCESS) {
          fprintf(stderr, "<rtl> Failed to create context for device %u.\n", i);
	  _clErrorCode (_status);
        }

        //Create a command queue for each context to communicate with the associated device
        _cmd_queue[i] = clCreateCommandQueue(_context[i], _device[i], properties, &_status);
        if (_status != CL_SUCCESS) {
          fprintf(stderr, "<rtl> Failed to create commandQueue for device %u.\n", i);
	  _clErrorCode (_status);
        }
      }
    }

  }

  // Allocate room to handle program and kernel objects
  _nkernels  = 16;
  _program   = (cl_program *) calloc(_nkernels, sizeof(cl_program));
  _kernel    = (cl_kernel *) calloc(_nkernels, sizeof(cl_kernel));
  _strprog   = (char **) calloc(_nkernels, sizeof(char*));
  _sentinel  = 0;  // points to first free slot to handle kernel/program objects
  _kerid     = -1; // points to invalid id of kernel/program

  // Allocate room to handle buffer memory locations
  _upperid = 16;
  _locs = (cl_mem *) calloc(_upperid, sizeof(cl_mem));
  _curid = -1;    // points to invalid location

  // initialize default device to 0 (CPU) unless CPU is not present
  if ( _cpu_present ) {
    _clid = 0;
  }
  else {
    _clid = 1; // At least, one accelerator is present & was mapped to device 1
  }
}

//
// Cleanup cldevice
//
void _cldevice_finish() {

  cl_uint i;

  // Clean up and wait for all the comands to complete
  for (i = 0; i< _ndevices; i++) {
    _status = clFlush(_cmd_queue[i]);
    _status = clFinish(_cmd_queue[i]);
  }

  // Release OpenCL allocated objects
  for (i = 0; i < _sentinel; i++) {
    _status = clReleaseKernel(_kernel[i]);
    _status = clReleaseProgram(_program[i]);
  }

  for (i = 0; i < _ndevices; i++) {
    _status = clReleaseCommandQueue(_cmd_queue[i]);
    _status = clReleaseContext(_context[i]);
  }
  free(_cmd_queue);
  free(_context);
  free(_device);
  free(_program);
  free(_kernel);
  free(_strprog);
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
      fprintf(stderr, "<rtl> Failed to open file for reading: %s\n", fileName);
      return NULL;
    }

    fseek(file, 0, SEEK_END);
    size_t fsize = ftell(file);
    rewind(file);

    char* buffer = (char*) calloc(fsize+1, sizeof(char));
    buffer[fsize] = '\0';
    fread(buffer, sizeof(char), fsize, file);
    fclose(file);

    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&buffer,
                                        NULL, NULL);
    if (program == NULL) {
      fprintf(stderr, "<rtl> Failed to create CL program from source.\n");
      return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
      // Determine the reason for the error
      char buildLog[16384];
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
          sizeof(buildLog), buildLog, NULL);

      fprintf(stderr, "<rtl> Error building %s : %s\n", fileName, buildLog);
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

  unsigned char *programBinary = calloc( binarySize, sizeof(unsigned char));
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
    fprintf(stderr, "<rtl> Error loading binary %s.\n", fileName);
    return NULL;
  }

  if (binaryStatus != CL_SUCCESS) {
    fprintf(stderr, "<rtl> Invalid binary %s for device.\n", fileName);
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
      fprintf(stderr, "<rtl> %s: This platform does not support cl_khr_spir extension!\n", buildLog);
    }
    else {
      fprintf(stderr, "<rtl> Error building %s : %s\n", fileName, buildLog);
    }
    clReleaseProgram(program);
    return NULL;
  }

  return program;
}

//
//  Retrieve program binary for all of the devices attached to
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
      fprintf(stderr, "<rtl> Error querying for number of devices.\n");
      return 0;
    }

    // 2 - Get all of the Device IDs
    cl_device_id *devices = calloc(numDevices, sizeof(cl_device_id));
    errNum = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                              sizeof(cl_device_id) * numDevices,
                              devices, NULL);
    if (errNum != CL_SUCCESS) {
      fprintf(stderr, "<rtl> Error querying for devices.\n");
      free(devices);
      return 0;
    }

    // 3 - Determine the size of each program binary
    size_t *programBinarySizes = calloc(numDevices, sizeof(size_t));
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                              sizeof(size_t) * numDevices,
                              programBinarySizes, NULL);
    if (errNum != CL_SUCCESS) {
      fprintf(stderr, "<rtl> Error querying for program binary sizes.\n");
      free(devices);
      free(programBinarySizes);
      return 0;
    }

    unsigned char **programBinaries = calloc(numDevices, sizeof(unsigned char));

    cl_uint i;
    for (i = 0; i < numDevices; i++) {
      programBinaries[i] = calloc(programBinarySizes[i], sizeof(unsigned char));
    }

    // 4 - Get all of the program binaries
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
            sizeof(unsigned char*) * numDevices,
                              programBinaries, NULL);
    if (errNum != CL_SUCCESS) {
      fprintf(stderr, "<rtl> Error querying for program binaries.\n");
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
  return _ndevices;
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

  cl_int  status = 0;
  size_t  param_size = 0;
  cl_uint maxWIDimensions;

  if ((id == 0) && (!_cpu_present)) {
      _clid = 1;
      fprintf(stderr, "<rtl> Warning: CPU is not set, run on device 1 instead.\n");
  }

  if (id == 1) {
    if (!_gpu_present && _cpu_present && _ndevices == 1 ) {
      _clid = 0;
      fprintf(stderr, "<rtl> Warning: Accelerator (GPU?) is not present, run on CPU instead.\n");
    }
    else {
      _clid = id;
    }
  }
  else if (id > _ndevices-1) {
    if ( _cpu_present ) {
      _clid = 0;
    }
    else {
      _clid = 1;
    }
    fprintf(stderr, "<rtl> Warning: Device id is invalid, run on device %u instead.\n", _clid);
  }
  else
    _clid = id;

  status = clGetDeviceInfo( _device[_clid], CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &param_size );
  if (status != CL_SUCCESS ) {
    fprintf(stderr, "<rtl> Warning: Unable to obtain MAX_WORK_ITEM_SIZES for device %u.\n", _clid);
    return;
  }

  size_t* ret = (size_t*) alloca(sizeof(size_t) * param_size);
  status = clGetDeviceInfo( _device[_clid], CL_DEVICE_MAX_WORK_ITEM_SIZES, param_size, ret, NULL );
  status = clGetDeviceInfo( _device[_clid], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWIDimensions, NULL );
  if (status != CL_SUCCESS ) {
    fprintf(stderr, "<rtl> Warning: Unable to obtain MAX_WORK_ITEM_DIMENSIONS for device %u.\n", _clid);
    return;
  }

  // checking the default work_group map.
  //     cpu     {0:128, 1:1, 2:1}
  //     gpu 1-d {3:512, 4:1, 5:1}
  //     gpu 2-d {6:32, 7:16, 8:1};

  // FIX-ME: The third dimmension is not being checked

  int i = 0;
  int j = 0;
  if ( _clid == 1 ) j = 3;  // >=1 ??

  // assert y-dimmension according selected device
  if ((int)ret[i+1] < _work_group[j+1]) {
    _work_group[j+1] = (int)ret[i+1];
  }

  // assert x-dimmension * y-dimmension for selected device
  if ((int)ret[i] < _work_group[j]*_work_group[j+1]) {
    _work_group[j+1] /= 2;
  }
  if ((int)ret[i] < _work_group[j]*_work_group[j+1]) {
    _work_group[j] /= 2;
  }
  if ((int)ret[i] > 2*_work_group[j]*_work_group[j+1]) {
    if ((int)ret[i+1] > 2*_work_group[j+1])
      _work_group[j+1] *= 2;
  }
  if ((int)ret[i] > 2*_work_group[j]*_work_group[j+1] &&
      _clid != 0) {
    _work_group[j] *= 2;
  }
  if ((int)ret[i] > 2*_work_group[j]*_work_group[j+1]) {
    if ((int)ret[i+1] > 2*_work_group[j+1])
      _work_group[j+1] *= 2;
  }

}

//
// Auxiliary Function. Increments the current Id. Resize the room if necessary
//
void _inc_curid () {
  _curid++;
  if (_curid == _upperid) {
    _upperid *= 2;
    _locs = (cl_mem *) realloc(_locs, _upperid * sizeof(cl_mem));
  }
}

//
// Create a write-only memory buffer on the selected device of a given size
//
int _cl_create_write_only (uint64_t size) {
  _inc_curid();
  _locs[_curid] = clCreateBuffer(_context[_clid], CL_MEM_READ_WRITE,
         size, NULL, &_status);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<rtl> Failed creating a %llu bytes write-only buffer on device.\n", size);
    _clErrorCode (_status);
    _curid--;
    return 0;
  }
  if (_verbose) printf("<rtl> Creating a write-only buffer %d of size: %llu\n", _curid, size);
  return 1;
}

//
// Create a read-only memory buffer and copy the host loc to the buffer
//
int _cl_offloading_write_only (uint64_t size, void* loc) {
  _inc_curid();
  _locs[_curid] = clCreateBuffer(_context[_clid], CL_MEM_READ_WRITE,
         size, NULL, &_status);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<rtl> Failed creating a %llu bytes write-only buffer %d.\n", size, _curid);
    _clErrorCode (_status);
    _curid--;
    return 0;
  }
  if (_verbose) printf("<rtl> Creating a write-only buffer %d of %llu bytes\n", _curid, size);
  return 1;
}

//
// Create a read-only memory buffer to offloading host locations
//
int _cl_create_read_only (uint64_t size) {
  _inc_curid();
  _locs[_curid] = clCreateBuffer(_context[_clid], CL_MEM_READ_ONLY,
         size, NULL, &_status);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<rtl> Failed creating a %llu read-only buffer on device.\n", size);
    _clErrorCode (_status);
    _curid--;
    return 0;
  }
  if (_verbose) printf("<rtl> Creating a read-only buffer %d of size: %llu\n", _curid, size);
  return 1;
}

//
// Create a read-only memory buffer and copy the host loc to the buffer
//
int _cl_offloading_read_only (uint64_t size, void* loc) {
  _inc_curid();

  _locs[_curid] = clCreateBuffer(_context[_clid], CL_MEM_READ_ONLY,
         size, NULL, &_status);

  _status = clEnqueueWriteBuffer
              (
                _cmd_queue[_clid],
                _locs[_curid], CL_TRUE,
                0,
                size,
                loc,
                0,
                NULL,
                (_profile) ? &_global_event : NULL
              );

  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<rtl> Failed writing %llu bytes into buffer %d.\n", size, _curid);
    _clErrorCode (_status);
    _curid--;
    return 0;
  }

  if (_profile) {
    _cl_profile("_cl_offloading_read_only", _global_event);
  }
  if (_verbose) {
    printf("<rtl> Offloading %llu bytes to buffer %d\n", size, _curid);
  }

  return 1;
}

//
// Create a read-write memory buffer
//
int _cl_create_read_write (uint64_t size) {
  _inc_curid();
  _locs[_curid] = clCreateBuffer(_context[_clid], CL_MEM_READ_WRITE,
         size, NULL, &_status);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<rtl> Failed creating a read-write buffer of size %llu.\n", size);
    _clErrorCode (_status);
    _curid--;
    return 0;
  }
  if (_verbose) printf("<rtl> Creating a read-write buffer %d of size: %llu\n", _curid, size);
  return 1;
}

//
// Create a read-write memory buffer and copy the host loc to the buffer
//
int _cl_offloading_read_write (uint64_t size, void* loc) {
  _inc_curid();

  _locs[_curid] = clCreateBuffer(_context[_clid], CL_MEM_READ_WRITE,
         size, NULL, &_status);

  _status = clEnqueueWriteBuffer
              (
                _cmd_queue[_clid],
                _locs[_curid],
                CL_TRUE,
                0,
                size,
                loc,
                0,
                NULL,
                (_profile) ? &_global_event : NULL
              );

  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<rtl> Failed writing %llu bytes into buffer %d.\n", size, _curid);
    _clErrorCode (_status);
    _curid--;
    return 0;
  }

  if (_profile) {
    _cl_profile("_cl_offloading_read_write", _global_event);
  }
  if (_verbose) {
    printf("<rtl> Creating read-write buffer %d of size: %llu\n", _curid, size);
  }

  return 1;
}

//
// Read the cl_memory given by index on selected device to the host variable
//
int _cl_read_buffer (uint64_t size, int id, void* loc) {

  _status = clEnqueueReadBuffer(_cmd_queue[_clid],
              _locs[id],
              CL_TRUE,
              0,
              size,
              loc,
              0,
              NULL,
              (_profile) ? &_global_event : NULL
            );

  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<rtl> Failed reading %llu bytes from buffer %d.\n", size, id);
    _clErrorCode (_status);
    return 0;
  }

  if (_profile) {
    _cl_profile("_cl_read_buffer", _global_event);
  }
  if (_verbose) {
    printf("<rtl> Reading %llu bytes from buffer %d\n", size, id);
  }

  return 1;
}

//
// Write (sync) the host variable to cl_memory given by index
//
int _cl_write_buffer (uint64_t size, int id, void* loc) {

  _status = clEnqueueWriteBuffer(_cmd_queue[_clid],
				 _locs[id],
				 CL_TRUE,
				 0,
				 size,
				 loc,
				 0,
				 NULL,
				 (_profile) ? &_global_event : NULL);
  
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<rtl> Failed writing %llu bytes into buffer %d.\n", size, id);
    _clErrorCode (_status);
    return 0;
  }

  if (_profile) {
    _cl_profile("_cl_write_buffer", _global_event);
  }
  if (_verbose) {
    printf("<rtl> Writing %llu bytes into buffer %d\n", size, id);
  }
  return 1;
}

// Auxiliary Function. Return true if program object was created before.
int _program_created(const char* str) {

  cl_uint i;

  for ( i = 0; i < _sentinel; i++)
    if (strcmp (str, _strprog[i]) == 0) {
      _kerid = i;
      return 1;
    }

  _kerid = _sentinel++;
  if (_sentinel == _nkernels) {
    _nkernels *= 2;
    _program = (cl_program *) realloc(_program, _nkernels * sizeof(cl_program));
    _kernel  = (cl_kernel *) realloc(_kernel, _nkernels * sizeof(cl_kernel));
  }
  _strprog[_kerid] = (char *) calloc(strlen(str), sizeof(char));
  strcpy (_strprog[_kerid],str);
  return 0;
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

  // Sets the handle (_kerid) if program was created before
  if (_program_created(str)) return 1;

  // otherwise, create it
  int fsize = strlen(str);

  char* cl_file   = calloc(fsize + 4, sizeof(char));
  char* bc_file   = calloc(fsize + 4, sizeof(char));
  char* aocx_file = calloc(fsize + 6, sizeof(char));

  strcpy(cl_file, str);
  strcpy(bc_file, str);
  strcpy(aocx_file, str);

  strcat(bc_file, ".bc");
  strcat(cl_file, ".cl");
  strcat(aocx_file, ".aocx");

  if (_does_file_exist(bc_file)) {
    //Attempting to create program from binary
    if (_verbose)
      printf("<rtl> Creating the program object for %s.\n", str);

    _program[_kerid] = _create_fromBinary(_context[_clid],
            _device[_clid],
            bc_file);
    if (_program[_kerid] != NULL) return 1;
  } else if (_does_file_exist(aocx_file)) {
    //Attempting to create program from aocx
    if (_verbose)
      printf("<rtl> Creating the program object for %s.\n", str);

    _program[_kerid] = _create_fromBinary(_context[_clid],
            _device[_clid],
            aocx_file);

    if (_program[_kerid] != NULL) return 1;
  }

  //Binary not loaded, create from source
  _program[_kerid] = _create_fromSource(_context[_clid],
          _device[_clid],
          cl_file);
  if (_program[_kerid] == NULL) {
    fprintf(stderr, "<rtl> Attempting to create program object failed.\n");
    return 0;
  }
  if (_save_toBinary(_program[_kerid], _device[_clid], bc_file) == 0) {
    fprintf(stderr, "<rtl> Failed to save program object in binary form.\n");
    return 0;
  }
  return 1;
}

//
// Create OpenCL kernel. Return 1 (=true), if success
//
int _cl_create_kernel (char* str) {

  if (_kernel[_kerid] != NULL) {
    clReleaseKernel(_kernel[_kerid]);
  }

  if (_verbose) printf("<rtl> Creating the kernel object for %s.\n", str);
  _kernel[_kerid] = clCreateKernel(_program[_kerid], str, NULL);
  if (_kernel[_kerid] == NULL) {
    fprintf(stderr, "<rtl> Failed to create kernel object.\n");
    return 0;
  }
  return 1;
}

//
// Set the kernel arguments for cl_mem buffers
//
int _cl_set_kernel_args (int nargs) {
  int i;
  for (i = 0; i<nargs; i++) {
    _status |= clSetKernelArg (_kernel[_kerid], i, sizeof(cl_mem), &_locs[i]);
    if (_status != CL_SUCCESS) {
      fprintf(stderr, "<rtl> Error setting buffer %d to kernel in pos %d.\n", i, i);
      _clErrorCode (_status);
      return 0;
    }
    if (_verbose) printf("<rtl> Pass buffer %d to kernel in pos %d\n", i, i);
  }
  return 1;
}

//
// Set the kernel argument for cl_mem buffer given by index
//
int _cl_set_kernel_arg (int pos, int index) {
  _status |= clSetKernelArg (_kernel[_kerid], pos, sizeof(cl_mem), &_locs[index]);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<rtl> Error setting buffer %d to kernel in pos %d.\n", index, pos);
    _clErrorCode (_status);
    return 0;
  }
  if (_verbose) printf("<rtl> Pass buffer %d to kernel in pos %d\n", index, pos);
  return 1;
}

//
// Set the kernel arguments for host args
//
int _cl_set_kernel_hostArg (int pos, int size, void* loc) {
  _status = clSetKernelArg (_kernel[_kerid], pos, size, loc);
  if (_status != CL_SUCCESS) {
    fprintf(stderr, "<rtl> Error setting host args on device.\n");
    _clErrorCode (_status);
    return 0;
  }
  return 1;
}

//
// Enqueues a command to execute a kernel on a device (without tiling).
//
int _cl_execute_kernel(uint64_t size1, uint64_t size2, uint64_t size3, int dim) {

  size_t  *global_size;
  size_t  *local_size;
  cl_uint  wd = dim;

  // work_group map:
  //     cpu     {0:128, 1:1, 2:1}
  //     gpu 1-d {3:256, 4:1, 5:1}
  //     gpu 2-d {6:32, 7:16, 8:1};
  int idx = 0;
  if (_clid == 1 ) idx  = 3; // >=1 ??
  if ( dim  == 2 ) idx *= 2;

  global_size = (size_t *) calloc(3, sizeof(size_t));
  global_size[0] = (size_t)ceil(((float)size1) / ((float)_work_group[idx])) * _work_group[idx];
  global_size[1] = (size_t)ceil(((float)size2) / ((float)_work_group[idx+1])) * _work_group[idx+1];
  global_size[2] = (size_t)ceil(((float)size2) / ((float)_work_group[idx+2])) * _work_group[idx+2];

  local_size = (size_t *) calloc(3, sizeof(size_t));
  local_size[0] = _work_group[idx];
  local_size[1] = _work_group[idx+1];
  local_size[2] = _work_group[idx+2];

  if (_verbose) {
    printf("<rtl> %s will be executed on device: %d\n", _strprog[_kerid], _clid);
    printf("<rtl> Work Group was configured to:\n");
    printf("\tX-size=%llu\t,Local X-WGS=%lu\t,Global X-WGS=%lu\n", size1, local_size[0], global_size[0]);
    if (dim >= 2)
      printf("\tY-size=%llu\t,Local Y-WGS=%lu\t,Global Y-WGS=%lu\n", size2, local_size[1], global_size[1]);
    if (dim == 3)
      printf("\tZ-size=%llu\t,Local Z-WGS=%lu\t,Global Z-WGS=%lu\n", size3, local_size[2], global_size[2]);
  }

  _status = clEnqueueNDRangeKernel
              (
                _cmd_queue[_clid],
                _kernel[_kerid],
                wd,                                // number of dimmensions
                NULL,                              // global_work_offset
                global_size,                       // global_work_size
                local_size,                        // local_work_size
                0,                                 // num_events_in_wait_list
                NULL,                              // event_wait_list
                (_profile) ? &_global_event : NULL // event
              );

  if (_status == CL_SUCCESS) {
    if (_profile) {
      _cl_profile("_cl_execute_kernel", _global_event);
    }
    if (_verbose) {    
      printf("<rtl> %s has been running successfully.\n", _strprog[_kerid]);
    }
    return 1;
  } else {
    if (_status == CL_INVALID_WORK_DIMENSION)
      fprintf(stderr, "<rtl> Error executing kernel. Number of dimmensions is not a valid value.\n");
    else if (_status == CL_INVALID_GLOBAL_WORK_SIZE)
      fprintf(stderr, "<rtl> Error executing kernel. Global Work Size is NULL or exceeded valid range.\n");
    else if (_status == CL_INVALID_WORK_GROUP_SIZE)
      fprintf(stderr, "<rtl> Error executing kernel. Local Work Size does not match the Work Group size.\n");
    else if (_status == CL_INVALID_WORK_ITEM_SIZE)
      fprintf(stderr, "<rtl> Error executing kernel. The number of work-items is greater than Max Work-items.\n");
    else
      fprintf(stderr, "<rtl> Error executing kernel on device %d\n", _clid);
    _clErrorCode (_status);
  }
  return 0;
}

//
// Enqueues a command to execute a (possible optimized w/ tilling) kernel.
//
int _cl_execute_tiled_kernel(int wsize0, int wsize1, int wsize2,
           int block0, int block1, int block2,
           int dim) {

  size_t  *global_size;
  size_t  *local_size;
  cl_uint  wd = dim;
  
  global_size = (size_t *) calloc(3, sizeof(size_t));
  local_size = (size_t *) calloc(3, sizeof(size_t));

  global_size[0] = (size_t)(wsize0 * block0);
  local_size[0]  = (size_t)block0;

  if (dim == 2) {
    global_size[1] = (size_t)(wsize1 * block1);
    local_size[1]  = block1;
    if (block2 == 0) {
      global_size[2] = 0;
      local_size[2]  = 0;
    }
    else {
      global_size[2] = block2;
      local_size[2] = block2;
      wd = 3;
    }
  }
  else if (dim == 3) {
    global_size[1] = (size_t)(wsize1 * block1);
    local_size[1]  = block1;
    global_size[2] = (size_t)(wsize2 * block2);
    local_size[2]  = block2;
  }

  if (_verbose) {
    printf("<rtl> %s will be executed on device: %d\n", _strprog[_kerid], _clid);
    printf("<rtl> Work Group was configured to:\n");
    printf("\tX-size=%d\t,Local X-WGS=%lu\t,Global X-WGS=%lu\n", wsize0, local_size[0], global_size[0]);
    if (wd >= 2)
      printf("\tY-size=%d\t,Local Y-WGS=%lu\t,Global Y-WGS=%lu\n", wsize1, local_size[1], global_size[1]);
    if (wd == 3)
      printf("\tZ-size=%d\t,Local Z-WGS=%lu\t,Global Z-WGS=%lu\n", wsize2, local_size[2], global_size[2]);
  }

  _status = clEnqueueNDRangeKernel
              (
                _cmd_queue[_clid],
                _kernel[_kerid],
                wd,                                // number of dimmensions
                NULL,                              // global_work_offset
                global_size,                       // global_work_size
                local_size,                        // local_work_size
                0,                                 // num_events_in_wait_list
                NULL,                              // event_wait_list
                (_profile) ? &_global_event : NULL // event
              );

  if (_status == CL_SUCCESS) {
    if (_profile) {
      _cl_profile("_cl_execute_tiled_kernel", _global_event);
    }
    if (_verbose) {
      printf("<rtl> %s has been running successfully.\n", _strprog[_kerid]);
    }
    return 1;
  }

  if (_status == CL_INVALID_WORK_DIMENSION)
    fprintf(stderr, "<rtl> Error executing kernel. Number of dimmensions is not a valid value.\n");
  else if (_status == CL_INVALID_GLOBAL_WORK_SIZE)
    fprintf(stderr, "<rtl> Error executing kernel. Global Work Size is NULL or exceeded valid range.\n");
  else if (_status == CL_INVALID_WORK_GROUP_SIZE)
    fprintf(stderr, "<rtl> Error executing kernel. Local Work Size does not match the Work Group size.\n");
  else if (_status == CL_INVALID_WORK_ITEM_SIZE)
    fprintf(stderr, "<rtl> Error executing kernel. The number of work-items is greater than Max Work-items.\n");
  else
    fprintf(stderr, "<rtl> Error executing kernel on device %d\n", _clid);
  _clErrorCode (_status);
  return 0;
}

//
// Release all OpenCL allocated buffer inside the map region.
//
void _cl_release_buffers(int upper) {
  int i;
  for (i=0; i<upper; i++) {
    if (_locs[i]) {
      _status = clReleaseMemObject(_locs[i]);
      if (_verbose) printf("<rtl> Releasing buffer %d\n", i);
      _locs[i] = NULL;
    }
  }
  _curid = -1;
}

//
// Release an OpenCL allocated buffer inside the map region, given an index.
//
void _cl_release_buffer(int index) {
  if (_locs[index]) {
    _status = clReleaseMemObject(_locs[index]);
    if (_verbose) printf("<rtl> Releasing buffer %d\n", index);
    _locs[index] = NULL;
    _curid--;
  }
}

void _cl_profile(const char* str, cl_event event) {
  cl_ulong time_start;
  cl_ulong time_end;
  cl_ulong time_elapsed;

  if (!_profile) {
    return;
  }
  
  _status = clFinish(_cmd_queue[_clid]);
  if (_status != CL_SUCCESS ) {
    fprintf(stderr, "<rtl> unable to finish command queue.\n");
    _clErrorCode (_status);
    return;
  }

  _status = clGetEventProfilingInfo
              (
                event,
                CL_PROFILING_COMMAND_START,
                sizeof(cl_ulong),
                &time_start,
                NULL
              );

  if (_status != CL_SUCCESS ) {
    fprintf(stderr, "<rtl> unable to start profile.\n");
    _clErrorCode (_status);
    return;
  }

  _status = clGetEventProfilingInfo
              (
                event,
                CL_PROFILING_COMMAND_END,
                sizeof(cl_ulong),
                &time_end,
                NULL
              );

  if (_status != CL_SUCCESS ) {
    fprintf(stderr, "<rtl> unable to finish profile.\n");
    _clErrorCode (_status);
    return;
  }

  time_elapsed = time_end - time_start;

  printf("<rtl><profile> %s = %llu ns\n", str, time_elapsed);
}


//
// Return the threads & blocks used to allocate auxiliary buffers
// and the adequate size of the vector to scan
//
int _cl_get_threads_blocks (int* threads, int* blocks, int size) {
  
  int n = (int) sqrt(size-1);
  int t = 1;
  for (int i = 0 ; i <= 10 ; i++) {
    if (_block_sizes[i] >= n) {
      t = _block_sizes[i];
      break;
    }
  }
  
  int r = size / t;
  n = r * t;
  if (size % t != 0) n += t;
  *threads = t;
  *blocks = n/t;
  return n;
}
