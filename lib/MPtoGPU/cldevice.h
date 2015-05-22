// NAME
//   cldevice.h
// VERSION
//    0.01
// SYNOPSIS
//   header file for the library that manage OpenCL programs,
//   creating contexts and command queues for each plataform
//   available on the host and manage opencl source and binary files
// AUTHOR
//    Marcio Machado Pereira
// COPYLEFT
//   Copyleft (C) 2015 -- UNICAMP & Samsumg R&D

#ifndef __CLDEVICE_H
#define __CLDEVICE_H

#ifdef __cplusplus
  extern "C" {
#endif
  
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

extern cl_device_id     *_device;
extern cl_context       *_context;
extern cl_command_queue *_cmd_queue;
extern cl_mem           *_locs;
extern int               _upperid;
extern int               _curid;
extern cl_platform_id    _platform;
extern cl_program        _program;
extern cl_kernel         _kernel;
extern cl_uint           _npairs;
extern cl_uint           _clid;
extern cl_int            _status;
  
void _cldevice_init ();

void _cldevice_finish ();

cl_program _create_fromSource(cl_context context,
			      cl_device_id device,
			      const char* fileName);

cl_program _create_fromBinary(cl_context context,
			      cl_device_id device,
			      const char* fileName);

int _save_toBinary(cl_program program,
		    cl_device_id device,
		    const char* fileName);

cl_uint _get_num_devices ();
    
cl_uint _get_default_device ();

void _set_default_device (cl_uint id);

int _cl_create_write_only (long size);

int _cl_create_read_only (long size);

int _cl_offloading_read_only (long size, void* loc);

int _cl_create_read_write (long size);

int _cl_offloading_read_write (long size, void* loc);

int _cl_read_buffer (long size, int id, void* loc);

int _cl_write_buffer (long size, int id, void* loc);

int _cl_create_program (char* str);

int _cl_create_kernel (char* str);

int _cl_set_kernel_args (int nargs);

int _cl_set_kernel_hostArg (int pos, int size, void* loc);
    
int _cl_execute_kernel (long work_size);

void _cl_release_buffers (int upper);    
    
#ifdef __cplusplus
  }
#endif
    
#endif /* __CLDEVICE_H */
