// NAME
//   cldevice.h
// VERSION
//    2.1
// SYNOPSIS
//   header file for the library that manage OpenCL programs,
//   creating contexts and command queues for each plataform
//   available on the host and manage opencl source and binary files
// AUTHOR
//    Marcio Machado Pereira
// COPYLEFT
//   Copyleft (C) 2015--2016, UNICAMP & Samsumg R&D

#ifndef __CLDEVICE_H
#define __CLDEVICE_H

#ifdef __cplusplus
  extern "C" {
#endif

#ifdef __APPLE__
#include <OpenCL/cl.h>
#elif __ALTERA__
#include <CL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <sys/time.h>

extern cl_device_id     *_device;
extern cl_context       *_context;
extern cl_command_queue *_cmd_queue;
extern cl_mem           *_locs;

extern cl_platform_id    _platform;
extern cl_program       *_program;
extern cl_kernel        *_kernel;
extern cl_uint           _ndevices;
extern cl_uint           _clid;
extern cl_int            _status;

extern cl_uint           _kerid;
extern cl_uint           _nkernels;
extern cl_uint           _sentinel;
extern char            **_strprog;

extern int               _spir_support;
extern int               _gpu_present;
extern int               _cpu_present;
extern int               _upperid;
extern int               _curid;
extern int               _verbose;
extern int               _profile;

extern int               _work_group[9];
extern int               _block_sizes[11];
    
extern cl_event         _global_event;

void _cldevice_details(cl_device_id   id,
                       cl_device_info param_name,
                       const char*    param_str);

void _cldevice_init (int rtlmode);

void _cldevice_finish ();

cl_program _create_fromSource(cl_context   context,
                              cl_device_id device,
                              const char*  fileName);

cl_program _create_fromBinary(cl_context   context,
                              cl_device_id device,
                              const char*  fileName);

int _save_toBinary(cl_program    program,
                   cl_device_id device,
                   const char*  fileName);

cl_uint _get_num_cores (int A, int B, int C, int T);

cl_uint _get_num_devices ();

cl_uint _get_default_device ();

void _set_default_device (cl_uint id);

int _cl_create_read_only (uint64_t size);

int _cl_create_write_only (uint64_t size);

int _cl_create_read_write (uint64_t size);

int _cl_offloading_read_only (uint64_t size, void* loc);

int _cl_offloading_write_only (uint64_t size, void* loc);

int _cl_offloading_read_write (uint64_t size, void* loc);

int _cl_read_buffer (uint64_t size, int id, void* loc);

int _cl_write_buffer (uint64_t size, int id, void* loc);

int _cl_create_program (char* str);

int _cl_create_kernel (char* str);

int _cl_set_kernel_args (int nargs);

int _cl_set_kernel_arg (int pos, int index);

int _cl_set_kernel_hostArg (int pos, int size, void* loc);

int _cl_execute_kernel (uint64_t size1, uint64_t size2, uint64_t size3, int dim);

int _cl_execute_tiled_kernel (int wsize0, int wsize1, int wsize2, int block0, int block1, int block2, int dim);

void _cl_release_buffers (int upper);

double _cl_profile(const char* str, cl_event event);

int _cl_get_threads_blocks(int *threads, int *blocks, int *sthreads, int *sblocks, uint64_t size, int bytes);


/* DCAO runtime */
void _cl_init_shared_buffer (int DCAO_dbg);
void _inc_curid_shared_buffer ();
void _cl_release_buffers_shared_buffer();
void _cl_release_shared_buffer(int index);
void _cl_create_shared_buffer_write_only (long size, int position);
void _cl_create_shared_buffer_read_only (long size, int position);
void _cl_create_shared_buffer_read_write (long size, int position);
int _cl_set_kernel_arg_shared_buffer (int pos, int index);
void *_cl_map_buffer_write(int index);
void *_cl_map_buffer_read(int index);
void *_cl_map_buffer_read_write(int index);
void _cl_unmap_buffer(int index);
double _cl_rtclock();
void _cl_prints();

#ifdef __cplusplus
  }
#endif

#endif /* __CLDEVICE_H */
