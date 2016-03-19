//===--- CGMPtoGPURuntime.h - Interface to OpenMP to GPU Runtime -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for OpenMP to GPU code generation.
// Concrete subclasses of this implement code generation for specific
// MPtoGPU runtime libraries.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_MPTOGPU_H
#define CLANG_CODEGEN_MPTOGPU_H

#include "clang/AST/Type.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "CodeGenModule.h"
#include "CodeGenFunction.h"

namespace llvm {
class AllocaInst;
class CallInst;
class GlobalVariable;
class Constant;
class Function;
class Module;
class StructLayout;
class FunctionType;
class StructType;
class Type;
class Value;
} // namespace llvm

namespace {
  typedef void(_set_default_device)(int32_t id);
  typedef int32_t(_get_num_devices)();
  typedef int32_t(_get_num_cores)(int32_t A, int32_t B, int32_t C, int32_t T);
  typedef int32_t(_get_default_device)();
  typedef void(_cldevice_init)(int32_t verbose);
  typedef void(_cldevice_finish)();
  typedef int32_t(_cl_create_write_only)(int64_t size);
  typedef int32_t(_cl_create_read_only)(int64_t size);
  typedef int32_t(_cl_offloading_read_only)(int64_t size, void* loc);
  typedef int32_t(_cl_create_read_write)(int64_t size);
  typedef int32_t(_cl_offloading_read_write)(int64_t size, void* loc);
  typedef int32_t(_cl_read_buffer)(int64_t size, int32_t id, void* loc);
  typedef int32_t(_cl_write_buffer)(int64_t size, int32_t id, void* loc);
  typedef int32_t(_cl_create_program)(char* str);
  typedef int32_t(_cl_create_kernel)(char* str);
  typedef int32_t(_cl_set_kernel_args)(int32_t nargs);
  typedef int32_t(_cl_set_kernel_arg)(int32_t pos, int32_t index);
  typedef int32_t(_cl_set_kernel_hostArg)(int32_t pos, int32_t size, void* loc);
  typedef int32_t(_cl_execute_kernel)(int64_t size1, int64_t size2, int64_t size3, int32_t dim);
  typedef int32_t(_cl_execute_tiled_kernel)(int64_t size1, int64_t size2, int64_t size3, int32_t tile, int32_t dim);
  typedef void(_cl_release_buffers)(int32_t upper);
  typedef void(_cl_release_buffer)(int32_t index);
}

namespace clang {
namespace CodeGen {

class CodeGenFunction;
class CodeGenModule;

/// Implements runtime-specific code generation functions.
class CGMPtoGPURuntime {

protected:
  CodeGenModule &CGM;
  
public:
  enum MPtoGPURTLFunction {
    MPtoGPURTL_set_default_device,
    MPtoGPURTL_get_num_devices,
    MPtoGPURTL_get_num_cores,
    MPtoGPURTL_get_default_device,
    MPtoGPURTL_cldevice_init,
    MPtoGPURTL_cldevice_finish,
    MPtoGPURTL_cl_create_write_only,
    MPtoGPURTL_cl_create_read_only,
    MPtoGPURTL_cl_offloading_read_only,
    MPtoGPURTL_cl_create_read_write,
    MPtoGPURTL_cl_offloading_read_write,
    MPtoGPURTL_cl_read_buffer,
    MPtoGPURTL_cl_write_buffer,
    MPtoGPURTL_cl_create_program,
    MPtoGPURTL_cl_create_kernel,
    MPtoGPURTL_cl_set_kernel_args,
    MPtoGPURTL_cl_set_kernel_arg,
    MPtoGPURTL_cl_set_kernel_hostArg,
    MPtoGPURTL_cl_execute_kernel,
    MPtoGPURTL_cl_execute_tiled_kernel,
    MPtoGPURTL_cl_release_buffers,
    MPtoGPURTL_cl_release_buffer
  };
  
  explicit CGMPtoGPURuntime(CodeGenModule &CGM);
  virtual ~CGMPtoGPURuntime() {}

  /// \brief Returns specified OpenMP to GPU runtime function.
  /// \param Function MPtoGPU runtime function.
  /// \return Specified function.
  llvm::Value *CreateRuntimeFunction(MPtoGPURTLFunction Function);

  virtual llvm::Value* Set_default_device();
  virtual llvm::Value* Get_num_devices();
  virtual llvm::Value* Get_num_cores();
  virtual llvm::Value* Get_default_device();
  virtual llvm::Value* cldevice_init();
  virtual llvm::Value* cldevice_finish();
  virtual llvm::Value* cl_create_write_only();
  virtual llvm::Value* cl_create_read_only();
  virtual llvm::Value* cl_offloading_read_only();
  virtual llvm::Value* cl_create_read_write();
  virtual llvm::Value* cl_offloading_read_write();
  virtual llvm::Value* cl_read_buffer();
  virtual llvm::Value* cl_write_buffer();
  virtual llvm::Value* cl_create_program();
  virtual llvm::Value* cl_create_kernel();
  virtual llvm::Value* cl_set_kernel_args();
  virtual llvm::Value* cl_set_kernel_arg();
  virtual llvm::Value* cl_set_kernel_hostArg();
  virtual llvm::Value* cl_execute_kernel();
  virtual llvm::Value* cl_execute_tiled_kernel();
  virtual llvm::Value* cl_release_buffers();  
  virtual llvm::Value* cl_release_buffer();
};
  
/// \brief Returns an implementation of the OpenMP to GPU RTL for a given target
CGMPtoGPURuntime *CreateMPtoGPURuntime(CodeGenModule &CGM);

} // namespace CodeGen
} // namespace clang

#endif
