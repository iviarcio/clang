//==-- CGMPtoGPURuntime.cpp - Interface to OpenMP to GPU Runtime -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for OpenMP to GPU code generation
// Concrete subclasses of this implement code generation for specific
// OpenMP to GPGPU runtime libraries.
//
//===----------------------------------------------------------------------===//

#include "CGMPtoGPURuntime.h"
#include "CodeGenFunction.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace clang;
using namespace CodeGen;

CGMPtoGPURuntime::CGMPtoGPURuntime(CodeGenModule &CGM) : CGM(CGM) {
}

llvm::Value *
CGMPtoGPURuntime::CreateRuntimeFunction(MPtoGPURTLFunction Function) {
  llvm::Value *RTLFn = nullptr;
  switch (Function) {
  case MPtoGPURTL_set_default_device: {
    // Build void _set_default_device(cl_uint id);
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, CGM.Int32Ty, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_set_default_device");
    break;
  }
  case MPtoGPURTL_get_num_devices: {
    // Build cl_uint _get_num_devices();
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, CGM.VoidTy, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_get_num_devices");
    break;
  }
  case MPtoGPURTL_get_num_cores: {
    // Build cl_uint _get_num_cores();
    llvm::Type *TParams[] = {CGM.Int32Ty, CGM.Int32Ty, CGM.Int32Ty, CGM.Int32Ty};
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_get_num_cores");
    break;
  }
  case MPtoGPURTL_get_default_device: {
    // Build cl_uint _get_default_device();
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, CGM.VoidTy, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_get_default_device");
    break;
  }
  case MPtoGPURTL_cldevice_init: {
    // Build void _cldevice_init(int verbose);
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, CGM.Int32Ty, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cldevice_init");
    break;
  }
  case MPtoGPURTL_cldevice_finish: {
    // Build void _cldevice_finish();
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, CGM.VoidTy, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cldevice_finish");
    break;
  }
  case MPtoGPURTL_cl_create_write_only: {
    // Build int _cl_create_write_only(long size);
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, CGM.Int64Ty, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_create_write_only");
    break;
  }
  case MPtoGPURTL_cl_create_read_only: {
    // Build int _cl_create_read_only(long size);
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, CGM.Int64Ty, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_create_read_only");
    break;
  }
  case MPtoGPURTL_cl_offloading_read_only: {
    // Build int _cl_offloading_read_only(long size, void* loc);
    llvm::Type *TParams[] = {CGM.Int64Ty, CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_offloading_read_only");
    break;
  }
  case MPtoGPURTL_cl_create_read_write: {
    // Build int _cl_create_read_write(long size, void* loc);
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, CGM.Int64Ty, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_create_read_write");
    break;
  }
  case MPtoGPURTL_cl_offloading_read_write: {
    // Build int _cl_offloading_read_write(long size, void* loc);
    llvm::Type *TParams[] = {CGM.Int64Ty, CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_offloading_read_write");
    break;
  }
  case MPtoGPURTL_cl_read_buffer: {
    // Build int _cl_read_buffer(long size, int id, void* loc);
    llvm::Type *TParams[] = {CGM.Int64Ty, CGM.Int32Ty, CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_read_buffer");
    break;
  }
  case MPtoGPURTL_cl_write_buffer: {
    // Build int _cl_write_buffer(long size, int id, void* loc);
    llvm::Type *TParams[] = {CGM.Int64Ty, CGM.Int32Ty, CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_write_buffer");
    break;
  }
  case MPtoGPURTL_cl_create_program: {
    // Build int _cl_create_program(char* str);
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, CGM.Int8PtrTy, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_create_program");
    break;
  }
  case MPtoGPURTL_cl_create_kernel: {
    // Build int _cl_create_kernel(char* str);
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, CGM.Int8PtrTy, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_create_kernel");
    break;
  }
  case MPtoGPURTL_cl_set_kernel_args: {
    // Build int _set_kernel_args(int nargs);
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, CGM.Int32Ty, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_set_kernel_args");
    break;
  }
  case MPtoGPURTL_cl_set_kernel_arg: {
    // Build int _set_kernel_arg(int pos, int index);
    llvm::Type *TParams[] = {CGM.Int32Ty, CGM.Int32Ty};
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, TParams,  false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_set_kernel_arg");
    break;
  }
  case MPtoGPURTL_cl_set_kernel_hostArg: {
    // Build int _set_kernel_hostArg(int pos, int size, void* loc);
    llvm::Type *TParams[] = {CGM.Int32Ty, CGM.Int32Ty, CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_set_kernel_hostArg");
    break;
  }
  case MPtoGPURTL_cl_execute_kernel: {
    // Build int _cl_execute_kernel(long size1, long size2, long size3, int tile, int dim);
    llvm::Type *TParams[] = {CGM.Int64Ty, CGM.Int64Ty, CGM.Int64Ty, CGM.Int32Ty, CGM.Int32Ty};
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_execute_kernel");
    break;
  }
  case MPtoGPURTL_cl_release_buffers: {
    // Build void _set_release_buffers(int upper);
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, CGM.Int32Ty, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_set_release_buffers");
    break;
  }
  case MPtoGPURTL_cl_release_buffer: {
    // Build void _set_release_buffer(int index);
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, CGM.Int32Ty, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_set_release_buffer");
    break;
  }  
  }
  return RTLFn;
}

llvm::Value*
CGMPtoGPURuntime::cldevice_init() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cldevice_init, false>::get(CGM.getLLVMContext())
	 , "_cldevice_init");
}

llvm::Value*
CGMPtoGPURuntime::cldevice_finish() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cldevice_finish, false>::get(CGM.getLLVMContext())
	 , "_cldevice_finish");
}

llvm::Value*
CGMPtoGPURuntime::Set_default_device() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_set_default_device, false>::get(CGM.getLLVMContext())
	 , "_set_default_device");
}

llvm::Value*
CGMPtoGPURuntime::Get_num_devices() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_get_num_devices, false>::get(CGM.getLLVMContext())
	 , "_get_num_devices");
}

llvm::Value*
CGMPtoGPURuntime::Get_num_cores() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_get_num_cores, false>::get(CGM.getLLVMContext())
	 , "_get_num_cores");
}

llvm::Value*
CGMPtoGPURuntime::Get_default_device() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_get_default_device, false>::get(CGM.getLLVMContext())
	 , "_get_default_device");
}

llvm::Value*
CGMPtoGPURuntime::cl_create_write_only() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_create_write_only, false>::get(CGM.getLLVMContext())
	 , "_cl_create_write_only");
}

llvm::Value*
CGMPtoGPURuntime::cl_create_read_only() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_create_read_only, false>::get(CGM.getLLVMContext())
	 , "_cl_create_read_only");
}

llvm::Value*
CGMPtoGPURuntime::cl_offloading_read_only() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_offloading_read_only, false>::get(CGM.getLLVMContext())
	 , "_cl_offloading_read_only");
}

llvm::Value*
CGMPtoGPURuntime::cl_create_read_write() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_create_read_write, false>::get(CGM.getLLVMContext())
	 , "_cl_create_read_write");
}

llvm::Value*
CGMPtoGPURuntime::cl_offloading_read_write() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_offloading_read_write, false>::get(CGM.getLLVMContext())
	 , "_cl_offloading_read_write");
}

llvm::Value*
CGMPtoGPURuntime::cl_read_buffer() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_read_buffer, false>::get(CGM.getLLVMContext())
	 , "_cl_read_buffer");
}

llvm::Value*
CGMPtoGPURuntime::cl_write_buffer() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_write_buffer, false>::get(CGM.getLLVMContext())
	 , "_cl_write_buffer");
}

llvm::Value*
CGMPtoGPURuntime::cl_create_program() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_create_program, false>::get(CGM.getLLVMContext())
	 , "_cl_create_program");
}

llvm::Value*
CGMPtoGPURuntime::cl_create_kernel() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_create_kernel, false>::get(CGM.getLLVMContext())
	 , "_cl_create_kernel");
}

llvm::Value*
CGMPtoGPURuntime::cl_set_kernel_args() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_set_kernel_args, false>::get(CGM.getLLVMContext())
	 , "_cl_set_kernel_args");
}

llvm::Value*
CGMPtoGPURuntime::cl_set_kernel_arg() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_set_kernel_arg, false>::get(CGM.getLLVMContext())
	 , "_cl_set_kernel_arg");
}

llvm::Value*
CGMPtoGPURuntime::cl_set_kernel_hostArg() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_set_kernel_hostArg, false>::get(CGM.getLLVMContext())
	 , "_cl_set_kernel_hostArg");
}

llvm::Value*
CGMPtoGPURuntime::cl_execute_kernel() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_execute_kernel, false>::get(CGM.getLLVMContext())
	 , "_cl_execute_kernel");
}

llvm::Value*
CGMPtoGPURuntime::cl_release_buffers() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_release_buffers, false>::get(CGM.getLLVMContext())
	 , "_cl_release_buffers");
}

llvm::Value*
CGMPtoGPURuntime::cl_release_buffer() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_cl_release_buffer, false>::get(CGM.getLLVMContext())
	 , "_cl_release_buffer");
}

//
// Create runtime for the target used in the Module
//
CGMPtoGPURuntime *CodeGen::CreateMPtoGPURuntime(CodeGenModule &CGM) {
  return new CGMPtoGPURuntime(CGM);
}
