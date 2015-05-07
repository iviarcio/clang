//===-- CGMPtoGPURuntime.h - Interface to OpenMP to GPU Runtime -*- C++ -*-===//
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
  _npairs = 0;
  _clid =0;
  _status = 0;
}

llvm::Value *
CGMPtoGPURuntime::CreateRuntimeFunction(MPtoGPURTLFunction Function) {
  llvm::Value *RTLFn = nullptr;
  switch (Function) {
  case MPtoGPURTL_set_default_device: {
    // Build void _set_default_device(cl_uint id);
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, CGM.Int32Ty, true);
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
  case MPtoGPURTL_get_default_device: {
    // Build cl_uint _get_default_device();
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, CGM.VoidTy, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_get_default_device");
    break;
  }
  case MPtoGPURTL_cldevice_init: {
    // Build void _cldevice_init(cl_uint id);
    llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, CGM.Int32Ty, true);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cldevice_init");
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
CGMPtoGPURuntime::Set_default_device() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_set_default_device, false>::get(CGM.getLLVMContext())
	 , "_set_default_device");

//	llvm::FunctionType *FnTy =
//	      llvm::FunctionType::get(CGM.VoidTy, CGM.Int32Ty, false);
//	    return CGM.CreateRuntimeFunction(FnTy, "_set_default_device");
}

llvm::Value*
CGMPtoGPURuntime::Get_num_devices() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_get_num_devices, false>::get(CGM.getLLVMContext())
	 , "_get_num_devices");
}

llvm::Value*
CGMPtoGPURuntime::Get_default_device() {
  return CGM.CreateRuntimeFunction(
	 llvm::TypeBuilder<_get_default_device, false>::get(CGM.getLLVMContext())
	 , "_get_default_device");
}

//
// Create runtime for the target used in the Module
//
CGMPtoGPURuntime *CodeGen::CreateMPtoGPURuntime(CodeGenModule &CGM) {
  return new CGMPtoGPURuntime(CGM);
}
