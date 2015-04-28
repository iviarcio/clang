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
  typedef int32_t(_get_default_device)();
  typedef void(_cldevice_init)();
}

namespace clang {
namespace CodeGen {

class CodeGenFunction;
class CodeGenModule;

/// Implements runtime-specific code generation functions.
class CGMPtoGPURuntime {

protected:
  CodeGenModule &CGM;
  
  int32_t _npairs;
  int32_t _clid;
  int32_t _status;

public:
  enum MPtoGPURTLFunction {
    MPtoGPURTL_set_default_device,
    MPtoGPURTL_get_num_devices,
    MPtoGPURTL_get_default_device,
    MPtoGPURTL_cldevice_init
  };
  
  explicit CGMPtoGPURuntime(CodeGenModule &CGM);
  virtual ~CGMPtoGPURuntime() {}

  /// \brief Returns specified OpenMP to GPU runtime function.
  /// \param Function MPtoGPU runtime function.
  /// \return Specified function.
  llvm::Constant *CreateRuntimeFunction(MPtoGPURTLFunction Function);

  virtual llvm::Constant* CLdevice_init();
  virtual llvm::Constant* Set_default_device();
  virtual llvm::Constant* Get_num_devices();
  virtual llvm::Constant* Get_default_device();

};
  
/// \brief Returns an implementation of the OpenMP to GPU RTL for a given target
CGMPtoGPURuntime *CreateMPtoGPURuntime(CodeGenModule &CGM);

} // namespace CodeGen
} // namespace clang

#endif
