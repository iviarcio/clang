##===- lib/Transforms/Hello/Makefile -----------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../..
LIBRARYNAME := mptogpu
LOADABLE_MODULE = 1
LINK_LIBS_IN_SHARED = 1
SHARED_LIBRARY = 1

# Build and install this archive.                                                                                                                  
BUILD_ARCHIVE = 1

CFLAGS += -O2 -lm

include $(CLANG_LEVEL)/Makefile
