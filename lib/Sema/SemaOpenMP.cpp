//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file implements semantic analysis for OpenMP directives and
/// clauses.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/OpenMPKinds.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Stack of data-sharing attributes for variables
//===----------------------------------------------------------------------===//

namespace {
/// \brief Default data sharing attributes, which can be applied to directive.
enum DefaultDataSharingAttributes {
  DSA_unspecified = 0, /// \brief Data sharing attribute not specified.
  DSA_none = 1 << 0,   /// \brief Default data sharing attribute 'none'.
  DSA_shared = 1 << 1  /// \brief Default data sharing attribute 'shared'.
};

/// \brief Stack for tracking declarations used in OpenMP directives and
/// clauses and their data-sharing attributes.
class DSAStackTy {
public:
  struct MapInfo {
    Expr *RefExpr;
    bool IsCEAN;
  };

private:
  struct DSAInfo {
    OpenMPClauseKind Attributes;
    DeclRefExpr *RefExpr;
  };
  typedef llvm::SmallDenseMap<VarDecl *, DSAInfo, 64> DeclSAMapTy;
  typedef llvm::SmallDenseMap<VarDecl *, MapInfo, 64> MappedDeclsTy;

  struct SharingMapTy {
    DeclSAMapTy SharingMap;
    DeclSAMapTy AlignedMap;
    MappedDeclsTy MappedDecls;
    DefaultDataSharingAttributes DefaultAttr;
    OpenMPDirectiveKind Directive;
    DeclarationNameInfo DirectiveName;
    bool IsOrdered;
    bool IsNowait;
    Scope *CurScope;
    SharingMapTy(OpenMPDirectiveKind DKind, const DeclarationNameInfo &Name,
                 Scope *CurScope)
        : SharingMap(), AlignedMap(), MappedDecls(),
          DefaultAttr(DSA_unspecified), Directive(DKind), DirectiveName(Name),
          IsOrdered(false), IsNowait(false), CurScope(CurScope) {}
    SharingMapTy()
        : SharingMap(), AlignedMap(), MappedDecls(),
          DefaultAttr(DSA_unspecified), Directive(OMPD_unknown),
          DirectiveName(), IsOrdered(false), IsNowait(false), CurScope(0) {}
  };

  typedef SmallVector<SharingMapTy, 4> StackTy;

  /// \brief Stack of used declaration and their data-sharing attributes.
  StackTy Stack;
  Sema &Actions;

  typedef SmallVector<SharingMapTy, 4>::reverse_iterator reverse_iterator;

  typedef llvm::DenseSet<Decl *> DeclaredTargetDeclsTy;

  DeclaredTargetDeclsTy DeclaredTargetDecls;

  OpenMPClauseKind getDSA(StackTy::reverse_iterator Iter, VarDecl *D,
                          OpenMPDirectiveKind &Kind, DeclRefExpr *&E);
  /// \brief Checks if the variable is a local for OpenMP region.
  bool isOpenMPLocal(VarDecl *D, StackTy::reverse_iterator Iter);

public:
  DSAStackTy(Sema &S) : Stack(1), Actions(S) {}

  void push(OpenMPDirectiveKind DKind, const DeclarationNameInfo &DirName,
            Scope *CurScope) {
    Stack.push_back(SharingMapTy(DKind, DirName, CurScope));
  }

  void pop() {
    assert(Stack.size() > 1 && "Stack is empty!");
    Stack.pop_back();
  }

  /// \brief Adds unique 'aligned' declaration of a given VarDecl, or,
  /// if it already exists, returns false.
  bool addUniqueAligned(VarDecl *D, DeclRefExpr *&E);

  /// \brief Adds explicit data sharing attribute to the specified declaration.
  void addDSA(VarDecl *D, DeclRefExpr *E, OpenMPClauseKind A);

  /// \brief Adds explicit data sharing attribute to the specified declaration
  /// to parent scope.
  void addParentDSA(VarDecl *D, DeclRefExpr *E, OpenMPClauseKind A);

  bool IsThreadprivate(VarDecl *D, DeclRefExpr *&E);

  /// \brief Returns data sharing attributes from top of the stack for the
  /// specified declaration.
  OpenMPClauseKind getTopDSA(VarDecl *D, DeclRefExpr *&E);
  /// \brief Returns data-sharing attributes for the specified declaration.
  OpenMPClauseKind getImplicitDSA(VarDecl *D, OpenMPDirectiveKind &Kind,
                                  DeclRefExpr *&E);

  /// \brief Checks if the specified variables has \a CKind data-sharing
  /// attribute in \a DKind directive.
  bool hasDSA(VarDecl *D, OpenMPClauseKind CKind, OpenMPDirectiveKind DKind,
              DeclRefExpr *&E);

  /// \brief Checks if the specified variables has \a CKind data-sharing
  /// attribute in an innermost \a DKind directive.
  bool hasInnermostDSA(VarDecl *D, OpenMPClauseKind CKind,
                       OpenMPDirectiveKind DKind, DeclRefExpr *&E);

  /// \brief Returns currently analized directive.
  OpenMPDirectiveKind getCurrentDirective() const {
    return Stack.back().Directive;
  }

  /// \brief Returns parent directive.
  OpenMPDirectiveKind getParentDirective() const {
    if (Stack.size() > 2)
      return Stack[Stack.size() - 2].Directive;
    return OMPD_unknown;
  }

  /// \brief Returns true if region is an ordered parallel or
  /// worksharing region.
  bool isRegionOrdered() const {
    if (Stack.size() > 1)
      return Stack[Stack.size() - 1].IsOrdered;
    return false;
  }

  /// \brief Returns true if parent region is an ordered parallel or
  /// worksharing region.
  bool isParentRegionOrdered() const {
    if (Stack.size() > 2)
      return Stack[Stack.size() - 2].IsOrdered;
    return false;
  }

  /// \brief Marks current regions as ordered.
  void setRegionOrdered() { Stack.back().IsOrdered = true; }

  /// \brief Returns true if region has nowait clause.
  bool isRegionNowait() const {
    if (Stack.size() > 1)
      return Stack[Stack.size() - 1].IsNowait;
    return false;
  }

  /// \brief Marks current regions as nowait.
  void setRegionNowait() { Stack.back().IsNowait = true; }

  /// \brief Checks if the specified kind of directive with the given name
  /// already exists.
  bool hasDirectiveWithName(OpenMPDirectiveKind Kind,
                            DeclarationNameInfo DirName);

  /// \brief Checks if the specified kind of directive exists.
  bool hasDirective(OpenMPDirectiveKind Kind);

  /// \brief Set default data sharing attribute to none.
  void setDefaultDSANone() { Stack.back().DefaultAttr = DSA_none; }
  /// \brief Set default data sharing attribute to shared.
  void setDefaultDSAShared() { Stack.back().DefaultAttr = DSA_shared; }
  DefaultDataSharingAttributes getDefaultDSA() {
    return Stack.back().DefaultAttr;
  }

  Scope *getCurScope() { return Stack.back().CurScope; }

  DeclContext *GetOpenMPFunctionRegion();

  void addDeclareTargetDecl(Decl *D) { DeclaredTargetDecls.insert(D); }

  bool isDeclareTargetDecl(Decl *D) { return DeclaredTargetDecls.count(D); }

  MapInfo getMapInfoForVar(VarDecl *VD) {
    MapInfo Tmp = {0, false};
    for (unsigned Cnt = Stack.size() - 1; Cnt > 0; --Cnt) {
      if (Stack[Cnt].MappedDecls.count(VD)) {
        Tmp = Stack[Cnt].MappedDecls[VD];
        break;
      }
    }
    return Tmp;
  }

  void addMapInfoForVar(VarDecl *VD, MapInfo MI) {
    if (Stack.size() > 1) {
      Stack.back().MappedDecls[VD] = MI;
    }
  }

  MapInfo IsMappedInCurrentRegion(VarDecl *VD) {
    assert(Stack.size() > 1 && "Target level is 0");
    MapInfo Tmp = {0, false};
    if (Stack.size() > 1 && Stack.back().MappedDecls.count(VD)) {
      Tmp = Stack.back().MappedDecls[VD];
    }
    return Tmp;
  }
};
} // end anonymous namespace.

OpenMPClauseKind DSAStackTy::getDSA(StackTy::reverse_iterator Iter, VarDecl *D,
                                    OpenMPDirectiveKind &Kind,
                                    DeclRefExpr *&E) {
  E = 0;
  if (Iter == Stack.rend() - 1) {
    Kind = OMPD_unknown;
    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a region but not in construct]
    //  File-scope or namespace-scope variables referenced in called routines
    //  in the region are shared unless they appear in a threadprivate
    //  directive.
    if (!D->isFunctionOrMethodVarDecl() && D->getKind() != Decl::ParmVar)
      return OMPC_shared;

    // OpenMP [2.9.1.2, Data-sharing Attribute Rules for Variables Referenced
    // in a region but not in construct]
    //  Variables with static storage duration that are declared in called
    //  routines in the region are shared.
    if (D->hasGlobalStorage())
      return OMPC_shared;

    return OMPC_unknown;
  }
  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.1]
  // Variables with automatic storage duration that are declared in a scope
  // inside the construct are private.
  Kind = Iter->Directive;
  if (isOpenMPLocal(D, Iter) && D->isLocalVarDecl() &&
      (D->getStorageClass() == SC_Auto || D->getStorageClass() == SC_None))
    return OMPC_private;
  // Explicitly specified attributes and local variables with predetermined
  // attributes.
  if (Iter->SharingMap.count(D)) {
    E = Iter->SharingMap[D].RefExpr;
    return Iter->SharingMap[D].Attributes;
  }

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, implicitly determined, p.1]
  //  In a parallel or task construct, the data-sharing attributes of these
  //  variables are determined by the default clause, if present.
  switch (Iter->DefaultAttr) {
  case DSA_shared:
    return OMPC_shared;
  case DSA_none:
    return OMPC_unknown;
  case DSA_unspecified:
    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, implicitly determined, p.2]
    //  In a parallel construct, if no default clause is present, these
    //  variables are shared.
    if (Kind == OMPD_parallel || Kind == OMPD_teams ||
        Kind == OMPD_parallel_for || Kind == OMPD_parallel_for_simd ||
        Kind == OMPD_parallel_sections ||
        Kind == OMPD_distribute_parallel_for ||
        Kind == OMPD_distribute_parallel_for_simd ||
        Kind == OMPD_teams_distribute_parallel_for ||
        Kind == OMPD_teams_distribute_parallel_for_simd ||
        Kind == OMPD_target_teams_distribute_parallel_for ||
        Kind == OMPD_target_teams_distribute_parallel_for_simd ||
        Kind == OMPD_target_teams || Kind == OMPD_teams_distribute ||
        Kind == OMPD_teams_distribute_simd ||
        Kind == OMPD_target_teams_distribute ||
        Kind == OMPD_target_teams_distribute_simd)
      return OMPC_shared;

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, implicitly determined, p.4]
    //  In a task construct, if no default clause is present, a variable that in
    //  the enclosing context is determined to be shared by all implicit tasks
    //  bound to the current team is shared.
    if (Kind == OMPD_task) {
      OpenMPClauseKind CKind = OMPC_unknown;
      for (StackTy::reverse_iterator I = Iter + 1, EE = Stack.rend() - 1;
           I != EE; ++I) {
        // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables
        // Referenced
        // in a Construct, implicitly determined, p.6]
        //  In a task construct, if no default clause is present, a variable
        //  whose data-sharing attribute is not determined by the rules above is
        //  firstprivate.
        CKind = getDSA(I, D, Kind, E);
        if (CKind != OMPC_shared) {
          E = 0;
          Kind = OMPD_task;
          return OMPC_firstprivate;
        }
        if (I->Directive == OMPD_parallel ||
            I->Directive == OMPD_parallel_for ||
            I->Directive == OMPD_parallel_for_simd ||
            I->Directive == OMPD_parallel_sections ||
            I->Directive == OMPD_distribute_parallel_for ||
            I->Directive == OMPD_distribute_parallel_for_simd ||
            I->Directive == OMPD_teams_distribute_parallel_for ||
            I->Directive == OMPD_teams_distribute_parallel_for_simd ||
            I->Directive == OMPD_target_teams_distribute_parallel_for ||
            I->Directive == OMPD_target_teams_distribute_parallel_for_simd)
          break;
      }
      Kind = OMPD_task;
      return (CKind == OMPC_unknown) ? OMPC_firstprivate : OMPC_shared;
    }
  }
  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, implicitly determined, p.3]
  //  For constructs other than task, if no default clause is present, these
  //  variables inherit their data-sharing attributes from the enclosing
  //  context.
  return getDSA(Iter + 1, D, Kind, E);
}

bool DSAStackTy::addUniqueAligned(VarDecl *D, DeclRefExpr *&E) {
  assert(Stack.size() > 1 && "Data sharing attributes stack is empty");
  DeclSAMapTy::iterator It = Stack.back().AlignedMap.find(D);
  if (It == Stack.back().AlignedMap.end()) {
    Stack.back().AlignedMap[D].Attributes = OMPC_aligned;
    Stack.back().AlignedMap[D].RefExpr = E;
    return true;
  } else {
    assert(Stack.back().AlignedMap[D].Attributes == OMPC_aligned);
    E = Stack.back().AlignedMap[D].RefExpr;
    return false;
  }
}

void DSAStackTy::addDSA(VarDecl *D, DeclRefExpr *E, OpenMPClauseKind A) {
  if (A == OMPC_threadprivate) {
    Stack[0].SharingMap[D].Attributes = A;
    Stack[0].SharingMap[D].RefExpr = E;
  } else {
    assert(Stack.size() > 1 && "Data sharing attributes stack is empty");
    Stack.back().SharingMap[D].Attributes = A;
    Stack.back().SharingMap[D].RefExpr = E;
  }
}

void DSAStackTy::addParentDSA(VarDecl *D, DeclRefExpr *E, OpenMPClauseKind A) {
  assert(Stack.size() > 2 &&
         "Data sharing attributes stack does not have parent");
  Stack[Stack.size() - 2].SharingMap[D].Attributes = A;
  Stack[Stack.size() - 2].SharingMap[D].RefExpr = E;
}

bool DSAStackTy::isOpenMPLocal(VarDecl *D, StackTy::reverse_iterator Iter) {
  if (Stack.size() > 2) {
    reverse_iterator I = Iter, E = Stack.rend() - 1;
    Scope *TopScope = 0;
    while (I != E && I->Directive != OMPD_parallel &&
           I->Directive != OMPD_parallel_for &&
           I->Directive != OMPD_parallel_for_simd &&
           I->Directive != OMPD_parallel_sections &&
           I->Directive != OMPD_distribute_parallel_for &&
           I->Directive != OMPD_distribute_parallel_for_simd &&
           I->Directive != OMPD_teams_distribute_parallel_for &&
           I->Directive != OMPD_teams_distribute_parallel_for_simd &&
           I->Directive != OMPD_target_teams_distribute_parallel_for &&
           I->Directive != OMPD_target_teams_distribute_parallel_for_simd &&
           I->Directive != OMPD_task && I->Directive != OMPD_teams &&
           I->Directive != OMPD_target_teams &&
           I->Directive != OMPD_teams_distribute &&
           I->Directive != OMPD_teams_distribute_simd &&
           I->Directive != OMPD_target_teams_distribute &&
           I->Directive != OMPD_target_teams_distribute_simd) {
      ++I;
    }
    if (I == E)
      return false;
    TopScope = I->CurScope ? I->CurScope->getParent() : 0;
    Scope *CurScope = getCurScope();
    while (CurScope != TopScope && !CurScope->isDeclScope(D)) {
      CurScope = CurScope->getParent();
    }
    return CurScope != TopScope;
  }
  return false;
}

bool DSAStackTy::IsThreadprivate(VarDecl *D, DeclRefExpr *&E) {
  E = 0;
  if (D->getTLSKind() != VarDecl::TLS_None)
    return true;
  if (Stack[0].SharingMap.count(D)) {
    E = Stack[0].SharingMap[D].RefExpr;
    return true;
  }
  return false;
}

OpenMPClauseKind DSAStackTy::getTopDSA(VarDecl *D, DeclRefExpr *&E) {
  E = 0;

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.1]
  //  Variables appearing in threadprivate directives are threadprivate.
  if (IsThreadprivate(D, E))
    return OMPC_threadprivate;

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.1]
  // Variables with automatic storage duration that are declared in a scope
  // inside the construct are private.
  OpenMPDirectiveKind Kind = getCurrentDirective();
  if (Kind != OMPD_parallel && Kind != OMPD_parallel_for &&
      Kind != OMPD_parallel_for_simd && Kind != OMPD_distribute_parallel_for &&
      Kind != OMPD_distribute_parallel_for_simd &&
      Kind != OMPD_teams_distribute_parallel_for &&
      Kind != OMPD_teams_distribute_parallel_for_simd &&
      Kind != OMPD_target_teams_distribute_parallel_for &&
      Kind != OMPD_target_teams_distribute_parallel_for_simd &&
      Kind != OMPD_task && Kind != OMPD_teams &&
      Kind != OMPD_parallel_sections && Kind != OMPD_target_teams &&
      Kind != OMPD_teams_distribute && Kind != OMPD_teams_distribute_simd &&
      Kind != OMPD_target_teams_distribute &&
      Kind != OMPD_target_teams_distribute_simd) {
    if (isOpenMPLocal(D, Stack.rbegin() + 1) && D->isLocalVarDecl() &&
        (D->getStorageClass() == SC_Auto || D->getStorageClass() == SC_None))
      return OMPC_private;
  }

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.4]
  //  Static data memebers are shared.
  if (D->isStaticDataMember()) {
    DeclRefExpr *E;
    // Variables with const-qualified type having no mutable member may be
    // listed
    // in a firstprivate clause, even if they are static data members.
    if (hasDSA(D, OMPC_firstprivate, OMPD_unknown, E) && E)
      return OMPC_unknown;
    return OMPC_shared;
  }

  QualType Type = D->getType().getNonReferenceType().getCanonicalType();
  bool IsConstant = Type.isConstant(Actions.getASTContext());
  while (Type->isArrayType()) {
    QualType ElemType = cast<ArrayType>(Type.getTypePtr())->getElementType();
    Type = ElemType.getNonReferenceType().getCanonicalType();
  }
  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.6]
  //  Variables with const qualified type having no mutable member are
  //  shared.
  CXXRecordDecl *RD =
      Actions.getLangOpts().CPlusPlus ? Type->getAsCXXRecordDecl() : 0;
  if (IsConstant &&
      !(Actions.getLangOpts().CPlusPlus && RD && RD->hasMutableFields())) {
    DeclRefExpr *E;
    // Variables with const-qualified type having no mutable member may be
    // listed
    // in a firstprivate clause, even if they are static data members.
    if (hasDSA(D, OMPC_firstprivate, OMPD_unknown, E) && E)
      return OMPC_unknown;
    return OMPC_shared;
  }

  // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++, predetermined, p.7]
  //  Variables with static storage duration that are declared in a scope
  //  inside the construct are shared.
  if (D->isStaticLocal())
    return OMPC_shared;

  // Explicitly specified attributes and local variables with predetermined
  // attributes.
  if (Stack.back().SharingMap.count(D)) {
    E = Stack.back().SharingMap[D].RefExpr;
    return Stack.back().SharingMap[D].Attributes;
  }

  return OMPC_unknown;
}

OpenMPClauseKind DSAStackTy::getImplicitDSA(VarDecl *D,
                                            OpenMPDirectiveKind &Kind,
                                            DeclRefExpr *&E) {
  return getDSA(Stack.rbegin() + 1, D, Kind, E);
}

bool DSAStackTy::hasDSA(VarDecl *D, OpenMPClauseKind CKind,
                        OpenMPDirectiveKind DKind, DeclRefExpr *&E) {
  for (StackTy::reverse_iterator I = Stack.rbegin() + 1, EE = Stack.rend() - 1;
       I != EE; ++I) {
    if (DKind != OMPD_unknown && DKind != I->Directive)
      continue;
    OpenMPDirectiveKind K;
    if (getDSA(I, D, K, E) == CKind)
      return true;
  }
  E = 0;
  return false;
}

bool DSAStackTy::hasInnermostDSA(VarDecl *D, OpenMPClauseKind CKind,
                                 OpenMPDirectiveKind DKind, DeclRefExpr *&E) {
  assert(DKind != OMPD_unknown && "Directive must be specified explicitly");
  for (StackTy::reverse_iterator I = Stack.rbegin(), EE = Stack.rend() - 1;
       I != EE; ++I) {
    if (DKind != I->Directive)
      continue;
    if (getDSA(I, D, DKind, E) == CKind)
      return true;
    return false;
  }
  return false;
}

bool DSAStackTy::hasDirectiveWithName(OpenMPDirectiveKind Kind,
                                      DeclarationNameInfo DirName) {
  for (reverse_iterator I = Stack.rbegin() + 1, E = Stack.rend() - 1; I != E;
       ++I) {
    if (I->Directive == Kind &&
        !DeclarationName::compare(I->DirectiveName.getName(),
                                  DirName.getName()))
      return true;
  }
  return false;
}

bool DSAStackTy::hasDirective(OpenMPDirectiveKind Kind) {
  for (reverse_iterator I = Stack.rbegin(), E = Stack.rend() - 1; I != E; ++I) {
    if (I->Directive == Kind)
      return true;
  }
  return false;
}

DeclContext *DSAStackTy::GetOpenMPFunctionRegion() {
  for (reverse_iterator I = Stack.rbegin(), E = Stack.rend() - 1; I != E; ++I) {
    if (I->Directive == OMPD_parallel || I->Directive == OMPD_parallel_for ||
        I->Directive == OMPD_parallel_for_simd ||
        I->Directive == OMPD_distribute_parallel_for ||
        I->Directive == OMPD_distribute_parallel_for_simd ||
        I->Directive == OMPD_teams_distribute_parallel_for ||
        I->Directive == OMPD_teams_distribute_parallel_for_simd ||
        I->Directive == OMPD_target_teams_distribute_parallel_for ||
        I->Directive == OMPD_target_teams_distribute_parallel_for_simd ||
        I->Directive == OMPD_teams || I->Directive == OMPD_task ||
        I->Directive == OMPD_parallel_sections ||
        I->Directive == OMPD_target_teams ||
        I->Directive == OMPD_teams_distribute ||
        I->Directive == OMPD_teams_distribute_simd ||
        I->Directive == OMPD_target_teams_distribute ||
        I->Directive == OMPD_target_teams_distribute_simd)
      return I->CurScope->getEntity();
  }
  return 0;
}

void Sema::InitDataSharingAttributesStack() {
  VarDataSharingAttributesStack = new DSAStackTy(*this);
}

#define DSAStack static_cast<DSAStackTy *>(VarDataSharingAttributesStack)

void Sema::DestroyDataSharingAttributesStack() { delete DSAStack; }

bool Sema::HasOpenMPRegion(OpenMPDirectiveKind Kind) {
  return DSAStack->hasDirective(Kind);
}

bool Sema::HasOpenMPSimdRegion() {
  return HasOpenMPRegion(OMPD_simd) || HasOpenMPRegion(OMPD_for_simd) ||
         HasOpenMPRegion(OMPD_parallel_for_simd) ||
         HasOpenMPRegion(OMPD_distribute_simd) ||
         HasOpenMPRegion(OMPD_for_simd) ||
         HasOpenMPRegion(OMPD_distribute_parallel_for_simd) ||
         HasOpenMPRegion(OMPD_teams_distribute_parallel_for_simd) ||
         HasOpenMPRegion(OMPD_target_teams_distribute_parallel_for_simd) ||
         HasOpenMPRegion(OMPD_teams_distribute_simd) ||
         HasOpenMPRegion(OMPD_target_teams_distribute_simd);
}

bool Sema::IsDeclContextInOpenMPTarget(DeclContext *DC) {
  while (DC && !isa<OMPDeclareTargetDecl>(DC)) {
    DC = DC->getParent();
  }
  return DC != 0;
}

DeclContext *Sema::GetOpenMPFunctionRegion() {
  return DSAStack->GetOpenMPFunctionRegion();
}

void Sema::StartOpenMPDSABlock(OpenMPDirectiveKind DKind,
                               const DeclarationNameInfo &DirName,
                               Scope *CurScope) {
  DSAStack->push(DKind, DirName, CurScope);

  PushExpressionEvaluationContext(PotentiallyEvaluated);
}

void Sema::EndOpenMPDSABlock(Stmt *CurDirective) {
  //  if (!getCurScope()->isOpenMPDirectiveScope()) return;
  // OpenMP [2.9.3.5, Restrictions, C/C++, p.1]
  //  A variable of class type (or array thereof) that appears in a lastprivate
  //  clause requires an accessible, unambiguous default constructor for the
  //  class type, unless the list item is also specified in a firstprivate
  //  clause.

  if (OMPExecutableDirective *D =
          dyn_cast_or_null<OMPExecutableDirective>(CurDirective)) {
    for (ArrayRef<OMPClause *>::iterator I = D->clauses().begin(),
                                         E = D->clauses().end();
         I != E; ++I) {
      if (OMPLastPrivateClause *Clause = dyn_cast<OMPLastPrivateClause>(*I)) {
        SmallVector<Expr *, 4> DefaultInits;
        ArrayRef<Expr *>::iterator PVIter = Clause->getPseudoVars1().begin();
        for (OMPLastPrivateClause::varlist_iterator
                 VI = Clause->varlist_begin(),
                 VE = Clause->varlist_end();
             VI != VE; ++VI, ++PVIter) {
          if ((*VI)->isValueDependent() || (*VI)->isTypeDependent() ||
              (*VI)->isInstantiationDependent() ||
              (*VI)->containsUnexpandedParameterPack()) {
            DefaultInits.push_back(0);
            continue;
          }
          DeclRefExpr *DE;
          VarDecl *VD = cast<VarDecl>(cast<DeclRefExpr>(*VI)->getDecl());
          QualType Type = (*VI)->getType().getCanonicalType();
          if (DSAStack->getTopDSA(VD, DE) == OMPC_lastprivate) {
            SourceLocation ELoc = (*VI)->getExprLoc();
            while (Type->isArrayType()) {
              QualType ElemType =
                  cast<ArrayType>(Type.getTypePtr())->getElementType();
              Type = ElemType.getNonReferenceType().getCanonicalType();
            }
            CXXRecordDecl *RD =
                getLangOpts().CPlusPlus ? Type->getAsCXXRecordDecl() : 0;
            if (RD) {
              CXXConstructorDecl *CD = LookupDefaultConstructor(RD);
              PartialDiagnostic PD =
                  PartialDiagnostic(PartialDiagnostic::NullDiagnostic());
              if (!CD ||
                  CheckConstructorAccess(
                      ELoc, CD, InitializedEntity::InitializeTemporary(Type),
                      CD->getAccess(), PD) == AR_inaccessible ||
                  CD->isDeleted()) {
                Diag(ELoc, diag::err_omp_required_method)
                    << getOpenMPClauseName(OMPC_lastprivate) << 0;
                bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                              VarDecl::DeclarationOnly;
                Diag(VD->getLocation(), IsDecl ? diag::note_previous_decl
                                               : diag::note_defined_here)
                    << VD;
                Diag(RD->getLocation(), diag::note_previous_decl) << RD;
                continue;
              }
              MarkFunctionReferenced(ELoc, CD);
              DiagnoseUseOfDecl(CD, ELoc);
            }
            VD = cast<VarDecl>(cast<DeclRefExpr>(*PVIter)->getDecl());
            InitializedEntity Entity =
                InitializedEntity::InitializeVariable(VD);
            InitializationKind InitKind =
                InitializationKind::CreateDefault(ELoc);
            InitializationSequence InitSeq(*this, Entity, InitKind,
                                           MultiExprArg());
            ExprResult Res =
                InitSeq.Perform(*this, Entity, InitKind, MultiExprArg());
            if (Res.isInvalid())
              continue;
            DefaultInits.push_back(ActOnFinishFullExpr(Res.get()).get());
          } else {
            DefaultInits.push_back(0);
          }
        }
        if (DefaultInits.size() == Clause->numberOfVariables())
          Clause->setDefaultInits(DefaultInits);
      }
    }
  }

  DSAStack->pop();
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();
}

namespace {
class VarDeclFilterCCC : public CorrectionCandidateCallback {
private:
  Sema &Actions;

public:
  VarDeclFilterCCC(Sema &S) : Actions(S) {}
  virtual bool ValidateCandidate(const TypoCorrection &Candidate) {
    NamedDecl *ND = Candidate.getCorrectionDecl();
    if (VarDecl *VD = dyn_cast_or_null<VarDecl>(ND)) {
      return VD->hasGlobalStorage() &&
             Actions.isDeclInScope(ND, Actions.getCurLexicalContext(),
                                   Actions.getCurScope());
    }
    return false;
  }
};
}

ExprResult Sema::ActOnOpenMPIdExpression(Scope *CurScope,
                                         CXXScopeSpec &ScopeSpec,
                                         const DeclarationNameInfo &Id) {
  LookupResult Lookup(*this, Id, LookupOrdinaryName);
  LookupParsedName(Lookup, CurScope, &ScopeSpec, true);

  if (Lookup.isAmbiguous())
    return ExprError();

  VarDecl *VD;
  if (!Lookup.isSingleResult()) {
    VarDeclFilterCCC Validator(*this);
    if (TypoCorrection Corrected =
            CorrectTypo(Id, LookupOrdinaryName, CurScope, nullptr, Validator,
                        CTK_ErrorRecovery)) {
      diagnoseTypo(Corrected,
                   PDiag(Lookup.empty()
                             ? diag::err_undeclared_var_use_suggest
                             : diag::err_omp_expected_var_arg_suggest)
                       << Id.getName());
      VD = Corrected.getCorrectionDeclAs<VarDecl>();
    } else {
      Diag(Id.getLoc(), Lookup.empty() ? diag::err_undeclared_var_use
                                       : diag::err_omp_expected_var_arg)
          << Id.getName();
      return ExprError();
    }
  } else {
    if (!(VD = Lookup.getAsSingle<VarDecl>())) {
      Diag(Id.getLoc(), diag::err_omp_expected_var_arg) << Id.getName();
      Diag(Lookup.getFoundDecl()->getLocation(), diag::note_declared_at);
      return ExprError();
    }
  }
  Lookup.suppressDiagnostics();

  // OpenMP [2.9.2, Syntax, C/C++]
  //   Variables must be file-scope, namespace-scope, or static block-scope.
  if (!VD->hasGlobalStorage()) {
    Diag(Id.getLoc(), diag::err_omp_global_var_arg)
        << getOpenMPDirectiveName(OMPD_threadprivate) << !VD->isStaticLocal();
    bool IsDecl =
        VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here)
        << VD;
    return ExprError();
  }

  // OpenMP [2.9.2, Restrictions, C/C++, p.2]
  //   A threadprivate directive for file-scope variables must appear outside
  //   any definition or declaration.
  // OpenMP [2.9.2, Restrictions, C/C++, p.3]
  //   A threadprivate directive for static class member variables must appear
  //   in the class definition, in the same scope in which the member
  //   variables are declared.
  // OpenMP [2.9.2, Restrictions, C/C++, p.4]
  //   A threadprivate directive for namespace-scope variables must appear
  //   outside any definition or declaration other than the namespace
  //   definition itself.
  // OpenMP [2.9.2, Restrictions, C/C++, p.6]
  //   A threadprivate directive for static block-scope variables must appear
  //   in the scope of the variable and not in a nested scope.
  NamedDecl *ND = cast<NamedDecl>(VD);
  if ((!getCurLexicalContext()->isFileContext() ||
       !VD->getDeclContext()->isFileContext()) &&
      !isDeclInScope(ND, getCurLexicalContext(), getCurScope())) {
    Diag(Id.getLoc(), diag::err_omp_var_scope)
        << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
    bool IsDecl =
        VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
    Diag(VD->getLocation(),
         IsDecl ? diag::note_previous_decl : diag::note_defined_here)
        << VD;
    return ExprError();
  }

  // OpenMP [2.9.2, Restrictions, C/C++, p.2-6]
  //   A threadprivate directive must lexically precede all references to any
  //   of the variables in its list.
  if (VD->isUsed()) {
    Diag(Id.getLoc(), diag::err_omp_var_used)
        << getOpenMPDirectiveName(OMPD_threadprivate) << VD;
    return ExprError();
  }

  QualType ExprType = VD->getType().getNonReferenceType();
  ExprResult DE = BuildDeclRefExpr(VD, ExprType, VK_LValue, Id.getLoc());
  return DE;
}

Sema::DeclGroupPtrTy
Sema::ActOnOpenMPThreadprivateDirective(SourceLocation Loc,
                                        ArrayRef<Expr *> VarList) {
  if (OMPThreadPrivateDecl *D = CheckOMPThreadPrivateDecl(Loc, VarList)) {
    D->setAccess(AS_public);
    CurContext->addDecl(D);
    return DeclGroupPtrTy::make(DeclGroupRef(D));
  }
  return DeclGroupPtrTy();
}

OMPThreadPrivateDecl *
Sema::CheckOMPThreadPrivateDecl(SourceLocation Loc, ArrayRef<Expr *> VarList) {
  SmallVector<Expr *, 4> Vars;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    DeclRefExpr *DE = cast<DeclRefExpr>(*I);
    VarDecl *VD = cast<VarDecl>(DE->getDecl());
    SourceLocation ILoc = DE->getExprLoc();

    // OpenMP [2.9.2, Restrictions, C/C++, p.10]
    //   A threadprivate variable must not have an incomplete type.
    if (RequireCompleteType(ILoc, VD->getType(),
                            diag::err_omp_threadprivate_incomplete_type)) {
      continue;
    }

    // OpenMP [2.9.2, Restrictions, C/C++, p.10]
    //   A threadprivate variable must not have a reference type.
    if (VD->getType()->isReferenceType()) {
      Diag(ILoc, diag::err_omp_ref_type_arg)
          << getOpenMPDirectiveName(OMPD_threadprivate) << VD->getType();
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // Check if this is a TLS variable.
    if (VD->getTLSKind()) {
      Diag(ILoc, diag::err_omp_var_thread_local) << VD;
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    QualType Type = VD->getType().getNonReferenceType().getCanonicalType();
    while (Type->isArrayType()) {
      QualType ElemType = cast<ArrayType>(Type.getTypePtr())->getElementType();
      Type = ElemType.getNonReferenceType().getCanonicalType();
    }
    CXXRecordDecl *RD =
        getLangOpts().CPlusPlus ? Type->getAsCXXRecordDecl() : 0;
    if (RD) {
      SourceLocation ELoc = (*I)->getExprLoc();
      CXXDestructorDecl *DD = RD->getDestructor();
      PartialDiagnostic PD =
          PartialDiagnostic(PartialDiagnostic::NullDiagnostic());
      if (DD && (CheckDestructorAccess(ELoc, DD, PD) == AR_inaccessible ||
                 DD->isDeleted())) {
        Diag(ELoc, diag::err_omp_required_method)
            << getOpenMPClauseName(OMPC_threadprivate) << 4;
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
        Diag(RD->getLocation(), diag::note_previous_decl) << RD;
        continue;
      } else if (DD) {
        MarkFunctionReferenced(ELoc, DD);
        DiagnoseUseOfDecl(DD, ELoc);
      }
    }

    DSAStack->addDSA(VD, DE, OMPC_threadprivate);
    Vars.push_back(*I);
  }
  return Vars.empty() ? 0 : OMPThreadPrivateDecl::Create(
                                Context, getCurLexicalContext(), Loc, Vars);
}

Sema::DeclGroupPtrTy Sema::ActOnOpenMPDeclareSimdDirective(
    SourceLocation Loc, Decl *FuncDecl, ArrayRef<SourceRange> SrcRanges,
    ArrayRef<unsigned> BeginIdx, ArrayRef<unsigned> EndIdx,
    ArrayRef<OMPClause *> CL) {
  DeclContext *CurDC = getCurLexicalContext();
  if (OMPDeclareSimdDecl *D = CheckOMPDeclareSimdDecl(
          Loc, FuncDecl, SrcRanges, BeginIdx, EndIdx, CL, CurDC)) {
    D->setAccess(AS_public);
    CurContext->addDecl(D);
    if (FunctionTemplateDecl *FTDecl =
            dyn_cast<FunctionTemplateDecl>(FuncDecl)) {
      OMPDSimdMap[FTDecl] = D;
    }
    return DeclGroupPtrTy::make(DeclGroupRef(D));
  }
  return DeclGroupPtrTy();
}

OMPDeclareSimdDecl *Sema::CheckOMPDeclareSimdDecl(
    SourceLocation Loc, Decl *FuncDecl, ArrayRef<SourceRange> SrcRanges,
    ArrayRef<unsigned> BeginIdx, ArrayRef<unsigned> EndIdx,
    ArrayRef<OMPClause *> CL, DeclContext *CurDC) {
  // Checks the clauses and their arguments.
  //
  typedef llvm::SmallDenseMap<VarDecl *, SourceLocation, 64> SeenVarMap;
  SeenVarMap SeenVarsLinear, SeenVarsAligned;
  // Build NewBeginIdx/NewEndIdx to remove the dead (NULL) clauses.
  //
  SmallVector<unsigned, 4> NewBeginIdx;
  SmallVector<unsigned, 4> NewEndIdx;
  SmallVector<OMPClause *, 4> NewCL;
  unsigned NumDeadClauses = 0;
  for (unsigned J = 0; J < BeginIdx.size(); ++J) {
    unsigned BeginI = BeginIdx[J];
    unsigned EndI = EndIdx[J];
    SeenVarsLinear.clear();
    SeenVarsAligned.clear();
    bool hasInBranch = false;
    bool hasNotInBranch = false;
    SourceLocation PrevLocInBranch;
    NewBeginIdx.push_back(BeginI - NumDeadClauses);
    // Walk the current variant's clauses.
    for (unsigned Idx = BeginI; Idx < EndI; ++Idx) {
      OMPClause *Clause = CL[Idx];
      if (OMPUniformClause *C = dyn_cast_or_null<OMPUniformClause>(Clause)) {
        for (OMPUniformClause::varlist_iterator I = C->varlist_begin(),
                                                E = C->varlist_end();
             I != E; ++I) {
          DeclRefExpr *DE = cast<DeclRefExpr>(*I);
          VarDecl *VD = cast<VarDecl>(DE->getDecl());
          SeenVarMap::iterator SVI = SeenVarsLinear.find(VD);
          if (SVI != SeenVarsLinear.end()) {
            Diag(DE->getLocation(),
                 diag::err_omp_at_most_one_uniform_or_linear);
            Diag(SVI->second, diag::note_omp_referenced);
          } else {
            SeenVarsLinear.insert(std::make_pair(VD, DE->getLocation()));
          }
        }
      } else if (OMPLinearClause *C =
                     dyn_cast_or_null<OMPLinearClause>(Clause)) {
        for (OMPLinearClause::varlist_iterator I = C->varlist_begin(),
                                               E = C->varlist_end();
             I != E; ++I) {
          DeclRefExpr *DE = cast<DeclRefExpr>(*I);
          VarDecl *VD = cast<VarDecl>(DE->getDecl());
          SeenVarMap::iterator SVI = SeenVarsLinear.find(VD);
          if (SVI != SeenVarsLinear.end()) {
            Diag(DE->getLocation(),
                 diag::err_omp_at_most_one_uniform_or_linear);
            Diag(SVI->second, diag::note_omp_referenced);
          } else {
            SeenVarsLinear.insert(std::make_pair(VD, DE->getLocation()));
          }
        }
      } else if (OMPAlignedClause *C =
                     dyn_cast_or_null<OMPAlignedClause>(Clause)) {
        for (OMPAlignedClause::varlist_iterator I = C->varlist_begin(),
                                                E = C->varlist_end();
             I != E; ++I) {
          DeclRefExpr *DE = cast<DeclRefExpr>(*I);
          VarDecl *VD = cast<VarDecl>(DE->getDecl());
          SeenVarMap::iterator SVI = SeenVarsAligned.find(VD);
          if (SVI != SeenVarsAligned.end()) {
            Diag(DE->getLocation(), diag::err_omp_at_most_one_aligned);
            Diag(SVI->second, diag::note_omp_referenced);
          } else {
            SeenVarsAligned.insert(std::make_pair(VD, DE->getLocation()));
          }
        }
      } else if (OMPInBranchClause *C =
                     dyn_cast_or_null<OMPInBranchClause>(Clause)) {
        if (hasNotInBranch) {
          Diag(C->getLocStart(), diag::err_omp_inbranch);
          Diag(PrevLocInBranch, diag::note_omp_specified);
          Clause = 0;
        }
        hasInBranch = true;
        PrevLocInBranch = C->getLocStart();
      } else if (OMPNotInBranchClause *C =
                     dyn_cast_or_null<OMPNotInBranchClause>(Clause)) {
        if (hasInBranch) {
          Diag(C->getLocStart(), diag::err_omp_inbranch);
          Diag(PrevLocInBranch, diag::note_omp_specified);
          Clause = 0;
        }
        hasNotInBranch = true;
        PrevLocInBranch = C->getLocStart();
      }
      if (Clause == 0) {
        ++NumDeadClauses;
      } else {
        NewCL.push_back(Clause);
      }
    }
    NewEndIdx.push_back(EndI - NumDeadClauses);
  }

  OMPDeclareSimdDecl *D = OMPDeclareSimdDecl::Create(
      Context, CurDC, Loc, FuncDecl, SrcRanges.size(), NewCL);
  CompleteOMPDeclareSimdDecl(D, SrcRanges, NewBeginIdx, NewEndIdx);
  return D;
}

void Sema::CompleteOMPDeclareSimdDecl(OMPDeclareSimdDecl *D,
                                      ArrayRef<SourceRange> SrcRanges,
                                      ArrayRef<unsigned> BeginIdx,
                                      ArrayRef<unsigned> EndIdx) {
  SmallVector<OMPDeclareSimdDecl::SimdVariant, 4> Data;
  ArrayRef<SourceRange>::iterator IS = SrcRanges.begin();
  ArrayRef<unsigned>::iterator IB = BeginIdx.begin();
  ArrayRef<unsigned>::iterator IE = EndIdx.begin();
  for (ArrayRef<SourceRange>::iterator ES = SrcRanges.end(); IS != ES;
       ++IS, ++IB, ++IE) {
    Data.push_back(OMPDeclareSimdDecl::SimdVariant(*IS, *IB, *IE));
  }
  D->setVariants(Data);
}

OMPDeclareReductionDecl *Sema::OMPDeclareReductionRAII::InitDeclareReduction(
    Scope *CS, DeclContext *DC, SourceLocation Loc, DeclarationName Name,
    unsigned NumTypes, AccessSpecifier AS) {
  OMPDeclareReductionDecl *D =
      OMPDeclareReductionDecl::Create(S.Context, DC, Loc, Name, NumTypes);
  if (CS)
    S.PushOnScopeChains(D, CS);
  else
    DC->addDecl(D);
  D->setAccess(AS);
  return D;
}

Decl *Sema::OMPDeclareReductionRAII::getDecl() { return D; }

Sema::OMPDeclareReductionRAII::OMPDeclareReductionRAII(
    Sema &S, Scope *CS, DeclContext *DC, SourceLocation Loc, DeclarationName DN,
    unsigned NumTypes, AccessSpecifier AS)
    : S(S), D(InitDeclareReduction(CS, DC, Loc, DN, NumTypes, AS)),
      SavedContext(S, D) {}

OMPDeclareScanDecl *Sema::OMPDeclareScanRAII::InitDeclareScan(
        Scope *CS, DeclContext *DC, SourceLocation Loc, DeclarationName Name,
        unsigned NumTypes, AccessSpecifier AS) {
    OMPDeclareScanDecl *D =
            OMPDeclareScanDecl::Create(S.Context, DC, Loc, Name, NumTypes);
    if (CS)
        S.PushOnScopeChains(D, CS);
    else
        DC->addDecl(D);
    D->setAccess(AS);
    return D;
}

Decl *Sema::OMPDeclareScanRAII::getDecl() { return D; }

Sema::OMPDeclareScanRAII::OMPDeclareScanRAII(
        Sema &S, Scope *CS, DeclContext *DC, SourceLocation Loc, DeclarationName DN,
        unsigned NumTypes, AccessSpecifier AS)
        : S(S), D(InitDeclareScan(CS, DC, Loc, DN, NumTypes, AS)),
          SavedContext(S, D) {}

FunctionDecl *Sema::OMPDeclareReductionFunctionScope::ActOnOMPDeclareReductionFunction(
        Sema &S, SourceLocation Loc, DeclarationName Name, QualType QTy) {
    QualType PtrQTy = S.Context.getPointerType(QTy);
    QualType Args[] = {PtrQTy, PtrQTy};
    FunctionProtoType::ExtProtoInfo EPI;
    QualType FuncType = S.Context.getFunctionType(S.Context.VoidTy, Args, EPI);
    TypeSourceInfo *TI = S.Context.getTrivialTypeSourceInfo(FuncType);
    FunctionTypeLoc FTL = TI->getTypeLoc().getAs<FunctionTypeLoc>();
    FunctionDecl *FD =
            FunctionDecl::Create(S.Context, S.CurContext, Loc, Loc, Name, FuncType,
                                 TI, SC_PrivateExtern, false, false);
    FD->setImplicit();
    S.CurContext->addDecl(FD);
    if (S.CurContext->isDependentContext()) {
        DeclContext *DC = S.CurContext->getParent();
        TemplateParameterList *TPL = 0;
        if (ClassTemplatePartialSpecializationDecl *CTPSD =
                dyn_cast<ClassTemplatePartialSpecializationDecl>(DC)) {
            TPL = CTPSD->getTemplateParameters();
        } else if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(DC)) {
            TPL = RD->getDescribedClassTemplate()
                    ->getCanonicalDecl()
                    ->getTemplateParameters();
        } else if (FunctionDecl *RD = dyn_cast<FunctionDecl>(DC)) {
            TPL = RD->getDescribedFunctionTemplate()
                    ->getCanonicalDecl()
                    ->getTemplateParameters();
        }
        FunctionTemplateDecl *FTD = FunctionTemplateDecl::Create(
                S.Context, S.CurContext, Loc, Name, TPL, FD);
        FD->setDescribedFunctionTemplate(FTD);
    }
    ParLHS = ParmVarDecl::Create(S.Context, FD, Loc, Loc, 0, PtrQTy,
                                 S.Context.getTrivialTypeSourceInfo(PtrQTy),
                                 SC_None, 0);
    ParLHS->setScopeInfo(0, 0);
    ParRHS = ParmVarDecl::Create(S.Context, FD, Loc, Loc, 0, PtrQTy,
                                 S.Context.getTrivialTypeSourceInfo(PtrQTy),
                                 SC_None, 0);
    ParRHS->setScopeInfo(0, 1);
    ParmVarDecl *Params[] = {ParLHS, ParRHS};
    FD->setParams(Params);
    FTL.setParam(0, ParLHS);
    FTL.setParam(1, ParRHS);
    OmpIn =
            VarDecl::Create(S.Context, FD, Loc, Loc, &S.Context.Idents.get("omp_in"),
                            QTy, S.Context.getTrivialTypeSourceInfo(QTy), SC_Auto);
    OmpOut =
            VarDecl::Create(S.Context, FD, Loc, Loc, &S.Context.Idents.get("omp_out"),
                            QTy, S.Context.getTrivialTypeSourceInfo(QTy), SC_Auto);
    S.AddKnownFunctionAttributes(FD);
    if (S.CurScope) {
        S.PushFunctionScope();
        S.PushDeclContext(S.CurScope, FD);
        S.PushOnScopeChains(OmpOut, S.CurScope);
        S.PushOnScopeChains(OmpIn, S.CurScope);
        S.PushExpressionEvaluationContext(PotentiallyEvaluated);
    } else {
        S.CurContext = FD;
        FD->addDecl(OmpIn);
        FD->addDecl(OmpOut);
    }
    ExprResult LHS =
            S.BuildDeclRefExpr(ParLHS, ParLHS->getType(), VK_LValue, Loc);
    ExprResult RHS =
            S.BuildDeclRefExpr(ParRHS, ParRHS->getType(), VK_LValue, Loc);
    LHS = S.DefaultLvalueConversion(LHS.get());
    RHS = S.DefaultLvalueConversion(RHS.get());
    LHS = S.CreateBuiltinUnaryOp(Loc, UO_Deref, LHS.get());
    RHS = S.CreateBuiltinUnaryOp(Loc, UO_Deref, RHS.get());
    LHS = S.ActOnFinishFullExpr(LHS.get());
    RHS = S.ActOnFinishFullExpr(RHS.get());
    S.AddInitializerToDecl(OmpOut, LHS.get(), true, false);
    S.AddInitializerToDecl(OmpIn, RHS.get(), true, false);
    return FD;
}

FunctionDecl *Sema::OMPDeclareScanFunctionScope::ActOnOMPDeclareScanFunction(
        Sema &S, SourceLocation Loc, DeclarationName Name, QualType QTy) {
  QualType PtrQTy = S.Context.getPointerType(QTy);
  QualType Args[] = {PtrQTy, PtrQTy};
  FunctionProtoType::ExtProtoInfo EPI;
  QualType FuncType = S.Context.getFunctionType(S.Context.VoidTy, Args, EPI);
  TypeSourceInfo *TI = S.Context.getTrivialTypeSourceInfo(FuncType);
  FunctionTypeLoc FTL = TI->getTypeLoc().getAs<FunctionTypeLoc>();
  FunctionDecl *FD =
          FunctionDecl::Create(S.Context, S.CurContext, Loc, Loc, Name, FuncType,
                               TI, SC_PrivateExtern, false, false);
  FD->setImplicit();
  S.CurContext->addDecl(FD);
  if (S.CurContext->isDependentContext()) {
    DeclContext *DC = S.CurContext->getParent();
    TemplateParameterList *TPL = 0;
    if (ClassTemplatePartialSpecializationDecl *CTPSD =
            dyn_cast<ClassTemplatePartialSpecializationDecl>(DC)) {
      TPL = CTPSD->getTemplateParameters();
    } else if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(DC)) {
      TPL = RD->getDescribedClassTemplate()
              ->getCanonicalDecl()
              ->getTemplateParameters();
    } else if (FunctionDecl *RD = dyn_cast<FunctionDecl>(DC)) {
      TPL = RD->getDescribedFunctionTemplate()
              ->getCanonicalDecl()
              ->getTemplateParameters();
    }
    FunctionTemplateDecl *FTD = FunctionTemplateDecl::Create(
            S.Context, S.CurContext, Loc, Name, TPL, FD);
    FD->setDescribedFunctionTemplate(FTD);
  }
  ParLHS = ParmVarDecl::Create(S.Context, FD, Loc, Loc, 0, PtrQTy,
                               S.Context.getTrivialTypeSourceInfo(PtrQTy),
                               SC_None, 0);
  ParLHS->setScopeInfo(0, 0);
  ParRHS = ParmVarDecl::Create(S.Context, FD, Loc, Loc, 0, PtrQTy,
                               S.Context.getTrivialTypeSourceInfo(PtrQTy),
                               SC_None, 0);
  ParRHS->setScopeInfo(0, 1);
  ParmVarDecl *Params[] = {ParLHS, ParRHS};
  FD->setParams(Params);
  FTL.setParam(0, ParLHS);
  FTL.setParam(1, ParRHS);
  OmpIn =
          VarDecl::Create(S.Context, FD, Loc, Loc, &S.Context.Idents.get("omp_in"),
                          QTy, S.Context.getTrivialTypeSourceInfo(QTy), SC_Auto);
  OmpOut =
          VarDecl::Create(S.Context, FD, Loc, Loc, &S.Context.Idents.get("omp_out"),
                          QTy, S.Context.getTrivialTypeSourceInfo(QTy), SC_Auto);
  S.AddKnownFunctionAttributes(FD);
  if (S.CurScope) {
    S.PushFunctionScope();
    S.PushDeclContext(S.CurScope, FD);
    S.PushOnScopeChains(OmpOut, S.CurScope);
    S.PushOnScopeChains(OmpIn, S.CurScope);
    S.PushExpressionEvaluationContext(PotentiallyEvaluated);
  } else {
    S.CurContext = FD;
    FD->addDecl(OmpIn);
    FD->addDecl(OmpOut);
  }
  ExprResult LHS =
          S.BuildDeclRefExpr(ParLHS, ParLHS->getType(), VK_LValue, Loc);
  ExprResult RHS =
          S.BuildDeclRefExpr(ParRHS, ParRHS->getType(), VK_LValue, Loc);
  LHS = S.DefaultLvalueConversion(LHS.get());
  RHS = S.DefaultLvalueConversion(RHS.get());
  LHS = S.CreateBuiltinUnaryOp(Loc, UO_Deref, LHS.get());
  RHS = S.CreateBuiltinUnaryOp(Loc, UO_Deref, RHS.get());
  LHS = S.ActOnFinishFullExpr(LHS.get());
  RHS = S.ActOnFinishFullExpr(RHS.get());
  S.AddInitializerToDecl(OmpOut, LHS.get(), true, false);
  S.AddInitializerToDecl(OmpIn, RHS.get(), true, false);
  return FD;
}

void Sema::OMPDeclareReductionFunctionScope::setBody(Expr *E) {
  if (!E) {
    FD->setBody(S.ActOnNullStmt(SourceLocation()).get());
    FD->setInvalidDecl();
    return;
  }
  StmtResult S1 = S.ActOnDeclStmt(DeclGroupPtrTy::make(DeclGroupRef(OmpIn)),
                                  E->getExprLoc(), E->getExprLoc());
  StmtResult S2 = S.ActOnDeclStmt(DeclGroupPtrTy::make(DeclGroupRef(OmpOut)),
                                  E->getExprLoc(), E->getExprLoc());
  ExprResult S3 = S.IgnoredValueConversions(E);
  ExprResult LHS =
      S.BuildDeclRefExpr(ParLHS, ParLHS->getType(), VK_LValue, E->getExprLoc());
  LHS = S.DefaultLvalueConversion(LHS.get());
  LHS = S.CreateBuiltinUnaryOp(E->getExprLoc(), UO_Deref, LHS.get());
  ExprResult RHS =
      S.BuildDeclRefExpr(OmpOut, OmpOut->getType(), VK_LValue, E->getExprLoc());
  ExprResult Res =
      S.BuildBinOp(0, E->getExprLoc(), BO_Assign, LHS.get(), RHS.get());
  ExprResult S4 = S.IgnoredValueConversions(Res.get());
  if (S1.isInvalid() || S2.isInvalid() || S3.isInvalid() || S4.isInvalid()) {
    FD->setBody(S.ActOnNullStmt(SourceLocation()).get());
    FD->setInvalidDecl();
  } else {
    CompoundScopeRAII CompoundScope(S);
    Stmt *Stmts[] = {S1.get(), S2.get(), S3.get(), S4.get()};
    StmtResult Body =
        S.ActOnCompoundStmt(E->getExprLoc(), E->getExprLoc(), Stmts, false);
    FD->setBody(Body.get());
  }
}

void Sema::OMPDeclareScanFunctionScope::setBody(Expr *E) {
    if (!E) {
        FD->setBody(S.ActOnNullStmt(SourceLocation()).get());
        FD->setInvalidDecl();
        return;
    }
    StmtResult S1 = S.ActOnDeclStmt(DeclGroupPtrTy::make(DeclGroupRef(OmpIn)),
                                    E->getExprLoc(), E->getExprLoc());
    StmtResult S2 = S.ActOnDeclStmt(DeclGroupPtrTy::make(DeclGroupRef(OmpOut)),
                                    E->getExprLoc(), E->getExprLoc());
    ExprResult S3 = S.IgnoredValueConversions(E);
    ExprResult LHS =
            S.BuildDeclRefExpr(ParLHS, ParLHS->getType(), VK_LValue, E->getExprLoc());
    LHS = S.DefaultLvalueConversion(LHS.get());
    LHS = S.CreateBuiltinUnaryOp(E->getExprLoc(), UO_Deref, LHS.get());
    ExprResult RHS =
            S.BuildDeclRefExpr(OmpOut, OmpOut->getType(), VK_LValue, E->getExprLoc());
    ExprResult Res =
            S.BuildBinOp(0, E->getExprLoc(), BO_Assign, LHS.get(), RHS.get());
    ExprResult S4 = S.IgnoredValueConversions(Res.get());
    if (S1.isInvalid() || S2.isInvalid() || S3.isInvalid() || S4.isInvalid()) {
        FD->setBody(S.ActOnNullStmt(SourceLocation()).get());
        FD->setInvalidDecl();
    } else {
        CompoundScopeRAII CompoundScope(S);
        Stmt *Stmts[] = {S1.get(), S2.get(), S3.get(), S4.get()};
        StmtResult Body =
                S.ActOnCompoundStmt(E->getExprLoc(), E->getExprLoc(), Stmts, false);
        FD->setBody(Body.get());
    }
}

Expr *Sema::OMPDeclareReductionFunctionScope::getCombiner() {
  ExprResult Res =
      S.BuildDeclRefExpr(FD, FD->getType(), VK_LValue, FD->getLocation());
  return Res.get();
}

Expr *Sema::OMPDeclareScanFunctionScope::getCombiner() {
    ExprResult Res =
            S.BuildDeclRefExpr(FD, FD->getType(), VK_LValue, FD->getLocation());
    return Res.get();
}

FunctionDecl *Sema::OMPDeclareReductionInitFunctionScope::
    ActOnOMPDeclareReductionInitFunction(Sema &S, SourceLocation Loc,
                                         DeclarationName Name, QualType QTy) {
  QualType PtrQTy = S.Context.getPointerType(QTy);
  QualType Args[] = {PtrQTy, PtrQTy};
  FunctionProtoType::ExtProtoInfo EPI;
  QualType FuncType = S.Context.getFunctionType(S.Context.VoidTy, Args, EPI);
  TypeSourceInfo *TI = S.Context.getTrivialTypeSourceInfo(FuncType);
  FunctionTypeLoc FTL = TI->getTypeLoc().getAs<FunctionTypeLoc>();
  FunctionDecl *FD =
      FunctionDecl::Create(S.Context, S.CurContext, Loc, Loc,
                           DeclarationName(&S.Context.Idents.get("init")),
                           FuncType, TI, SC_PrivateExtern, false, false);
  FD->setImplicit();
  S.CurContext->addDecl(FD);
  if (S.CurContext->isDependentContext()) {
    DeclContext *DC = S.CurContext->getParent();
    TemplateParameterList *TPL = 0;
    if (ClassTemplatePartialSpecializationDecl *CTPSD =
            dyn_cast<ClassTemplatePartialSpecializationDecl>(DC)) {
      TPL = CTPSD->getTemplateParameters();
    } else if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(DC)) {
      TPL = RD->getDescribedClassTemplate()
                ->getCanonicalDecl()
                ->getTemplateParameters();
    } else if (FunctionDecl *RD = dyn_cast<FunctionDecl>(DC)) {
      TPL = RD->getDescribedFunctionTemplate()
                ->getCanonicalDecl()
                ->getTemplateParameters();
    }
    FunctionTemplateDecl *FTD = FunctionTemplateDecl::Create(
        S.Context, S.CurContext, Loc, Name, TPL, FD);
    FD->setDescribedFunctionTemplate(FTD);
  }
  ParLHS = ParmVarDecl::Create(S.Context, FD, Loc, Loc, 0, PtrQTy,
                               S.Context.getTrivialTypeSourceInfo(PtrQTy),
                               SC_None, 0);
  ParLHS->setScopeInfo(0, 0);
  ParRHS = ParmVarDecl::Create(S.Context, FD, Loc, Loc, 0, PtrQTy,
                               S.Context.getTrivialTypeSourceInfo(PtrQTy),
                               SC_None, 0);
  ParRHS->setScopeInfo(0, 1);
  ParmVarDecl *Params[] = {ParLHS, ParRHS};
  FD->setParams(Params);
  FTL.setParam(0, ParLHS);
  FTL.setParam(1, ParRHS);
  OmpOrig = VarDecl::Create(S.Context, FD, Loc, Loc,
                            &S.Context.Idents.get("omp_orig"), QTy,
                            S.Context.getTrivialTypeSourceInfo(QTy), SC_Auto);
  OmpPriv = VarDecl::Create(S.Context, FD, OmpPrivLoc, OmpPrivLoc,
                            &S.Context.Idents.get("omp_priv"), QTy,
                            S.Context.getTrivialTypeSourceInfo(QTy), SC_Auto);
  S.AddKnownFunctionAttributes(FD);
  if (S.CurScope) {
    S.PushFunctionScope();
    S.PushDeclContext(S.CurScope, FD);
    S.PushOnScopeChains(OmpPriv, S.CurScope);
    S.PushOnScopeChains(OmpOrig, S.CurScope);
    S.PushExpressionEvaluationContext(PotentiallyEvaluated);
  } else {
    S.CurContext = FD;
    FD->addDecl(OmpOrig);
    FD->addDecl(OmpPriv);
  }
  ExprResult RHS =
      S.BuildDeclRefExpr(ParRHS, ParRHS->getType(), VK_LValue, Loc);
  RHS = S.DefaultLvalueConversion(RHS.get());
  RHS = S.CreateBuiltinUnaryOp(Loc, UO_Deref, RHS.get());
  RHS = S.ActOnFinishFullExpr(RHS.get());
  S.AddInitializerToDecl(OmpOrig, RHS.get(), true, false);
  return FD;
}

FunctionDecl *Sema::OMPDeclareScanInitFunctionScope::
ActOnOMPDeclareScanInitFunction(Sema &S, SourceLocation Loc,
                                DeclarationName Name, QualType QTy) {
    QualType PtrQTy = S.Context.getPointerType(QTy);
    QualType Args[] = {PtrQTy, PtrQTy};
    FunctionProtoType::ExtProtoInfo EPI;
    QualType FuncType = S.Context.getFunctionType(S.Context.VoidTy, Args, EPI);
    TypeSourceInfo *TI = S.Context.getTrivialTypeSourceInfo(FuncType);
    FunctionTypeLoc FTL = TI->getTypeLoc().getAs<FunctionTypeLoc>();
    FunctionDecl *FD =
            FunctionDecl::Create(S.Context, S.CurContext, Loc, Loc,
                                 DeclarationName(&S.Context.Idents.get("init")),
                                 FuncType, TI, SC_PrivateExtern, false, false);
    FD->setImplicit();
    S.CurContext->addDecl(FD);
    if (S.CurContext->isDependentContext()) {
        DeclContext *DC = S.CurContext->getParent();
        TemplateParameterList *TPL = 0;
        if (ClassTemplatePartialSpecializationDecl *CTPSD =
                dyn_cast<ClassTemplatePartialSpecializationDecl>(DC)) {
            TPL = CTPSD->getTemplateParameters();
        } else if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(DC)) {
            TPL = RD->getDescribedClassTemplate()
                    ->getCanonicalDecl()
                    ->getTemplateParameters();
        } else if (FunctionDecl *RD = dyn_cast<FunctionDecl>(DC)) {
            TPL = RD->getDescribedFunctionTemplate()
                    ->getCanonicalDecl()
                    ->getTemplateParameters();
        }
        FunctionTemplateDecl *FTD = FunctionTemplateDecl::Create(
                S.Context, S.CurContext, Loc, Name, TPL, FD);
        FD->setDescribedFunctionTemplate(FTD);
    }
    ParLHS = ParmVarDecl::Create(S.Context, FD, Loc, Loc, 0, PtrQTy,
                                 S.Context.getTrivialTypeSourceInfo(PtrQTy),
                                 SC_None, 0);
    ParLHS->setScopeInfo(0, 0);
    ParRHS = ParmVarDecl::Create(S.Context, FD, Loc, Loc, 0, PtrQTy,
                                 S.Context.getTrivialTypeSourceInfo(PtrQTy),
                                 SC_None, 0);
    ParRHS->setScopeInfo(0, 1);
    ParmVarDecl *Params[] = {ParLHS, ParRHS};
    FD->setParams(Params);
    FTL.setParam(0, ParLHS);
    FTL.setParam(1, ParRHS);
    OmpOrig = VarDecl::Create(S.Context, FD, Loc, Loc,
                              &S.Context.Idents.get("omp_orig"), QTy,
                              S.Context.getTrivialTypeSourceInfo(QTy), SC_Auto);
    OmpPriv = VarDecl::Create(S.Context, FD, OmpPrivLoc, OmpPrivLoc,
                              &S.Context.Idents.get("omp_priv"), QTy,
                              S.Context.getTrivialTypeSourceInfo(QTy), SC_Auto);
    S.AddKnownFunctionAttributes(FD);
    if (S.CurScope) {
        S.PushFunctionScope();
        S.PushDeclContext(S.CurScope, FD);
        S.PushOnScopeChains(OmpPriv, S.CurScope);
        S.PushOnScopeChains(OmpOrig, S.CurScope);
        S.PushExpressionEvaluationContext(PotentiallyEvaluated);
    } else {
        S.CurContext = FD;
        FD->addDecl(OmpOrig);
        FD->addDecl(OmpPriv);
    }
    ExprResult RHS =
            S.BuildDeclRefExpr(ParRHS, ParRHS->getType(), VK_LValue, Loc);
    RHS = S.DefaultLvalueConversion(RHS.get());
    RHS = S.CreateBuiltinUnaryOp(Loc, UO_Deref, RHS.get());
    RHS = S.ActOnFinishFullExpr(RHS.get());
    S.AddInitializerToDecl(OmpOrig, RHS.get(), true, false);
    return FD;
}

void Sema::CreateDefaultDeclareReductionInitFunctionBody(FunctionDecl *FD,
                                                         VarDecl *OmpPriv,
                                                         ParmVarDecl *ParLHS) {
  ExprResult MemCall;
  SourceLocation Loc = OmpPriv->getLocation();
  if (!getLangOpts().CPlusPlus || OmpPriv->getType().isPODType(Context)) {
    // Perform explicit initialization of POD types.
    ExprResult OmpPrivDRE =
        BuildDeclRefExpr(OmpPriv, OmpPriv->getType(), VK_LValue, Loc);
    Expr *OmpPrivDREExpr = OmpPrivDRE.get();
    ExprResult OmpPrivAddr =
        CreateBuiltinUnaryOp(Loc, UO_AddrOf, OmpPrivDREExpr);
    OmpPrivAddr = PerformImplicitConversion(OmpPrivAddr.get(),
                                            Context.VoidPtrTy, AA_Casting);
    ExprResult OmpPrivSizeOf;
    {
      EnterExpressionEvaluationContext Unevaluated(
          *this, Sema::Unevaluated, Sema::ReuseLambdaContextDecl);

      OmpPrivSizeOf =
          CreateUnaryExprOrTypeTraitExpr(OmpPrivDREExpr, Loc, UETT_SizeOf);
    }
    UnqualifiedId Name;
    CXXScopeSpec SS;
    SourceLocation TemplateKwLoc;
    Name.setIdentifier(PP.getIdentifierInfo("__builtin_memset"), Loc);
    ExprResult MemSetFn =
        ActOnIdExpression(TUScope, SS, TemplateKwLoc, Name, true, false);
    Expr *Args[] = {OmpPrivAddr.get(), ActOnIntegerConstant(Loc, 0).get(),
                    OmpPrivSizeOf.get()};
    MemCall = ActOnCallExpr(0, MemSetFn.get(), Loc, Args, Loc);
    MemCall = IgnoredValueConversions(MemCall.get());
  } else {
    ActOnUninitializedDecl(OmpPriv, false);
  }
  StmtResult S1 =
      ActOnDeclStmt(DeclGroupPtrTy::make(DeclGroupRef(OmpPriv)), Loc, Loc);
  ExprResult LHS = BuildDeclRefExpr(ParLHS, ParLHS->getType(), VK_LValue, Loc);
  LHS = DefaultLvalueConversion(LHS.get());
  LHS = CreateBuiltinUnaryOp(Loc, UO_Deref, LHS.get());
  ExprResult RHS =
      BuildDeclRefExpr(OmpPriv, OmpPriv->getType(), VK_LValue, Loc);
  ExprResult Res = BuildBinOp(0, Loc, BO_Assign, LHS.get(), RHS.get());
  ExprResult S2 = IgnoredValueConversions(ActOnFinishFullExpr(Res.get()).get());
  if (S1.isInvalid() || S2.isInvalid()) {
    FD->setBody(ActOnNullStmt(Loc).get());
    FD->setInvalidDecl();
  } else {
    CompoundScopeRAII CompoundScope(*this);
    SmallVector<Stmt *, 4> Stmts;
    Stmts.push_back(S1.get());
    if (MemCall.isUsable())
      Stmts.push_back(MemCall.get());
    Stmts.push_back(S2.get());
    StmtResult Body = ActOnCompoundStmt(Loc, Loc, Stmts, false);
    FD->setBody(Body.get());
  }
}

void Sema::CreateDefaultDeclareScanInitFunctionBody(FunctionDecl *FD,
                                                    VarDecl *OmpPriv,
                                                    ParmVarDecl *ParLHS) {
    ExprResult MemCall;
    SourceLocation Loc = OmpPriv->getLocation();
    if (!getLangOpts().CPlusPlus || OmpPriv->getType().isPODType(Context)) {
        // Perform explicit initialization of POD types.
        ExprResult OmpPrivDRE =
                BuildDeclRefExpr(OmpPriv, OmpPriv->getType(), VK_LValue, Loc);
        Expr *OmpPrivDREExpr = OmpPrivDRE.get();
        ExprResult OmpPrivAddr =
                CreateBuiltinUnaryOp(Loc, UO_AddrOf, OmpPrivDREExpr);
        OmpPrivAddr = PerformImplicitConversion(OmpPrivAddr.get(),
                                                Context.VoidPtrTy, AA_Casting);
        ExprResult OmpPrivSizeOf;
        {
            EnterExpressionEvaluationContext Unevaluated(
                    *this, Sema::Unevaluated, Sema::ReuseLambdaContextDecl);

            OmpPrivSizeOf =
                    CreateUnaryExprOrTypeTraitExpr(OmpPrivDREExpr, Loc, UETT_SizeOf);
        }
        UnqualifiedId Name;
        CXXScopeSpec SS;
        SourceLocation TemplateKwLoc;
        Name.setIdentifier(PP.getIdentifierInfo("__builtin_memset"), Loc);
        ExprResult MemSetFn =
                ActOnIdExpression(TUScope, SS, TemplateKwLoc, Name, true, false);
        Expr *Args[] = {OmpPrivAddr.get(), ActOnIntegerConstant(Loc, 0).get(),
                        OmpPrivSizeOf.get()};
        MemCall = ActOnCallExpr(0, MemSetFn.get(), Loc, Args, Loc);
        MemCall = IgnoredValueConversions(MemCall.get());
    } else {
        ActOnUninitializedDecl(OmpPriv, false);
    }
    StmtResult S1 =
            ActOnDeclStmt(DeclGroupPtrTy::make(DeclGroupRef(OmpPriv)), Loc, Loc);
    ExprResult LHS = BuildDeclRefExpr(ParLHS, ParLHS->getType(), VK_LValue, Loc);
    LHS = DefaultLvalueConversion(LHS.get());
    LHS = CreateBuiltinUnaryOp(Loc, UO_Deref, LHS.get());
    ExprResult RHS =
            BuildDeclRefExpr(OmpPriv, OmpPriv->getType(), VK_LValue, Loc);
    ExprResult Res = BuildBinOp(0, Loc, BO_Assign, LHS.get(), RHS.get());
    ExprResult S2 = IgnoredValueConversions(ActOnFinishFullExpr(Res.get()).get());
    if (S1.isInvalid() || S2.isInvalid()) {
        FD->setBody(ActOnNullStmt(Loc).get());
        FD->setInvalidDecl();
    } else {
        CompoundScopeRAII CompoundScope(*this);
        SmallVector<Stmt *, 4> Stmts;
        Stmts.push_back(S1.get());
        if (MemCall.isUsable())
            Stmts.push_back(MemCall.get());
        Stmts.push_back(S2.get());
        StmtResult Body = ActOnCompoundStmt(Loc, Loc, Stmts, false);
        FD->setBody(Body.get());
    }
}

void Sema::OMPDeclareReductionInitFunctionScope::setInit(Expr *E) {
  ExprResult MemCall;
  if (!E) {
    if (OmpPriv->getType()->isDependentType() ||
        OmpPriv->getType()->isInstantiationDependentType())
      // It will be handled later on instantiation.
      return;
    S.CreateDefaultDeclareReductionInitFunctionBody(FD, OmpPriv, ParLHS);
    return;
  } else {
    if (IsInit)
      S.AddInitializerToDecl(OmpPriv, E, true, false);
    else {
      if (!isa<CallExpr>(E->IgnoreParenImpCasts())) {
        FD->setInvalidDecl();
        S.Diag(E->getExprLoc(), diag::err_omp_reduction_non_function_init)
            << E->getSourceRange();
        return;
      }
      MemCall = S.IgnoredValueConversions(E);
    }
  }
  SourceLocation Loc = E->getExprLoc();
  StmtResult S1 =
      S.ActOnDeclStmt(DeclGroupPtrTy::make(DeclGroupRef(OmpOrig)), Loc, Loc);
  StmtResult S2 =
      S.ActOnDeclStmt(DeclGroupPtrTy::make(DeclGroupRef(OmpPriv)), Loc, Loc);
  ExprResult LHS =
      S.BuildDeclRefExpr(ParLHS, ParLHS->getType(), VK_LValue, Loc);
  LHS = S.DefaultLvalueConversion(LHS.get());
  LHS = S.CreateBuiltinUnaryOp(Loc, UO_Deref, LHS.get());
  ExprResult RHS =
      S.BuildDeclRefExpr(OmpPriv, OmpPriv->getType(), VK_LValue, Loc);
  ExprResult Res = S.BuildBinOp(0, Loc, BO_Assign, LHS.get(), RHS.get());
  Res = S.ActOnFinishFullExpr(Res.get());
  ExprResult S3 = S.IgnoredValueConversions(Res.get());
  if (S1.isInvalid() || S2.isInvalid() || S3.isInvalid()) {
    FD->setBody(S.ActOnNullStmt(Loc).get());
    FD->setInvalidDecl();
  } else {
    CompoundScopeRAII CompoundScope(S);
    SmallVector<Stmt *, 4> Stmts;
    Stmts.push_back(S1.get());
    Stmts.push_back(S2.get());
    if (MemCall.isUsable())
      Stmts.push_back(MemCall.get());
    Stmts.push_back(S3.get());
    StmtResult Body = S.ActOnCompoundStmt(Loc, Loc, Stmts, false);
    FD->setBody(Body.get());
  }
}

void Sema::OMPDeclareScanInitFunctionScope::setInit(Expr *E) {
    ExprResult MemCall;
    if (!E) {
        if (OmpPriv->getType()->isDependentType() ||
            OmpPriv->getType()->isInstantiationDependentType())
            // It will be handled later on instantiation.
            return;
        S.CreateDefaultDeclareScanInitFunctionBody(FD, OmpPriv, ParLHS);
        return;
    } else {
        if (IsInit)
            S.AddInitializerToDecl(OmpPriv, E, true, false);
        else {
            if (!isa<CallExpr>(E->IgnoreParenImpCasts())) {
                FD->setInvalidDecl();
                S.Diag(E->getExprLoc(), diag::err_omp_reduction_non_function_init)
                        << E->getSourceRange();
                return;
            }
            MemCall = S.IgnoredValueConversions(E);
        }
    }
    SourceLocation Loc = E->getExprLoc();
    StmtResult S1 =
            S.ActOnDeclStmt(DeclGroupPtrTy::make(DeclGroupRef(OmpOrig)), Loc, Loc);
    StmtResult S2 =
            S.ActOnDeclStmt(DeclGroupPtrTy::make(DeclGroupRef(OmpPriv)), Loc, Loc);
    ExprResult LHS =
            S.BuildDeclRefExpr(ParLHS, ParLHS->getType(), VK_LValue, Loc);
    LHS = S.DefaultLvalueConversion(LHS.get());
    LHS = S.CreateBuiltinUnaryOp(Loc, UO_Deref, LHS.get());
    ExprResult RHS =
            S.BuildDeclRefExpr(OmpPriv, OmpPriv->getType(), VK_LValue, Loc);
    ExprResult Res = S.BuildBinOp(0, Loc, BO_Assign, LHS.get(), RHS.get());
    Res = S.ActOnFinishFullExpr(Res.get());
    ExprResult S3 = S.IgnoredValueConversions(Res.get());
    if (S1.isInvalid() || S2.isInvalid() || S3.isInvalid()) {
        FD->setBody(S.ActOnNullStmt(Loc).get());
        FD->setInvalidDecl();
    } else {
        CompoundScopeRAII CompoundScope(S);
        SmallVector<Stmt *, 4> Stmts;
        Stmts.push_back(S1.get());
        Stmts.push_back(S2.get());
        if (MemCall.isUsable())
            Stmts.push_back(MemCall.get());
        Stmts.push_back(S3.get());
        StmtResult Body = S.ActOnCompoundStmt(Loc, Loc, Stmts, false);
        FD->setBody(Body.get());
    }
}

Expr *Sema::OMPDeclareReductionInitFunctionScope::getInitializer() {
  ExprResult Res =
      S.BuildDeclRefExpr(FD, FD->getType(), VK_LValue, FD->getLocation());
  return Res.get();
}

Expr *Sema::OMPDeclareScanInitFunctionScope::getInitializer() {
    ExprResult Res =
            S.BuildDeclRefExpr(FD, FD->getType(), VK_LValue, FD->getLocation());
    return Res.get();
}

bool Sema::IsOMPDeclareReductionTypeAllowed(SourceRange Range, QualType QTy,
                                            ArrayRef<QualType> Types,
                                            ArrayRef<SourceRange> TyRanges) {
  if (QTy.isNull())
    return false;

  if (QTy.getCanonicalType().hasQualifiers()) {
    Diag(Range.getBegin(), diag::err_omp_reduction_qualified_type) << Range;
    return false;
  }

  QTy = QTy.getCanonicalType();
  if (QTy->isFunctionType() || QTy->isFunctionNoProtoType() ||
      QTy->isFunctionProtoType() || QTy->isFunctionPointerType() ||
      QTy->isMemberFunctionPointerType()) {
    Diag(Range.getBegin(), diag::err_omp_reduction_function_type) << Range;
    return false;
  }
  if (QTy->isReferenceType()) {
    Diag(Range.getBegin(), diag::err_omp_reduction_reference_type) << Range;
    return false;
  }
  if (QTy->isArrayType()) {
    Diag(Range.getBegin(), diag::err_omp_reduction_array_type) << Range;
    return false;
  }

  bool IsValid = true;
  ArrayRef<SourceRange>::iterator IR = TyRanges.begin();
  for (ArrayRef<QualType>::iterator I = Types.begin(), E = Types.end(); I != E;
       ++I, ++IR) {
    if (Context.hasSameType(QTy, *I)) {
      Diag(Range.getBegin(), diag::err_omp_reduction_redeclared) << *I << Range;
      Diag(IR->getBegin(), diag::note_previous_declaration) << *IR;
      IsValid = false;
    }
  }
  return IsValid;
}

bool Sema::IsOMPDeclareScanTypeAllowed(SourceRange Range, QualType QTy,
                                       ArrayRef<QualType> Types,
                                       ArrayRef<SourceRange> TyRanges) {
    if (QTy.isNull())
        return false;

    if (QTy.getCanonicalType().hasQualifiers()) {
        Diag(Range.getBegin(), diag::err_omp_reduction_qualified_type) << Range;
        return false;
    }

    QTy = QTy.getCanonicalType();
    if (QTy->isFunctionType() || QTy->isFunctionNoProtoType() ||
        QTy->isFunctionProtoType() || QTy->isFunctionPointerType() ||
        QTy->isMemberFunctionPointerType()) {
        Diag(Range.getBegin(), diag::err_omp_reduction_function_type) << Range;
        return false;
    }
    if (QTy->isReferenceType()) {
        Diag(Range.getBegin(), diag::err_omp_reduction_reference_type) << Range;
        return false;
    }
    if (QTy->isArrayType()) {
        Diag(Range.getBegin(), diag::err_omp_reduction_array_type) << Range;
        return false;
    }

    bool IsValid = true;
    ArrayRef<SourceRange>::iterator IR = TyRanges.begin();
    for (ArrayRef<QualType>::iterator I = Types.begin(), E = Types.end(); I != E;
         ++I, ++IR) {
        if (Context.hasSameType(QTy, *I)) {
            Diag(Range.getBegin(), diag::err_omp_reduction_redeclared) << *I << Range;
            Diag(IR->getBegin(), diag::note_previous_declaration) << *IR;
            IsValid = false;
        }
    }
    return IsValid;
}

Sema::DeclGroupPtrTy Sema::ActOnOpenMPDeclareReductionDirective(
    Decl *D, ArrayRef<QualType> Types, ArrayRef<SourceRange> TyRanges,
    ArrayRef<Expr *> Combiners, ArrayRef<Expr *> Inits) {
  OMPDeclareReductionDecl *DR = cast<OMPDeclareReductionDecl>(D);

  LookupResult Found(*this, DR->getDeclName(), DR->getLocation(),
                     LookupOMPDeclareReduction);
  Found.suppressDiagnostics();
  LookupName(Found, CurScope);
  for (LookupResult::iterator I = Found.begin(), E = Found.end(); I != E; ++I) {
    OMPDeclareReductionDecl *DRI = cast<OMPDeclareReductionDecl>(*I);
    if (DRI == D)
      continue;
    for (OMPDeclareReductionDecl::datalist_iterator II = DRI->datalist_begin(),
                                                    EE = DRI->datalist_end();
         II != EE; ++II) {
      ArrayRef<SourceRange>::iterator IR = TyRanges.begin();
      for (ArrayRef<QualType>::iterator IT = Types.begin(), IE = Types.end();
           IT != IE; ++IT, ++IR) {
        if (!II->QTy.isNull() && !IT->isNull() &&
            Context.hasSameType(II->QTy, *IT)) {
          Diag(IR->getBegin(), diag::err_omp_reduction_redeclared) << II->QTy
                                                                   << *IR;
          Diag(II->TyRange.getBegin(), diag::note_previous_declaration)
              << II->TyRange;
          D->setInvalidDecl();
        }
      }
    }
  }

  if (!D->isInvalidDecl()) {
    CompleteOMPDeclareReductionDecl(DR, Types, TyRanges, Combiners, Inits);
    PushOnScopeChains(DR, CurScope, false);
    return DeclGroupPtrTy::make(DeclGroupRef(DR));
  }
  return DeclGroupPtrTy();
}

Sema::DeclGroupPtrTy Sema::ActOnOpenMPDeclareScanDirective(
        Decl *D, ArrayRef<QualType> Types, ArrayRef<SourceRange> TyRanges,
        ArrayRef<Expr *> Combiners, ArrayRef<Expr *> Inits) {
    OMPDeclareScanDecl *DR = cast<OMPDeclareScanDecl>(D);

    LookupResult Found(*this, DR->getDeclName(), DR->getLocation(),
                       LookupOMPDeclareScan);
    Found.suppressDiagnostics();
    LookupName(Found, CurScope);
    for (LookupResult::iterator I = Found.begin(), E = Found.end(); I != E; ++I) {
        OMPDeclareScanDecl *DRI = cast<OMPDeclareScanDecl>(*I);
        if (DRI == D)
            continue;
        for (OMPDeclareScanDecl::datalist_iterator II = DRI->datalist_begin(),
                     EE = DRI->datalist_end();
             II != EE; ++II) {
            ArrayRef<SourceRange>::iterator IR = TyRanges.begin();
            for (ArrayRef<QualType>::iterator IT = Types.begin(), IE = Types.end();
                 IT != IE; ++IT, ++IR) {
                if (!II->QTy.isNull() && !IT->isNull() &&
                    Context.hasSameType(II->QTy, *IT)) {
                    Diag(IR->getBegin(), diag::err_omp_reduction_redeclared) << II->QTy
                                                                             << *IR;
                    Diag(II->TyRange.getBegin(), diag::note_previous_declaration)
                            << II->TyRange;
                    D->setInvalidDecl();
                }
            }
        }
    }

    if (!D->isInvalidDecl()) {
        CompleteOMPDeclareScanDecl(DR, Types, TyRanges, Combiners, Inits);
        PushOnScopeChains(DR, CurScope, false);
        return DeclGroupPtrTy::make(DeclGroupRef(DR));
    }
    return DeclGroupPtrTy();
}

void Sema::CompleteOMPDeclareReductionDecl(OMPDeclareReductionDecl *D,
                                           ArrayRef<QualType> Types,
                                           ArrayRef<SourceRange> TyRanges,
                                           ArrayRef<Expr *> Combiners,
                                           ArrayRef<Expr *> Inits) {
  SmallVector<OMPDeclareReductionDecl::ReductionData, 4> Data;
  ArrayRef<Expr *>::iterator IC = Combiners.begin();
  ArrayRef<Expr *>::iterator II = Inits.begin();
  ArrayRef<SourceRange>::iterator IR = TyRanges.begin();
  for (ArrayRef<QualType>::iterator IT = Types.begin(), ET = Types.end();
       IT != ET; ++IT, ++IC, ++II, ++IR) {
    Data.push_back(OMPDeclareReductionDecl::ReductionData(*IT, *IR, *IC, *II));
  }
  D->setData(Data);
}

void Sema::CompleteOMPDeclareScanDecl(OMPDeclareScanDecl *D,
                                      ArrayRef<QualType> Types,
                                      ArrayRef<SourceRange> TyRanges,
                                      ArrayRef<Expr *> Combiners,
                                      ArrayRef<Expr *> Inits) {
    SmallVector<OMPDeclareScanDecl::ScanData, 4> Data;
    ArrayRef<Expr *>::iterator IC = Combiners.begin();
    ArrayRef<Expr *>::iterator II = Inits.begin();
    ArrayRef<SourceRange>::iterator IR = TyRanges.begin();
    for (ArrayRef<QualType>::iterator IT = Types.begin(), ET = Types.end();
         IT != ET; ++IT, ++IC, ++II, ++IR) {
        Data.push_back(OMPDeclareScanDecl::ScanData(*IT, *IR, *IC, *II));
    }
    D->setData(Data);
}

bool Sema::ActOnStartOpenMPDeclareTargetDirective(Scope *S,
                                                  SourceLocation Loc) {
  if (CurContext && !CurContext->isFileContext()) {
    Diag(Loc, diag::err_omp_region_not_file_context);
    return false;
  }
  OMPDeclareTargetDecl *DT =
      OMPDeclareTargetDecl::Create(Context, CurContext, Loc);
  DT->setAccess(AS_public);
  CurContext->addDecl(DT);
  if (CurScope)
    PushDeclContext(S, DT);
  else
    CurContext = DT;
  return true;
}

void Sema::ActOnOpenMPDeclareTargetDecls(Sema::DeclGroupPtrTy Decls) {
  if (!Decls)
    return;
  DeclGroupRef DGR = Decls.get();
  if (DGR.isNull())
    return;
  for (DeclGroupRef::iterator I = DGR.begin(), E = DGR.end(); I != E; ++I) {
    if (*I)
      DSAStack->addDeclareTargetDecl(*I);
  }
}

Sema::DeclGroupPtrTy Sema::ActOnFinishOpenMPDeclareTargetDirective() {
  if (CurContext && isa<OMPDeclareTargetDecl>(CurContext)) {
    OMPDeclareTargetDecl *DT = cast<OMPDeclareTargetDecl>(CurContext);
    PopDeclContext();
    return DeclGroupPtrTy::make(DeclGroupRef(DT));
  }
  return DeclGroupPtrTy();
}

void Sema::ActOnOpenMPDeclareTargetDirectiveError() {
  if (CurContext && isa<OMPDeclareTargetDecl>(CurContext)) {
    PopDeclContext();
  }
}

static void CheckDeclInTargetContext(SourceLocation SL, SourceRange SR,
                                     Sema &SemaRef, DSAStackTy *Stack,
                                     Decl *D) {
  if (!D)
    return;
  Decl *LD = 0;
  if (isa<TagDecl>(D)) {
    LD = cast<TagDecl>(D)->getDefinition();
  } else if (isa<VarDecl>(D)) {
    LD = cast<VarDecl>(D)->getDefinition();
  } else if (isa<FunctionDecl>(D)) {
    const FunctionDecl *FD = 0;
    if (cast<FunctionDecl>(D)->hasBody(FD))
      LD = const_cast<FunctionDecl *>(FD);
  }
  if (!LD)
    LD = D;
  if (LD) {
    if (!Stack->isDeclareTargetDecl(LD)) {
      // Outlined declaration is not declared target.
      if (LD->isOutOfLine()) {
        SemaRef.Diag(LD->getLocation(), diag::warn_omp_not_in_target_context);
        SemaRef.Diag(SL, diag::note_used_here) << SR;
      } else {
        DeclContext *DC = LD->getDeclContext();
        while (DC) {
          if (isa<OMPDeclareTargetDecl>(DC))
            break;
          DC = DC->getParent();
        }
        // Is not declared in target context.
        if (!DC) {
          SemaRef.Diag(LD->getLocation(), diag::warn_omp_not_in_target_context);
          SemaRef.Diag(SL, diag::note_used_here) << SR;
        }
      }
    }
    // Mark decl as declared to prevent further diagnostic.
    if (isa<VarDecl>(LD) || isa<FunctionDecl>(LD))
      Stack->addDeclareTargetDecl(LD);
  }
}

static bool IsCXXRecordForMappable(Sema &SemaRef, SourceLocation Loc,
                                   DSAStackTy *Stack, CXXRecordDecl *RD);

static bool CheckTypeMappable(SourceLocation SL, SourceRange SR, Sema &SemaRef,
                              DSAStackTy *Stack, QualType QTy) {
  NamedDecl *ND;
  if (QTy->isIncompleteType(&ND)) {
    SemaRef.Diag(SL, diag::err_incomplete_type) << QTy << SR;
    return false;
  } else if (CXXRecordDecl *RD = dyn_cast_or_null<CXXRecordDecl>(ND)) {
    if (!RD->isInvalidDecl() &&
        !IsCXXRecordForMappable(SemaRef, SL, Stack, RD)) {
      return false;
    }
  }
  return true;
}

static bool CheckValueDeclInTarget(SourceLocation SL, SourceRange SR,
                                   Sema &SemaRef, DSAStackTy *Stack,
                                   ValueDecl *VD) {
  if (Stack->isDeclareTargetDecl(VD))
    return true;
  if (!CheckTypeMappable(SL, SR, SemaRef, Stack, VD->getType())) {
    return false;
  }
  return true;
}

static bool IsCXXRecordForMappable(Sema &SemaRef, SourceLocation Loc,
                                   DSAStackTy *Stack, CXXRecordDecl *RD) {
  if (!RD || RD->isInvalidDecl())
    return true;

  QualType QTy = SemaRef.Context.getRecordType(RD);
  if (RD->isDynamicClass()) {
    SemaRef.Diag(Loc, diag::err_omp_not_mappable_type) << QTy;
    SemaRef.Diag(RD->getLocation(), diag::note_omp_polymorphic_in_target);
    return false;
  }
  DeclContext *DC = RD;
  bool IsCorrect = true;
  for (DeclContext::decl_iterator I = DC->noload_decls_begin(),
                                  E = DC->noload_decls_end();
       I != E; ++I) {
    if (*I) {
      if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(*I)) {
        if (MD->isStatic()) {
          SemaRef.Diag(Loc, diag::err_omp_not_mappable_type) << QTy;
          SemaRef.Diag(MD->getLocation(),
                       diag::note_omp_static_member_in_target);
          IsCorrect = false;
        }
      } else if (VarDecl *VD = dyn_cast<VarDecl>(*I)) {
        if (VD->isStaticDataMember()) {
          SemaRef.Diag(Loc, diag::err_omp_not_mappable_type) << QTy;
          SemaRef.Diag(VD->getLocation(),
                       diag::note_omp_static_member_in_target);
          IsCorrect = false;
        }
      }
    }
  }
  for (CXXRecordDecl::base_class_iterator I = RD->bases_begin(),
                                          E = RD->bases_end();
       I != E; ++I) {
    if (!IsCXXRecordForMappable(SemaRef, I->getLocStart(), Stack,
                                I->getType()->getAsCXXRecordDecl())) {
      IsCorrect = false;
    }
  }
  return IsCorrect;
}

void Sema::CheckDeclIsAllowedInOpenMPTarget(Expr *E, Decl *D) {
  if (!D || D->isInvalidDecl())
    return;
  SourceRange SR = E ? E->getSourceRange() : D->getSourceRange();
  SourceLocation SL = E ? E->getLocStart() : D->getLocation();
  if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    DeclRefExpr *DRE;
    if (DSAStack->IsThreadprivate(VD, DRE)) {
      SourceLocation Loc = DRE ? DRE->getLocation() : VD->getLocation();
      Diag(Loc, diag::err_omp_threadprivate_in_target);
      Diag(SL, diag::note_used_here) << SR;
      D->setInvalidDecl();
      return;
    }
  }
  if (ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
    if (!CheckValueDeclInTarget(SL, SR, *this, DSAStack, VD)) {
      VD->setInvalidDecl();
      return;
    }
  }
  if (!E) {
    // Checking declaration.
    if (isa<VarDecl>(D) || isa<FunctionDecl>(D))
      DSAStack->addDeclareTargetDecl(D);
    return;
  }
  CheckDeclInTargetContext(E->getExprLoc(), E->getSourceRange(), *this,
                           DSAStack, D);
}

void Sema::MarkOpenMPClauses(ArrayRef<OMPClause *> Clauses) {
  for (ArrayRef<OMPClause *>::iterator I = Clauses.begin(), E = Clauses.end();
       I != E; ++I)
    for (Stmt::child_range S = (*I)->children(); S; ++S) {
      if (*S && isa<Expr>(*S))
        MarkDeclarationsReferencedInExpr(cast<Expr>(*S));
    }
}

namespace {
class DSAAttrChecker : public StmtVisitor<DSAAttrChecker, void> {
  DSAStackTy *Stack;
  Sema &Actions;
  llvm::SmallVector<Expr *, 2> ImplicitFirstprivate;
  bool ErrorFound;
  CapturedStmt *CS;

public:
  void VisitDeclRefExpr(DeclRefExpr *E) {
    if (VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
      if (VD->isImplicit() && VD->hasAttr<UnusedAttr>())
        return;
      // Skip internally declared variables.
      if (VD->isLocalVarDecl() && !CS->capturesVariable(VD))
        return;
      // NamedDecl *ND = VD;
      // if (
      //    Actions.isDeclInScope(ND, Actions.CurContext,
      //                          Stack->getCurScope())) return;
      SourceLocation ELoc = E->getExprLoc();
      DeclRefExpr *PrevRef;

      OpenMPDirectiveKind DKind = Stack->getCurrentDirective();
      OpenMPClauseKind Kind = Stack->getTopDSA(VD, PrevRef);

      // The default(none) clause requires that each variable that is referenced
      // in the construct, and does not have a predetermined data-sharing
      // attribute, must have its data-sharing attribute explicitly determined
      // by being listed in a data-sharing attribute clause.
      if (Kind == OMPC_unknown && Stack->getDefaultDSA() == DSA_none &&
          (DKind == OMPD_parallel || DKind == OMPD_parallel_for ||
           DKind == OMPD_parallel_for_simd ||
           DKind == OMPD_distribute_parallel_for ||
           DKind == OMPD_distribute_parallel_for_simd || DKind == OMPD_task ||
           DKind == OMPD_teams_distribute_parallel_for ||
           DKind == OMPD_teams_distribute_parallel_for_simd ||
           DKind == OMPD_target_teams_distribute_parallel_for ||
           DKind == OMPD_target_teams_distribute_parallel_for_simd ||
           DKind == OMPD_teams || DKind == OMPD_parallel_sections ||
           DKind == OMPD_target_teams || DKind == OMPD_teams_distribute ||
           DKind == OMPD_teams_distribute_simd ||
           DKind == OMPD_target_teams_distribute ||
           DKind == OMPD_target_teams_distribute_simd)) {
        ErrorFound = true;
        Actions.Diag(ELoc, diag::err_omp_no_dsa_for_variable) << VD;
        return;
      }

      // OpenMP [2.9.3.6, Restrictions, p.2]
      //  A list item that appears in a reduction clause of the innermost
      //  enclosing worksharing or parallel construct may not be accessed in an
      //  explicit task.
      if (DKind == OMPD_task &&
          (Stack->hasInnermostDSA(VD, OMPC_reduction, OMPD_for, PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction, OMPD_for_simd, PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction, OMPD_sections, PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction, OMPD_parallel, PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction, OMPD_parallel_for,
                                  PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction, OMPD_parallel_for_simd,
                                  PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction,
                                  OMPD_distribute_parallel_for, PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction,
                                  OMPD_distribute_parallel_for_simd, PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction,
                                  OMPD_teams_distribute_parallel_for,
                                  PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction,
                                  OMPD_teams_distribute_parallel_for_simd,
                                  PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction,
                                  OMPD_target_teams_distribute_parallel_for,
                                  PrevRef) ||
           Stack->hasInnermostDSA(
               VD, OMPC_reduction,
               OMPD_target_teams_distribute_parallel_for_simd, PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction, OMPD_parallel_sections,
                                  PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction, OMPD_teams, PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction, OMPD_target_teams,
                                  PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction, OMPD_teams_distribute,
                                  PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction,
                                  OMPD_teams_distribute_simd, PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction,
                                  OMPD_target_teams_distribute, PrevRef) ||
           Stack->hasInnermostDSA(VD, OMPC_reduction,
                                  OMPD_target_teams_distribute_simd,
                                  PrevRef))) {
        ErrorFound = true;
        Actions.Diag(ELoc, diag::err_omp_reduction_in_task);
        if (PrevRef) {
          Actions.Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
              << getOpenMPClauseName(OMPC_reduction);
        }
        return;
      }
      // Define implicit data-sharing attributes for task.
      if (DKind == OMPD_task && Kind == OMPC_unknown) {
        Kind = Stack->getImplicitDSA(VD, DKind, PrevRef);
        if (Kind != OMPC_shared)
          ImplicitFirstprivate.push_back(E);
      }
    }
  }
  void VisitOMPExecutableDirective(OMPExecutableDirective *S) {
    for (ArrayRef<OMPClause *>::iterator I = S->clauses().begin(),
                                         E = S->clauses().end();
         I != E; ++I) {
      if (OMPClause *C = *I)
        for (StmtRange R = C->children(); R; ++R) {
          if (Stmt *Child = *R)
            Visit(Child);
        }
    }
  }
  void VisitStmt(Stmt *S) {
    for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E;
         ++I) {
      if (Stmt *Child = *I) {
        if (!isa<OMPExecutableDirective>(Child))
          Visit(Child);
      }
    }
  }

  ArrayRef<Expr *> getImplicitFirstprivate() { return ImplicitFirstprivate; }
  bool isErrorFound() { return ErrorFound; }

  DSAAttrChecker(DSAStackTy *S, Sema &Actions, CapturedStmt *CS)
      : Stack(S), Actions(Actions), ImplicitFirstprivate(), ErrorFound(false),
        CS(CS) {}
};
}

StmtResult Sema::ActOnOpenMPExecutableDirective(
    OpenMPDirectiveKind Kind, const DeclarationNameInfo &DirName,
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc, OpenMPDirectiveKind ConstructType) {
  // OpenMP [2.16, Nesting of Regions]
  llvm::SmallVector<OMPClause *, 4> ClausesWithImplicit;
  bool ErrorFound = false;
  if (DSAStack->getCurScope()) {
    OpenMPDirectiveKind ParentKind = DSAStack->getParentDirective();
    bool NestingProhibited = false;
    bool CloseNesting = true;
    bool HasNamedDirective = false;
    StringRef Region;
    bool ConstructTypeMatches = false;
    if (Kind == OMPD_cancel || Kind == OMPD_cancellation_point) {
      switch (ConstructType) {
      case OMPD_parallel:
        ConstructTypeMatches = ParentKind == OMPD_parallel;
        break;
      case OMPD_for:
        ConstructTypeMatches =
            ParentKind == OMPD_for || ParentKind == OMPD_parallel_for ||
            ParentKind == OMPD_distribute_parallel_for ||
            ParentKind == OMPD_teams_distribute_parallel_for ||
            ParentKind == OMPD_target_teams_distribute_parallel_for;
        break;
      case OMPD_sections:
        ConstructTypeMatches =
            ParentKind == OMPD_sections || ParentKind == OMPD_parallel_sections;
        break;
      case OMPD_taskgroup:
        ConstructTypeMatches = ParentKind == OMPD_task;
        break;
      default:
        break;
      }
    }
    switch (ParentKind) {
    case OMPD_parallel:
      NestingProhibited =
          (Kind == OMPD_cancel && !ConstructTypeMatches) ||
          (Kind == OMPD_cancellation_point && !ConstructTypeMatches);
      Region = "a parallel";
      break;
    case OMPD_for:
    case OMPD_sections:
    case OMPD_distribute_parallel_for:
    case OMPD_teams_distribute_parallel_for:
    case OMPD_target_teams_distribute_parallel_for:
    case OMPD_parallel_for:
    case OMPD_parallel_sections:
    case OMPD_single:
      // Worksharing region
      // OpenMP [2.16, Nesting of Regions, p. 1]
      //  A worksharing region may not be closely nested inside a worksharing,
      //  explicit task, critical, ordered, atomic, or master region.
      // OpenMP [2.16, Nesting of Regions, p. 2]
      //  A barrier region may not be closely nested inside a worksharing,
      //  explicit task, critical, ordered, atomic, or master region.
      // OpenMP [2.16, Nesting of Regions, p. 3]
      //  A master region may not be closely nested inside a worksharing,
      //  atomic, or explicit task region.
      NestingProhibited =
          Kind == OMPD_for || Kind == OMPD_sections || Kind == OMPD_for_simd ||
          Kind == OMPD_distribute_simd || Kind == OMPD_distribute ||
          Kind == OMPD_distribute_parallel_for ||
          Kind == OMPD_distribute_parallel_for_simd || Kind == OMPD_single ||
          Kind == OMPD_master || Kind == OMPD_barrier ||
          (Kind == OMPD_cancel && !ConstructTypeMatches) ||
          (Kind == OMPD_cancellation_point && !ConstructTypeMatches);
      Region = "a worksharing";
      break;
    case OMPD_task:
      // Task region
      // OpenMP [2.16, Nesting of Regions, p. 1]
      //  A worksharing region may not be closely nested inside a worksharing,
      //  explicit task, critical, ordered, atomic, or master region.
      // OpenMP [2.16, Nesting of Regions, p. 2]
      //  A barrier region may not be closely nested inside a worksharing,
      //  explicit task, critical, ordered, atomic, or master region.
      // OpenMP [2.16, Nesting of Regions, p. 3]
      //  A master region may not be closely nested inside a worksharing,
      // atomic,
      //  or explicit task region.
      // OpenMP [2.16, Nesting of Regions, p. 4]
      //  An ordered region may not be closely nested inside a critical, atomic,
      //  or explicit task region.
      NestingProhibited =
          Kind == OMPD_for || Kind == OMPD_sections || Kind == OMPD_for_simd ||
          Kind == OMPD_distribute_simd || Kind == OMPD_distribute ||
          Kind == OMPD_distribute_parallel_for ||
          Kind == OMPD_distribute_parallel_for_simd || Kind == OMPD_single ||
          Kind == OMPD_master || Kind == OMPD_barrier || Kind == OMPD_ordered ||
          (Kind == OMPD_cancel && !ConstructTypeMatches) ||
          (Kind == OMPD_cancellation_point && !ConstructTypeMatches);
      Region = "explicit task";
      break;
    case OMPD_master:
      // OpenMP [2.16, Nesting of Regions, p. 1]
      //  A worksharing region may not be closely nested inside a worksharing,
      //  explicit task, critical, ordered, atomic, or master region.
      // OpenMP [2.16, Nesting of Regions, p. 2]
      //  A barrier region may not be closely nested inside a worksharing,
      //  explicit task, critical, ordered, atomic, or master region.
      NestingProhibited =
          Kind == OMPD_for || Kind == OMPD_sections || Kind == OMPD_for_simd ||
          Kind == OMPD_distribute_simd || Kind == OMPD_distribute ||
          Kind == OMPD_distribute_parallel_for ||
          Kind == OMPD_distribute_parallel_for_simd || Kind == OMPD_single ||
          Kind == OMPD_barrier || Kind == OMPD_cancel ||
          Kind == OMPD_cancellation_point;
      Region = "a master";
      break;
    case OMPD_critical:
      // OpenMP [2.16, Nesting of Regions, p. 1]
      //  A worksharing region may not be closely nested inside a worksharing,
      //  explicit task, critical, ordered, atomic, or master region.
      // OpenMP [2.16, Nesting of Regions, p. 2]
      //  A barrier region may not be closely nested inside a worksharing,
      //  explicit task, critical, ordered, atomic, or master region.
      // OpenMP [2.16, Nesting of Regions, p. 4]
      //  An ordered region may not be closely nested inside a critical, atomic,
      //  or explicit task region.
      NestingProhibited =
          Kind == OMPD_for || Kind == OMPD_sections || Kind == OMPD_for_simd ||
          Kind == OMPD_distribute_simd || Kind == OMPD_distribute ||
          Kind == OMPD_distribute_parallel_for ||
          Kind == OMPD_distribute_parallel_for_simd || Kind == OMPD_single ||
          HasNamedDirective || Kind == OMPD_barrier || Kind == OMPD_ordered ||
          Kind == OMPD_cancel || Kind == OMPD_cancellation_point;
      Region = "a critical";
      break;
    case OMPD_atomic:
      // OpenMP [2.16, Nesting of Regions, p. 7]
      //  OpenMP constructs may not be nested inside an atomic region.
      NestingProhibited = true;
      Region = "an atomic";
      break;
    case OMPD_simd:
      // OpenMP [2.16, Nesting of Regions, p. 8]
      //  OpenMP constructs may not be nested inside a simd region.
      NestingProhibited = true;
      Region = "a simd";
      break;
    case OMPD_for_simd:
      // OpenMP [2.16, Nesting of Regions, p. 8]
      //  OpenMP constructs may not be nested inside a simd region.
      NestingProhibited = true;
      Region = "a for simd";
      break;
    case OMPD_distribute_simd:
      // OpenMP [2.16, Nesting of Regions, p. 8]
      //  OpenMP constructs may not be nested inside a simd region.
      NestingProhibited = true;
      Region = "a distribute simd";
      break;
    case OMPD_parallel_for_simd:
      // OpenMP [2.16, Nesting of Regions, p. 8]
      //  OpenMP constructs may not be nested inside a simd region.
      NestingProhibited = true;
      Region = "a parallel for simd";
      break;
    case OMPD_distribute_parallel_for_simd:
      // OpenMP [2.16, Nesting of Regions, p. 8]
      //  OpenMP constructs may not be nested inside a simd region.
      NestingProhibited = true;
      Region = "a distribute parallel for simd";
      break;
    case OMPD_teams_distribute_parallel_for_simd:
      // OpenMP [2.16, Nesting of Regions, p. 8]
      //  OpenMP constructs may not be nested inside a simd region.
      NestingProhibited = true;
      Region = "a teams distribute parallel for simd";
      break;
    case OMPD_target_teams_distribute_parallel_for_simd:
      // OpenMP [2.16, Nesting of Regions, p. 8]
      //  OpenMP constructs may not be nested inside a simd region.
      NestingProhibited = true;
      Region = "a target teams distribute parallel for simd";
      break;
    case OMPD_ordered:
      // OpenMP [2.16, Nesting of Regions, p. 1]
      //  A worksharing region may not be closely nested inside a worksharing,
      //  explicit task, critical, ordered, atomic, or master region.
      // OpenMP [2.16, Nesting of Regions, p. 2]
      //  A barrier region may not be closely nested inside a worksharing,
      //  explicit task, critical, ordered, atomic, or master region.
      // OpenMP [2.16, Nesting of Regions, p. 3]
      //  A master region may not be closely nested inside a worksharing,
      // atomic,
      //  or explicit task region.
      NestingProhibited =
          Kind == OMPD_for || Kind == OMPD_sections || Kind == OMPD_for_simd ||
          Kind == OMPD_distribute_simd || Kind == OMPD_distribute ||
          Kind == OMPD_distribute_parallel_for ||
          Kind == OMPD_distribute_parallel_for_simd || Kind == OMPD_single ||
          Kind == OMPD_master || Kind == OMPD_barrier || Kind == OMPD_cancel ||
          Kind == OMPD_cancellation_point;
      Region = "an ordered";
      break;
    case OMPD_teams:
      // OpenMP [2.16, Nesting of Regions, p. 11]
      // distribute, parallel, parallel sections, parallel workshare, and
      // the parallel loop and parallel loop SIMD constructs are the only
      // OpenMP constructs that can be closely nested in the teams region.
      NestingProhibited =
          Kind == OMPD_for || Kind == OMPD_sections || Kind == OMPD_single ||
          Kind == OMPD_for_simd || Kind == OMPD_simd || Kind == OMPD_master ||
          Kind == OMPD_barrier || Kind == OMPD_task || Kind == OMPD_ordered ||
          Kind == OMPD_teams || Kind == OMPD_atomic || Kind == OMPD_critical ||
          Kind == OMPD_taskgroup || Kind == OMPD_cancel ||
          Kind == OMPD_cancellation_point || Kind == OMPD_target_teams ||
          Kind == OMPD_teams_distribute || Kind == OMPD_teams_distribute_simd ||
          Kind == OMPD_target_teams_distribute ||
          Kind == OMPD_target_teams_distribute_simd ||
          Kind == OMPD_teams_distribute_parallel_for ||
          Kind == OMPD_teams_distribute_parallel_for_simd ||
          Kind == OMPD_target_teams_distribute_parallel_for ||
          Kind == OMPD_target_teams_distribute_parallel_for_simd;
      Region = "a teams";
      break;
    case OMPD_teams_distribute:
      NestingProhibited =
          Kind == OMPD_for || Kind == OMPD_sections || Kind == OMPD_for_simd ||
          Kind == OMPD_distribute_simd || Kind == OMPD_distribute ||
          Kind == OMPD_distribute_parallel_for ||
          Kind == OMPD_distribute_parallel_for_simd || Kind == OMPD_single ||
          Kind == OMPD_master || Kind == OMPD_barrier ||
          (Kind == OMPD_cancel && !ConstructTypeMatches) ||
          (Kind == OMPD_cancellation_point && !ConstructTypeMatches);
      Region = "a teams distribute";
      break;
    case OMPD_target_teams_distribute:
      NestingProhibited =
          Kind == OMPD_for || Kind == OMPD_sections || Kind == OMPD_for_simd ||
          Kind == OMPD_distribute_simd || Kind == OMPD_distribute ||
          Kind == OMPD_distribute_parallel_for ||
          Kind == OMPD_distribute_parallel_for_simd || Kind == OMPD_single ||
          Kind == OMPD_master || Kind == OMPD_barrier ||
          (Kind == OMPD_cancel && !ConstructTypeMatches) ||
          (Kind == OMPD_cancellation_point && !ConstructTypeMatches);
      Region = "a target teams distribute";
      break;
    case OMPD_teams_distribute_simd:
      // OpenMP [2.16, Nesting of Regions, p. 8]
      //  OpenMP constructs may not be nested inside a simd region.
      NestingProhibited = true;
      Region = "a teams distribute simd";
      break;
    case OMPD_target_teams_distribute_simd:
      // OpenMP [2.16, Nesting of Regions, p. 8]
      //  OpenMP constructs may not be nested inside a simd region.
      NestingProhibited = true;
      Region = "a target teams distribute simd";
      break;
    case OMPD_target_teams:
      // OpenMP [2.16, Nesting of Regions, p. 11]
      // distribute, parallel, parallel sections, parallel workshare, and
      // the parallel loop and parallel loop SIMD constructs are the only
      // OpenMP constructs that can be closely nested in the teams region.
      NestingProhibited =
          Kind == OMPD_for || Kind == OMPD_sections || Kind == OMPD_single ||
          Kind == OMPD_for_simd || Kind == OMPD_simd || Kind == OMPD_master ||
          Kind == OMPD_barrier || Kind == OMPD_task || Kind == OMPD_ordered ||
          Kind == OMPD_teams || Kind == OMPD_atomic || Kind == OMPD_critical ||
          Kind == OMPD_taskgroup || Kind == OMPD_cancel ||
          Kind == OMPD_cancellation_point || Kind == OMPD_target_teams ||
          Kind == OMPD_teams_distribute || Kind == OMPD_teams_distribute_simd ||
          Kind == OMPD_target_teams_distribute ||
          Kind == OMPD_target_teams_distribute_simd ||
          Kind == OMPD_teams_distribute_parallel_for ||
          Kind == OMPD_teams_distribute_parallel_for_simd ||
          Kind == OMPD_target_teams_distribute_parallel_for ||
          Kind == OMPD_target_teams_distribute_parallel_for_simd;
      Region = "a target teams";
      break;
    case OMPD_distribute:
      NestingProhibited =
          Kind == OMPD_for || Kind == OMPD_sections || Kind == OMPD_for_simd ||
          Kind == OMPD_distribute_simd || Kind == OMPD_distribute ||
          Kind == OMPD_distribute_parallel_for ||
          Kind == OMPD_distribute_parallel_for_simd || Kind == OMPD_single ||
          Kind == OMPD_master || Kind == OMPD_barrier ||
          (Kind == OMPD_cancel && !ConstructTypeMatches) ||
          (Kind == OMPD_cancellation_point && !ConstructTypeMatches);
      Region = "a distribute";
      break;
    case OMPD_taskgroup:
      NestingProhibited =
          (Kind == OMPD_cancel) || (Kind == OMPD_cancellation_point);
      Region = "a taskgroup";
      break;
    default:
      break;
    }
    // OpenMP [2.16, Nesting of Regions, p. 6]
    //  A critical region may not be nested (closely or otherwise) inside a
    //  critical region with the same name. Note that this restriction is not
    //  sufficient to prevent deadlock.
    if (DirName.getName() && Kind == OMPD_critical) {
      HasNamedDirective = DSAStack->hasDirectiveWithName(Kind, DirName);
      CloseNesting = false;
      NestingProhibited = HasNamedDirective;
      Region = "a critical";
    }
    if (NestingProhibited) {
      Diag(StartLoc, diag::err_omp_prohibited_region)
          << CloseNesting << Region << HasNamedDirective << DirName.getName();
      return StmtError();
    }
    // OpenMP [2.16, Nesting of Regions, p. 5]
    //  An ordered region must be closely nested inside a loop region (or
    //  parallel loop region) with an ordered clause.
    if (Kind == OMPD_ordered &&
        (ParentKind != OMPD_unknown && !DSAStack->isParentRegionOrdered())) {
      Diag(StartLoc, diag::err_omp_prohibited_ordered_region);
      return StmtError();
    }
    if (Kind == OMPD_cancel && ParentKind != OMPD_unknown) {
      // OpenMP [2.16, Nesting of Regions, p. 13]
      // the cancel construct must be nested inside a taskgroup region.
      if (ConstructType == OMPD_taskgroup &&
          !DSAStack->hasDirective(OMPD_taskgroup)) {
        Diag(StartLoc, diag::err_omp_prohibited_cancel_region);
        return StmtError();
      }
      // OpenMP [2.13.1, cancel Construct, Restriction]
      // A worksharing construct that is cancelled must not have a nowait
      // clause.
      if ((ConstructType == OMPD_for || ConstructType == OMPD_sections) &&
          DSAStack->isRegionNowait()) {
        Diag(StartLoc, diag::err_omp_prohibited_cancel_region_nowait);
        return StmtError();
      }
      // OpenMP [2.13.1, cancel Construct, Restriction]
      // A loop construct that is cancelled must not have an ordered clause.
      if (ConstructType == OMPD_for && DSAStack->isRegionOrdered()) {
        Diag(StartLoc, diag::err_omp_prohibited_cancel_region_ordered);
        return StmtError();
      }
    }
    // OpenMP [2.16, Nesting of Regions, p. 5]
    //  A distribute construct must be closely nested in a teams region.
    if ((Kind == OMPD_distribute || Kind == OMPD_distribute_simd ||
         Kind == OMPD_distribute_parallel_for ||
         Kind == OMPD_distribute_parallel_for_simd) &&
        (ParentKind != OMPD_unknown && ParentKind != OMPD_teams &&
         ParentKind != OMPD_target_teams)) {
      Diag(StartLoc, diag::err_omp_prohibited_distribute_region);
      return StmtError();
    }
    // OpenMP [2.16, Nesting of Regions, p. 10]
    //  If specified, a teams construct must be contained within a target
    // construct.
    if ((Kind == OMPD_teams || Kind == OMPD_teams_distribute ||
         Kind == OMPD_teams_distribute_simd ||
         Kind == OMPD_teams_distribute_parallel_for ||
         Kind == OMPD_teams_distribute_parallel_for_simd) &&
        ParentKind != OMPD_target) {
      Diag(StartLoc, diag::err_omp_prohibited_teams_region);
      return StmtError();
    }
  }
  if (Kind == OMPD_task) {
    assert(AStmt && isa<CapturedStmt>(AStmt) && "Captured statement expected");
    // Check default data sharing attributes for captured variables.
    DSAAttrChecker DSAChecker(DSAStack, *this, cast<CapturedStmt>(AStmt));
    DSAChecker.Visit(cast<CapturedStmt>(AStmt)->getCapturedStmt());
    if (DSAChecker.isErrorFound())
      return StmtError();
    if (DSAChecker.getImplicitFirstprivate().size() > 0) {
      if (OMPClause *Implicit = ActOnOpenMPFirstPrivateClause(
              DSAChecker.getImplicitFirstprivate(), SourceLocation(),
              SourceLocation())) {
        ClausesWithImplicit.push_back(Implicit);
        if (Implicit &&
            cast<OMPFirstPrivateClause>(Implicit)->varlist_size() !=
                DSAChecker.getImplicitFirstprivate().size())
          ErrorFound = true;
      } else
        ErrorFound = true;
    }
  }
  ClausesWithImplicit.append(Clauses.begin(), Clauses.end());

  StmtResult Res = StmtError();
  switch (Kind) {
  case OMPD_parallel:
    Res = ActOnOpenMPParallelDirective(ClausesWithImplicit, AStmt, StartLoc,
                                       EndLoc);
    break;
  case OMPD_parallel_for:
    Res = ActOnOpenMPParallelForDirective(Kind, ClausesWithImplicit, AStmt,
                                          StartLoc, EndLoc);
    break;
  case OMPD_for:
    Res = ActOnOpenMPForDirective(Kind, ClausesWithImplicit, AStmt, StartLoc,
                                  EndLoc);
    break;
  case OMPD_parallel_sections:
    Res = ActOnOpenMPParallelSectionsDirective(Kind, ClausesWithImplicit, AStmt,
                                               StartLoc, EndLoc);
    break;
  case OMPD_sections:
    Res = ActOnOpenMPSectionsDirective(Kind, ClausesWithImplicit, AStmt,
                                       StartLoc, EndLoc);
    break;
  case OMPD_section:
    assert(Clauses.empty() && "Clauses are not allowed for section");
    Res = ActOnOpenMPSectionDirective(AStmt, StartLoc, EndLoc);
    break;
  case OMPD_single:
    Res = ActOnOpenMPSingleDirective(ClausesWithImplicit, AStmt, StartLoc,
                                     EndLoc);
    break;
  case OMPD_task:
    Res =
        ActOnOpenMPTaskDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OMPD_taskyield:
    assert(Clauses.empty() && !AStmt &&
           "Clauses and statement are not allowed for taskyield");
    Res = ActOnOpenMPTaskyieldDirective(StartLoc, EndLoc);
    break;
  case OMPD_master:
    assert(Clauses.empty() && "Clauses are not allowed for master");
    Res = ActOnOpenMPMasterDirective(AStmt, StartLoc, EndLoc);
    break;
  case OMPD_critical:
    assert(Clauses.empty() && "Clauses are not allowed for critical");
    Res = ActOnOpenMPCriticalDirective(DirName, AStmt, StartLoc, EndLoc);
    break;
  case OMPD_barrier:
    assert(Clauses.empty() && !AStmt &&
           "Clauses and statement are not allowed for barrier");
    Res = ActOnOpenMPBarrierDirective(StartLoc, EndLoc);
    break;
  case OMPD_taskwait:
    assert(Clauses.empty() && !AStmt &&
           "Clauses and statement are not allowed for taskwait");
    Res = ActOnOpenMPTaskwaitDirective(StartLoc, EndLoc);
    break;
  case OMPD_taskgroup:
    assert(Clauses.empty() && "Clauses are not allowed for taskgroup");
    Res = ActOnOpenMPTaskgroupDirective(AStmt, StartLoc, EndLoc);
    break;
  case OMPD_atomic:
    Res = ActOnOpenMPAtomicDirective(ClausesWithImplicit, AStmt, StartLoc,
                                     EndLoc);
    break;
  case OMPD_flush:
    assert(!AStmt && "Statement is not allowed for flush");
    Res = ActOnOpenMPFlushDirective(ClausesWithImplicit, StartLoc, EndLoc);
    break;
  case OMPD_ordered:
    assert(Clauses.empty() && "Clauses are not allowed for ordered");
    Res = ActOnOpenMPOrderedDirective(AStmt, StartLoc, EndLoc);
    break;
  case OMPD_simd:
    Res = ActOnOpenMPSimdDirective(Kind, ClausesWithImplicit, AStmt, StartLoc,
                                   EndLoc);
    break;
  case OMPD_for_simd:
    Res = ActOnOpenMPForSimdDirective(Kind, ClausesWithImplicit, AStmt,
                                      StartLoc, EndLoc);
    break;
  case OMPD_parallel_for_simd:
    Res = ActOnOpenMPParallelForSimdDirective(Kind, ClausesWithImplicit, AStmt,
                                              StartLoc, EndLoc);
    break;
  case OMPD_distribute_simd:
    Res = ActOnOpenMPDistributeSimdDirective(Kind, ClausesWithImplicit, AStmt,
                                             StartLoc, EndLoc);
    break;
  case OMPD_distribute_parallel_for:
    Res = ActOnOpenMPDistributeParallelForDirective(Kind, ClausesWithImplicit,
                                                    AStmt, StartLoc, EndLoc);
    break;
  case OMPD_distribute_parallel_for_simd:
    Res = ActOnOpenMPDistributeParallelForSimdDirective(
        Kind, ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OMPD_teams_distribute_parallel_for:
    Res = ActOnOpenMPTeamsDistributeParallelForDirective(
        Kind, ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OMPD_teams_distribute_parallel_for_simd:
    Res = ActOnOpenMPTeamsDistributeParallelForSimdDirective(
        Kind, ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OMPD_target_teams_distribute_parallel_for:
    Res = ActOnOpenMPTargetTeamsDistributeParallelForDirective(
        Kind, ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OMPD_target_teams_distribute_parallel_for_simd:
    Res = ActOnOpenMPTargetTeamsDistributeParallelForSimdDirective(
        Kind, ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OMPD_teams:
    Res =
        ActOnOpenMPTeamsDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OMPD_target_teams:
    Res = ActOnOpenMPTargetTeamsDirective(ClausesWithImplicit, AStmt, StartLoc,
                                          EndLoc);
    break;
  case OMPD_distribute:
    Res = ActOnOpenMPDistributeDirective(ClausesWithImplicit, AStmt, StartLoc,
                                         EndLoc);
    break;
  case OMPD_target:
    Res = ActOnOpenMPTargetDirective(ClausesWithImplicit, AStmt, StartLoc,
                                     EndLoc);
    break;
  case OMPD_target_data:
    Res = ActOnOpenMPTargetDataDirective(ClausesWithImplicit, AStmt, StartLoc,
                                         EndLoc);
    break;
  case OMPD_target_update:
    assert(!AStmt && "Statement is not allowed for target update");
    Res =
        ActOnOpenMPTargetUpdateDirective(ClausesWithImplicit, StartLoc, EndLoc);
    break;
  case OMPD_cancel:
    assert(!AStmt && "Statement is not allowed for cancel");
    if (ConstructType == OMPD_unknown)
      return StmtError();
    Res = ActOnOpenMPCancelDirective(ClausesWithImplicit, StartLoc, EndLoc,
                                     ConstructType);
    break;
  case OMPD_cancellation_point:
    assert(!AStmt && "Statement is not allowed for cancellation point");
    assert(Clauses.empty() && "Clauses are not allowed for cancellation point");
    if (ConstructType == OMPD_unknown)
      return StmtError();
    Res =
        ActOnOpenMPCancellationPointDirective(StartLoc, EndLoc, ConstructType);
    break;
  case OMPD_teams_distribute:
    Res = ActOnOpenMPTeamsDistributeDirective(ClausesWithImplicit, AStmt,
                                              StartLoc, EndLoc);
    break;
  case OMPD_teams_distribute_simd:
    Res = ActOnOpenMPTeamsDistributeSimdDirective(Kind, ClausesWithImplicit,
                                                  AStmt, StartLoc, EndLoc);
    break;
  case OMPD_target_teams_distribute:
    Res = ActOnOpenMPTargetTeamsDistributeDirective(ClausesWithImplicit, AStmt,
                                                    StartLoc, EndLoc);
    break;
  case OMPD_target_teams_distribute_simd:
    Res = ActOnOpenMPTargetTeamsDistributeSimdDirective(
        Kind, ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  default:
    break;
  }
  // Additional analysis for all directives except for task
  switch (Kind) {
  case OMPD_taskyield:
  case OMPD_barrier:
  case OMPD_taskwait:
  case OMPD_flush:
  case OMPD_cancel:
  case OMPD_cancellation_point:
  case OMPD_target_update:
  case OMPD_task:
    break;
  default: {
    assert(AStmt && isa<CapturedStmt>(AStmt) && "Captured statement expected");
    // Check default data sharing attributes for captured variables.
    DSAAttrChecker DSAChecker(DSAStack, *this, cast<CapturedStmt>(AStmt));
    DSAChecker.Visit(cast<CapturedStmt>(AStmt)->getCapturedStmt());
    if (DSAChecker.isErrorFound())
      return StmtError();
    if (DSAChecker.getImplicitFirstprivate().size() > 0) {
      if (OMPClause *Implicit = ActOnOpenMPFirstPrivateClause(
              DSAChecker.getImplicitFirstprivate(), SourceLocation(),
              SourceLocation())) {
        ClausesWithImplicit.push_back(Implicit);
        if (Implicit &&
            cast<OMPFirstPrivateClause>(Implicit)->varlist_size() !=
                DSAChecker.getImplicitFirstprivate().size())
          ErrorFound = true;
      } else
        ErrorFound = true;
    }
    break;
  }
  }

  if (ErrorFound)
    return StmtError();

  return Res;
}

StmtResult Sema::ActOnOpenMPParallelDirective(ArrayRef<OMPClause *> Clauses,
                                              Stmt *AStmt,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc) {

  getCurFunction()->setHasBranchProtectedScope();
  return OMPParallelDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                      AStmt);
}

namespace {
class ForBreakStmtChecker : public StmtVisitor<ForBreakStmtChecker, bool> {
  Stmt *Break;

public:
  bool VisitBreakStmt(BreakStmt *S) {
    Break = S;
    return true;
  }
  bool VisitSwitchStmt(SwitchStmt *S) { return false; }
  bool VisitWhileStmt(WhileStmt *S) { return false; }
  bool VisitDoStmt(DoStmt *S) { return false; }
  bool VisitForStmt(ForStmt *S) { return false; }
  bool VisitCXXForRangeStmt(CXXForRangeStmt *S) { return false; }
  bool VisitStmt(Stmt *S) {
    for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E;
         ++I) {
      if (*I && Visit(*I))
        return true;
    }
    return false;
  }
  ForBreakStmtChecker() {}
  Stmt *getBreak() { return Break; }
};
}

namespace {
class EhChecker : public StmtVisitor<EhChecker, bool> {
  Stmt *BadStmt;

public:
  bool VisitCXXCatchStmt(CXXCatchStmt *S) {
    BadStmt = S;
    return true;
  }
  bool VisitCXXThrowExpr(CXXThrowExpr *S) {
    BadStmt = S;
    return true;
  }
  bool VisitCXXTryStmt(CXXTryStmt *S) {
    BadStmt = S;
    return true;
  }
  bool VisitStmt(Stmt *S) {
    for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E;
         ++I) {
      if (*I && Visit(*I))
        return true;
    }
    return false;
  }
  EhChecker() {}
  Stmt *getBadStmt() { return BadStmt; }
};
}

bool Sema::CollapseOpenMPLoop(OpenMPDirectiveKind Kind,
                              ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
                              SourceLocation StartLoc, SourceLocation EndLoc,
                              Expr *&NewVar, Expr *&NewEnd,
                              Expr *&NewVarCntExpr, Expr *&NewFinal,
                              SmallVector<Expr *, 4> &VarCnts) {
  // This is helper routine to process collapse clause that
  // can be met in directives 'for', 'simd', 'for simd' and others.
  //
  // OpenMP [2.7.1, Loop construct, Description]
  //  The collapse clause may be used to specify how many loops are
  //  associated with the loop construct.
  //
  NewVar = 0;
  NewEnd = 0;
  NewVarCntExpr = 0;
  NewFinal = 0;
  VarCnts.clear();
  FunctionDecl *FD = getCurFunctionDecl();
  if (FD && FD->isDependentContext())
    return true;
  SmallVector<Expr *, 4> Ends;
  SmallVector<Expr *, 4> Incrs;
  SmallVector<Expr *, 4> Inits;
  SmallVector<BinaryOperatorKind, 4> OpKinds;
  unsigned StmtCount = 1;
  for (ArrayRef<OMPClause *>::iterator I = Clauses.begin(), E = Clauses.end();
       I != E; ++I) {
    if (OMPCollapseClause *Clause = dyn_cast_or_null<OMPCollapseClause>(*I)) {
      IntegerLiteral *IL = cast<IntegerLiteral>(Clause->getNumForLoops());
      StmtCount = IL->getValue().getLimitedValue();
      break;
    }
  }
  Stmt *CStmt = AStmt;
  while (CapturedStmt *CS = dyn_cast_or_null<CapturedStmt>(CStmt))
    CStmt = CS->getCapturedStmt();
  while (AttributedStmt *AS = dyn_cast_or_null<AttributedStmt>(CStmt))
    CStmt = AS->getSubStmt();
  bool SkipExprCount = false;
  for (unsigned Cnt = 0; Cnt < StmtCount; ++Cnt) {
    Expr *NewEnd;
    Expr *NewIncr;
    Expr *Init;
    Expr *VarCnt;
    BinaryOperatorKind OpKind;
    if (isNotOpenMPCanonicalLoopForm(CStmt, Kind, NewEnd, NewIncr, Init, VarCnt,
                                     OpKind))
      return false;
    if (NewEnd->getType()->isDependentType() ||
        NewIncr->getType()->isDependentType() ||
        Init->getType()->isDependentType() ||
        VarCnt->getType()->isDependentType())
      SkipExprCount = true;
    Ends.push_back(NewEnd);
    Incrs.push_back(NewIncr);
    Inits.push_back(Init);
    VarCnts.push_back(VarCnt);
    OpKinds.push_back(OpKind);
    CStmt = cast<ForStmt>(CStmt)->getBody();
    bool SkippedContainers = false;
    while (!SkippedContainers) {
      if (AttributedStmt *AS = dyn_cast_or_null<AttributedStmt>(CStmt))
        CStmt = AS->getSubStmt();
      else if (CompoundStmt *CS = dyn_cast_or_null<CompoundStmt>(CStmt)) {
        if (CS->size() != 1) {
          SkippedContainers = true;
        } else {
          CStmt = CS->body_back();
        }
      } else
        SkippedContainers = true;
    }
  }

  ForBreakStmtChecker Check;
  if (CStmt && Check.Visit(CStmt)) {
    Diag(Check.getBreak()->getLocStart(), diag::err_omp_for_cannot_break)
        << getOpenMPDirectiveName(Kind);
    return false;
  }

  if (Kind == OMPD_simd || Kind == OMPD_for_simd ||
      Kind == OMPD_parallel_for_simd || Kind == OMPD_distribute_simd ||
      Kind == OMPD_distribute_parallel_for_simd ||
      Kind == OMPD_teams_distribute_simd ||
      Kind == OMPD_target_teams_distribute_simd ||
      Kind == OMPD_teams_distribute_parallel_for_simd ||
      Kind == OMPD_target_teams_distribute_parallel_for_simd) {
    // OpenMP [2.8.1] No exception can be raised in the simd region.
    EhChecker Check;
    if (CStmt && Check.Visit(CStmt)) {
      Diag(Check.getBadStmt()->getLocStart(), diag::err_omp_for_cannot_have_eh)
          << getOpenMPDirectiveName(Kind);
      return false;
    }
  }

  // Build ending for Idx var;
  NewEnd = 0;
  NewVar = 0;
  NewVarCntExpr = 0;
  NewFinal = 0;

  if (!SkipExprCount) {
    NewEnd = Ends[0];
    for (unsigned I = 1; I < StmtCount; ++I) {
      ExprResult Res = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Mul,
                                  Ends[I], NewEnd);
      if (!Res.isUsable())
        return false;
      NewEnd = Res.get();
    }
    QualType IdxTy = NewEnd->getType();
    TypeSourceInfo *TI = Context.getTrivialTypeSourceInfo(IdxTy, StartLoc);
    VarDecl *Idx = VarDecl::Create(Context, Context.getTranslationUnitDecl(),
                                   StartLoc, StartLoc, 0, IdxTy, TI, SC_Static);
    Idx->setImplicit();
    Idx->addAttr(new (Context) UnusedAttr(SourceLocation(), Context, 0));
    Context.getTranslationUnitDecl()->addHiddenDecl(Idx);
    ExprResult IdxExprRes = BuildDeclRefExpr(Idx, IdxTy, VK_LValue, StartLoc);
    NewVar = IdxExprRes.get();

    // Build new values for actual indexes.

    // We can go either from outer loop to inner [0, StmtCount, 1] or reverse
    // [StmtCount-1, -1, -1] in the case of 'omp for', but in an 'omp simd'
    // directive the reverse order is required because we may have loop-carried
    // dependencies (as specified by 'safelen' clause).
    // For cache locality reasons this may be also preferred for 'omp for', as
    // usually programs walk inner array dimensions first.
    int LoopIdBegin = StmtCount - 1;
    int LoopIdEnd = -1;
    int LoopIdStep = -1;

    Expr *NewDiv = Ends[LoopIdBegin];
    Expr *IdxRVal = DefaultLvalueConversion(NewVar).get();
    if (!IdxRVal)
      return false;
    ExprResult Res =
        BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Sub, NewEnd,
                   ActOnIntegerConstant(SourceLocation(), 1).get());
    if (!Res.isUsable())
      return false;
    NewEnd = Res.get();

    Expr *NewIncr = IdxRVal;
    if (StmtCount != 1) {
      NewIncr = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Rem, IdxRVal,
                           Ends[LoopIdBegin]).get();
      if (!NewIncr)
        return false;
    }

    NewIncr = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Mul, NewIncr,
                         Incrs[LoopIdBegin]).get();
    if (!NewIncr)
      return false;
    NewFinal = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Assign,
                          VarCnts[LoopIdBegin], Inits[LoopIdBegin]).get();
    if (!NewFinal)
      return false;
    NewFinal = IgnoredValueConversions(NewFinal).get();
    if (!NewFinal)
      return false;
    Expr *NewFinal1 = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Mul,
                                 Ends[LoopIdBegin], Incrs[LoopIdBegin]).get();
    if (!NewFinal1)
      return false;
    NewFinal1 = BuildBinOp(DSAStack->getCurScope(), StartLoc,
                           (OpKinds[LoopIdBegin] == BO_Add) ? BO_AddAssign
                                                            : BO_SubAssign,
                           VarCnts[LoopIdBegin], NewFinal1).get();
    if (!NewFinal1)
      return false;
    NewFinal1 = IgnoredValueConversions(NewFinal1).get();
    if (!NewFinal1)
      return false;
    NewFinal =
        CreateBuiltinBinOp(StartLoc, BO_Comma, NewFinal, NewFinal1).get();
    if (!NewFinal)
      return false;
    // Expr *NewStep = BuildBinOp(DSAStack->getCurScope(), StartLoc,
    //                  OpKinds[LoopIdBegin], Inits[LoopIdBegin],
    // NewIncr).get();
    // if (!NewStep) return false;
    // NewVarCntExpr = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Assign,
    //                           VarCnts[LoopIdBegin], NewStep).get();
    NewVarCntExpr = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Assign,
                               VarCnts[LoopIdBegin], Inits[LoopIdBegin]).get();
    if (!NewVarCntExpr)
      return false;
    NewVarCntExpr = IgnoredValueConversions(NewVarCntExpr).get();
    if (!NewVarCntExpr)
      return false;
    Expr *NewVarCntExpr1 =
        BuildBinOp(DSAStack->getCurScope(), StartLoc,
                   (OpKinds[LoopIdBegin] == BO_Add) ? BO_AddAssign
                                                    : BO_SubAssign,
                   VarCnts[LoopIdBegin], NewIncr).get();
    if (!NewVarCntExpr1)
      return false;
    NewVarCntExpr1 = IgnoredValueConversions(NewVarCntExpr1).get();
    if (!NewVarCntExpr1)
      return false;
    NewVarCntExpr = CreateBuiltinBinOp(StartLoc, BO_Comma, NewVarCntExpr,
                                       NewVarCntExpr1).get();
    if (!NewVarCntExpr)
      return false;

    for (int I = LoopIdBegin + LoopIdStep; I != LoopIdEnd; I += LoopIdStep) {
      NewIncr = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Div, IdxRVal,
                           NewDiv).get();
      if (!NewIncr)
        return false;

      if (I + LoopIdStep != LoopIdEnd) {
        NewIncr = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Rem, NewIncr,
                             Ends[I]).get();
        if (!NewIncr)
          return false;
      }

      NewIncr = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Mul, NewIncr,
                           Incrs[I]).get();
      if (!NewIncr)
        return false;
      NewFinal1 = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Assign,
                             VarCnts[I], Inits[I]).get();
      if (!NewFinal1)
        return false;
      NewFinal =
          CreateBuiltinBinOp(StartLoc, BO_Comma, NewFinal, NewFinal1).get();
      if (!NewFinal)
        return false;
      NewFinal1 = IgnoredValueConversions(NewFinal1).get();
      if (!NewFinal1)
        return false;
      NewFinal1 = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Mul, Ends[I],
                             Incrs[I]).get();
      if (!NewFinal1)
        return false;
      NewFinal1 =
          BuildBinOp(DSAStack->getCurScope(), StartLoc,
                     (OpKinds[I] == BO_Add) ? BO_AddAssign : BO_SubAssign,
                     VarCnts[I], NewFinal1).get();
      if (!NewFinal1)
        return false;
      NewFinal1 = IgnoredValueConversions(NewFinal1).get();
      if (!NewFinal1)
        return false;
      NewFinal =
          CreateBuiltinBinOp(StartLoc, BO_Comma, NewFinal, NewFinal1).get();
      if (!NewFinal)
        return false;
      //      NewStep = BuildBinOp(DSAStack->getCurScope(), StartLoc,
      // OpKinds[I],
      //                           Inits[I], NewIncr).get();
      //      if (!NewStep) return false;
      //      Expr *NewVarCntExpr1 = BuildBinOp(DSAStack->getCurScope(),
      // StartLoc, BO_Assign,
      //                                        VarCnts[I], NewStep).get();
      NewVarCntExpr1 = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Assign,
                                  VarCnts[I], Inits[I]).get();
      if (!NewVarCntExpr1)
        return false;
      NewVarCntExpr1 = IgnoredValueConversions(NewVarCntExpr1).get();
      if (!NewVarCntExpr1)
        return false;
      NewVarCntExpr = CreateBuiltinBinOp(StartLoc, BO_Comma, NewVarCntExpr,
                                         NewVarCntExpr1).get();
      if (!NewVarCntExpr)
        return false;
      NewVarCntExpr1 =
          BuildBinOp(DSAStack->getCurScope(), StartLoc,
                     (OpKinds[I] == BO_Add) ? BO_AddAssign : BO_SubAssign,
                     VarCnts[I], NewIncr).get();
      if (!NewVarCntExpr1)
        return false;
      NewVarCntExpr1 = IgnoredValueConversions(NewVarCntExpr1).get();
      if (!NewVarCntExpr1)
        return false;
      NewVarCntExpr = CreateBuiltinBinOp(StartLoc, BO_Comma, NewVarCntExpr,
                                         NewVarCntExpr1).get();
      if (!NewVarCntExpr)
        return false;
      NewDiv = BuildBinOp(DSAStack->getCurScope(), StartLoc, BO_Mul, NewDiv,
                          Ends[I]).get();
      if (!NewDiv)
        return false;
    }
    NewVarCntExpr = IgnoredValueConversions(NewVarCntExpr).get();
    NewFinal = IgnoredValueConversions(NewFinal).get();
    NewFinal = ActOnFinishFullExpr(NewFinal).get();
    NewVarCntExpr = ActOnFinishFullExpr(NewVarCntExpr).get();
    NewEnd = ActOnFinishFullExpr(NewEnd).get();
  }
  return true;
}

StmtResult Sema::ActOnOpenMPForDirective(OpenMPDirectiveKind Kind,
                                         ArrayRef<OMPClause *> Clauses,
                                         Stmt *AStmt, SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();
  return OMPForDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                 NewVar, NewEnd, NewVarCntExpr, NewFinal,
                                 VarCnts);
}

StmtResult Sema::ActOnOpenMPParallelForDirective(OpenMPDirectiveKind Kind,
                                                 ArrayRef<OMPClause *> Clauses,
                                                 Stmt *AStmt,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();
  return OMPParallelForDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                         AStmt, NewVar, NewEnd, NewVarCntExpr,
                                         NewFinal, VarCnts);
}

CapturedStmt *Sema::AddSimdArgsIntoCapturedStmt(CapturedStmt *Cap,
                                                Expr *NewVar) {
  CapturedDecl *CD = Cap->getCapturedDecl();
  DeclContext *DC = CapturedDecl::castToDeclContext(CD);
  assert(CD->getNumParams() == 3);
  if (!DC->isDependentContext()) {
    assert(NewVar);
    QualType IndexType = NewVar->getType();
    ImplicitParamDecl *Index = 0, *LastIter = 0;
    Index = ImplicitParamDecl::Create(getASTContext(), DC, SourceLocation(), 0,
                                      IndexType);
    DC->addDecl(Index);
    CD->setParam(1, Index);
    LastIter = ImplicitParamDecl::Create(getASTContext(), DC, SourceLocation(),
                                         0, Context.BoolTy);
    DC->addDecl(LastIter);
    CD->setParam(2, LastIter);
  }
  RecordDecl *RD = const_cast<RecordDecl *>(Cap->getCapturedRecordDecl());

  // Extract the captures from AStmt and insert them into CapturedBody.
  SmallVector<CapturedStmt::Capture, 4> Captures;
  SmallVector<Expr *, 4> CaptureInits;
  CapturedStmt::capture_iterator I;
  CapturedStmt::capture_init_iterator J;
  for (I = Cap->capture_begin(), J = Cap->capture_init_begin();
       (I != Cap->capture_end()) && (J != Cap->capture_init_end()); ++I, ++J) {
    // Assuming that copy constructors are OK here.
    Captures.push_back(*I);
    CaptureInits.push_back(*J);
  }
  CapturedRegionKind CapKind = Cap->getCapturedRegionKind();
  Stmt *Body = Cap->getCapturedStmt();
  // Rebuild the captured stmt.
  CapturedStmt *CapturedBody = CapturedStmt::Create(
      getASTContext(), Body, CapKind, Captures, CaptureInits, CD, RD);
  CD->setBody(Body);

  return CapturedBody;
}

Stmt *Sema::AddDistributedParallelArgsIntoCapturedStmt(CapturedStmt *Cap,
                                                       Expr *NewVar,
                                                       Expr *&LowerBound,
                                                       Expr *&UpperBound) {
  CapturedDecl *CD = Cap->getCapturedDecl();
  DeclContext *DC = CapturedDecl::castToDeclContext(CD);
  VarDecl *LowerBoundVar = 0;
  VarDecl *UpperBoundVar = 0;
  if (!DC->isDependentContext()) {
    assert(NewVar);
    QualType VDTy = NewVar->getType();
    uint64_t TypeSize = 32;
    if (Context.getTypeSize(VDTy) > TypeSize)
      TypeSize = 64;
    VDTy = Context.getIntTypeForBitwidth(TypeSize, true);
    TypeSourceInfo *TI =
        Context.getTrivialTypeSourceInfo(VDTy, SourceLocation());
    LowerBoundVar = VarDecl::Create(Context, CurContext, SourceLocation(),
                                    SourceLocation(), 0, VDTy, TI, SC_Auto);
    UpperBoundVar = VarDecl::Create(Context, CurContext, SourceLocation(),
                                    SourceLocation(), 0, VDTy, TI, SC_Auto);
    LowerBound = DeclRefExpr::Create(Context, NestedNameSpecifierLoc(),
                                     SourceLocation(), LowerBoundVar, false,
                                     SourceLocation(), VDTy, VK_LValue);
    UpperBound = DeclRefExpr::Create(Context, NestedNameSpecifierLoc(),
                                     SourceLocation(), UpperBoundVar, false,
                                     SourceLocation(), VDTy, VK_LValue);
  }

  Stmt *Body = Cap->getCapturedStmt();

  ActOnCapturedRegionStart(Cap->getLocStart(), 0, Cap->getCapturedRegionKind(),
                           Cap->getCapturedDecl()->getNumParams());
  MarkVariableReferenced(Cap->getLocStart(), LowerBoundVar);
  MarkVariableReferenced(Cap->getLocStart(), UpperBoundVar);
  for (CapturedStmt::capture_iterator I = Cap->capture_begin(),
                                      E = Cap->capture_end();
       I != E; ++I) {
    if (I->capturesVariable())
      MarkVariableReferenced(I->getLocation(), I->getCapturedVar());
    else
      CheckCXXThisCapture(I->getLocation(), /*explicit*/ false);
  }
  StmtResult CapturedBody = ActOnCapturedRegionEnd(Body);

  return CapturedBody.get();
}

StmtResult Sema::ActOnOpenMPSimdDirective(OpenMPDirectiveKind Kind,
                                          ArrayRef<OMPClause *> Clauses,
                                          Stmt *AStmt, SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  // Add two arguments into captured stmt for index and last_iter.
  CapturedStmt *CapturedBody =
      AddSimdArgsIntoCapturedStmt(cast<CapturedStmt>(AStmt), NewVar);

  getCurFunction()->setHasBranchProtectedScope();

  // Rebuild the directive.
  return OMPSimdDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                  CapturedBody, NewVar, NewEnd, NewVarCntExpr,
                                  NewFinal, VarCnts);
}

StmtResult Sema::ActOnOpenMPForSimdDirective(OpenMPDirectiveKind Kind,
                                             ArrayRef<OMPClause *> Clauses,
                                             Stmt *AStmt,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  // Add two arguments into captured stmt for index and last_iter.
  CapturedStmt *CapturedBody =
      AddSimdArgsIntoCapturedStmt(cast<CapturedStmt>(AStmt), NewVar);

  getCurFunction()->setHasBranchProtectedScope();

  // Rebuild the directive.
  return OMPForSimdDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                     CapturedBody, NewVar, NewEnd,
                                     NewVarCntExpr, NewFinal, VarCnts);
}

StmtResult Sema::ActOnOpenMPParallelForSimdDirective(
    OpenMPDirectiveKind Kind, ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  // Add two arguments into captured stmt for index and last_iter.
  CapturedStmt *CapturedBody =
      AddSimdArgsIntoCapturedStmt(cast<CapturedStmt>(AStmt), NewVar);

  getCurFunction()->setHasBranchProtectedScope();

  // Rebuild the directive.
  return OMPParallelForSimdDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                             CapturedBody, NewVar, NewEnd,
                                             NewVarCntExpr, NewFinal, VarCnts);
}

StmtResult Sema::ActOnOpenMPDistributeSimdDirective(
    OpenMPDirectiveKind Kind, ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  // Add two arguments into captured stmt for index and last_iter.
  CapturedStmt *CapturedBody =
      AddSimdArgsIntoCapturedStmt(cast<CapturedStmt>(AStmt), NewVar);

  getCurFunction()->setHasBranchProtectedScope();

  // Rebuild the directive.
  return OMPDistributeSimdDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                            CapturedBody, NewVar, NewEnd,
                                            NewVarCntExpr, NewFinal, VarCnts);
}

StmtResult Sema::ActOnOpenMPDistributeParallelForDirective(
    OpenMPDirectiveKind Kind, ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();

  // Create variables for lower/upper bound
  Expr *LowerBound = 0;
  Expr *UpperBound = 0;
  if (NewVar && AStmt) {
    AStmt = AddDistributedParallelArgsIntoCapturedStmt(
        cast<CapturedStmt>(AStmt), NewVar, LowerBound, UpperBound);
  }
  // Rebuild the directive.
  return OMPDistributeParallelForDirective::Create(
      Context, StartLoc, EndLoc, Clauses, AStmt, NewVar, NewEnd, NewVarCntExpr,
      NewFinal, LowerBound, UpperBound, VarCnts);
}

StmtResult Sema::ActOnOpenMPDistributeParallelForSimdDirective(
    OpenMPDirectiveKind Kind, ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();

  // Create variables for lower/upper bound
  Expr *LowerBound = 0;
  Expr *UpperBound = 0;
  if (NewVar && AStmt) {
    AStmt = AddDistributedParallelArgsIntoCapturedStmt(
        cast<CapturedStmt>(AStmt), NewVar, LowerBound, UpperBound);
  }

  // Add two arguments into captured stmt for index and last_iter.
  CapturedStmt *CapturedBody =
      AddSimdArgsIntoCapturedStmt(cast<CapturedStmt>(AStmt), NewVar);

  // Rebuild the directive.
  return OMPDistributeParallelForSimdDirective::Create(
      Context, StartLoc, EndLoc, Clauses, CapturedBody, NewVar, NewEnd,
      NewVarCntExpr, NewFinal, LowerBound, UpperBound, VarCnts);
}

StmtResult Sema::ActOnOpenMPTeamsDistributeParallelForDirective(
    OpenMPDirectiveKind Kind, ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();

  // Create variables for lower/upper bound
  Expr *LowerBound = 0;
  Expr *UpperBound = 0;
  if (NewVar && AStmt) {
    AStmt = AddDistributedParallelArgsIntoCapturedStmt(
        cast<CapturedStmt>(AStmt), NewVar, LowerBound, UpperBound);
  }
  // Rebuild the directive.
  return OMPTeamsDistributeParallelForDirective::Create(
      Context, StartLoc, EndLoc, Clauses, AStmt, NewVar, NewEnd, NewVarCntExpr,
      NewFinal, LowerBound, UpperBound, VarCnts);
}

StmtResult Sema::ActOnOpenMPTeamsDistributeParallelForSimdDirective(
    OpenMPDirectiveKind Kind, ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();

  // Create variables for lower/upper bound
  Expr *LowerBound = 0;
  Expr *UpperBound = 0;
  if (NewVar && AStmt) {
    AStmt = AddDistributedParallelArgsIntoCapturedStmt(
        cast<CapturedStmt>(AStmt), NewVar, LowerBound, UpperBound);
  }

  // Add two arguments into captured stmt for index and last_iter.
  CapturedStmt *CapturedBody =
      AddSimdArgsIntoCapturedStmt(cast<CapturedStmt>(AStmt), NewVar);

  // Rebuild the directive.
  return OMPTeamsDistributeParallelForSimdDirective::Create(
      Context, StartLoc, EndLoc, Clauses, CapturedBody, NewVar, NewEnd,
      NewVarCntExpr, NewFinal, LowerBound, UpperBound, VarCnts);
}

StmtResult Sema::ActOnOpenMPTargetTeamsDistributeParallelForDirective(
    OpenMPDirectiveKind Kind, ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();

  // Create variables for lower/upper bound
  Expr *LowerBound = 0;
  Expr *UpperBound = 0;
  if (NewVar && AStmt) {
    AStmt = AddDistributedParallelArgsIntoCapturedStmt(
        cast<CapturedStmt>(AStmt), NewVar, LowerBound, UpperBound);
  }
  // Rebuild the directive.
  return OMPTargetTeamsDistributeParallelForDirective::Create(
      Context, StartLoc, EndLoc, Clauses, AStmt, NewVar, NewEnd, NewVarCntExpr,
      NewFinal, LowerBound, UpperBound, VarCnts);
}

StmtResult Sema::ActOnOpenMPTargetTeamsDistributeParallelForSimdDirective(
    OpenMPDirectiveKind Kind, ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();

  // Create variables for lower/upper bound
  Expr *LowerBound = 0;
  Expr *UpperBound = 0;
  if (NewVar && AStmt) {
    AStmt = AddDistributedParallelArgsIntoCapturedStmt(
        cast<CapturedStmt>(AStmt), NewVar, LowerBound, UpperBound);
  }

  // Add two arguments into captured stmt for index and last_iter.
  CapturedStmt *CapturedBody =
      AddSimdArgsIntoCapturedStmt(cast<CapturedStmt>(AStmt), NewVar);

  // Rebuild the directive.
  return OMPTargetTeamsDistributeParallelForSimdDirective::Create(
      Context, StartLoc, EndLoc, Clauses, CapturedBody, NewVar, NewEnd,
      NewVarCntExpr, NewFinal, LowerBound, UpperBound, VarCnts);
}

StmtResult Sema::ActOnOpenMPSectionsDirective(OpenMPDirectiveKind Kind,
                                              ArrayRef<OMPClause *> Clauses,
                                              Stmt *AStmt,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  Stmt *BaseStmt = AStmt;
  while (CapturedStmt *CS = dyn_cast_or_null<CapturedStmt>(BaseStmt))
    BaseStmt = CS->getCapturedStmt();
  CompoundStmt *C = dyn_cast_or_null<CompoundStmt>(BaseStmt);
  if (!C) {
    Diag(AStmt->getLocStart(), diag::err_omp_sections_not_compound_stmt)
        << getOpenMPDirectiveName(Kind);
    return StmtError();
  }
  // All associated statements must be '#pragma omp section' except for
  // the first one.
  Stmt::child_range S = C->children();
  if (!S)
    return StmtError();
  for (++S; S; ++S) {
    Stmt *SectionStmt = *S;
    if (!SectionStmt || !isa<OMPSectionDirective>(SectionStmt)) {
      if (SectionStmt)
        Diag(SectionStmt->getLocStart(), diag::err_omp_sections_not_section)
            << getOpenMPDirectiveName(Kind);
      return StmtError();
    }
  }

  getCurFunction()->setHasBranchProtectedScope();

  return OMPSectionsDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                      AStmt);
}

StmtResult Sema::ActOnOpenMPParallelSectionsDirective(
    OpenMPDirectiveKind Kind, ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  Stmt *BaseStmt = AStmt;
  while (CapturedStmt *CS = dyn_cast_or_null<CapturedStmt>(BaseStmt))
    BaseStmt = CS->getCapturedStmt();
  CompoundStmt *C = dyn_cast_or_null<CompoundStmt>(BaseStmt);
  if (!C) {
    Diag(AStmt->getLocStart(), diag::err_omp_sections_not_compound_stmt)
        << getOpenMPDirectiveName(Kind);
    return StmtError();
  }
  // All associated statements must be '#pragma omp section' except for
  // the first one.
  Stmt::child_range S = C->children();
  if (!S)
    return StmtError();
  for (++S; S; ++S) {
    Stmt *SectionStmt = *S;
    if (!SectionStmt || !isa<OMPSectionDirective>(SectionStmt)) {
      if (SectionStmt)
        Diag(SectionStmt->getLocStart(), diag::err_omp_sections_not_section)
            << getOpenMPDirectiveName(Kind);
      return StmtError();
    }
  }

  getCurFunction()->setHasBranchProtectedScope();

  return OMPParallelSectionsDirective::Create(Context, StartLoc, EndLoc,
                                              Clauses, AStmt);
}

StmtResult Sema::ActOnOpenMPSectionDirective(Stmt *AStmt,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  // OpenMP [2.6.2, Sections Construct, Restrictions, p.1]
  //  Orphaned section directives are prohibited. That is, the section
  //  directives must appear within the sections construct and must not
  //  be encountered elsewhere in the sections region.
  // OpenMP scope for current directive.
  if (DSAStack->getCurScope()) {
    Scope *ParentScope = DSAStack->getCurScope()->getParent();
    // CompoundStmt scope for sections scope.
    ParentScope = ParentScope ? getCurScope()->getParent() : 0;
    // Sections scope.
    ParentScope = ParentScope ? ParentScope->getParent() : 0;
    if (!ParentScope || !ParentScope->isOpenMPDirectiveScope() ||
        (DSAStack->getParentDirective() != OMPD_sections &&
         DSAStack->getParentDirective() != OMPD_parallel_sections)) {
      Diag(StartLoc, diag::err_omp_section_orphaned);
      return StmtError();
    }
  }

  getCurFunction()->setHasBranchProtectedScope();

  return OMPSectionDirective::Create(Context, StartLoc, EndLoc, AStmt);
}

StmtResult Sema::ActOnOpenMPSingleDirective(ArrayRef<OMPClause *> Clauses,
                                            Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  getCurFunction()->setHasBranchProtectedScope();

  return OMPSingleDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

StmtResult Sema::ActOnOpenMPTaskDirective(ArrayRef<OMPClause *> Clauses,
                                          Stmt *AStmt, SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  getCurFunction()->setHasBranchProtectedScope();

  return OMPTaskDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

StmtResult Sema::ActOnOpenMPTaskyieldDirective(SourceLocation StartLoc,
                                               SourceLocation EndLoc) {
  getCurFunction()->setHasBranchProtectedScope();

  return OMPTaskyieldDirective::Create(Context, StartLoc, EndLoc);
}

StmtResult Sema::ActOnOpenMPMasterDirective(Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  getCurFunction()->setHasBranchProtectedScope();

  return OMPMasterDirective::Create(Context, StartLoc, EndLoc, AStmt);
}

StmtResult
Sema::ActOnOpenMPCriticalDirective(const DeclarationNameInfo &DirName,
                                   Stmt *AStmt, SourceLocation StartLoc,
                                   SourceLocation EndLoc) {
  getCurFunction()->setHasBranchProtectedScope();

  return OMPCriticalDirective::Create(Context, DirName, StartLoc, EndLoc,
                                      AStmt);
}

StmtResult Sema::ActOnOpenMPBarrierDirective(SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  getCurFunction()->setHasBranchProtectedScope();

  return OMPBarrierDirective::Create(Context, StartLoc, EndLoc);
}

StmtResult Sema::ActOnOpenMPTaskwaitDirective(SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  getCurFunction()->setHasBranchProtectedScope();

  return OMPTaskwaitDirective::Create(Context, StartLoc, EndLoc);
}

StmtResult Sema::ActOnOpenMPTaskgroupDirective(Stmt *AStmt,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc) {
  getCurFunction()->setHasBranchProtectedScope();

  return OMPTaskgroupDirective::Create(Context, StartLoc, EndLoc, AStmt);
}

namespace {
class ExprUseChecker : public StmtVisitor<ExprUseChecker, bool> {
  const llvm::FoldingSetNodeID &ExprID;
  const ASTContext &Context;

public:
  bool VisitStmt(Stmt *S) {
    if (!S)
      return false;
    for (Stmt::child_range R = S->children(); R; ++R) {
      if (Visit(*R))
        return true;
    }
    llvm::FoldingSetNodeID ID;
    S->Profile(ID, Context, true);
    return ID == ExprID;
  }
  ExprUseChecker(const llvm::FoldingSetNodeID &ExprID,
                 const ASTContext &Context)
      : ExprID(ExprID), Context(Context) {}
};
}

StmtResult Sema::ActOnOpenMPAtomicDirective(ArrayRef<OMPClause *> Clauses,
                                            Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  // OpenMP [2.10.6, atomic Construct, Syntax]
  //  There should not be no more than 1 clause 'read', 'write', 'update'
  //  or 'capture'.
  OpenMPClauseKind Kind = OMPC_update;
  if (!Clauses.empty()) {
    bool FoundClauses = false;
    for (ArrayRef<OMPClause *>::iterator I = Clauses.begin(), E = Clauses.end();
         I != E; ++I) {
      if ((*I)->getClauseKind() != OMPC_seq_cst) {
        Kind = (*I)->getClauseKind();
        bool CurFoundClauses = Kind == OMPC_read || Kind == OMPC_write ||
                               Kind == OMPC_update || Kind == OMPC_capture;
        if (FoundClauses && CurFoundClauses) {
          Diag(StartLoc, diag::err_omp_atomic_more_one_clause);
          Kind = OMPC_unknown;
          return StmtError();
        }
        FoundClauses = FoundClauses || CurFoundClauses;
      }
    }
  }

  // OpenMP [2.10.6, atomic Construct, Syntax]
  //  For 'read', 'write', 'update' clauses only expression statements are
  //  allowed.
  Stmt *BaseStmt = AStmt;
  while (CapturedStmt *CS = dyn_cast_or_null<CapturedStmt>(BaseStmt))
    BaseStmt = CS->getCapturedStmt();
  while (ExprWithCleanups *EWC = dyn_cast_or_null<ExprWithCleanups>(BaseStmt))
    BaseStmt = EWC->getSubExpr();
  while (AttributedStmt *AS = dyn_cast_or_null<AttributedStmt>(BaseStmt))
    BaseStmt = AS->getSubStmt();
  bool ExprStmt = isa<Expr>(BaseStmt);
  if (Kind != OMPC_capture && !ExprStmt) {
    Diag(BaseStmt->getLocStart(), diag::err_omp_atomic_not_expression)
        << getOpenMPClauseName(Kind);
    return StmtError();
  }
  bool WrongStmt = false;
  Expr *V = 0;
  Expr *X = 0;
  Expr *OpExpr = 0;
  BinaryOperatorKind Op = BO_Assign;
  bool CaptureAfter = false;
  bool Reversed = false;
  switch (Kind) {
  case OMPC_read: {
    // expr : v = x, where x and v are both l-value with scalar type.
    BinaryOperator *BinOp = dyn_cast_or_null<BinaryOperator>(BaseStmt);
    ImplicitCastExpr *ImpCast;
    WrongStmt =
        !BinOp || BinOp->getOpcode() != BO_Assign || !BinOp->getLHS() ||
        !BinOp->getRHS() ||
        (!BinOp->getLHS()->getType().getCanonicalType()->isScalarType() &&
         !BinOp->getLHS()->getType().getCanonicalType()->isDependentType()) ||
        (!BinOp->getRHS()->getType().getCanonicalType()->isScalarType() &&
         !BinOp->getRHS()->getType().getCanonicalType()->isDependentType()) ||
        !(ImpCast = dyn_cast_or_null<ImplicitCastExpr>(BinOp->getRHS())) ||
        ImpCast->getCastKind() != CK_LValueToRValue;
    if (!WrongStmt) {
      llvm::FoldingSetNodeID ID;
      BinOp->getLHS()->IgnoreParenCasts()->Profile(ID, Context, true);
      ExprUseChecker UseCheck(ID, Context);
      WrongStmt = UseCheck.Visit(BinOp->getRHS()->IgnoreParenCasts());
      if (!WrongStmt) {
        V = BinOp->getLHS();
        X = BinOp->getRHS();
      }
    }
    break;
  }
  case OMPC_write: {
    // expr : x = expr, where x is an l-value with scalar type and expr has
    // scalar type.
    BinaryOperator *BinOp = dyn_cast_or_null<BinaryOperator>(BaseStmt);
    WrongStmt =
        !BinOp || BinOp->getOpcode() != BO_Assign || !BinOp->getLHS() ||
        !BinOp->getRHS() ||
        (!BinOp->getLHS()->getType().getCanonicalType()->isScalarType() &&
         !BinOp->getLHS()->getType().getCanonicalType()->isDependentType()) ||
        (!BinOp->getRHS()->getType().getCanonicalType()->isScalarType() &&
         !BinOp->getRHS()->getType().getCanonicalType()->isDependentType());
    if (!WrongStmt) {
      llvm::FoldingSetNodeID ID;
      BinOp->getLHS()->IgnoreParenCasts()->Profile(ID, Context, true);
      ExprUseChecker UseCheck(ID, Context);
      WrongStmt = UseCheck.Visit(BinOp->getRHS()->IgnoreParenCasts());
      if (!WrongStmt) {
        X = BinOp->getLHS();
        OpExpr = BinOp->getRHS();
      }
    }
    break;
  }
  case OMPC_update: {
    // expr : x++, where x is an l-value with scalar type.
    // expr : x--, where x is an l-value with scalar type.
    // expr : ++x, where x is an l-value with scalar type.
    // expr : --x, where x is an l-value with scalar type.
    // expr : x binop= expr, where x is an l-value with scalar type and expr is
    // scalar.
    // expr : x = x binop expr, where x is an l-value with scalar type and expr
    // is scalar.
    // expr : x = expr binop x, where x is an l-value with scalar type and expr
    // is scalar.
    // binop : +, *, -, /, &, ^, |, << or >>.
    UnaryOperator *UnOp = dyn_cast_or_null<UnaryOperator>(BaseStmt);
    BinaryOperator *BinOp = dyn_cast_or_null<BinaryOperator>(BaseStmt);
    BinaryOperator *RHSBinOp = BinOp ? dyn_cast_or_null<BinaryOperator>(
                                           BinOp->getRHS()->IgnoreParenCasts())
                                     : 0;
    WrongStmt =
        (!UnOp && !BinOp) ||
        (UnOp && ((!UnOp->getType().getCanonicalType()->isScalarType() &&
                   !UnOp->getType().getCanonicalType()->isDependentType()) ||
                  !UnOp->isIncrementDecrementOp())) ||
        (BinOp &&
         ((!BinOp->getLHS()->getType().getCanonicalType()->isScalarType() &&
           !BinOp->getLHS()->getType().getCanonicalType()->isDependentType()) ||
          (!BinOp->getRHS()->getType().getCanonicalType()->isScalarType() &&
           !BinOp->getRHS()
                ->getType()
                .getCanonicalType()
                ->isDependentType()))) ||
        (BinOp &&
         (!BinOp->isCompoundAssignmentOp() && !BinOp->isShiftAssignOp()) &&
         RHSBinOp &&
         (BinOp->getOpcode() != BO_Assign ||
          (!RHSBinOp->isAdditiveOp() && RHSBinOp->getOpcode() != BO_Mul &&
           RHSBinOp->getOpcode() != BO_Div && !RHSBinOp->isBitwiseOp() &&
           !RHSBinOp->isShiftOp()))) ||
        (BinOp && !RHSBinOp &&
         ((!BinOp->isCompoundAssignmentOp() && !BinOp->isShiftAssignOp()) ||
          BinOp->getOpcode() == BO_RemAssign));
    if (!WrongStmt && UnOp) {
      X = UnOp->getSubExpr();
      OpExpr = ActOnIntegerConstant(BaseStmt->getLocStart(), 1).get();
      if (UnOp->isIncrementOp())
        Op = BO_Add;
      else
        Op = BO_Sub;
    } else if (!WrongStmt && BinOp &&
               (BinOp->isCompoundAssignmentOp() || BinOp->isShiftAssignOp())) {
      llvm::FoldingSetNodeID ID;
      BinOp->getLHS()->IgnoreParenCasts()->Profile(ID, Context, true);
      ExprUseChecker UseCheck(ID, Context);
      WrongStmt = UseCheck.Visit(BinOp->getRHS()->IgnoreParenCasts());
      if (!WrongStmt) {
        X = BinOp->getLHS();
        OpExpr = BinOp->getRHS();
        switch (BinOp->getOpcode()) {
        case BO_AddAssign:
          Op = BO_Add;
          break;
        case BO_MulAssign:
          Op = BO_Mul;
          break;
        case BO_SubAssign:
          Op = BO_Sub;
          break;
        case BO_DivAssign:
          Op = BO_Div;
          break;
        case BO_AndAssign:
          Op = BO_And;
          break;
        case BO_XorAssign:
          Op = BO_Xor;
          break;
        case BO_OrAssign:
          Op = BO_Or;
          break;
        case BO_ShlAssign:
          Op = BO_Shl;
          break;
        case BO_ShrAssign:
          Op = BO_Shr;
          break;
        default:
          WrongStmt = true;
          break;
        }
      }
    } else if (!WrongStmt && RHSBinOp) {
      llvm::FoldingSetNodeID ID1, ID2;
      BinOp->getLHS()->IgnoreParenCasts()->Profile(ID1, Context, true);
      RHSBinOp->getLHS()->IgnoreParenCasts()->Profile(ID2, Context, true);
      if (ID1 == ID2) {
        ExprUseChecker UseCheck(ID1, Context);
        WrongStmt = UseCheck.Visit(RHSBinOp->getRHS()->IgnoreParenCasts());
        if (!WrongStmt) {
          X = BinOp->getLHS();
          OpExpr = RHSBinOp->getRHS();
          Op = RHSBinOp->getOpcode();
        }
      } else {
        ID2.clear();
        RHSBinOp->getRHS()->IgnoreParenCasts()->Profile(ID2, Context, true);
        if (ID1 == ID2) {
          ExprUseChecker UseCheck(ID2, Context);
          WrongStmt = UseCheck.Visit(RHSBinOp->getLHS()->IgnoreParenCasts());
          if (!WrongStmt) {
            X = BinOp->getLHS();
            OpExpr = RHSBinOp->getLHS();
            Op = RHSBinOp->getOpcode();
            Reversed = true;
          }
        } else
          WrongStmt = true;
      }
    }
    break;
  }
  case OMPC_capture: {
    // expr : v = x++, where v and x are l-values with scalar types.
    // expr : v = x--, where v and x are l-values with scalar types.
    // expr : v = ++x, where v and x are l-values with scalar types.
    // expr : v = --x, where v and x are l-values with scalar types.
    // expr : v = x binop= expr, where v and x are l-values with scalar types
    // and expr is scalar.
    // expr : v = x = x binop expr, where v and x are l-values with scalar type
    // and expr is scalar.
    // expr : v = x = expr binop x, where v and x are l-values with scalar type
    // and expr is scalar.
    // stmt : {v = x; x binop= expr;}
    // stmt : {x binop= expr; v = x;}
    // stmt : {v = x; x = x binop expr;}
    // stmt : {v = x; x = expr binop x;}
    // stmt : {x = x binop expr; v = x;}
    // stmt : {x = expr binop x; v = x;}
    // stmt : {v = x; x = expr;}
    // stmt : {v = x; x++;}
    // stmt : {v = x; ++x;}
    // stmt : {x++; v = x;}
    // stmt : {++x; v = x;}
    // stmt : {v = x; x--;}
    // stmt : {v = x; --x;}
    // stmt : {x--; v = x;}
    // stmt : {--x; v = x;}
    // binop : +, *, -, /, &, ^, |, << or >>.

    // Expr *V = 0;
    // Expr *X = 0;
    llvm::FoldingSetNodeID VID, XID;
    BinaryOperator *BinOp = dyn_cast_or_null<BinaryOperator>(BaseStmt);
    if (ExprStmt && (!BinOp || BinOp->getOpcode() != BO_Assign)) {
      WrongStmt = true;
      break;
    }
    if (ExprStmt) {
      V = BinOp->getLHS();
      V->IgnoreParenCasts()->Profile(VID, Context, true);
      ExprUseChecker UseCheck(VID, Context);
      WrongStmt =
          (!V->getType().getCanonicalType()->isScalarType() &&
           !V->getType().getCanonicalType()->isDependentType()) ||
          (!BinOp->getRHS()->getType().getCanonicalType()->isScalarType() &&
           !BinOp->getRHS()->getType().getCanonicalType()->isDependentType());
      Expr *RHS = BinOp->getRHS()->IgnoreParenLValueCasts();
      if (UnaryOperator *XOp = dyn_cast_or_null<UnaryOperator>(RHS)) {
        X = XOp->getSubExpr();
        X->IgnoreParenCasts()->Profile(XID, Context, true);
        OpExpr = ActOnIntegerConstant(X->getLocStart(), 1).get();
        if (XOp->isIncrementOp())
          Op = BO_Add;
        else
          Op = BO_Sub;
        CaptureAfter = XOp->isPrefix();
      } else if (BinaryOperator *XOp = dyn_cast_or_null<BinaryOperator>(RHS)) {
        X = XOp->getLHS();
        X->IgnoreParenCasts()->Profile(XID, Context, true);
        CaptureAfter = true;
      } else
        WrongStmt = true;
      if (WrongStmt)
        break;
      BaseStmt = RHS;
    } else if (CompoundStmt *CStmt = dyn_cast_or_null<CompoundStmt>(BaseStmt)) {
      WrongStmt = CStmt->size() != 2;
      if (WrongStmt)
        break;
      Stmt *S1 = *(CStmt->body_begin());
      Stmt *S2 = CStmt->body_back();
      BinaryOperator *VXOp1 = dyn_cast_or_null<BinaryOperator>(S1);
      BinaryOperator *VXOp2 = dyn_cast_or_null<BinaryOperator>(S2);
      UnaryOperator *XOp1 = dyn_cast_or_null<UnaryOperator>(S1);
      UnaryOperator *XOp2 = dyn_cast_or_null<UnaryOperator>(S2);
      if (VXOp1 && VXOp2 && VXOp1->getOpcode() == BO_Assign &&
          VXOp2->getOpcode() == BO_Assign) {
        V = VXOp1->getLHS();
        X = VXOp1->getRHS()->IgnoreParenLValueCasts();
        V->IgnoreParenCasts()->Profile(VID, Context, true);
        X->IgnoreParenCasts()->Profile(XID, Context, true);
        llvm::FoldingSetNodeID X2ID;
        VXOp2->getLHS()->IgnoreParenCasts()->Profile(X2ID, Context, true);
        if (!(XID == X2ID)) {
          llvm::FoldingSetNodeID ExprID;
          VXOp2->getRHS()->IgnoreParenCasts()->Profile(ExprID, Context, true);
          if (ExprID == VID) {
            X = VXOp1->getLHS();
            XID = VID;
            V = VXOp2->getLHS();
            VID = X2ID;
            BaseStmt = S1;
            CaptureAfter = true;
          } else {
            WrongStmt = true;
            break;
          }
        } else {
          BaseStmt = S2;
        }
      } else if (VXOp1 && VXOp2 && VXOp1->getOpcode() == BO_Assign &&
                 VXOp2->isCompoundAssignmentOp()) {
        V = VXOp1->getLHS();
        X = VXOp1->getRHS()->IgnoreParenLValueCasts();
        V->IgnoreParenCasts()->Profile(VID, Context, true);
        X->IgnoreParenCasts()->Profile(XID, Context, true);
        llvm::FoldingSetNodeID X2ID;
        VXOp2->getLHS()->IgnoreParenCasts()->Profile(X2ID, Context, true);
        if (!(XID == X2ID)) {
          WrongStmt = true;
          break;
        }
        BaseStmt = S2;
      } else if (VXOp1 && VXOp2 && VXOp2->getOpcode() == BO_Assign &&
                 VXOp1->isCompoundAssignmentOp()) {
        V = VXOp2->getLHS();
        X = VXOp2->getRHS()->IgnoreParenLValueCasts();
        V->IgnoreParenCasts()->Profile(VID, Context, true);
        X->IgnoreParenCasts()->Profile(XID, Context, true);
        llvm::FoldingSetNodeID X2ID;
        VXOp1->getLHS()->IgnoreParenCasts()->Profile(X2ID, Context, true);
        if (!(XID == X2ID)) {
          WrongStmt = true;
          break;
        }
        BaseStmt = S1;
        CaptureAfter = true;
      } else if (VXOp1 && XOp2 && VXOp1->getOpcode() == BO_Assign) {
        V = VXOp1->getLHS();
        X = VXOp1->getRHS()->IgnoreParenLValueCasts();
        V->IgnoreParenCasts()->Profile(VID, Context, true);
        X->IgnoreParenCasts()->Profile(XID, Context, true);
        llvm::FoldingSetNodeID X2ID;
        XOp2->getSubExpr()->IgnoreParenCasts()->Profile(X2ID, Context, true);
        if (!(XID == X2ID)) {
          WrongStmt = true;
          break;
        }
        BaseStmt = S2;
      } else if (VXOp2 && XOp1 && VXOp2->getOpcode() == BO_Assign) {
        V = VXOp2->getLHS();
        X = VXOp2->getRHS()->IgnoreParenLValueCasts();
        V->IgnoreParenCasts()->Profile(VID, Context, true);
        X->IgnoreParenCasts()->Profile(XID, Context, true);
        llvm::FoldingSetNodeID X2ID;
        XOp1->getSubExpr()->IgnoreParenCasts()->Profile(X2ID, Context, true);
        if (!(XID == X2ID)) {
          WrongStmt = true;
          break;
        }
        BaseStmt = S1;
        CaptureAfter = true;
      } else {
        WrongStmt = true;
        break;
      }
      if ((!V->getType().getCanonicalType()->isScalarType() &&
           !V->getType().getCanonicalType()->isDependentType()) ||
          (!X->getType().getCanonicalType()->isScalarType() &&
           !X->getType().getCanonicalType()->isDependentType())) {
        WrongStmt = true;
        break;
      }
    } else {
      WrongStmt = true;
      break;
    }
    ExprUseChecker UseCheckV(VID, Context);
    ExprUseChecker UseCheckX(XID, Context);
    WrongStmt = UseCheckV.Visit(X->IgnoreParenCasts()) ||
                UseCheckX.Visit(V->IgnoreParenCasts());
    if (WrongStmt)
      break;
    UnaryOperator *UnOp = dyn_cast_or_null<UnaryOperator>(BaseStmt);
    BinOp = dyn_cast_or_null<BinaryOperator>(BaseStmt);
    BinaryOperator *RHSBinOp = BinOp ? dyn_cast_or_null<BinaryOperator>(
                                           BinOp->getRHS()->IgnoreParenCasts())
                                     : 0;
    WrongStmt =
        (!UnOp && !BinOp) ||
        (UnOp && ((!UnOp->getType().getCanonicalType()->isScalarType() &&
                   !UnOp->getType().getCanonicalType()->isDependentType()) ||
                  !UnOp->isIncrementDecrementOp())) ||
        (BinOp &&
         ((!BinOp->getLHS()->getType().getCanonicalType()->isScalarType() &&
           !BinOp->getLHS()->getType().getCanonicalType()->isDependentType()) ||
          (!BinOp->getRHS()->getType().getCanonicalType()->isScalarType() &&
           !BinOp->getRHS()
                ->getType()
                .getCanonicalType()
                ->isDependentType()))) ||
        (BinOp &&
         (!BinOp->isCompoundAssignmentOp() && !BinOp->isShiftAssignOp()) &&
         RHSBinOp &&
         (BinOp->getOpcode() != BO_Assign ||
          (!RHSBinOp->isAdditiveOp() && RHSBinOp->getOpcode() != BO_Mul &&
           RHSBinOp->getOpcode() != BO_Div && !RHSBinOp->isBitwiseOp() &&
           !RHSBinOp->isShiftOp()))) ||
        (BinOp && !RHSBinOp &&
         ((!BinOp->isCompoundAssignmentOp() && !BinOp->isShiftAssignOp() &&
           BinOp->getOpcode() != BO_Assign) ||
          BinOp->getOpcode() == BO_RemAssign));
    if (!WrongStmt && UnOp) {
      OpExpr = ActOnIntegerConstant(BaseStmt->getLocStart(), 1).get();
      if (UnOp->isIncrementOp())
        Op = BO_Add;
      else
        Op = BO_Sub;
    } else if (!WrongStmt && BinOp && !RHSBinOp &&
               BinOp->getOpcode() == BO_Assign) {
      Op = BO_Assign;
      OpExpr = BinOp->getRHS();
    } else if (!WrongStmt && BinOp &&
               (BinOp->isCompoundAssignmentOp() || BinOp->isShiftAssignOp())) {
      ExprUseChecker UseCheckX(XID, Context);
      ExprUseChecker UseCheckV(VID, Context);
      WrongStmt = UseCheckX.Visit(BinOp->getRHS()->IgnoreParenCasts()) ||
                  UseCheckV.Visit(BinOp->getRHS()->IgnoreParenCasts());
      if (!WrongStmt) {
        OpExpr = BinOp->getRHS();
        switch (BinOp->getOpcode()) {
        case BO_AddAssign:
          Op = BO_Add;
          break;
        case BO_MulAssign:
          Op = BO_Mul;
          break;
        case BO_SubAssign:
          Op = BO_Sub;
          break;
        case BO_DivAssign:
          Op = BO_Div;
          break;
        case BO_AndAssign:
          Op = BO_And;
          break;
        case BO_XorAssign:
          Op = BO_Xor;
          break;
        case BO_OrAssign:
          Op = BO_Or;
          break;
        case BO_ShlAssign:
          Op = BO_Shl;
          break;
        case BO_ShrAssign:
          Op = BO_Shr;
          break;
        default:
          WrongStmt = true;
          break;
        }
      }
    } else if (!WrongStmt && RHSBinOp) {
      llvm::FoldingSetNodeID ID;
      RHSBinOp->getLHS()->IgnoreParenCasts()->Profile(ID, Context, true);
      if (XID == ID) {
        ExprUseChecker UseCheckX(XID, Context);
        ExprUseChecker UseCheckV(VID, Context);
        WrongStmt = UseCheckX.Visit(RHSBinOp->getRHS()->IgnoreParenCasts()) ||
                    UseCheckV.Visit(RHSBinOp->getRHS()->IgnoreParenCasts());
        if (!WrongStmt) {
          OpExpr = RHSBinOp->getRHS();
          Op = RHSBinOp->getOpcode();
        }
      } else {
        ID.clear();
        RHSBinOp->getRHS()->IgnoreParenCasts()->Profile(ID, Context, true);
        if (XID == ID) {
          ExprUseChecker UseCheckX(XID, Context);
          ExprUseChecker UseCheckV(VID, Context);
          WrongStmt = UseCheckX.Visit(RHSBinOp->getLHS()->IgnoreParenCasts()) ||
                      UseCheckV.Visit(RHSBinOp->getLHS()->IgnoreParenCasts());
          if (!WrongStmt) {
            OpExpr = RHSBinOp->getLHS();
            Op = RHSBinOp->getOpcode();
            Reversed = true;
          }
        } else
          WrongStmt = true;
      }
    }
    break;
  }
  default:
    break;
  }
  if (WrongStmt) {
    Diag(BaseStmt->getLocStart(), diag::err_omp_atomic_wrong_statement)
        << getOpenMPClauseName(Kind);
    return StmtError();
  }
  //  if (OpExpr && !X->getType()->isDependentType() &&
  //      !OpExpr->getType()->isDependentType()) {
  //    ExprResult Res = OpExpr;
  //    CastKind CK = PrepareScalarCast(Res, X->getType());
  //    if (CK != CK_NoOp)
  //      OpExpr = ImpCastExprToType(Res.get(), X->getType(), CK).get();
  //  }
  //  if (V && !V->getType()->isDependentType()) {
  //    ExprResult Res = X;
  //    CastKind CK = PrepareScalarCast(Res, V->getType());
  //    if (CK != CK_NoOp)
  //      X = ImpCastExprToType(Res.get(), V->getType(), CK).get();
  //  }

  getCurFunction()->setHasBranchProtectedScope();

  return OMPAtomicDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt,
                                    V, X, OpExpr, Op, CaptureAfter, Reversed);
}

StmtResult Sema::ActOnOpenMPFlushDirective(ArrayRef<OMPClause *> Clauses,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  getCurFunction()->setHasBranchProtectedScope();

  return OMPFlushDirective::Create(Context, StartLoc, EndLoc, Clauses);
}

StmtResult Sema::ActOnOpenMPOrderedDirective(Stmt *AStmt,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  getCurFunction()->setHasBranchProtectedScope();

  return OMPOrderedDirective::Create(Context, StartLoc, EndLoc, AStmt);
}

StmtResult Sema::ActOnOpenMPTeamsDirective(ArrayRef<OMPClause *> Clauses,
                                           Stmt *AStmt, SourceLocation StartLoc,
                                           SourceLocation EndLoc) {

  getCurFunction()->setHasBranchProtectedScope();
  return OMPTeamsDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

StmtResult Sema::ActOnOpenMPTargetTeamsDirective(ArrayRef<OMPClause *> Clauses,
                                                 Stmt *AStmt,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc) {

  getCurFunction()->setHasBranchProtectedScope();
  return OMPTargetTeamsDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                         AStmt);
}

StmtResult Sema::ActOnOpenMPDistributeDirective(ArrayRef<OMPClause *> Clauses,
                                                Stmt *AStmt,
                                                SourceLocation StartLoc,
                                                SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(OMPD_distribute, Clauses, AStmt, StartLoc, EndLoc,
                          NewVar, NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();
  return OMPDistributeDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                        AStmt, NewVar, NewEnd, NewVarCntExpr,
                                        NewFinal, VarCnts);
}

StmtResult Sema::ActOnOpenMPCancelDirective(ArrayRef<OMPClause *> Clauses,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc,
                                            OpenMPDirectiveKind ConstructType) {
  getCurFunction()->setHasBranchProtectedScope();
  return OMPCancelDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                    ConstructType);
}

StmtResult
Sema::ActOnOpenMPCancellationPointDirective(SourceLocation StartLoc,
                                            SourceLocation EndLoc,
                                            OpenMPDirectiveKind ConstructType) {
  getCurFunction()->setHasBranchProtectedScope();
  return OMPCancellationPointDirective::Create(Context, StartLoc, EndLoc,
                                               ConstructType);
}

namespace {
class TeamsChecker : public StmtVisitor<TeamsChecker, bool> {
  Stmt *FoundTeams;

public:
  bool VisitOMPTeamsDirective(OMPTeamsDirective *D) {
    FoundTeams = D;
    return false;
  }
  bool VisitOMPTeamsDistributeDirective(OMPTeamsDistributeDirective *D) {
    FoundTeams = D;
    return false;
  }
  bool
  VisitOMPTeamsDistributeSimdDirective(OMPTeamsDistributeSimdDirective *D) {
    FoundTeams = D;
    return false;
  }
  bool VisitOMPTeamsDistributeParallelForDirective(
      OMPTeamsDistributeParallelForDirective *D) {
    FoundTeams = D;
    return false;
  }
  bool VisitOMPTeamsDistributeParallelForSimdDirective(
      OMPTeamsDistributeParallelForSimdDirective *D) {
    FoundTeams = D;
    return false;
  }
  bool VisitCompoundStmt(CompoundStmt *S) {
    bool Flag = false;
    for (Stmt::child_range R = S->children(); R; ++R) {
      Flag |= Visit(*R);
      if (Flag && FoundTeams)
        return true;
    }
    return Flag;
  }
  bool VisitNullStmt(NullStmt *) { return false; }
  bool VisitStmt(Stmt *) { return true; }
  TeamsChecker() : FoundTeams(0) {}
  Stmt *getFoundTeams() { return FoundTeams; }
};
}

StmtResult Sema::ActOnOpenMPTargetDirective(ArrayRef<OMPClause *> Clauses,
                                            Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  TeamsChecker Checker;
  // If specified, a teams construct must be contained within a target
  // construct. That target construct must contain no statements or directives
  // outside of the teams construct.
  if (Checker.Visit(cast<CapturedStmt>(AStmt)->getCapturedStmt())) {
    if (Stmt *S = Checker.getFoundTeams()) {
      Diag(S->getLocStart(), diag::err_omp_teams_not_single_in_target);
      return StmtError();
    }
  }

  getCurFunction()->setHasBranchProtectedScope();
  return OMPTargetDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

StmtResult Sema::ActOnOpenMPTargetDataDirective(ArrayRef<OMPClause *> Clauses,
                                                Stmt *AStmt,
                                                SourceLocation StartLoc,
                                                SourceLocation EndLoc) {
  getCurFunction()->setHasBranchProtectedScope();
  return OMPTargetDataDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                        AStmt);
}

StmtResult Sema::ActOnOpenMPTargetUpdateDirective(ArrayRef<OMPClause *> Clauses,
                                                  SourceLocation StartLoc,
                                                  SourceLocation EndLoc) {
  // FIXME Add checking that at least one 'from' or 'to' clause is specified

  getCurFunction()->setHasBranchProtectedScope();
  return OMPTargetUpdateDirective::Create(Context, StartLoc, EndLoc, Clauses);
}

StmtResult
Sema::ActOnOpenMPTeamsDistributeDirective(ArrayRef<OMPClause *> Clauses,
                                          Stmt *AStmt, SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(OMPD_teams_distribute, Clauses, AStmt, StartLoc,
                          EndLoc, NewVar, NewEnd, NewVarCntExpr, NewFinal,
                          VarCnts)) {
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();
  return OMPTeamsDistributeDirective::Create(Context, StartLoc, EndLoc, Clauses,
                                             AStmt, NewVar, NewEnd,
                                             NewVarCntExpr, NewFinal, VarCnts);
}

StmtResult Sema::ActOnOpenMPTeamsDistributeSimdDirective(
    OpenMPDirectiveKind Kind, ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  // Add two arguments into captured stmt for index and last_iter.
  CapturedStmt *CapturedBody =
      AddSimdArgsIntoCapturedStmt(cast<CapturedStmt>(AStmt), NewVar);

  getCurFunction()->setHasBranchProtectedScope();

  // Rebuild the directive.
  return OMPTeamsDistributeSimdDirective::Create(
      Context, StartLoc, EndLoc, Clauses, CapturedBody, NewVar, NewEnd,
      NewVarCntExpr, NewFinal, VarCnts);
}

StmtResult Sema::ActOnOpenMPTargetTeamsDistributeDirective(
    ArrayRef<OMPClause *> Clauses, Stmt *AStmt, SourceLocation StartLoc,
    SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(OMPD_target_teams_distribute, Clauses, AStmt,
                          StartLoc, EndLoc, NewVar, NewEnd, NewVarCntExpr,
                          NewFinal, VarCnts)) {
    return StmtError();
  }

  getCurFunction()->setHasBranchProtectedScope();
  return OMPTargetTeamsDistributeDirective::Create(
      Context, StartLoc, EndLoc, Clauses, AStmt, NewVar, NewEnd, NewVarCntExpr,
      NewFinal, VarCnts);
}

StmtResult Sema::ActOnOpenMPTargetTeamsDistributeSimdDirective(
    OpenMPDirectiveKind Kind, ArrayRef<OMPClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  // Prepare the output arguments for routine CollapseOpenMPLoop
  Expr *NewEnd = 0;
  Expr *NewVar = 0;
  Expr *NewVarCntExpr = 0;
  Expr *NewFinal = 0;
  SmallVector<Expr *, 4> VarCnts;

  // Do the collapse.
  if (!CollapseOpenMPLoop(Kind, Clauses, AStmt, StartLoc, EndLoc, NewVar,
                          NewEnd, NewVarCntExpr, NewFinal, VarCnts)) {
    return StmtError();
  }

  // Add two arguments into captured stmt for index and last_iter.
  CapturedStmt *CapturedBody =
      AddSimdArgsIntoCapturedStmt(cast<CapturedStmt>(AStmt), NewVar);

  getCurFunction()->setHasBranchProtectedScope();

  // Rebuild the directive.
  return OMPTargetTeamsDistributeSimdDirective::Create(
      Context, StartLoc, EndLoc, Clauses, CapturedBody, NewVar, NewEnd,
      NewVarCntExpr, NewFinal, VarCnts);
}

OMPClause *Sema::ActOnOpenMPSingleExprClause(OpenMPClauseKind Kind, Expr *Expr,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  OMPClause *Res = 0;
  switch (Kind) {
  case OMPC_if:
    Res = ActOnOpenMPIfClause(Expr, StartLoc, EndLoc);
    break;
  case OMPC_num_threads:
    Res = ActOnOpenMPNumThreadsClause(Expr, StartLoc, EndLoc);
    break;
  case OMPC_collapse:
    Res = ActOnOpenMPCollapseClause(Expr, StartLoc, EndLoc);
    break;
  case OMPC_final:
    Res = ActOnOpenMPFinalClause(Expr, StartLoc, EndLoc);
    break;
  case OMPC_safelen:
    Res = ActOnOpenMPSafelenClause(Expr, StartLoc, EndLoc);
    break;
  case OMPC_simdlen:
    Res = ActOnOpenMPSimdlenClause(Expr, StartLoc, EndLoc);
    break;
  case OMPC_num_teams:
    Res = ActOnOpenMPNumTeamsClause(Expr, StartLoc, EndLoc);
    break;
  case OMPC_thread_limit:
    Res = ActOnOpenMPThreadLimitClause(Expr, StartLoc, EndLoc);
    break;
  case OMPC_device:
    Res = ActOnOpenMPDeviceClause(Expr, StartLoc, EndLoc);
    break;
  default:
    break;
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPIfClause(Expr *Condition, SourceLocation StartLoc,
                                     SourceLocation EndLoc) {
  QualType Type = Condition->getType();
  Expr *ValExpr = Condition;
  if (!Type->isDependentType() && !Type->isInstantiationDependentType()) {
    ExprResult Val = ActOnBooleanCondition(DSAStack->getCurScope(),
                                           Condition->getExprLoc(), Condition);
    if (Val.isInvalid())
      return 0;

    ValExpr = Val.get();
  }

  return new (Context) OMPIfClause(ValExpr, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPFinalClause(Expr *Condition,
                                        SourceLocation StartLoc,
                                        SourceLocation EndLoc) {
  QualType Type = Condition->getType();
  Expr *ValExpr = Condition;
  if (!Type->isDependentType() && !Type->isInstantiationDependentType()) {
    ExprResult Val = ActOnBooleanCondition(DSAStack->getCurScope(),
                                           Condition->getExprLoc(), Condition);
    if (Val.isInvalid())
      return 0;

    ValExpr = Val.get();
  }

  return new (Context) OMPFinalClause(ValExpr, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPNumThreadsClause(Expr *NumThreads,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  class CConvertDiagnoser : public ICEConvertDiagnoser {
  public:
    CConvertDiagnoser() : ICEConvertDiagnoser(true, false, true) {}
    virtual SemaDiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                                 QualType T) {
      return S.Diag(Loc, diag::err_typecheck_statement_requires_integer) << T;
    }
    virtual SemaDiagnosticBuilder
    diagnoseIncomplete(Sema &S, SourceLocation Loc, QualType T) {
      return S.Diag(Loc, diag::err_incomplete_class_type) << T;
    }
    virtual SemaDiagnosticBuilder diagnoseExplicitConv(Sema &S,
                                                       SourceLocation Loc,
                                                       QualType T,
                                                       QualType ConvTy) {
      return S.Diag(Loc, diag::err_explicit_conversion) << T << ConvTy;
    }

    virtual SemaDiagnosticBuilder
    noteExplicitConv(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) {
      return S.Diag(Conv->getLocation(), diag::note_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    virtual SemaDiagnosticBuilder diagnoseAmbiguous(Sema &S, SourceLocation Loc,
                                                    QualType T) {
      return S.Diag(Loc, diag::err_multiple_conversions) << T;
    }

    virtual SemaDiagnosticBuilder
    noteAmbiguous(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) {
      return S.Diag(Conv->getLocation(), diag::note_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }

    virtual SemaDiagnosticBuilder diagnoseConversion(Sema &S,
                                                     SourceLocation Loc,
                                                     QualType T,
                                                     QualType ConvTy) {
      llvm_unreachable("conversion functions are permitted");
    }
  } ConvertDiagnoser;

  if (!NumThreads)
    return 0;

  Expr *ValExpr = NumThreads;
  if (!ValExpr->isTypeDependent() && !ValExpr->isValueDependent() &&
      !ValExpr->isInstantiationDependent()) {
    SourceLocation Loc = NumThreads->getExprLoc();
    ExprResult Value =
        PerformContextualImplicitConversion(Loc, NumThreads, ConvertDiagnoser);
    if (Value.isInvalid() ||
        !Value.get()->getType()->isIntegralOrUnscopedEnumerationType())
      return 0;

    llvm::APSInt Result;
    if (Value.get()->isIntegerConstantExpr(Result, Context) &&
        !Result.isStrictlyPositive()) {
      Diag(Loc, diag::err_negative_expression_in_clause)
          << NumThreads->getSourceRange();
      return 0;
    }
    Value = DefaultLvalueConversion(Value.get());
    if (Value.isInvalid())
      return 0;
    Value = PerformImplicitConversion(
        Value.get(), Context.getIntTypeForBitwidth(32, true), AA_Converting);
    ValExpr = Value.get();
  }

  return new (Context) OMPNumThreadsClause(ValExpr, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPDeviceClause(Expr *Device, SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  class CConvertDiagnoser : public ICEConvertDiagnoser {
  public:
    CConvertDiagnoser() : ICEConvertDiagnoser(true, false, true) {}
    virtual SemaDiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                                 QualType T) {
      return S.Diag(Loc, diag::err_typecheck_statement_requires_integer) << T;
    }
    virtual SemaDiagnosticBuilder
    diagnoseIncomplete(Sema &S, SourceLocation Loc, QualType T) {
      return S.Diag(Loc, diag::err_incomplete_class_type) << T;
    }
    virtual SemaDiagnosticBuilder diagnoseExplicitConv(Sema &S,
                                                       SourceLocation Loc,
                                                       QualType T,
                                                       QualType ConvTy) {
      return S.Diag(Loc, diag::err_explicit_conversion) << T << ConvTy;
    }

    virtual SemaDiagnosticBuilder
    noteExplicitConv(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) {
      return S.Diag(Conv->getLocation(), diag::note_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    virtual SemaDiagnosticBuilder diagnoseAmbiguous(Sema &S, SourceLocation Loc,
                                                    QualType T) {
      return S.Diag(Loc, diag::err_multiple_conversions) << T;
    }

    virtual SemaDiagnosticBuilder
    noteAmbiguous(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) {
      return S.Diag(Conv->getLocation(), diag::note_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }

    virtual SemaDiagnosticBuilder diagnoseConversion(Sema &S,
                                                     SourceLocation Loc,
                                                     QualType T,
                                                     QualType ConvTy) {
      llvm_unreachable("conversion functions are permitted");
    }
  } ConvertDiagnoser;

  if (!Device)
    return 0;

  Expr *ValExpr = Device;
  if (!ValExpr->isTypeDependent() && !ValExpr->isValueDependent() &&
      !ValExpr->isInstantiationDependent()) {
    SourceLocation Loc = ValExpr->getExprLoc();
    ExprResult Value =
        PerformContextualImplicitConversion(Loc, ValExpr, ConvertDiagnoser);
    if (Value.isInvalid() ||
        !Value.get()->getType()->isIntegralOrUnscopedEnumerationType())
      return 0;

    llvm::APSInt Result;
    if (Value.get()->isIntegerConstantExpr(Result, Context) &&
        Result.isNegative()) {
      Diag(Loc, diag::err_negative_expression_in_clause)
          << ValExpr->getSourceRange();
      return 0;
    }
    Value = DefaultLvalueConversion(Value.get());
    if (Value.isInvalid())
      return 0;
    Value = PerformImplicitConversion(
        Value.get(), Context.getIntTypeForBitwidth(32, true), AA_Converting);
    ValExpr = Value.get();
  }

  return new (Context) OMPDeviceClause(ValExpr, StartLoc, EndLoc);
}

Expr *Sema::ActOnConstantPositiveSubExpressionInClause(Expr *E) {
  if (!E)
    return 0;
  if (E->isInstantiationDependent())
    return E;
  llvm::APSInt Result;
  ExprResult ICE = VerifyIntegerConstantExpression(E, &Result);
  if (ICE.isInvalid())
    return 0;
  if (!Result.isStrictlyPositive()) {
    Diag(E->getExprLoc(), diag::err_negative_expression_in_clause)
        << E->getSourceRange();
    return 0;
  }
  return IntegerLiteral::Create(Context, Result,
                                ICE.get()->getType().getNonReferenceType(),
                                E->getExprLoc());
}

Expr *Sema::ActOnConstantLinearStep(Expr *E) {
  if (!E)
    return 0;
  if (E->isInstantiationDependent())
    return E;
  llvm::APSInt Result;
  ExprResult ICE = VerifyIntegerConstantExpression(E, &Result);
  if (ICE.isInvalid())
    return 0;
  if (!Result.isStrictlyPositive() && !Result.isNegative()) {
    Diag(E->getExprLoc(), diag::err_zero_step_in_linear_clause)
        << E->getSourceRange();
    return 0;
  }
  return IntegerLiteral::Create(Context, Result,
                                ICE.get()->getType().getNonReferenceType(),
                                E->getExprLoc());
}

OMPClause *Sema::ActOnOpenMPCollapseClause(Expr *NumLoops,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  // OpenMP [2.7.1, Loop construct, Description]
  // The parameter of the collapse clause must be a constant
  // positive integer expression.
  Expr *Val = ActOnConstantPositiveSubExpressionInClause(NumLoops);
  if (!Val)
    return 0;

  return new (Context) OMPCollapseClause(Val, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSafelenClause(Expr *Len, SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  // OpenMP [2.8.1, simd construct, Description]
  // The parameter of the safelen clause must be a constant
  // positive integer expression.
  Expr *Val = ActOnConstantPositiveSubExpressionInClause(Len);
  if (!Val)
    return 0;

  return new (Context) OMPSafelenClause(Val, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSimdlenClause(Expr *Len, SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  // OpenMP [2.8.2, declare simd construct, Description]
  // The parameter of the simdlen clause must be a constant
  // positive integer expression.
  Expr *Val = ActOnConstantPositiveSubExpressionInClause(Len);
  if (!Val)
    return 0;

  return new (Context) OMPSimdlenClause(Val, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPNumTeamsClause(Expr *E, SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  class CConvertDiagnoser : public ICEConvertDiagnoser {
  public:
    CConvertDiagnoser() : ICEConvertDiagnoser(true, false, true) {}
    virtual SemaDiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                                 QualType T) {
      return S.Diag(Loc, diag::err_typecheck_statement_requires_integer) << T;
    }
    virtual SemaDiagnosticBuilder
    diagnoseIncomplete(Sema &S, SourceLocation Loc, QualType T) {
      return S.Diag(Loc, diag::err_incomplete_class_type) << T;
    }
    virtual SemaDiagnosticBuilder diagnoseExplicitConv(Sema &S,
                                                       SourceLocation Loc,
                                                       QualType T,
                                                       QualType ConvTy) {
      return S.Diag(Loc, diag::err_explicit_conversion) << T << ConvTy;
    }

    virtual SemaDiagnosticBuilder
    noteExplicitConv(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) {
      return S.Diag(Conv->getLocation(), diag::note_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    virtual SemaDiagnosticBuilder diagnoseAmbiguous(Sema &S, SourceLocation Loc,
                                                    QualType T) {
      return S.Diag(Loc, diag::err_multiple_conversions) << T;
    }

    virtual SemaDiagnosticBuilder
    noteAmbiguous(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) {
      return S.Diag(Conv->getLocation(), diag::note_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }

    virtual SemaDiagnosticBuilder diagnoseConversion(Sema &S,
                                                     SourceLocation Loc,
                                                     QualType T,
                                                     QualType ConvTy) {
      llvm_unreachable("conversion functions are permitted");
    }
  } ConvertDiagnoser;

  if (!E)
    return 0;

  Expr *ValExpr = E;
  if (!ValExpr->isTypeDependent() && !ValExpr->isValueDependent() &&
      !ValExpr->isInstantiationDependent()) {
    SourceLocation Loc = ValExpr->getExprLoc();
    ExprResult Value =
        PerformContextualImplicitConversion(Loc, ValExpr, ConvertDiagnoser);
    if (Value.isInvalid() ||
        !Value.get()->getType()->isIntegralOrUnscopedEnumerationType())
      return 0;

    llvm::APSInt Result;
    if (Value.get()->isIntegerConstantExpr(Result, Context) &&
        !Result.isStrictlyPositive()) {
      Diag(Loc, diag::err_negative_expression_in_clause)
          << ValExpr->getSourceRange();
      return 0;
    }
    Value = DefaultLvalueConversion(Value.get());
    if (Value.isInvalid())
      return 0;
    Value = PerformImplicitConversion(
        Value.get(), Context.getIntTypeForBitwidth(32, true), AA_Converting);
    if (Value.isInvalid())
      return 0;
    ValExpr = Value.get();
  }

  return new (Context) OMPNumTeamsClause(ValExpr, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPThreadLimitClause(Expr *E, SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  class CConvertDiagnoser : public ICEConvertDiagnoser {
  public:
    CConvertDiagnoser() : ICEConvertDiagnoser(true, false, true) {}
    virtual SemaDiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                                 QualType T) {
      return S.Diag(Loc, diag::err_typecheck_statement_requires_integer) << T;
    }
    virtual SemaDiagnosticBuilder
    diagnoseIncomplete(Sema &S, SourceLocation Loc, QualType T) {
      return S.Diag(Loc, diag::err_incomplete_class_type) << T;
    }
    virtual SemaDiagnosticBuilder diagnoseExplicitConv(Sema &S,
                                                       SourceLocation Loc,
                                                       QualType T,
                                                       QualType ConvTy) {
      return S.Diag(Loc, diag::err_explicit_conversion) << T << ConvTy;
    }

    virtual SemaDiagnosticBuilder
    noteExplicitConv(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) {
      return S.Diag(Conv->getLocation(), diag::note_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    virtual SemaDiagnosticBuilder diagnoseAmbiguous(Sema &S, SourceLocation Loc,
                                                    QualType T) {
      return S.Diag(Loc, diag::err_multiple_conversions) << T;
    }

    virtual SemaDiagnosticBuilder
    noteAmbiguous(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) {
      return S.Diag(Conv->getLocation(), diag::note_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }

    virtual SemaDiagnosticBuilder diagnoseConversion(Sema &S,
                                                     SourceLocation Loc,
                                                     QualType T,
                                                     QualType ConvTy) {
      llvm_unreachable("conversion functions are permitted");
    }
  } ConvertDiagnoser;

  if (!E)
    return 0;

  Expr *ValExpr = E;
  if (!ValExpr->isTypeDependent() && !ValExpr->isValueDependent() &&
      !ValExpr->isInstantiationDependent()) {
    SourceLocation Loc = ValExpr->getExprLoc();
    ExprResult Value =
        PerformContextualImplicitConversion(Loc, ValExpr, ConvertDiagnoser);
    if (Value.isInvalid() ||
        !Value.get()->getType()->isIntegralOrUnscopedEnumerationType())
      return 0;

    llvm::APSInt Result;
    if (Value.get()->isIntegerConstantExpr(Result, Context) &&
        !Result.isStrictlyPositive()) {
      Diag(Loc, diag::err_negative_expression_in_clause)
          << ValExpr->getSourceRange();
      return 0;
    }
    Value = DefaultLvalueConversion(Value.get());
    if (Value.isInvalid())
      return 0;
    Value = PerformImplicitConversion(
        Value.get(), Context.getIntTypeForBitwidth(32, true), AA_Converting);
    if (Value.isInvalid())
      return 0;
    ValExpr = Value.get();
  }

  return new (Context) OMPThreadLimitClause(ValExpr, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSimpleClause(OpenMPClauseKind Kind,
                                         unsigned Argument,
                                         SourceLocation ArgumentLoc,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  OMPClause *Res = 0;
  switch (Kind) {
  case OMPC_default:
    Res =
        ActOnOpenMPDefaultClause(static_cast<OpenMPDefaultClauseKind>(Argument),
                                 ArgumentLoc, StartLoc, EndLoc);
    break;
  case OMPC_proc_bind:
    Res = ActOnOpenMPProcBindClause(
        static_cast<OpenMPProcBindClauseKind>(Argument), ArgumentLoc, StartLoc,
        EndLoc);
    break;
  default:
    break;
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPDefaultClause(OpenMPDefaultClauseKind Kind,
                                          SourceLocation KindLoc,
                                          SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  if (Kind == OMPC_DEFAULT_unknown) {
    std::string Values;
    std::string Sep(NUM_OPENMP_DEFAULT_KINDS > 1 ? ", " : "");
    for (unsigned i = OMPC_DEFAULT_unknown + 1; i < NUM_OPENMP_DEFAULT_KINDS;
         ++i) {
      Values += "'";
      Values += getOpenMPSimpleClauseTypeName(OMPC_default, i);
      Values += "'";
      switch (i) {
      case NUM_OPENMP_DEFAULT_KINDS - 2:
        Values += " or ";
        break;
      case NUM_OPENMP_DEFAULT_KINDS - 1:
        break;
      default:
        Values += Sep;
        break;
      }
    }
    Diag(KindLoc, diag::err_omp_unexpected_clause_value)
        << Values << getOpenMPClauseName(OMPC_default);
    return 0;
  }
  switch (Kind) {
  case OMPC_DEFAULT_none:
    DSAStack->setDefaultDSANone();
    break;
  case OMPC_DEFAULT_shared:
    DSAStack->setDefaultDSAShared();
    break;
  default:
    break;
  }
  return new (Context) OMPDefaultClause(Kind, KindLoc, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPProcBindClause(OpenMPProcBindClauseKind Kind,
                                           SourceLocation KindLoc,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  if (Kind == OMPC_PROC_BIND_unknown) {
    std::string Values;
    std::string Sep(NUM_OPENMP_PROC_BIND_KINDS > 1 ? ", " : "");
    for (unsigned i = OMPC_PROC_BIND_unknown + 1;
         i < NUM_OPENMP_PROC_BIND_KINDS; ++i) {
      Values += "'";
      Values += getOpenMPSimpleClauseTypeName(OMPC_proc_bind, i);
      Values += "'";
      switch (i) {
      case NUM_OPENMP_PROC_BIND_KINDS - 2:
        Values += " or ";
        break;
      case NUM_OPENMP_PROC_BIND_KINDS - 1:
        break;
      default:
        Values += Sep;
        break;
      }
    }
    Diag(KindLoc, diag::err_omp_unexpected_clause_value)
        << Values << getOpenMPClauseName(OMPC_proc_bind);
    return 0;
  }
  return new (Context) OMPProcBindClause(Kind, KindLoc, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPClause(OpenMPClauseKind Kind,
                                   SourceLocation StartLoc,
                                   SourceLocation EndLoc) {
  OMPClause *Res = 0;
  switch (Kind) {
  case OMPC_ordered:
    Res = ActOnOpenMPOrderedClause(StartLoc, EndLoc);
    break;
  case OMPC_nowait:
    Res = ActOnOpenMPNowaitClause(StartLoc, EndLoc);
    break;
  case OMPC_untied:
    Res = ActOnOpenMPUntiedClause(StartLoc, EndLoc);
    break;
  case OMPC_mergeable:
    Res = ActOnOpenMPMergeableClause(StartLoc, EndLoc);
    break;
  case OMPC_read:
    Res = ActOnOpenMPReadClause(StartLoc, EndLoc);
    break;
  case OMPC_write:
    Res = ActOnOpenMPWriteClause(StartLoc, EndLoc);
    break;
  case OMPC_update:
    Res = ActOnOpenMPUpdateClause(StartLoc, EndLoc);
    break;
  case OMPC_capture:
    Res = ActOnOpenMPCaptureClause(StartLoc, EndLoc);
    break;
  case OMPC_seq_cst:
    Res = ActOnOpenMPSeqCstClause(StartLoc, EndLoc);
    break;
  case OMPC_inbranch:
    Res = ActOnOpenMPInBranchClause(StartLoc, EndLoc);
    break;
  case OMPC_notinbranch:
    Res = ActOnOpenMPNotInBranchClause(StartLoc, EndLoc);
    break;
  default:
    break;
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPOrderedClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  DSAStack->setRegionOrdered();
  return new (Context) OMPOrderedClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPNowaitClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  DSAStack->setRegionNowait();
  return new (Context) OMPNowaitClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPUntiedClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return new (Context) OMPUntiedClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPMergeableClause(SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  return new (Context) OMPMergeableClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSingleExprWithTypeClause(
    OpenMPClauseKind Kind, unsigned Argument, SourceLocation ArgumentLoc,
    Expr *Expr, SourceLocation StartLoc, SourceLocation EndLoc) {
  OMPClause *Res = 0;
  switch (Kind) {
  case OMPC_schedule:
    Res = ActOnOpenMPScheduleClause(
        static_cast<OpenMPScheduleClauseKind>(Argument), ArgumentLoc, Expr,
        StartLoc, EndLoc);
    break;
  case OMPC_dist_schedule:
    Res = ActOnOpenMPDistScheduleClause(
        static_cast<OpenMPDistScheduleClauseKind>(Argument), ArgumentLoc, Expr,
        StartLoc, EndLoc);
    break;
  default:
    break;
  }
  return Res;
}

OMPClause *Sema::ActOnOpenMPScheduleClause(OpenMPScheduleClauseKind Kind,
                                           SourceLocation KindLoc,
                                           Expr *ChunkSize,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  class CConvertDiagnoser : public ICEConvertDiagnoser {
  public:
    CConvertDiagnoser() : ICEConvertDiagnoser(true, false, true) {}
    virtual SemaDiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                                 QualType T) {
      return S.Diag(Loc, diag::err_typecheck_statement_requires_integer) << T;
    }
    virtual SemaDiagnosticBuilder
    diagnoseIncomplete(Sema &S, SourceLocation Loc, QualType T) {
      return S.Diag(Loc, diag::err_incomplete_class_type) << T;
    }
    virtual SemaDiagnosticBuilder diagnoseExplicitConv(Sema &S,
                                                       SourceLocation Loc,
                                                       QualType T,
                                                       QualType ConvTy) {
      return S.Diag(Loc, diag::err_explicit_conversion) << T << ConvTy;
    }

    virtual SemaDiagnosticBuilder
    noteExplicitConv(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) {
      return S.Diag(Conv->getLocation(), diag::note_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    virtual SemaDiagnosticBuilder diagnoseAmbiguous(Sema &S, SourceLocation Loc,
                                                    QualType T) {
      return S.Diag(Loc, diag::err_multiple_conversions) << T;
    }

    virtual SemaDiagnosticBuilder
    noteAmbiguous(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) {
      return S.Diag(Conv->getLocation(), diag::note_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }

    virtual SemaDiagnosticBuilder diagnoseConversion(Sema &S,
                                                     SourceLocation Loc,
                                                     QualType T,
                                                     QualType ConvTy) {
      llvm_unreachable("conversion functions are permitted");
    }
  } ConvertDiagnoser;

  if (Kind == OMPC_SCHEDULE_unknown) {
    std::string Values;
    std::string Sep(NUM_OPENMP_SCHEDULE_KINDS > 1 ? ", " : "");
    for (int i = OMPC_SCHEDULE_unknown + 1; i < NUM_OPENMP_SCHEDULE_KINDS;
         ++i) {
      Values += "'";
      Values += getOpenMPSimpleClauseTypeName(OMPC_schedule, i);
      Values += "'";
      switch (i) {
      case NUM_OPENMP_SCHEDULE_KINDS - 2:
        Values += " or ";
        break;
      case NUM_OPENMP_SCHEDULE_KINDS - 1:
        break;
      default:
        Values += Sep;
        break;
      }
    }
    Diag(KindLoc, diag::err_omp_unexpected_clause_value)
        << Values << getOpenMPClauseName(OMPC_schedule);
    return 0;
  }
  ExprResult Value;
  if (ChunkSize) {
    if (!ChunkSize->isTypeDependent() && !ChunkSize->isValueDependent() &&
        !ChunkSize->isInstantiationDependent()) {
      SourceLocation Loc = ChunkSize->getExprLoc();
      Value =
          PerformContextualImplicitConversion(Loc, ChunkSize, ConvertDiagnoser);
      if (Value.isInvalid())
        return 0;

      llvm::APSInt Result;
      if (Value.get()->isIntegerConstantExpr(Result, Context) &&
          !Result.isStrictlyPositive()) {
        Diag(Loc, diag::err_negative_expression_in_clause)
            << ChunkSize->getSourceRange();
        return 0;
      }
    }
  } else {
    // OpenMP [2.5.1, Loop Construct, Description, Table 2-1]
    //  dynamic       When no chunk_size is specified, it defaults to 1.
    //  guided        When no chunk_size is specified, it defaults to 1.
    switch (Kind) {
    case OMPC_SCHEDULE_dynamic:
    case OMPC_SCHEDULE_guided:
      Value = ActOnIntegerConstant(StartLoc, 1);
      break;
    default:
      break;
    }
  }
  Expr *ValExpr = Value.get();

  return new (Context)
      OMPScheduleClause(Kind, KindLoc, ValExpr, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPDistScheduleClause(
    OpenMPDistScheduleClauseKind Kind, SourceLocation KindLoc, Expr *ChunkSize,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  class CConvertDiagnoser : public ICEConvertDiagnoser {
  public:
    CConvertDiagnoser() : ICEConvertDiagnoser(true, false, true) {}
    virtual SemaDiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                                 QualType T) {
      return S.Diag(Loc, diag::err_typecheck_statement_requires_integer) << T;
    }
    virtual SemaDiagnosticBuilder
    diagnoseIncomplete(Sema &S, SourceLocation Loc, QualType T) {
      return S.Diag(Loc, diag::err_incomplete_class_type) << T;
    }
    virtual SemaDiagnosticBuilder diagnoseExplicitConv(Sema &S,
                                                       SourceLocation Loc,
                                                       QualType T,
                                                       QualType ConvTy) {
      return S.Diag(Loc, diag::err_explicit_conversion) << T << ConvTy;
    }

    virtual SemaDiagnosticBuilder
    noteExplicitConv(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) {
      return S.Diag(Conv->getLocation(), diag::note_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    virtual SemaDiagnosticBuilder diagnoseAmbiguous(Sema &S, SourceLocation Loc,
                                                    QualType T) {
      return S.Diag(Loc, diag::err_multiple_conversions) << T;
    }

    virtual SemaDiagnosticBuilder
    noteAmbiguous(Sema &S, CXXConversionDecl *Conv, QualType ConvTy) {
      return S.Diag(Conv->getLocation(), diag::note_conversion)
             << ConvTy->isEnumeralType() << ConvTy;
    }

    virtual SemaDiagnosticBuilder diagnoseConversion(Sema &S,
                                                     SourceLocation Loc,
                                                     QualType T,
                                                     QualType ConvTy) {
      llvm_unreachable("conversion functions are permitted");
    }
  } ConvertDiagnoser;

  if (Kind != OMPC_DIST_SCHEDULE_static) {
    std::string Values = "'";
    Values += getOpenMPSimpleClauseTypeName(OMPC_dist_schedule,
                                            OMPC_DIST_SCHEDULE_static);
    Values += "'";
    Diag(KindLoc, diag::err_omp_unexpected_clause_value)
        << Values << getOpenMPClauseName(OMPC_dist_schedule);
    return 0;
  }
  ExprResult Value;
  if (ChunkSize) {
    if (!ChunkSize->isTypeDependent() && !ChunkSize->isValueDependent() &&
        !ChunkSize->isInstantiationDependent()) {
      SourceLocation Loc = ChunkSize->getExprLoc();
      Value =
          PerformContextualImplicitConversion(Loc, ChunkSize, ConvertDiagnoser);
      if (Value.isInvalid())
        return 0;

      llvm::APSInt Result;
      if (Value.get()->isIntegerConstantExpr(Result, Context) &&
          !Result.isStrictlyPositive()) {
        Diag(Loc, diag::err_negative_expression_in_clause)
            << ChunkSize->getSourceRange();
        return 0;
      }
    }
  } else {
    Value = ExprEmpty();
  }
  Expr *ValExpr = Value.get();

  return new (Context)
      OMPDistScheduleClause(Kind, KindLoc, ValExpr, StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPVarListClause(
    OpenMPClauseKind Kind, ArrayRef<Expr *> VarList, SourceLocation StartLoc,
    SourceLocation EndLoc, unsigned Op, Expr *TailExpr, CXXScopeSpec &SS,
    const UnqualifiedId &OpName, SourceLocation OpLoc) {
  OMPClause *Res = 0;
  switch (Kind) {
  case OMPC_private:
    Res = ActOnOpenMPPrivateClause(VarList, StartLoc, EndLoc);
    break;
  case OMPC_lastprivate:
    Res = ActOnOpenMPLastPrivateClause(VarList, StartLoc, EndLoc);
    break;
  case OMPC_firstprivate:
    Res = ActOnOpenMPFirstPrivateClause(VarList, StartLoc, EndLoc);
    break;
  case OMPC_shared:
    Res = ActOnOpenMPSharedClause(VarList, StartLoc, EndLoc);
    break;
  case OMPC_copyin:
    Res = ActOnOpenMPCopyinClause(VarList, StartLoc, EndLoc);
    break;
  case OMPC_copyprivate:
    Res = ActOnOpenMPCopyPrivateClause(VarList, StartLoc, EndLoc);
    break;
  case OMPC_reduction:
    Res = ActOnOpenMPReductionClause(
        VarList, StartLoc, EndLoc,
        static_cast<OpenMPReductionClauseOperator>(Op), SS,
        GetNameFromUnqualifiedId(OpName));
    break;
      case OMPC_scan:
          Res = ActOnOpenMPScanClause(
                  VarList, StartLoc, EndLoc,
                  static_cast<OpenMPScanClauseOperator>(Op), SS,
                  GetNameFromUnqualifiedId(OpName));
          break;
  case OMPC_flush:
    Res = ActOnOpenMPFlushClause(VarList, StartLoc, EndLoc);
    break;
  case OMPC_depend:
    Res =
        ActOnOpenMPDependClause(VarList, StartLoc, EndLoc,
                                static_cast<OpenMPDependClauseType>(Op), OpLoc);
    break;
  case OMPC_uniform:
    Res = ActOnOpenMPUniformClause(VarList, StartLoc, EndLoc);
    break;
  case OMPC_linear:
    Res = ActOnOpenMPLinearClause(VarList, StartLoc, EndLoc, TailExpr, OpLoc);
    break;
  case OMPC_aligned:
    Res = ActOnOpenMPAlignedClause(VarList, StartLoc, EndLoc, TailExpr, OpLoc);
    break;
  case OMPC_map:
    Res = ActOnOpenMPMapClause(VarList, StartLoc, EndLoc,
                               static_cast<OpenMPMapClauseKind>(Op), OpLoc);
    break;
  case OMPC_to:
    Res = ActOnOpenMPToClause(VarList, StartLoc, EndLoc);
    break;
  case OMPC_from:
    Res = ActOnOpenMPFromClause(VarList, StartLoc, EndLoc);
    break;
  default:
    break;
  }
  return Res;
}

Expr *Sema::ActOnOpenMPParameterInDeclarativeVarListClause(SourceLocation Loc,
                                                           ParmVarDecl *Param) {
  QualType ExprType = Param->getType().getNonReferenceType();
  DeclContext *SavedCurContext = CurContext;
  CurContext = Param->getDeclContext();
  ExprResult DE = BuildDeclRefExpr(Param, ExprType, VK_RValue, Loc);
  CurContext = SavedCurContext;
  return DE.get();
}

Expr *Sema::FindOpenMPDeclarativeClauseParameter(StringRef Name,
                                                 SourceLocation Loc,
                                                 Decl *FuncDecl) {
  FunctionDecl *FDecl = dyn_cast<FunctionDecl>(FuncDecl);
  FunctionTemplateDecl *FTDecl = dyn_cast<FunctionTemplateDecl>(FuncDecl);
  if (FTDecl) {
    FDecl = FTDecl->getTemplatedDecl();
  }
  if (!FDecl)
    return 0;
  for (FunctionDecl::param_iterator PI = FDecl->param_begin(),
                                    PE = FDecl->param_end();
       PI != PE; ++PI) {
    ParmVarDecl *Param = *PI;
    if (Name == Param->getName()) {
      Expr *E = ActOnOpenMPParameterInDeclarativeVarListClause(Loc, Param);
      if (E) {
        return E;
      }
    }
  }
  return 0;
}

OMPClause *Sema::ActOnOpenMPDeclarativeVarListClause(
    OpenMPClauseKind CKind, ArrayRef<DeclarationNameInfo> NameInfos,
    SourceLocation StartLoc, SourceLocation EndLoc, Expr *TailExpr,
    SourceLocation TailLoc, Decl *FuncDecl) {
  // Vars for the clause.
  SmallVector<Expr *, 4> Vars;
  if (FuncDecl) {
    // Find each var among the function parameters.
    for (unsigned J = 0; J < NameInfos.size(); ++J) {
      Expr *Param = FindOpenMPDeclarativeClauseParameter(
          NameInfos[J].getName().getAsString(), NameInfos[J].getLoc(),
          FuncDecl);
      if (!Param) {
        Diag(NameInfos[J].getLoc(), diag::err_omp_arg_not_found);
      } else {
        Vars.push_back(Param);
      }
    }
  }

  switch (CKind) {
  case OMPC_linear:
    return ActOnOpenMPDeclarativeLinearClause(Vars, StartLoc, EndLoc, TailExpr,
                                              TailLoc);
  case OMPC_aligned:
    return ActOnOpenMPDeclarativeAlignedClause(Vars, StartLoc, EndLoc, TailExpr,
                                               TailLoc);
  case OMPC_uniform:
    return ActOnOpenMPDeclarativeUniformClause(Vars, StartLoc, EndLoc);
  default:
    assert(0 && "bad clause kind for a declarative clause");
  }
  return 0;
}

OMPClause *Sema::ActOnOpenMPDeclarativeLinearClause(ArrayRef<Expr *> VarList,
                                                    SourceLocation StartLoc,
                                                    SourceLocation EndLoc,
                                                    Expr *Step,
                                                    SourceLocation StepLoc) {
  if (VarList.empty())
    return 0;
  // OpenMP [2.8.2 declare simd Construct, Restrictions]
  // When a constant-linear-step expression is specified in a linear clause
  // it must be a constant positive integer expression
  if (Step) {
    Step = ActOnConstantPositiveSubExpressionInClause(Step);
    if (!Step)
      return 0;
  }

  // Check the vars.
  SmallVector<Expr *, 4> Vars;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {

    assert(*I && "Null expr in omp linear");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      continue;
    }
    SourceLocation ELoc = (*I)->getExprLoc();
    //  A list-item that appears in a linear clause must be of integral
    //   or pointer type.
    //
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I);
    QualType QTy = DE->getType().getUnqualifiedType().getCanonicalType();
    const Type *Ty = QTy.getTypePtrOrNull();
    if (!Ty || (!Ty->isDependentType() && !Ty->isIntegerType() &&
                !Ty->isPointerType())) {
      Diag(ELoc, diag::err_omp_expected_int_or_ptr) << (*I)->getSourceRange();
      continue;
    }

    Vars.push_back(DE);
  }

  if (Vars.empty())
    return 0;

  return OMPLinearClause::Create(Context, StartLoc, EndLoc, VarList, Step,
                                 StepLoc);
}

OMPClause *Sema::ActOnOpenMPDeclarativeAlignedClause(
    ArrayRef<Expr *> VarList, SourceLocation StartLoc, SourceLocation EndLoc,
    Expr *Alignment, SourceLocation AlignmentLoc) {
  SmallVector<Expr *, 4> Vars;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {

    assert(*I && "Null expr in omp aligned");
    if (*I && isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I);

    // OpenMP  [2.8.2, declare simd construct, Restrictions]
    // The type of list items appearing in the aligned clause must be
    // array, pointer, reference to array, or reference to pointer.
    QualType QTy = DE->getType()
                       .getNonReferenceType()
                       .getUnqualifiedType()
                       .getCanonicalType();
    const Type *Ty = QTy.getTypePtrOrNull();
    if (!Ty || (!Ty->isDependentType() && !Ty->isArrayType() &&
                !Ty->isPointerType())) {
      Diag(ELoc, diag::err_omp_expected_array_or_ptr) << (*I)->getSourceRange();
      continue;
    }

    Vars.push_back(DE);
  }

  if (Vars.empty())
    return 0;

  // OpenMP [2.8.2 declare simd Construct]
  // The optional parameter of the aligned clause, alignment, must be
  // a constant positive integer expression.
  if (Alignment) {
    Alignment = ActOnConstantPositiveSubExpressionInClause(Alignment);
    if (!Alignment)
      return 0;
  }
  return OMPAlignedClause::Create(Context, StartLoc, EndLoc, VarList, Alignment,
                                  AlignmentLoc);
}

OMPClause *Sema::ActOnOpenMPDeclarativeUniformClause(ArrayRef<Expr *> VarList,
                                                     SourceLocation StartLoc,
                                                     SourceLocation EndLoc) {
  if (VarList.empty())
    return 0;
  return OMPUniformClause::Create(Context, StartLoc, EndLoc, VarList);
}

OMPClause *Sema::ActOnOpenMPPrivateClause(ArrayRef<Expr *> VarList,
                                          SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  SmallVector<Expr *, 4> Vars;
  SmallVector<Expr *, 4> DefaultInits;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "Null expr in omp private");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      DefaultInits.push_back(0);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.9.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << (*I)->getSourceRange();
      continue;
    }
    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(*I);
      DefaultInits.push_back(0);
      continue;
    }

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.3]
    //  A variable that appears in a private clause must not have an incomplete
    //  type or a reference type.
    if (RequireCompleteType(ELoc, Type,
                            diag::err_omp_private_incomplete_type)) {
      continue;
    }
    if (Type->isReferenceType()) {
      Diag(ELoc, diag::err_omp_clause_ref_type_arg)
          << getOpenMPClauseName(OMPC_private);
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below. For these exceptions only, listing a predetermined
    //  variable in a data-sharing attribute clause is allowed and overrides
    //  the variable's predetermined data-sharing attributes.
    DeclRefExpr *PrevRef;
    OpenMPClauseKind Kind = DSAStack->getTopDSA(VD, PrevRef);
    if (Kind != OMPC_unknown && Kind != OMPC_private) {
      Diag(ELoc, diag::err_omp_wrong_dsa) << getOpenMPClauseName(Kind)
                                          << getOpenMPClauseName(OMPC_private);
      if (PrevRef) {
        Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
            << getOpenMPClauseName(Kind);
      } else {
        Diag(VD->getLocation(), diag::note_omp_predetermined_dsa)
            << getOpenMPClauseName(Kind);
      }
      continue;
    }

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.1]
    //  A variable of class type (or array thereof) that appears in a private
    //  clause requires an accesible, unambiguous default constructor for the
    //  class type.
    Type = Type.getNonReferenceType().getCanonicalType();
    while (Type->isArrayType()) {
      QualType ElemType = cast<ArrayType>(Type.getTypePtr())->getElementType();
      Type = ElemType.getNonReferenceType().getCanonicalType();
    }
    CXXRecordDecl *RD =
        getLangOpts().CPlusPlus ? Type->getAsCXXRecordDecl() : 0;
    if (RD) {
      CXXConstructorDecl *CD = LookupDefaultConstructor(RD);
      PartialDiagnostic PD =
          PartialDiagnostic(PartialDiagnostic::NullDiagnostic());
      if (!CD ||
          CheckConstructorAccess(ELoc, CD,
                                 InitializedEntity::InitializeTemporary(Type),
                                 CD->getAccess(), PD) == AR_inaccessible ||
          CD->isDeleted()) {
        Diag(ELoc, diag::err_omp_required_method)
            << getOpenMPClauseName(OMPC_private) << 0;
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
        Diag(RD->getLocation(), diag::note_previous_decl) << RD;
        continue;
      }
      MarkFunctionReferenced(ELoc, CD);
      DiagnoseUseOfDecl(CD, ELoc);

      CXXDestructorDecl *DD = RD->getDestructor();
      if (DD && (CheckDestructorAccess(ELoc, DD, PD) == AR_inaccessible ||
                 DD->isDeleted())) {
        Diag(ELoc, diag::err_omp_required_method)
            << getOpenMPClauseName(OMPC_private) << 4;
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
        Diag(RD->getLocation(), diag::note_previous_decl) << RD;
        continue;
      } else if (DD) {
        MarkFunctionReferenced(ELoc, DD);
        DiagnoseUseOfDecl(DD, ELoc);
      }
    }
    Type = Type.getUnqualifiedType();
    IdentifierInfo *Id = &Context.Idents.get(".private.");
    TypeSourceInfo *TI = Context.getTrivialTypeSourceInfo(Type, ELoc);
    VarDecl *PseudoVar = VarDecl::Create(
        Context, Context.getTranslationUnitDecl(), SourceLocation(),
        SourceLocation(), Id, Type, TI, SC_Static);
    PseudoVar->setImplicit();
    PseudoVar->addAttr(new (Context) UnusedAttr(SourceLocation(), Context, 0));
    InitializedEntity Entity = InitializedEntity::InitializeVariable(PseudoVar);
    InitializationKind InitKind = InitializationKind::CreateDefault(ELoc);
    InitializationSequence InitSeq(*this, Entity, InitKind, MultiExprArg());
    ExprResult Res = InitSeq.Perform(*this, Entity, InitKind, MultiExprArg());
    if (Res.isInvalid())
      continue;
    DefaultInits.push_back(ActOnFinishFullExpr(Res.get()).get());
    DSAStack->addDSA(VD, DE, OMPC_private);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return 0;

  return OMPPrivateClause::Create(Context, StartLoc, EndLoc, Vars,
                                  DefaultInits);
}

OMPClause *Sema::ActOnOpenMPFirstPrivateClause(ArrayRef<Expr *> VarList,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc) {
  SmallVector<Expr *, 4> Vars;
  SmallVector<DeclRefExpr *, 4> PseudoVars;
  SmallVector<Expr *, 4> Inits;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "Null expr in omp firstprivate");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      PseudoVars.push_back(0);
      Inits.push_back(0);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.9.3.4, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << (*I)->getSourceRange();
      continue;
    }
    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(*I);
      PseudoVars.push_back(0);
      Inits.push_back(0);
      continue;
    }

    // OpenMP [2.9.3.4, Restrictions, C/C++, p.2]
    //  A variable that appears in a firstprivate clause must not have an
    //  incomplete type or a reference type.
    if (RequireCompleteType(ELoc, Type,
                            diag::err_omp_firstprivate_incomplete_type)) {
      continue;
    }
    if (Type->isReferenceType()) {
      Diag(ELoc, diag::err_omp_clause_ref_type_arg)
          << getOpenMPClauseName(OMPC_firstprivate);
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below. For these exceptions only, listing a predetermined
    //  variable in a data-sharing attribute clause is allowed and overrides
    //  the variable's predetermined data-sharing attributes.
    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, C/C++, p.2]
    //  Variables with const-qualified type having no mutable member may be
    //  listed in a firstprivate clause, even if they are static data members.
    // OpenMP [2.9.3.4, Description]
    //  If a list item appears in both firstprivate and lastprivate clauses,
    //  the update requires for lastprivate occurs after all the initializations
    //  for firstprivate.
    DeclRefExpr *PrevRef;
    OpenMPDirectiveKind CurrDir = DSAStack->getCurrentDirective();
    OpenMPClauseKind Kind = DSAStack->getTopDSA(VD, PrevRef);
    Type = Type.getNonReferenceType().getCanonicalType();
    bool IsConstant = Type.isConstant(Context);
    bool IsArray = Type->isArrayType();
    while (Type->isArrayType()) {
      QualType ElemType = cast<ArrayType>(Type.getTypePtr())->getElementType();
      Type = ElemType.getNonReferenceType().getCanonicalType();
    }
    if (Kind != OMPC_unknown && Kind != OMPC_firstprivate &&
        Kind != OMPC_lastprivate &&
        !(Kind == OMPC_shared && !PrevRef &&
          (IsConstant || VD->isStaticDataMember()))) {
      if ((CurrDir != OMPD_task || PrevRef) && StartLoc.isValid() &&
          EndLoc.isValid()) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(Kind)
            << getOpenMPClauseName(OMPC_firstprivate);
        if (PrevRef) {
          Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
              << getOpenMPClauseName(Kind);
        } else {
          Diag(VD->getLocation(), diag::note_omp_predetermined_dsa)
              << getOpenMPClauseName(Kind);
        }
        continue;
      }
    }

    // OpenMP [2.9.3.4, Restrictions, p.2]
    //  A list item that is private within a parallel region must not appear in
    //  a firstprivate clause on a worksharing construct if any of the
    //  worksharing regions arising from the worksharing construct ever bind to
    //  any of the parallel regions arising from the parallel construct.
    // OpenMP [2.9.3.4, Restrictions, p.3]
    //  A list item that appears in a reduction clause of a parallel construct
    //  must not appear in a firstprivate clause on a worksharing or task
    //  construct if any of the worksharing or task regions arising from the
    //  worksharing or task construct ever bind to any of the parallel regions
    //  arising from the parallel construct.
    // OpenMP [2.9.3.4, Restrictions, p.4]
    //  A list item that appears in a reduction clause in worksharing construct
    //  must not appear in a firstprivate clause in a task construct encountered
    //  during execution of any of the worksharing regions arising from the
    //  worksharing construct.
    OpenMPDirectiveKind DKind;
    Kind = DSAStack->getImplicitDSA(VD, DKind, PrevRef);
    if ((Kind != OMPC_shared &&
         (CurrDir == OMPD_for || CurrDir == OMPD_sections ||
          CurrDir == OMPD_for_simd || CurrDir == OMPD_distribute_simd ||
          CurrDir == OMPD_single || CurrDir == OMPD_distribute)) ||
        (CurrDir == OMPD_task &&
         DSAStack->hasDSA(VD, OMPC_reduction, OMPD_parallel, PrevRef))) {
      if (Kind == OMPC_unknown) {
        Diag(ELoc, diag::err_omp_required_access)
            << getOpenMPClauseName(OMPC_firstprivate)
            << getOpenMPClauseName(OMPC_shared);
        if (PrevRef) {
          Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
              << getOpenMPClauseName(Kind);
        }
        continue;
      } else if (DKind == OMPD_unknown) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(Kind)
            << getOpenMPClauseName(OMPC_firstprivate);
        if (PrevRef) {
          Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
              << getOpenMPClauseName(Kind);
        }
        continue;
      } else {
        // Skip template instantiations for parallel for and parallel sections.
        if (Kind != OMPC_firstprivate || DKind != OMPD_parallel ||
            (CurrDir != OMPD_for && CurrDir != OMPD_sections) || !PrevRef ||
            PrevRef->getExprLoc() != ELoc) {
          Diag(ELoc, diag::err_omp_dsa_with_directives)
              << getOpenMPClauseName(Kind) << getOpenMPDirectiveName(DKind)
              << getOpenMPClauseName(OMPC_firstprivate)
              << getOpenMPDirectiveName(CurrDir);
          if (PrevRef) {
            Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
                << getOpenMPClauseName(Kind);
          }
          continue;
        }
      }
    }

    // OpenMP [2.9.3.4, Restrictions, C/C++, p.1]
    //  A variable of class type (or array thereof) that appears in a
    //  firstprivate clause requires an accesible, unambiguous copy constructor
    //  for the class type.
    CXXRecordDecl *RD =
        getLangOpts().CPlusPlus ? Type->getAsCXXRecordDecl() : 0;
    if (RD) {
      CXXConstructorDecl *CD = LookupCopyingConstructor(RD, 0);
      PartialDiagnostic PD =
          PartialDiagnostic(PartialDiagnostic::NullDiagnostic());
      if (!CD ||
          CheckConstructorAccess(ELoc, CD,
                                 InitializedEntity::InitializeTemporary(Type),
                                 CD->getAccess(), PD) == AR_inaccessible ||
          CD->isDeleted()) {
        Diag(ELoc, diag::err_omp_required_method)
            << getOpenMPClauseName(OMPC_firstprivate) << 1;
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
        Diag(RD->getLocation(), diag::note_previous_decl) << RD;
        continue;
      }
      MarkFunctionReferenced(ELoc, CD);
      DiagnoseUseOfDecl(CD, ELoc);

      CXXDestructorDecl *DD = RD->getDestructor();
      if (DD && (CheckDestructorAccess(ELoc, DD, PD) == AR_inaccessible ||
                 DD->isDeleted())) {
        Diag(ELoc, diag::err_omp_required_method)
            << getOpenMPClauseName(OMPC_firstprivate) << 4;
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
        Diag(RD->getLocation(), diag::note_previous_decl) << RD;
        continue;
      } else if (DD) {
        MarkFunctionReferenced(ELoc, DD);
        DiagnoseUseOfDecl(DD, ELoc);
      }
    }

    Type = Type.getUnqualifiedType();
    if ((RD && !RD->isTriviallyCopyable()) || IsArray) {
      DeclRefExpr *PseudoDE = DE;
      IdentifierInfo *Id = &Context.Idents.get(".firstprivate.");
      TypeSourceInfo *TI = Context.getTrivialTypeSourceInfo(Type, ELoc);
      VarDecl *PseudoVar = VarDecl::Create(
          Context, Context.getTranslationUnitDecl(), SourceLocation(),
          SourceLocation(), Id, Type, TI, SC_Static);
      PseudoVar->setImplicit();
      PseudoVar->addAttr(new (Context)
                             UnusedAttr(SourceLocation(), Context, 0));
      Context.getTranslationUnitDecl()->addHiddenDecl(PseudoVar);
      PseudoDE = cast<DeclRefExpr>(
          BuildDeclRefExpr(PseudoVar, Type, VK_LValue, ELoc).get());
      InitializedEntity Entity =
          InitializedEntity::InitializeVariable(PseudoVar);
      InitializationKind InitKind = InitializationKind::CreateCopy(ELoc, ELoc);
      Expr *Arg = DefaultLvalueConversion(PseudoDE).get();
      if (!Arg)
        continue;
      InitializationSequence InitSeq(*this, Entity, InitKind,
                                     MultiExprArg(&Arg, 1));
      ExprResult Res =
          InitSeq.Perform(*this, Entity, InitKind, MultiExprArg(&Arg, 1));
      if (Res.isInvalid())
        continue;
      PseudoVars.push_back(PseudoDE);
      Inits.push_back(ActOnFinishFullExpr(Res.get()).get());
    } else {
      PseudoVars.push_back(0);
      Inits.push_back(0);
    }
    DSAStack->addDSA(VD, DE, OMPC_firstprivate);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return 0;

  return OMPFirstPrivateClause::Create(Context, StartLoc, EndLoc, Vars,
                                       PseudoVars, Inits);
}

OMPClause *Sema::ActOnOpenMPLastPrivateClause(ArrayRef<Expr *> VarList,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  SmallVector<Expr *, 4> Vars;
  SmallVector<DeclRefExpr *, 4> PseudoVars1;
  SmallVector<DeclRefExpr *, 4> PseudoVars2;
  SmallVector<Expr *, 4> Assignments;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "Null expr in omp lastprivate");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      PseudoVars1.push_back(0);
      PseudoVars2.push_back(0);
      Assignments.push_back(0);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.11.3.5, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << (*I)->getSourceRange();
      continue;
    }
    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(*I);
      PseudoVars1.push_back(0);
      PseudoVars2.push_back(0);
      Assignments.push_back(0);
      continue;
    }

    // OpenMP [2.9.3.11, Restrictions, C/C++, p.4]
    //  A variable that appears in a firstprivate clause must not have an
    //  incomplete type or a reference type.
    if (RequireCompleteType(ELoc, Type,
                            diag::err_omp_lastprivate_incomplete_type)) {
      continue;
    }
    if (Type->isReferenceType()) {
      Diag(ELoc, diag::err_omp_clause_ref_type_arg)
          << getOpenMPClauseName(OMPC_lastprivate);
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below. For these exceptions only, listing a predetermined
    //  variable in a data-sharing attribute clause is allowed and overrides
    //  the variable's predetermined data-sharing attributes.
    // OpenMP [2.9.3.4, Description]
    //  If a list item appears in both firstprivate and lastprivate clauses,
    //  the update requires for lastprivate occurs after all the initializations
    //  for firstprivate.
    DeclRefExpr *PrevRef;
    OpenMPClauseKind Kind = DSAStack->getTopDSA(VD, PrevRef);
    Type = Type.getNonReferenceType().getCanonicalType();
    bool IsArray = Type->isArrayType();
    while (Type->isArrayType()) {
      QualType ElemType = cast<ArrayType>(Type.getTypePtr())->getElementType();
      Type = ElemType.getNonReferenceType().getCanonicalType();
    }
    if (Kind != OMPC_unknown && Kind != OMPC_firstprivate &&
        Kind != OMPC_lastprivate) {
      Diag(ELoc, diag::err_omp_wrong_dsa)
          << getOpenMPClauseName(Kind) << getOpenMPClauseName(OMPC_lastprivate);
      if (PrevRef) {
        Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
            << getOpenMPClauseName(Kind);
      } else {
        Diag(VD->getLocation(), diag::note_omp_predetermined_dsa)
            << getOpenMPClauseName(Kind);
      }
      continue;
    }
    bool IsNotFirstprivate = Kind != OMPC_firstprivate;

    // OpenMP [2.9.3.5, Restrictions, p.2]
    //  A list item that is private within a parallel region, or that appears
    //  in the reduction clause of a parallel construct,  must not appear in
    //  a lastprivate clause on a worksharing construct if any of the
    //  worksharing regions ever bind to any of the correspponding parallel
    //  regions.
    OpenMPDirectiveKind DKind;
    OpenMPDirectiveKind CurrDir = DSAStack->getCurrentDirective();
    Kind = DSAStack->getImplicitDSA(VD, DKind, PrevRef);
    if ((Kind != OMPC_shared && Kind != OMPC_unknown &&
         DKind != OMPD_unknown) &&
        (CurrDir == OMPD_for || CurrDir == OMPD_sections ||
         CurrDir == OMPD_for_simd)) {
      if (Kind == OMPC_unknown) {
        Diag(ELoc, diag::err_omp_required_access)
            << getOpenMPClauseName(OMPC_lastprivate)
            << getOpenMPClauseName(OMPC_shared);
      } else if (DKind == OMPD_unknown) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(Kind)
            << getOpenMPClauseName(OMPC_lastprivate);
      } else {
        Diag(ELoc, diag::err_omp_dsa_with_directives)
            << getOpenMPClauseName(Kind) << getOpenMPDirectiveName(DKind)
            << getOpenMPClauseName(OMPC_lastprivate)
            << getOpenMPDirectiveName(CurrDir);
      }
      if (PrevRef) {
        Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
            << getOpenMPClauseName(Kind);
      }
      continue;
    }

    // OpenMP [2.9.3.5, Restrictions, C/C++, p.2]
    //  A variable of class type (or array thereof) that appears in a
    //  lastprivate clause requires an accesible, unambiguous copy assignment
    //  operator for the class type.
    CXXRecordDecl *RD =
        getLangOpts().CPlusPlus ? Type->getAsCXXRecordDecl() : 0;
    if (RD) {
      CXXMethodDecl *MD = LookupCopyingAssignment(RD, 0, false, 0);
      if (!MD ||
          CheckMemberAccess(ELoc, RD,
                            DeclAccessPair::make(MD, MD->getAccess())) ==
              AR_inaccessible ||
          MD->isDeleted()) {
        Diag(ELoc, diag::err_omp_required_method)
            << getOpenMPClauseName(OMPC_lastprivate) << 2;
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
        Diag(RD->getLocation(), diag::note_previous_decl) << RD;
        continue;
      }
      MarkFunctionReferenced(ELoc, MD);
      DiagnoseUseOfDecl(MD, ELoc);
      PartialDiagnostic PD =
          PartialDiagnostic(PartialDiagnostic::NullDiagnostic());
      CXXDestructorDecl *DD = RD->getDestructor();
      if (DD && (CheckDestructorAccess(ELoc, DD, PD) == AR_inaccessible ||
                 DD->isDeleted())) {
        Diag(ELoc, diag::err_omp_required_method)
            << getOpenMPClauseName(OMPC_lastprivate) << 4;
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
        Diag(RD->getLocation(), diag::note_previous_decl) << RD;
        continue;
      } else if (DD) {
        MarkFunctionReferenced(ELoc, DD);
        DiagnoseUseOfDecl(DD, ELoc);
      }
    }

    Type = Type.getUnqualifiedType();
    IdentifierInfo *Id = &Context.Idents.get(".lastprivate.");
    TypeSourceInfo *TI = Context.getTrivialTypeSourceInfo(Type, ELoc);
    VarDecl *PseudoVar1 = VarDecl::Create(
        Context, Context.getTranslationUnitDecl(), SourceLocation(),
        SourceLocation(), Id, Type, TI, SC_Static);
    PseudoVar1->setImplicit();
    PseudoVar1->addAttr(new (Context) UnusedAttr(SourceLocation(), Context, 0));
    Context.getTranslationUnitDecl()->addHiddenDecl(PseudoVar1);
    DeclRefExpr *PseudoDE1 = cast<DeclRefExpr>(
        BuildDeclRefExpr(PseudoVar1, Type, VK_LValue, ELoc).get());
    if ((RD && !RD->isTriviallyCopyable()) || IsArray) {
      VarDecl *PseudoVar2 = VarDecl::Create(
          Context, Context.getTranslationUnitDecl(), SourceLocation(),
          SourceLocation(), Id, Type, TI, SC_Static);
      PseudoVar2->setImplicit();
      PseudoVar2->addAttr(new (Context)
                              UnusedAttr(SourceLocation(), Context, 0));
      Context.getTranslationUnitDecl()->addHiddenDecl(PseudoVar2);
      DeclRefExpr *PseudoDE2 = cast<DeclRefExpr>(
          BuildDeclRefExpr(PseudoVar2, Type, VK_LValue, ELoc).get());
      Expr *PseudoDE2RVal = DefaultLvalueConversion(PseudoDE2).get();
      if (!PseudoDE2RVal)
        continue;
      ExprResult Res = BuildBinOp(DSAStack->getCurScope(), ELoc, BO_Assign,
                                  PseudoDE1, PseudoDE2RVal).get();
      if (Res.isInvalid())
        continue;
      PseudoVars2.push_back(PseudoDE2);
      Assignments.push_back(
          ActOnFinishFullExpr(IgnoredValueConversions(Res.get()).get()).get());
    } else {
      PseudoVars2.push_back(0);
      Assignments.push_back(0);
    }
    PseudoVars1.push_back(PseudoDE1);
    if (IsNotFirstprivate)
      DSAStack->addDSA(VD, DE, OMPC_lastprivate);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return 0;

  return OMPLastPrivateClause::Create(Context, StartLoc, EndLoc, Vars,
                                      PseudoVars1, PseudoVars2, Assignments);
}

OMPClause *Sema::ActOnOpenMPSharedClause(ArrayRef<Expr *> VarList,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  SmallVector<Expr *, 4> Vars;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "Null expr in omp shared");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.9.3.4, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << (*I)->getSourceRange();
      continue;
    }
    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(*I);
      continue;
    }

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below. For these exceptions only, listing a predetermined
    //  variable in a data-sharing attribute clause is allowed and overrides
    //  the variable's predetermined data-sharing attributes.
    DeclRefExpr *PrevRef;
    OpenMPClauseKind Kind = DSAStack->getTopDSA(VD, PrevRef);
    if (Kind != OMPC_unknown && Kind != OMPC_shared && PrevRef) {
      Diag(ELoc, diag::err_omp_wrong_dsa) << getOpenMPClauseName(Kind)
                                          << getOpenMPClauseName(OMPC_shared);
      Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
          << getOpenMPClauseName(Kind);
      continue;
    }

    DSAStack->addDSA(VD, DE, OMPC_shared);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return 0;

  return OMPSharedClause::Create(Context, StartLoc, EndLoc, Vars);
}

OMPClause *Sema::ActOnOpenMPCopyinClause(ArrayRef<Expr *> VarList,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  SmallVector<Expr *, 4> Vars;
  SmallVector<DeclRefExpr *, 4> PseudoVars1;
  SmallVector<DeclRefExpr *, 4> PseudoVars2;
  SmallVector<Expr *, 4> Assignments;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "Null expr in omp copyin");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      PseudoVars1.push_back(0);
      PseudoVars2.push_back(0);
      Assignments.push_back(0);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << (*I)->getSourceRange();
      continue;
    }
    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(*I);
      PseudoVars1.push_back(0);
      PseudoVars2.push_back(0);
      Assignments.push_back(0);
      continue;
    }

    // OpenMP [2.9.2, Restrictions, p.1]
    //  A threadprivate variable must not appear in any clause except the
    //  copyin, copyprivate, schedule, num_threads, and if clauses.
    // OpenMP [2.9.4.1, Restrictions, C/C++, p.1]
    //  A list item that appears in a copyin clause must be threadprivate.
    DeclRefExpr *PrevRef;
    OpenMPClauseKind Kind = DSAStack->getTopDSA(VD, PrevRef);
    if (Kind != OMPC_threadprivate && Kind != OMPC_copyin) {
      Diag(ELoc, diag::err_omp_required_access)
          << getOpenMPClauseName(OMPC_copyin)
          << getOpenMPDirectiveName(OMPD_threadprivate);
      continue;
    }

    // OpenMP [2.9.3.4, Restrictions, C/C++, p.1]
    //  A variable of class type (or array thereof) that appears in a
    //  firstprivate clause requires an accesible, unambiguous copy assignment
    //  operator for the class type.
    Type = Type.getNonReferenceType().getCanonicalType();
    bool IsArray = Type->isArrayType();
    while (Type->isArrayType()) {
      QualType ElemType = cast<ArrayType>(Type.getTypePtr())->getElementType();
      Type = ElemType.getNonReferenceType().getCanonicalType();
    }
    CXXRecordDecl *RD =
        getLangOpts().CPlusPlus ? Type->getAsCXXRecordDecl() : 0;
    if (RD) {
      CXXMethodDecl *MD = LookupCopyingAssignment(RD, 0, false, 0);
      if (!MD ||
          CheckMemberAccess(ELoc, RD,
                            DeclAccessPair::make(MD, MD->getAccess())) ==
              AR_inaccessible ||
          MD->isDeleted()) {
        Diag(ELoc, diag::err_omp_required_method)
            << getOpenMPClauseName(OMPC_copyin) << 2;
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
        Diag(RD->getLocation(), diag::note_previous_decl) << RD;
        continue;
      }
      MarkFunctionReferenced(ELoc, MD);
      DiagnoseUseOfDecl(MD, ELoc);
    }

    Type = Type.getUnqualifiedType();
    IdentifierInfo *Id = &Context.Idents.get(".copyin.");
    TypeSourceInfo *TI = Context.getTrivialTypeSourceInfo(Type, ELoc);
    VarDecl *PseudoVar1 = VarDecl::Create(
        Context, Context.getTranslationUnitDecl(), SourceLocation(),
        SourceLocation(), Id, Type, TI, SC_Static);
    PseudoVar1->setImplicit();
    PseudoVar1->addAttr(new (Context) UnusedAttr(SourceLocation(), Context, 0));
    Context.getTranslationUnitDecl()->addHiddenDecl(PseudoVar1);
    DeclRefExpr *PseudoDE1 = cast<DeclRefExpr>(
        BuildDeclRefExpr(PseudoVar1, Type, VK_LValue, ELoc).get());
    if ((RD && !RD->isTriviallyCopyable()) || IsArray) {
      VarDecl *PseudoVar2 = VarDecl::Create(
          Context, Context.getTranslationUnitDecl(), SourceLocation(),
          SourceLocation(), Id, Type, TI, SC_Static);
      PseudoVar2->setImplicit();
      PseudoVar2->addAttr(new (Context)
                              UnusedAttr(SourceLocation(), Context, 0));
      Context.getTranslationUnitDecl()->addHiddenDecl(PseudoVar2);
      DeclRefExpr *PseudoDE2 = cast<DeclRefExpr>(
          BuildDeclRefExpr(PseudoVar2, Type, VK_LValue, ELoc).get());
      Expr *PseudoDE2RVal = DefaultLvalueConversion(PseudoDE2).get();
      if (!PseudoDE2RVal)
        continue;
      ExprResult Res = BuildBinOp(DSAStack->getCurScope(), ELoc, BO_Assign,
                                  PseudoDE1, PseudoDE2RVal).get();
      if (Res.isInvalid())
        continue;
      PseudoVars2.push_back(PseudoDE2);
      Assignments.push_back(
          ActOnFinishFullExpr(IgnoredValueConversions(Res.get()).get()).get());
    } else {
      PseudoVars2.push_back(0);
      Assignments.push_back(0);
    }
    PseudoVars1.push_back(PseudoDE1);
    DSAStack->addDSA(VD, DE, OMPC_copyin);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return 0;

  return OMPCopyinClause::Create(Context, StartLoc, EndLoc, Vars, PseudoVars1,
                                 PseudoVars2, Assignments);
}

OMPClause *Sema::ActOnOpenMPCopyPrivateClause(ArrayRef<Expr *> VarList,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  SmallVector<Expr *, 4> Vars;
  SmallVector<DeclRefExpr *, 4> PseudoVars1;
  SmallVector<DeclRefExpr *, 4> PseudoVars2;
  SmallVector<Expr *, 4> Assignments;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "Null expr in omp copyprivate");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      PseudoVars1.push_back(0);
      PseudoVars2.push_back(0);
      Assignments.push_back(0);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << (*I)->getSourceRange();
      continue;
    }
    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(*I);
      PseudoVars1.push_back(0);
      PseudoVars2.push_back(0);
      Assignments.push_back(0);
      continue;
    }

    // OpenMP [2.11.4.2, Restrictions, p.2]
    //  A list item that appears in a copyprivate clause may not appear in
    //  a private or firstprivate clause on the single construct.
    DeclRefExpr *PrevRef;
    OpenMPClauseKind Kind = DSAStack->getTopDSA(VD, PrevRef);
    if (Kind != OMPC_threadprivate && Kind != OMPC_copyprivate &&
        Kind != OMPC_unknown && !(Kind == OMPC_private && !PrevRef)) {
      Diag(ELoc, diag::err_omp_wrong_dsa)
          << getOpenMPClauseName(Kind) << getOpenMPClauseName(OMPC_copyprivate);
      if (PrevRef) {
        Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
            << getOpenMPClauseName(Kind);
      } else {
        Diag(VD->getLocation(), diag::note_omp_predetermined_dsa)
            << getOpenMPClauseName(Kind);
      }
      continue;
    }

    // OpenMP [2.11.4.2, Restrictions, p.1]
    //  All list items that appear in a copyprivate clause must be either
    //  threadprivate or private in the enclosing context.
    if (Kind == OMPC_unknown) {
      OpenMPDirectiveKind DKind;
      Kind = DSAStack->getImplicitDSA(VD, DKind, PrevRef);
      if (Kind == OMPC_shared) {
        Diag(ELoc, diag::err_omp_required_access)
            << getOpenMPClauseName(OMPC_copyprivate)
            << "threadprivate or private in the enclosing context";
        if (PrevRef) {
          Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
              << getOpenMPClauseName(Kind);
        }
        continue;
      }
    }

    // OpenMP [2.11.4.2, Restrictions, C/C++, p.1]
    //  A variable of class type (or array thereof) that appears in a
    //  copytprivate clause requires an accesible, unambiguous copy assignment
    //  operator for the class type.
    Type = Type.getNonReferenceType().getCanonicalType();
    while (Type->isArrayType()) {
      QualType ElemType = cast<ArrayType>(Type.getTypePtr())->getElementType();
      Type = ElemType.getNonReferenceType().getCanonicalType();
    }
    CXXRecordDecl *RD =
        getLangOpts().CPlusPlus ? Type->getAsCXXRecordDecl() : 0;
    if (RD) {
      CXXMethodDecl *MD = LookupCopyingAssignment(RD, 0, false, 0);
      if (!MD ||
          CheckMemberAccess(ELoc, RD,
                            DeclAccessPair::make(MD, MD->getAccess())) ==
              AR_inaccessible ||
          MD->isDeleted()) {
        Diag(ELoc, diag::err_omp_required_method)
            << getOpenMPClauseName(OMPC_copyprivate) << 2;
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
        Diag(RD->getLocation(), diag::note_previous_decl) << RD;
        continue;
      }
      MarkFunctionReferenced(ELoc, MD);
      DiagnoseUseOfDecl(MD, ELoc);
    }

    Type = Type.getUnqualifiedType();
    IdentifierInfo *Id = &Context.Idents.get(".copyin.");
    TypeSourceInfo *TI = Context.getTrivialTypeSourceInfo(Type, ELoc);
    VarDecl *PseudoVar1 = VarDecl::Create(
        Context, Context.getTranslationUnitDecl(), SourceLocation(),
        SourceLocation(), Id, Type, TI, SC_Static);
    PseudoVar1->setImplicit();
    PseudoVar1->addAttr(new (Context) UnusedAttr(SourceLocation(), Context, 0));
    Context.getTranslationUnitDecl()->addHiddenDecl(PseudoVar1);
    DeclRefExpr *PseudoDE1 = cast<DeclRefExpr>(
        BuildDeclRefExpr(PseudoVar1, Type, VK_LValue, ELoc).get());
    VarDecl *PseudoVar2 = VarDecl::Create(
        Context, Context.getTranslationUnitDecl(), SourceLocation(),
        SourceLocation(), Id, Type, TI, SC_Static);
    PseudoVar2->setImplicit();
    PseudoVar2->addAttr(new (Context) UnusedAttr(SourceLocation(), Context, 0));
    Context.getTranslationUnitDecl()->addHiddenDecl(PseudoVar2);
    DeclRefExpr *PseudoDE2 = cast<DeclRefExpr>(
        BuildDeclRefExpr(PseudoVar2, Type, VK_LValue, ELoc).get());
    Expr *PseudoDE2RVal = DefaultLvalueConversion(PseudoDE2).get();
    if (!PseudoDE2RVal)
      continue;
    ExprResult Res = BuildBinOp(DSAStack->getCurScope(), ELoc, BO_Assign,
                                PseudoDE1, PseudoDE2RVal).get();
    if (Res.isInvalid())
      continue;
    PseudoVars1.push_back(PseudoDE1);
    PseudoVars2.push_back(PseudoDE2);
    Assignments.push_back(
        ActOnFinishFullExpr(IgnoredValueConversions(Res.get()).get()).get());
    DSAStack->addDSA(VD, DE, OMPC_copyprivate);
    Vars.push_back(DE);
  }

  if (Vars.empty())
    return 0;

  return OMPCopyPrivateClause::Create(Context, StartLoc, EndLoc, Vars,
                                      PseudoVars1, PseudoVars2, Assignments);
}

namespace {
class DSARefChecker : public StmtVisitor<DSARefChecker, bool> {
  DSAStackTy *Stack;

public:
  bool VisitDeclRefExpr(DeclRefExpr *E) {
    if (VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
      DeclRefExpr *PrevRef;
      OpenMPClauseKind Kind = Stack->getTopDSA(VD, PrevRef);
      if (Kind == OMPC_shared && !PrevRef)
        return false;
      if (Kind != OMPC_unknown)
        return true;
      // OpenMPDirectiveKind DKind;
      // Kind = Stack->getImplicitDSA(VD, DKind, PrevRef);
      if (Stack->hasDSA(VD, OMPC_private, OMPD_unknown, PrevRef) ||
          Stack->hasDSA(VD, OMPC_firstprivate, OMPD_unknown, PrevRef) ||
          Stack->hasDSA(VD, OMPC_lastprivate, OMPD_unknown, PrevRef) ||
          Stack->hasDSA(VD, OMPC_reduction, OMPD_unknown, PrevRef) ||
          Stack->hasDSA(VD, OMPC_scan, OMPD_unknown, PrevRef) ||
          Stack->hasDSA(VD, OMPC_linear, OMPD_unknown, PrevRef))
        return true;
      return false;
      // return Kind != OMPC_shared && Kind != OMPC_unknown;
    }
    return false;
  }
  bool VisitStmt(Stmt *S) {
    for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E;
         ++I) {
      if (Stmt *Child = *I)
        if (Visit(Child))
          return true;
    }
    return false;
  }
  DSARefChecker(DSAStackTy *S) : Stack(S) {}
};
}

namespace {
class RedDeclFilterCCC : public CorrectionCandidateCallback {
private:
  Sema &Actions;
  QualType QTy;
  OMPDeclareReductionDecl::ReductionData *FoundData;

public:
  RedDeclFilterCCC(Sema &S, QualType QTy)
      : Actions(S), QTy(QTy), FoundData(0) {}
  virtual bool ValidateCandidate(const TypoCorrection &Candidate) {
    if (OMPDeclareReductionDecl *D = dyn_cast_or_null<OMPDeclareReductionDecl>(
            Candidate.getCorrectionDecl())) {
      if (D->isInvalidDecl())
        return false;
      bool Found = false;
      for (OMPDeclareReductionDecl::datalist_iterator IT = D->datalist_begin(),
                                                      ET = D->datalist_end();
           IT != ET; ++IT) {
        if (!IT->QTy.isNull() &&
            (Actions.Context.hasSameUnqualifiedType(IT->QTy, QTy) ||
             Actions.IsDerivedFrom(QTy, IT->QTy))) {
          Found = true;
          FoundData = IT;
        }
      }
      return Found;
    }
    return false;
  }
  OMPDeclareReductionDecl::ReductionData *getFoundData() { return FoundData; }
};
}

static OMPDeclareReductionDecl::ReductionData *
TryToFindDeclareReductionDecl(Sema &SemaRef, CXXScopeSpec &SS,
                              DeclarationNameInfo OpName, QualType QTy,
                              OpenMPReductionClauseOperator Op) {
  LookupResult Lookup(SemaRef, OpName, Sema::LookupOMPDeclareReduction);
  if (Op != OMPC_REDUCTION_custom) {
    Lookup.suppressDiagnostics();
  }
  if (SemaRef.LookupParsedName(Lookup, SemaRef.getCurScope(), &SS)) {
    LookupResult::Filter Filter = Lookup.makeFilter();
    SmallVector<OMPDeclareReductionDecl::ReductionData *, 4> Found;
    SmallVector<OMPDeclareReductionDecl *, 4> FoundDecl;
    while (Filter.hasNext()) {
      OMPDeclareReductionDecl *D = cast<OMPDeclareReductionDecl>(Filter.next());
      bool Remove = true;
      if (!D->isInvalidDecl()) {
        for (OMPDeclareReductionDecl::datalist_iterator
                 IT = D->datalist_begin(),
                 ET = D->datalist_end();
             IT != ET; ++IT) {
          if (!IT->QTy.isNull() &&
              SemaRef.Context.hasSameUnqualifiedType(IT->QTy, QTy)) {
            Found.push_back(IT);
            FoundDecl.push_back(D);
            Remove = false;
          }
        }
        if (Found.empty()) {
          for (OMPDeclareReductionDecl::datalist_iterator
                   IT = D->datalist_begin(),
                   ET = D->datalist_end();
               IT != ET; ++IT) {
            if (!IT->QTy.isNull() && SemaRef.IsDerivedFrom(QTy, IT->QTy)) {
              Found.push_back(IT);
              FoundDecl.push_back(D);
              Remove = false;
            }
          }
        }
      }
      if (Remove)
        Filter.erase();
    }
    Filter.done();
    if (Found.size() > 1) {
      // Ambiguous declaration found.
      SemaRef.Diag(OpName.getLoc(), diag::err_ambiguous_reference)
          << OpName.getName();
      SmallVectorImpl<OMPDeclareReductionDecl::ReductionData *>::iterator IT =
          Found.begin();
      for (SmallVectorImpl<OMPDeclareReductionDecl *>::iterator
               IR = FoundDecl.begin(),
               ER = FoundDecl.end();
           IR != ER; ++IR, ++IT) {
        SemaRef.Diag((*IR)->getLocation(), diag::note_ambiguous_candidate)
            << *IR << (*IT)->TyRange;
      }
    }
    if (!Found.empty())
      return Found.back();
  }
  assert(Lookup.empty() && "Lookup is not empty.");
  return 0;
}

static OMPDeclareScanDecl::ScanData *
TryToFindDeclareScanDecl(Sema &SemaRef, CXXScopeSpec &SS,
                         DeclarationNameInfo OpName, QualType QTy,
                         OpenMPScanClauseOperator Op) {
    LookupResult Lookup(SemaRef, OpName, Sema::LookupOMPDeclareScan);
    if (Op != OMPC_SCAN_custom) {
        Lookup.suppressDiagnostics();
    }
    if (SemaRef.LookupParsedName(Lookup, SemaRef.getCurScope(), &SS)) {
        LookupResult::Filter Filter = Lookup.makeFilter();
        SmallVector<OMPDeclareScanDecl::ScanData *, 4> Found;
        SmallVector<OMPDeclareScanDecl *, 4> FoundDecl;
        while (Filter.hasNext()) {
            OMPDeclareScanDecl *D = cast<OMPDeclareScanDecl>(Filter.next());
            bool Remove = true;
            if (!D->isInvalidDecl()) {
                for (OMPDeclareScanDecl::datalist_iterator
                             IT = D->datalist_begin(),
                             ET = D->datalist_end();
                     IT != ET; ++IT) {
                    if (!IT->QTy.isNull() &&
                        SemaRef.Context.hasSameUnqualifiedType(IT->QTy, QTy)) {
                        Found.push_back(IT);
                        FoundDecl.push_back(D);
                        Remove = false;
                    }
                }
                if (Found.empty()) {
                    for (OMPDeclareScanDecl::datalist_iterator
                                 IT = D->datalist_begin(),
                                 ET = D->datalist_end();
                         IT != ET; ++IT) {
                        if (!IT->QTy.isNull() && SemaRef.IsDerivedFrom(QTy, IT->QTy)) {
                            Found.push_back(IT);
                            FoundDecl.push_back(D);
                            Remove = false;
                        }
                    }
                }
            }
            if (Remove)
                Filter.erase();
        }
        Filter.done();
        if (Found.size() > 1) {
            // Ambiguous declaration found.
            SemaRef.Diag(OpName.getLoc(), diag::err_ambiguous_reference)
                    << OpName.getName();
            SmallVectorImpl<OMPDeclareScanDecl::ScanData *>::iterator IT =
                    Found.begin();
            for (SmallVectorImpl<OMPDeclareScanDecl *>::iterator
                         IR = FoundDecl.begin(),
                         ER = FoundDecl.end();
                 IR != ER; ++IR, ++IT) {
                SemaRef.Diag((*IR)->getLocation(), diag::note_ambiguous_candidate)
                        << *IR << (*IT)->TyRange;
            }
        }
        if (!Found.empty())
            return Found.back();
    }
    assert(Lookup.empty() && "Lookup is not empty.");
    return 0;
}

OMPClause *Sema::ActOnOpenMPReductionClause(ArrayRef<Expr *> VarList,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc,
                                            OpenMPReductionClauseOperator Op,
                                            CXXScopeSpec &SS,
                                            DeclarationNameInfo OpName) {
  BinaryOperatorKind NewOp = BO_Assign;
  switch (Op) {
  case OMPC_REDUCTION_add:
    NewOp = BO_AddAssign;
    break;
  case OMPC_REDUCTION_mult:
    NewOp = BO_MulAssign;
    break;
  case OMPC_REDUCTION_sub:
    NewOp = BO_SubAssign;
    break;
  case OMPC_REDUCTION_bitand:
    NewOp = BO_AndAssign;
    break;
  case OMPC_REDUCTION_bitor:
    NewOp = BO_OrAssign;
    break;
  case OMPC_REDUCTION_bitxor:
    NewOp = BO_XorAssign;
    break;
  case OMPC_REDUCTION_and:
    NewOp = BO_LAnd;
    break;
  case OMPC_REDUCTION_or:
    NewOp = BO_LOr;
    break;
  case OMPC_REDUCTION_min:
    NewOp = BO_LT;
    break;
  case OMPC_REDUCTION_max:
    NewOp = BO_GT;
    break;
  default:
    break;
  }
  SmallVector<Expr *, 4> Vars;
  SmallVector<Expr *, 4> DefaultInits;
  SmallVector<Expr *, 4> OpExprs;
  SmallVector<Expr *, 4> HelperParams1;
  SmallVector<Expr *, 4> HelperParams2;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "Null expr in omp reduction");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      DefaultInits.push_back(0);
      OpExprs.push_back(0);
      HelperParams1.push_back(0);
      HelperParams2.push_back(0);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.9.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << (*I)->getSourceRange();
      continue;
    }
    Decl *D = DE->getDecl();
    VarDecl *VD = cast<VarDecl>(D);

    QualType Type = VD->getType();
    if (Type->isDependentType() || Type->isInstantiationDependentType()) {
      // It will be analyzed later.
      Vars.push_back(*I);
      DefaultInits.push_back(0);
      OpExprs.push_back(0);
      HelperParams1.push_back(0);
      HelperParams2.push_back(0);
      continue;
    }

    // OpenMP [2.9.3.6, Restrictions, C/C++, p.4]
    //  If a list-item is a reference type then it must bind to the same object
    //  for all threads of the team.
    if (Type.getCanonicalType()->isReferenceType() && VD->hasInit()) {
      DSARefChecker Check(DSAStack);
      if (Check.Visit(VD->getInit())) {
        Diag(ELoc, diag::err_omp_reduction_ref_type_arg)
            << getOpenMPClauseName(OMPC_reduction);
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
        continue;
      }
    }

    // OpenMP [2.9.3.6, Restrictions, C/C++, p.2]
    //  Aggregate types (including arrays), pointer types and reference types
    //  may not appear in a reduction clause.
    if (RequireCompleteType(ELoc, Type,
                            diag::err_omp_reduction_incomplete_type))
      continue;
    Type = Type.getNonReferenceType().getCanonicalType();
    if (Type->isArrayType()) {
      Diag(ELoc, diag::err_omp_clause_array_type_arg)
          << getOpenMPClauseName(OMPC_reduction);
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP [2.9.3.6, Restrictions, C/C++, p.3]
    //  A list item that appears in a reduction clause must not be
    //  const-qualified.
    if (Type.isConstant(Context)) {
      Diag(ELoc, diag::err_omp_const_variable)
          << getOpenMPClauseName(OMPC_reduction);
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP [2.9.3.6, Restrictions, C/C++, p.1]
    //  The type of a list item that appears in a reduction clause must be valid
    //  for the reduction operator. For max or min reduction in C/C++ must be an
    //  arithmetic type.
    if (((Op == OMPC_REDUCTION_min || Op == OMPC_REDUCTION_max) &&
         !Type->isArithmeticType() && !Type->isDependentType()) ||
        (!getLangOpts().CPlusPlus && !Type->isScalarType() &&
         !Type->isDependentType())) {
      Diag(ELoc, diag::err_omp_clause_not_arithmetic_type_arg)
          << getOpenMPClauseName(OMPC_reduction) << getLangOpts().CPlusPlus;
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct]
    //  Variables with the predetermined data-sharing attributes may not be
    //  listed in data-sharing attributes clauses, except for the cases
    //  listed below. For these exceptions only, listing a predetermined
    //  variable in a data-sharing attribute clause is allowed and overrides
    //  the variable's predetermined data-sharing attributes.
    // OpenMP [2.9.3.6, Restrictions, p.3]
    //  Any number of reduction clauses can be specified on the directive,
    //  but a list item can appear only once in the reduction clauses for that
    //  directive.
    DeclRefExpr *PrevRef;
    OpenMPClauseKind Kind = DSAStack->getTopDSA(VD, PrevRef);
    if (Kind == OMPC_reduction) {
      Diag(ELoc, diag::err_omp_once_referenced)
          << getOpenMPClauseName(OMPC_reduction);
      if (PrevRef) {
        Diag(PrevRef->getExprLoc(), diag::note_omp_referenced);
      }
    } else if (Kind != OMPC_unknown) {
      Diag(ELoc, diag::err_omp_wrong_dsa)
          << getOpenMPClauseName(Kind) << getOpenMPClauseName(OMPC_reduction);
      if (PrevRef) {
        Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
            << getOpenMPClauseName(Kind);
      } else {
        Diag(VD->getLocation(), diag::note_omp_predetermined_dsa)
            << getOpenMPClauseName(Kind);
      }
      continue;
    }

    // OpenMP [2.9.3.6, Restrictions, p.1]
    //  A list item that appears in a reduction clause of a worksharing
    //  construct must be shared in the parallel regions to which any of the
    //  worksharing regions arising from the worksharing construct bind.
    OpenMPDirectiveKind DKind;
    OpenMPDirectiveKind CurrDir = DSAStack->getCurrentDirective();
    Kind = DSAStack->getImplicitDSA(VD, DKind, PrevRef);
    if ((Kind != OMPC_shared && Kind != OMPC_unknown &&
         DKind != OMPD_unknown) &&
        (CurrDir == OMPD_for || CurrDir == OMPD_sections ||
         CurrDir == OMPD_for_simd)) {
      if (Kind == OMPC_unknown) {
        Diag(ELoc, diag::err_omp_required_access)
            << getOpenMPClauseName(OMPC_reduction)
            << getOpenMPClauseName(OMPC_shared);
      } else if (DKind == OMPD_unknown) {
        Diag(ELoc, diag::err_omp_wrong_dsa)
            << getOpenMPClauseName(Kind) << getOpenMPClauseName(OMPC_reduction);
      } else {
        Diag(ELoc, diag::err_omp_dsa_with_directives)
            << getOpenMPClauseName(Kind) << getOpenMPDirectiveName(DKind)
            << getOpenMPClauseName(OMPC_reduction)
            << getOpenMPDirectiveName(CurrDir);
      }
      if (PrevRef) {
        Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
            << getOpenMPClauseName(Kind);
      }
      continue;
    }

    QualType RedTy = DE->getType().getUnqualifiedType();
    OMPDeclareReductionDecl::ReductionData *DRRD =
        TryToFindDeclareReductionDecl(*this, SS, OpName, RedTy, Op);
    if (Op == OMPC_REDUCTION_custom && !DRRD) {
      RedDeclFilterCCC CCC(*this, RedTy);
      LookupResult Lookup(*this, OpName, LookupOMPDeclareReduction);
      if (DiagnoseEmptyLookup(getCurScope(), SS, Lookup, CCC))
        continue;
      DRRD = CCC.getFoundData();
      if (!DRRD)
        continue;
    }
    if (DRRD) {
      Op = OMPC_REDUCTION_custom;
      QualType PtrQTy = Context.getPointerType(DE->getType());
      TypeSourceInfo *TI =
          Context.getTrivialTypeSourceInfo(PtrQTy, SourceLocation());
      IdentifierInfo *Id1 = &Context.Idents.get(".ptr1.");
      VarDecl *Parameter1 = VarDecl::Create(
          Context, Context.getTranslationUnitDecl(), SourceLocation(),
          SourceLocation(), Id1, PtrQTy, TI, SC_Static);
      Parameter1->setImplicit();
      Parameter1->addAttr(new (Context)
                              UnusedAttr(SourceLocation(), Context, 0));
      IdentifierInfo *Id2 = &Context.Idents.get(".ptr2.");
      VarDecl *Parameter2 = VarDecl::Create(
          Context, Context.getTranslationUnitDecl(), SourceLocation(),
          SourceLocation(), Id2, PtrQTy, TI, SC_Static);
      Parameter2->setImplicit();
      Parameter2->addAttr(new (Context)
                              UnusedAttr(SourceLocation(), Context, 0));
      Context.getTranslationUnitDecl()->addHiddenDecl(Parameter1);
      Context.getTranslationUnitDecl()->addHiddenDecl(Parameter2);
      ExprResult PtrDE1 =
          BuildDeclRefExpr(Parameter1, PtrQTy, VK_LValue, SourceLocation());
      ExprResult PtrDE2 =
          BuildDeclRefExpr(Parameter2, PtrQTy, VK_LValue, SourceLocation());
      Expr *PtrDE1Expr = PtrDE1.get();
      Expr *PtrDE2Expr = PtrDE2.get();
      ExprResult DE1 = DefaultLvalueConversion(PtrDE1Expr);
      ExprResult DE2 = DefaultLvalueConversion(PtrDE2Expr);
      Expr *Args[] = {DE1.get(), DE2.get()};
      ExprResult Res =
          ActOnCallExpr(DSAStack->getCurScope(), DRRD->CombinerFunction, ELoc,
                        Args, SourceLocation());
      if (Res.isInvalid())
        continue;

      DefaultInits.push_back(DRRD->InitFunction);
      Vars.push_back(DE);
      OpExprs.push_back(Res.get());
      HelperParams1.push_back(PtrDE1Expr);
      HelperParams2.push_back(PtrDE2Expr);
    } else {
      if ((Op == OMPC_REDUCTION_bitor || Op == OMPC_REDUCTION_bitand ||
           Op == OMPC_REDUCTION_bitxor) &&
          Type->isFloatingType()) {
        Diag(ELoc, diag::err_omp_clause_floating_type_arg);
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
            << VD;
        continue;
      }
      QualType PtrQTy = Context.getPointerType(DE->getType());
      TypeSourceInfo *TI =
          Context.getTrivialTypeSourceInfo(PtrQTy, SourceLocation());
      IdentifierInfo *Id1 = &Context.Idents.get(".ptr1.");
      VarDecl *Parameter1 = VarDecl::Create(
          Context, Context.getTranslationUnitDecl(), SourceLocation(),
          SourceLocation(), Id1, PtrQTy, TI, SC_Static);
      Parameter1->setImplicit();
      Parameter1->addAttr(new (Context)
                              UnusedAttr(SourceLocation(), Context, 0));
      IdentifierInfo *Id2 = &Context.Idents.get(".ptr2.");
      VarDecl *Parameter2 = VarDecl::Create(
          Context, Context.getTranslationUnitDecl(), SourceLocation(),
          SourceLocation(), Id2, PtrQTy, TI, SC_Static);
      Parameter2->setImplicit();
      Parameter2->addAttr(new (Context)
                              UnusedAttr(SourceLocation(), Context, 0));
      Context.getTranslationUnitDecl()->addHiddenDecl(Parameter1);
      Context.getTranslationUnitDecl()->addHiddenDecl(Parameter2);
      ExprResult PtrDE1 =
          BuildDeclRefExpr(Parameter1, PtrQTy, VK_LValue, SourceLocation());
      ExprResult PtrDE2 =
          BuildDeclRefExpr(Parameter2, PtrQTy, VK_LValue, SourceLocation());
      Expr *PtrDE1Expr = PtrDE1.get();
      Expr *PtrDE2Expr = PtrDE2.get();
      ExprResult DE1 = DefaultLvalueConversion(PtrDE1Expr);
      ExprResult DE2 = DefaultLvalueConversion(PtrDE2Expr);
      DE1 = CreateBuiltinUnaryOp(ELoc, UO_Deref, DE1.get());
      DE2 = CreateBuiltinUnaryOp(ELoc, UO_Deref, DE2.get());
      if (NewOp == BO_SubAssign) {
        NewOp = BO_AddAssign;
      }
      ExprResult Res = BuildBinOp(DSAStack->getCurScope(), ELoc, NewOp,
                                  DE1.get(), DE2.get());
      if (Res.isInvalid())
        continue;
      CXXRecordDecl *RD = Type->getAsCXXRecordDecl();
      if (RD) {
        CXXConstructorDecl *CD = LookupDefaultConstructor(RD);
        PartialDiagnostic PD =
            PartialDiagnostic(PartialDiagnostic::NullDiagnostic());
        if (!CD ||
            CheckConstructorAccess(ELoc, CD,
                                   InitializedEntity::InitializeTemporary(Type),
                                   CD->getAccess(), PD) == AR_inaccessible ||
            CD->isDeleted()) {
          Diag(ELoc, diag::err_omp_required_method)
              << getOpenMPClauseName(OMPC_reduction) << 0;
          bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                        VarDecl::DeclarationOnly;
          Diag(VD->getLocation(),
               IsDecl ? diag::note_previous_decl : diag::note_defined_here)
              << VD;
          Diag(RD->getLocation(), diag::note_previous_decl) << RD;
          continue;
        }
        MarkFunctionReferenced(ELoc, CD);
        DiagnoseUseOfDecl(CD, ELoc);
        CXXDestructorDecl *DD = RD->getDestructor();
        if (DD && (CheckDestructorAccess(ELoc, DD, PD) == AR_inaccessible ||
                   DD->isDeleted())) {
          Diag(ELoc, diag::err_omp_required_method)
              << getOpenMPClauseName(OMPC_reduction) << 4;
          bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                        VarDecl::DeclarationOnly;
          Diag(VD->getLocation(),
               IsDecl ? diag::note_previous_decl : diag::note_defined_here)
              << VD;
          Diag(RD->getLocation(), diag::note_previous_decl) << RD;
          continue;
        } else if (DD) {
          MarkFunctionReferenced(ELoc, DD);
          DiagnoseUseOfDecl(DD, ELoc);
        }
      }
      if (NewOp == BO_LAnd || NewOp == BO_LOr) {
        Res = BuildBinOp(DSAStack->getCurScope(), ELoc, BO_Assign, DE1.get(),
                         Res.get());
      } else if (NewOp == BO_LT || NewOp == BO_GT) {
        Res = ActOnConditionalOp(ELoc, ELoc, Res.get(), DE1.get(), DE2.get());
        if (Res.isInvalid())
          continue;
        Res = BuildBinOp(DSAStack->getCurScope(), ELoc, BO_Assign, DE1.get(),
                         Res.get());
      }
      if (Res.isInvalid())
        continue;
      Res = IgnoredValueConversions(Res.get());

      Type = Type.getUnqualifiedType();
      if (RD) {
        IdentifierInfo *Id = &Context.Idents.get(".firstprivate.");
        TypeSourceInfo *TI1 = Context.getTrivialTypeSourceInfo(Type, ELoc);
        VarDecl *PseudoVar = VarDecl::Create(
            Context, Context.getTranslationUnitDecl(), SourceLocation(),
            SourceLocation(), Id, Type, TI1, SC_Static);
        PseudoVar->setImplicit();
        PseudoVar->addAttr(new (Context)
                               UnusedAttr(SourceLocation(), Context, 0));
        InitializedEntity Entity =
            InitializedEntity::InitializeVariable(PseudoVar);
        InitializationKind InitKind = InitializationKind::CreateDefault(ELoc);
        InitializationSequence InitSeq(*this, Entity, InitKind, MultiExprArg());
        ExprResult CPRes =
            InitSeq.Perform(*this, Entity, InitKind, MultiExprArg());
        if (CPRes.isInvalid())
          continue;
        DefaultInits.push_back(ActOnFinishFullExpr(CPRes.get()).get());
      } else {
        DefaultInits.push_back(0);
      }
      Vars.push_back(DE);
      OpExprs.push_back(ActOnFinishFullExpr(Res.get()).get());
      HelperParams1.push_back(PtrDE1Expr);
      HelperParams2.push_back(PtrDE2Expr);
    }
    DSAStack->addDSA(VD, DE, OMPC_reduction);
  }

  if (Vars.empty())
    return 0;

  return OMPReductionClause::Create(
      Context, StartLoc, EndLoc, Vars, OpExprs, HelperParams1, HelperParams2,
      DefaultInits, Op, SS.getWithLocInContext(Context), OpName);
}

OMPClause *Sema::ActOnOpenMPScanClause(ArrayRef<Expr *> VarList,
                                       SourceLocation StartLoc,
                                       SourceLocation EndLoc,
                                       OpenMPScanClauseOperator Op,
                                       CXXScopeSpec &SS,
                                       DeclarationNameInfo OpName) {
    BinaryOperatorKind NewOp = BO_Assign;
    switch (Op) {
        case OMPC_SCAN_add:
            NewOp = BO_AddAssign;
            break;
        case OMPC_SCAN_mult:
            NewOp = BO_MulAssign;
            break;
        case OMPC_SCAN_sub:
            NewOp = BO_SubAssign;
            break;
        case OMPC_SCAN_bitand:
            NewOp = BO_AndAssign;
            break;
        case OMPC_SCAN_bitor:
            NewOp = BO_OrAssign;
            break;
        case OMPC_SCAN_bitxor:
            NewOp = BO_XorAssign;
            break;
        case OMPC_SCAN_and:
            NewOp = BO_LAnd;
            break;
        case OMPC_SCAN_or:
            NewOp = BO_LOr;
            break;
        case OMPC_SCAN_min:
            NewOp = BO_LT;
            break;
        case OMPC_SCAN_max:
            NewOp = BO_GT;
            break;
        default:
            break;
    }
    SmallVector<Expr *, 4> Vars;
    SmallVector<Expr *, 4> DefaultInits;
    SmallVector<Expr *, 4> OpExprs;
    SmallVector<Expr *, 4> HelperParams1;
    SmallVector<Expr *, 4> HelperParams2;
    for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
         I != E; ++I) {
        assert(*I && "Null expr in omp scan");
//        if (isa<DependentScopeDeclRefExpr>(*I)) {
        // It will be analyzed later.
        Vars.push_back(*I);
        DefaultInits.push_back(0);
        OpExprs.push_back(0);
        HelperParams1.push_back(0);
        HelperParams2.push_back(0);
        continue;
//        }

/*
        SourceLocation ELoc = (*I)->getExprLoc();
        // OpenMP [2.1, C/C++]
        //  A list item is a variable name.
        // OpenMP  [2.9.3.3, Restrictions, p.1]
        //  A variable that is part of another variable (as an array or
        //  structure element) cannot appear in a private clause.
        DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I);
        if (!DE || !isa<VarDecl>(DE->getDecl())) {
            Diag(ELoc, diag::err_omp_expected_var_name) << (*I)->getSourceRange();
            continue;
        }
        Decl *D = DE->getDecl();
        VarDecl *VD = cast<VarDecl>(D);

        QualType Type = VD->getType();
        if (Type->isDependentType() || Type->isInstantiationDependentType()) {
            // It will be analyzed later.
            Vars.push_back(*I);
            DefaultInits.push_back(0);
            OpExprs.push_back(0);
            HelperParams1.push_back(0);
            HelperParams2.push_back(0);
            continue;
        }

        // OpenMP [2.9.3.6, Restrictions, C/C++, p.4]
        //  If a list-item is a reference type then it must bind to the same object
        //  for all threads of the team.

    if (Type.getCanonicalType()->isReferenceType() && VD->hasInit()) {
      DSARefChecker Check(DSAStack);
      if (Check.Visit(VD->getInit())) {
        Diag(ELoc, diag::err_omp_reduction_ref_type_arg)
                << getOpenMPClauseName(OMPC_scan);
        bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                      VarDecl::DeclarationOnly;
        Diag(VD->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
                << VD;
        continue;
      }
    }

        // OpenMP [2.9.3.6, Restrictions, C/C++, p.2]
        if (RequireCompleteType(ELoc, Type,
                                diag::err_omp_reduction_incomplete_type))
            continue;

        Type = Type.getNonReferenceType().getCanonicalType();

        // OpenMP [2.9.3.6, Restrictions, C/C++, p.3]
        //  A list item that appears in a scan clause must not be
        //  const-qualified.
        if (Type.isConstant(Context)) {
            Diag(ELoc, diag::err_omp_const_variable)
                    << getOpenMPClauseName(OMPC_scan);
            bool IsDecl =
                    VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
            Diag(VD->getLocation(),
                 IsDecl ? diag::note_previous_decl : diag::note_defined_here)
                    << VD;
            continue;
        }

        // OpenMP [2.9.3.6, Restrictions, C/C++, p.1]
        //  The type of a list item that appears in a scan clause must be valid
        //  for the scan operator. For max or min scan in C/C++ must be an
        //  arithmetic type.

    if ((Op == OMPC_SCAN_min || Op == OMPC_SCAN_max) &&
         !Type->isArithmeticType() && !Type->isDependentType()) {
      Diag(ELoc, diag::err_omp_clause_not_arithmetic_type_arg)
              << getOpenMPClauseName(OMPC_scan);
      bool IsDecl =
              VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
              << VD;
      continue;
    }

        // OpenMP [2.9.1.1, Data-sharing Attribute Rules for Variables Referenced
        // in a Construct]
        //  Variables with the predetermined data-sharing attributes may not be
        //  listed in data-sharing attributes clauses, except for the cases
        //  listed below. For these exceptions only, listing a predetermined
        //  variable in a data-sharing attribute clause is allowed and overrides
        //  the variable's predetermined data-sharing attributes.
        // OpenMP [2.9.3.6, Restrictions, p.3]
        //  Any number of scan clauses can be specified on the directive,
        //  but a list item can appear only once in the scan clauses for that
        //  directive.
        DeclRefExpr *PrevRef;
        OpenMPClauseKind Kind = DSAStack->getTopDSA(VD, PrevRef);
        if (Kind == OMPC_scan) {
            Diag(ELoc, diag::err_omp_once_referenced)
                    << getOpenMPClauseName(OMPC_scan);
            if (PrevRef) {
                Diag(PrevRef->getExprLoc(), diag::note_omp_referenced);
            }
        } else if (Kind != OMPC_unknown) {
            Diag(ELoc, diag::err_omp_wrong_dsa)
                    << getOpenMPClauseName(Kind) << getOpenMPClauseName(OMPC_scan);
            if (PrevRef) {
                Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
                        << getOpenMPClauseName(Kind);
            } else {
                Diag(VD->getLocation(), diag::note_omp_predetermined_dsa)
                        << getOpenMPClauseName(Kind);
            }
            continue;
        }

        // OpenMP [2.9.3.6, Restrictions, p.1]
        //  A list item that appears in a scan clause of a worksharing
        //  construct must be shared in the parallel regions to which any of the
        //  worksharing regions arising from the worksharing construct bind.
        OpenMPDirectiveKind DKind;
        OpenMPDirectiveKind CurrDir = DSAStack->getCurrentDirective();
        Kind = DSAStack->getImplicitDSA(VD, DKind, PrevRef);
        if ((Kind != OMPC_shared && Kind != OMPC_unknown &&
             DKind != OMPD_unknown) &&
            (CurrDir == OMPD_for || CurrDir == OMPD_sections ||
             CurrDir == OMPD_for_simd)) {
            if (Kind == OMPC_unknown) {
                Diag(ELoc, diag::err_omp_required_access)
                        << getOpenMPClauseName(OMPC_scan)
                        << getOpenMPClauseName(OMPC_shared);
            } else if (DKind == OMPD_unknown) {
                Diag(ELoc, diag::err_omp_wrong_dsa)
                        << getOpenMPClauseName(Kind) << getOpenMPClauseName(OMPC_scan);
            } else {
                Diag(ELoc, diag::err_omp_dsa_with_directives)
                        << getOpenMPClauseName(Kind) << getOpenMPDirectiveName(DKind)
                        << getOpenMPClauseName(OMPC_scan)
                        << getOpenMPDirectiveName(CurrDir);
            }
            if (PrevRef) {
                Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
                        << getOpenMPClauseName(Kind);
            }
            continue;
        }

        if ((Op == OMPC_SCAN_bitor || Op == OMPC_SCAN_bitand ||
             Op == OMPC_SCAN_bitxor) &&
            Type->isFloatingType()) {
            Diag(ELoc, diag::err_omp_clause_floating_type_arg);
            bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                          VarDecl::DeclarationOnly;
            Diag(VD->getLocation(),
                 IsDecl ? diag::note_previous_decl : diag::note_defined_here)
                    << VD;
            continue;
        }

        QualType PtrQTy = Context.getPointerType(DE->getType());
        TypeSourceInfo *TI =
                Context.getTrivialTypeSourceInfo(PtrQTy, SourceLocation());
        IdentifierInfo *Id1 = &Context.Idents.get(".ptr1.");
        VarDecl *Parameter1 = VarDecl::Create(
                Context, Context.getTranslationUnitDecl(), SourceLocation(),
                SourceLocation(), Id1, PtrQTy, TI, SC_Static);
        Parameter1->setImplicit();
        Parameter1->addAttr(new(Context)
                                    UnusedAttr(SourceLocation(), Context, 0));
        IdentifierInfo *Id2 = &Context.Idents.get(".ptr2.");
        VarDecl *Parameter2 = VarDecl::Create(
                Context, Context.getTranslationUnitDecl(), SourceLocation(),
                SourceLocation(), Id2, PtrQTy, TI, SC_Static);
        Parameter2->setImplicit();
        Parameter2->addAttr(new(Context)
                                    UnusedAttr(SourceLocation(), Context, 0));
        Context.getTranslationUnitDecl()->addHiddenDecl(Parameter1);
        Context.getTranslationUnitDecl()->addHiddenDecl(Parameter2);
        ExprResult PtrDE1 =
                BuildDeclRefExpr(Parameter1, PtrQTy, VK_LValue, SourceLocation());
        ExprResult PtrDE2 =
                BuildDeclRefExpr(Parameter2, PtrQTy, VK_LValue, SourceLocation());
        Expr *PtrDE1Expr = PtrDE1.get();
        Expr *PtrDE2Expr = PtrDE2.get();
        ExprResult DE1 = DefaultLvalueConversion(PtrDE1Expr);
        ExprResult DE2 = DefaultLvalueConversion(PtrDE2Expr);
        DE1 = CreateBuiltinUnaryOp(ELoc, UO_Deref, DE1.get());
        DE2 = CreateBuiltinUnaryOp(ELoc, UO_Deref, DE2.get());
        if (NewOp == BO_SubAssign) {
            NewOp = BO_AddAssign;
        }
        ExprResult Res;

        Res = BuildBinOp(DSAStack->getCurScope(), ELoc, NewOp,
                                    DE1.get(), DE2.get());
        llvm::errs() << "Act2\n";
        if (Res.isInvalid())
            continue;

        CXXRecordDecl *RD = Type->getAsCXXRecordDecl();
        if (RD) {
            CXXConstructorDecl *CD = LookupDefaultConstructor(RD);
            PartialDiagnostic PD =
                    PartialDiagnostic(PartialDiagnostic::NullDiagnostic());
            if (!CD ||
                CheckConstructorAccess(ELoc, CD,
                                       InitializedEntity::InitializeTemporary(Type),
                                       CD->getAccess(), PD) == AR_inaccessible ||
                CD->isDeleted()) {
                Diag(ELoc, diag::err_omp_required_method)
                        << getOpenMPClauseName(OMPC_scan) << 0;
                bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                              VarDecl::DeclarationOnly;
                Diag(VD->getLocation(),
                     IsDecl ? diag::note_previous_decl : diag::note_defined_here)
                        << VD;
                Diag(RD->getLocation(), diag::note_previous_decl) << RD;
                continue;
            }
            MarkFunctionReferenced(ELoc, CD);
            DiagnoseUseOfDecl(CD, ELoc);
            CXXDestructorDecl *DD = RD->getDestructor();
            if (DD && (CheckDestructorAccess(ELoc, DD, PD) == AR_inaccessible ||
                       DD->isDeleted())) {
                Diag(ELoc, diag::err_omp_required_method)
                        << getOpenMPClauseName(OMPC_scan) << 4;
                bool IsDecl = VD->isThisDeclarationADefinition(Context) ==
                              VarDecl::DeclarationOnly;
                Diag(VD->getLocation(),
                     IsDecl ? diag::note_previous_decl : diag::note_defined_here)
                        << VD;
                Diag(RD->getLocation(), diag::note_previous_decl) << RD;
                continue;
                llvm::errs() << "Act14\n";

            } else if (DD) {
                MarkFunctionReferenced(ELoc, DD);
                DiagnoseUseOfDecl(DD, ELoc);
            }
        }
        llvm::errs() << "Act4\n";
        if (NewOp == BO_LAnd || NewOp == BO_LOr) {
            Res = BuildBinOp(DSAStack->getCurScope(), ELoc, BO_Assign, DE1.get(),
                             Res.get());
            llvm::errs() << "Act5\n";
        } else if (NewOp == BO_LT || NewOp == BO_GT) {
            Res = ActOnConditionalOp(ELoc, ELoc, Res.get(), DE1.get(), DE2.get());
            llvm::errs() << "Act6\n";
            if (Res.isInvalid())
                continue;
            Res = BuildBinOp(DSAStack->getCurScope(), ELoc, BO_Assign, DE1.get(),
                             Res.get());
            llvm::errs() << "Act7\n";
        }
        if (Res.isInvalid())
            continue;
        Res = IgnoredValueConversions(Res.get());
        llvm::errs() << "Act8\n";

        Type = Type.getUnqualifiedType();
        if (RD) {
            IdentifierInfo *Id = &Context.Idents.get(".firstprivate.");
            TypeSourceInfo *TI1 = Context.getTrivialTypeSourceInfo(Type, ELoc);
            VarDecl *PseudoVar = VarDecl::Create(
                    Context, Context.getTranslationUnitDecl(), SourceLocation(),
                    SourceLocation(), Id, Type, TI1, SC_Static);
            PseudoVar->setImplicit();
            PseudoVar->addAttr(new(Context)
                                       UnusedAttr(SourceLocation(), Context, 0));
            InitializedEntity Entity =
                    InitializedEntity::InitializeVariable(PseudoVar);
            InitializationKind InitKind = InitializationKind::CreateDefault(ELoc);
            InitializationSequence InitSeq(*this, Entity, InitKind, MultiExprArg());
            ExprResult CPRes =
                    InitSeq.Perform(*this, Entity, InitKind, MultiExprArg());
            if (CPRes.isInvalid())
                continue;
            DefaultInits.push_back(ActOnFinishFullExpr(CPRes.get()).get());
            llvm::errs() << "Act9\n";
        } else {
            DefaultInits.push_back(0);
        }
        Vars.push_back(DE);
        OpExprs.push_back(ActOnFinishFullExpr(Res.get()).get());
        HelperParams1.push_back(PtrDE1Expr);
        HelperParams2.push_back(PtrDE2Expr);
        DSAStack->addDSA(VD, DE, OMPC_scan);
        llvm::errs() << "Act10\n";
*/
    }

    if (Vars.empty())
        return 0;

    return OMPScanClause::Create(
            Context, StartLoc, EndLoc, Vars, OpExprs, HelperParams1, HelperParams2,
            DefaultInits, Op, SS.getWithLocInContext(Context), OpName);
}

namespace {
class ArrayItemChecker : public StmtVisitor<ArrayItemChecker, bool> {
private:
  Sema &SemaRef;
  Expr *End;

public:
  bool VisitDeclRefExpr(DeclRefExpr *E) {
    if (isa<VarDecl>(E->getDecl())) {
      End = E;
      return false;
    }
    return true;
  }
  bool VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
    Expr *Base = E->getBase()->IgnoreImplicit();
    bool Result = Visit(Base);
    if (!End)
      return Result;
    if (CEANIndexExpr *CIE = dyn_cast_or_null<CEANIndexExpr>(E->getIdx())) {
      llvm::APSInt Value;
      // OpenMP [2.11.1.1, Restrictions]
      //  List items used in dependent clauses cannot be zero-length array
      //  sections.
      if (CIE->getLength()->EvaluateAsInt(Value, SemaRef.getASTContext()) &&
          ((Value.isSigned() && Value.isNegative()) || !Value)) {
        SemaRef.Diag(CIE->getExprLoc(),
                     diag::err_omp_array_section_length_not_greater_zero)
            << CIE->getSourceRange();
        End = 0;
        return Result;
      }
      ExprResult Idx = SemaRef.CreateBuiltinBinOp(
          E->getExprLoc(), BO_Add, CIE->getLowerBound(), CIE->getLength());
      if (Idx.isInvalid()) {
        End = 0;
        return Result;
      }
      Idx = SemaRef.CreateBuiltinBinOp(
          E->getExprLoc(), BO_Sub, Idx.get(),
          SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get());
      if (Idx.isInvalid()) {
        End = 0;
        return Result;
      }
      End = SemaRef.CreateBuiltinArraySubscriptExpr(
                        End, E->getExprLoc(), Idx.get(), E->getExprLoc()).get();
      CIE->setIndexExpr(CIE->getLowerBound());
    } else if (End != Base) {
      End = SemaRef.CreateBuiltinArraySubscriptExpr(End, E->getExprLoc(),
                                                    E->getIdx(),
                                                    E->getExprLoc()).get();
    } else {
      End = E;
    }
    return Result;
  }
  bool VisitStmt(Stmt *S) { return true; }

  ArrayItemChecker(Sema &SemaRef) : SemaRef(SemaRef), End(0) {}

  std::pair<Expr *, Expr *> CalculateSize(Expr *Begin) {
    if (!Begin)
      return std::make_pair<Expr *, Expr *>(0, 0);
    QualType CharPtrTy =
        SemaRef.getASTContext().getPointerType(SemaRef.getASTContext().CharTy);
    if (!End || Begin == End) {
      Expr *Size;
      {
        EnterExpressionEvaluationContext Unevaluated(
            SemaRef, Sema::Unevaluated, Sema::ReuseLambdaContextDecl);

        Size = SemaRef.CreateUnaryExprOrTypeTraitExpr(Begin, SourceLocation(),
                                                      UETT_SizeOf).get();
      }
      ExprResult AddrBegin =
          SemaRef.CreateBuiltinUnaryOp(Begin->getExprLoc(), UO_AddrOf, Begin);
      if (AddrBegin.isInvalid())
        return std::make_pair<Expr *, Expr *>(0, 0);
      AddrBegin =
          SemaRef.ImpCastExprToType(AddrBegin.get(), CharPtrTy, CK_BitCast);
      if (AddrBegin.isInvalid())
        return std::make_pair<Expr *, Expr *>(0, 0);
      Expr *AB = SemaRef.DefaultLvalueConversion(AddrBegin.get()).get();
      return std::make_pair(AB, Size);
    }

    ExprResult AddrEnd =
        SemaRef.CreateBuiltinUnaryOp(End->getExprLoc(), UO_AddrOf, End);
    if (AddrEnd.isInvalid())
      return std::make_pair<Expr *, Expr *>(0, 0);
    AddrEnd = SemaRef.CreateBuiltinBinOp(
        End->getExprLoc(), BO_Add, AddrEnd.get(),
        SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get());
    if (AddrEnd.isInvalid())
      return std::make_pair<Expr *, Expr *>(0, 0);
    ExprResult AddrBegin =
        SemaRef.CreateBuiltinUnaryOp(Begin->getExprLoc(), UO_AddrOf, Begin);
    if (AddrBegin.isInvalid())
      return std::make_pair<Expr *, Expr *>(0, 0);
    Expr *AE = SemaRef.DefaultLvalueConversion(AddrEnd.get()).get();
    Expr *AB = SemaRef.DefaultLvalueConversion(AddrBegin.get()).get();
    return std::make_pair(AB, AE);
  }
};
}

OMPClause *Sema::ActOnOpenMPDependClause(ArrayRef<Expr *> VarList,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc,
                                         OpenMPDependClauseType Ty,
                                         SourceLocation TyLoc) {
  SmallVector<Expr *, 4> Vars;
  SmallVector<Expr *, 4> Begins;
  SmallVector<Expr *, 4> SizeInBytes;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "Null expr in omp depend");
    if ((*I)->isValueDependent() || (*I)->isTypeDependent() ||
        (*I)->isInstantiationDependent()) {
      // It will be analyzed later.
      Vars.push_back(*I);
      Begins.push_back(0);
      SizeInBytes.push_back(0);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();

    // OpenMP [2.11.1.1, Restrictions]
    //  A variable that is part of another variable (such as field of a
    //  structure) but is not an array element or an array section cannot appear
    //  in a depend clause.
    // OpenMP  [2.9.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    Expr *VE = (*I)->IgnoreParenLValueCasts();

    if (VE->isRValue()) {
      Diag(ELoc, diag::err_omp_depend_arg_not_lvalue) << (*I)->getSourceRange();
      continue;
    }

    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(VE);
    ArraySubscriptExpr *ASE = dyn_cast_or_null<ArraySubscriptExpr>(VE);
    ArrayItemChecker Checker(*this);
    if ((!DE || !isa<VarDecl>(DE->getDecl())) && (!ASE || Checker.Visit(ASE))) {
      Diag(ELoc, diag::err_omp_expected_var_name_or_array_item)
          << (*I)->getSourceRange();
      continue;
    }

    std::pair<Expr *, Expr *> BeginSize = Checker.CalculateSize(VE);
    if (!BeginSize.first || !BeginSize.second)
      continue;

    Vars.push_back(VE);
    Begins.push_back(BeginSize.first);
    SizeInBytes.push_back(BeginSize.second);
  }

  if (Vars.empty() || Vars.size() != Begins.size())
    return 0;
  return OMPDependClause::Create(Context, StartLoc, EndLoc, Vars, Begins,
                                 SizeInBytes, Ty, TyLoc);
}

namespace {
class MapArrayItemChecker : public StmtVisitor<MapArrayItemChecker, bool> {
private:
  Sema &SemaRef;
  Expr *CopyBegin;
  Expr *CopyEnd;
  Expr *WholeBegin;
  Expr *WholeEnd;
  VarDecl *VD;
  DeclRefExpr *DRE;
  bool IsCEAN;

  std::pair<Expr *, Expr *> CalculateSize(Expr *Begin, Expr *End) {
    if (!Begin || !End)
      return std::make_pair<Expr *, Expr *>(0, 0);
    QualType CharPtrTy =
        SemaRef.getASTContext().getPointerType(SemaRef.getASTContext().CharTy);
    if (Begin == End) {
      Expr *Size;
      {
        EnterExpressionEvaluationContext Unevaluated(
            SemaRef, Sema::Unevaluated, Sema::ReuseLambdaContextDecl);

        Size = SemaRef.CreateUnaryExprOrTypeTraitExpr(Begin, SourceLocation(),
                                                      UETT_SizeOf).get();
      }
      ExprResult AddrBegin =
          SemaRef.CreateBuiltinUnaryOp(Begin->getExprLoc(), UO_AddrOf, Begin);
      if (AddrBegin.isInvalid())
        return std::make_pair<Expr *, Expr *>(0, 0);
      AddrBegin =
          SemaRef.ImpCastExprToType(AddrBegin.get(), CharPtrTy, CK_BitCast);
      if (AddrBegin.isInvalid())
        return std::make_pair<Expr *, Expr *>(0, 0);
      Expr *AB = SemaRef.DefaultLvalueConversion(AddrBegin.get()).get();
      return std::make_pair(AB, Size);
    }

    ExprResult AddrEnd =
        SemaRef.CreateBuiltinUnaryOp(End->getExprLoc(), UO_AddrOf, End);
    if (AddrEnd.isInvalid())
      return std::make_pair<Expr *, Expr *>(0, 0);
    AddrEnd = SemaRef.CreateBuiltinBinOp(
        End->getExprLoc(), BO_Add, AddrEnd.get(),
        SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get());
    if (AddrEnd.isInvalid())
      return std::make_pair<Expr *, Expr *>(0, 0);
    ExprResult AddrBegin =
        SemaRef.CreateBuiltinUnaryOp(Begin->getExprLoc(), UO_AddrOf, Begin);
    if (AddrBegin.isInvalid())
      return std::make_pair<Expr *, Expr *>(0, 0);
    Expr *AE = SemaRef.DefaultLvalueConversion(AddrEnd.get()).get();
    Expr *AB = SemaRef.DefaultLvalueConversion(AddrBegin.get()).get();
    return std::make_pair(AB, AE);
  }

public:
  bool VisitDeclRefExpr(DeclRefExpr *E) {
    if (isa<VarDecl>(E->getDecl())) {
      CopyBegin = CopyEnd = E;
      WholeBegin = WholeEnd = E;
      VD = cast<VarDecl>(E->getDecl());
      DRE = E;
      return false;
    }
    return true;
  }
  bool VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
    Expr *Base = E->getBase()->IgnoreImplicit();
    bool Result = Visit(Base);
    if (!CopyEnd || !CopyBegin)
      return Result;
    if (!WholeEnd || !WholeBegin)
      return Result;
    WholeBegin =
        SemaRef.CreateBuiltinArraySubscriptExpr(
                    WholeBegin, E->getExprLoc(),
                    SemaRef.ActOnIntegerConstant(SourceLocation(), 0).get(),
                    E->getExprLoc()).get();
    QualType QTy = Base->getType();
    Expr *Idx = 0;
    if (const ArrayType *AT = QTy->getAsArrayTypeUnsafe()) {
      if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(AT)) {
        Idx = SemaRef.ActOnIntegerConstant(
                          SourceLocation(),
                          (CAT->getSize() - 1).getLimitedValue()).get();
      } else if (const VariableArrayType *VAT =
                     dyn_cast<VariableArrayType>(AT)) {
        Idx = VAT->getSizeExpr();
        Idx = SemaRef.CreateBuiltinBinOp(E->getExprLoc(), BO_Sub, Idx,
                                         SemaRef.ActOnIntegerConstant(
                                                     SourceLocation(), 1).get())
                  .get();
      } else if (const DependentSizedArrayType *DSAT =
                     dyn_cast<DependentSizedArrayType>(AT)) {
        Idx = DSAT->getSizeExpr();
        Idx = SemaRef.CreateBuiltinBinOp(E->getExprLoc(), BO_Sub, Idx,
                                         SemaRef.ActOnIntegerConstant(
                                                     SourceLocation(), 1).get())
                  .get();
      }
    }
    Expr *LastIdx = 0;
    if (CEANIndexExpr *CIE = dyn_cast_or_null<CEANIndexExpr>(E->getIdx())) {
      IsCEAN = true;
      LastIdx = SemaRef.CreateBuiltinBinOp(E->getExprLoc(), BO_Add,
                                           CIE->getLowerBound(),
                                           CIE->getLength()).get();
      if (LastIdx == 0) {
        CopyBegin = CopyEnd = 0;
        WholeBegin = WholeEnd = 0;
        return Result;
      }
      LastIdx = SemaRef.CreateBuiltinBinOp(
                            E->getExprLoc(), BO_Sub, LastIdx,
                            SemaRef.ActOnIntegerConstant(SourceLocation(), 1)
                                .get()).get();
      CopyBegin = SemaRef.CreateBuiltinArraySubscriptExpr(
                              CopyBegin, E->getExprLoc(), CIE->getLowerBound(),
                              E->getExprLoc()).get();
    } else {
      LastIdx = E->getIdx();
      CopyBegin = SemaRef.CreateBuiltinArraySubscriptExpr(
                              CopyBegin, E->getExprLoc(), LastIdx,
                              E->getExprLoc()).get();
    }
    CopyEnd =
        SemaRef.CreateBuiltinArraySubscriptExpr(CopyEnd, E->getExprLoc(),
                                                LastIdx, E->getExprLoc()).get();
    if (Idx == 0) {
      Idx = LastIdx;
    }
    if (Idx == 0) {
      CopyBegin = CopyEnd = 0;
      WholeBegin = WholeEnd = 0;
      return Result;
    }
    WholeEnd =
        SemaRef.CreateBuiltinArraySubscriptExpr(WholeEnd, E->getExprLoc(), Idx,
                                                E->getExprLoc()).get();
    return Result;
  }
  bool VisitStmt(Stmt *S) { return true; }

  MapArrayItemChecker(Sema &SemaRef)
      : SemaRef(SemaRef), CopyBegin(0), CopyEnd(0), WholeBegin(0), WholeEnd(0),
        VD(0), DRE(0), IsCEAN(false) {}

  VarDecl *getBaseDecl() { return VD; }
  DeclRefExpr *getDeclRefExprForBaseDecl() { return DRE; }
  bool IsCEANExpr() const { return IsCEAN; }

  std::pair<Expr *, Expr *> CalculateCopySize() {
    return CalculateSize(CopyBegin, CopyEnd);
  }
  std::pair<Expr *, Expr *> CalculateWholeSize() {
    return CalculateSize(WholeBegin, WholeEnd);
  }
};
}

OMPClause *Sema::ActOnOpenMPMapClause(ArrayRef<Expr *> VarList,
                                      SourceLocation StartLoc,
                                      SourceLocation EndLoc,
                                      OpenMPMapClauseKind Kind,
                                      SourceLocation KindLoc) {
  SmallVector<Expr *, 4> Vars;
  SmallVector<Expr *, 4> WholeBegins;
  SmallVector<Expr *, 4> WholeEnds;
  SmallVector<Expr *, 4> CopyBegins;
  SmallVector<Expr *, 4> CopyEnds;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "Null expr in omp map");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      WholeBegins.push_back(0);
      WholeEnds.push_back(0);
      CopyBegins.push_back(0);
      CopyEnds.push_back(0);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();

    // OpenMP [2.14.5, Restrictions]
    //  A variable that is part of another variable (such as field of a
    //  structure) but is not an array element or an array section cannot appear
    //  in a map clause.
    Expr *VE = (*I)->IgnoreParenLValueCasts();

    if (VE->isValueDependent() || VE->isTypeDependent() ||
        VE->isInstantiationDependent() ||
        VE->containsUnexpandedParameterPack()) {
      // It will be analyzed later.
      Vars.push_back(*I);
      WholeBegins.push_back(0);
      WholeEnds.push_back(0);
      CopyBegins.push_back(0);
      CopyEnds.push_back(0);
      continue;
    }

    MapArrayItemChecker Checker(*this);
    VarDecl *VD = 0;
    DeclRefExpr *DE = 0;
    if (Checker.Visit(VE) || !(VD = Checker.getBaseDecl()) ||
        !(DE = Checker.getDeclRefExprForBaseDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name_or_array_item)
          << (*I)->getSourceRange();
      continue;
    }

    // OpenMP [2.14.5, Restrictions, p.8]
    // threadprivate variables cannot appear in a map clause.
    DeclRefExpr *DRE = 0;
    if (DSAStack->IsThreadprivate(VD, DRE)) {
      SourceLocation Loc = DRE ? DRE->getLocation() : VD->getLocation();
      Diag(Loc, diag::err_omp_threadprivate_in_target);
      Diag(DE->getLocStart(), diag::note_used_here) << DE->getSourceRange();
      continue;
    }

    // OpenMP [2.14.5, map Clause]
    //  If a corresponding list item of the original list item is in the
    //  enclosing device data environment, the new device data environment uses
    //  the corresponding list item from the enclosing device data environment.
    //  No additional storage is allocated in the new device data environment
    //  and neither initialization nor assignment is performed, regardless of
    //  the map-type that is specified.
    if (DSAStack->isDeclareTargetDecl(VD)) {
      // Use original variable.
      continue;
    }
    // OpenMP [2.14.5, Restrictions, p.2]
    //  At most one list item can be an array item derived from a given variable
    //  in map clauses of the same construct.
    // OpenMP [2.14.5, Restrictions, p.3]
    //  List items of map clauses in the same construct must not share original
    //  storage.
    // OpenMP [2.14.5, Restrictions, C/C++, p.2]
    //  A variable for which the type is pointer, reference to array, or
    //  reference to pointer and an array section derived from that variable
    //  must not appear as list items of map clauses of the same construct.
    DSAStackTy::MapInfo MI = DSAStack->IsMappedInCurrentRegion(VD);
    if (MI.RefExpr) {
      Diag(DE->getExprLoc(), diag::err_omp_map_shared_storage)
          << DE->getSourceRange();
      Diag(MI.RefExpr->getExprLoc(), diag::note_used_here)
          << MI.RefExpr->getSourceRange();
      continue;
    }

    // OpenMP [2.14.5, Restrictions, C/C++, p.3,4]
    //  A variable for which the type is pointer, reference to array, or
    //  reference to pointer must not appear as a list item if the enclosing
    //  device data environment already contains an array section derived from
    //  that variable.
    //  An array section derived from a variable for which the type is pointer,
    //  reference to array, or reference to pointer must not appear as a list
    //  item if the enclosing device data environment already contains that
    //  variable.
    QualType Type = VD->getType();
    MI = DSAStack->getMapInfoForVar(VD);
    if (MI.RefExpr && (isa<DeclRefExpr>(MI.RefExpr->IgnoreParenLValueCasts()) !=
                       isa<DeclRefExpr>(VE)) &&
        (MI.IsCEAN || Checker.IsCEANExpr()) &&
        (Type->isPointerType() || Type->isReferenceType())) {
      Diag(DE->getExprLoc(), diag::err_omp_map_shared_storage)
          << DE->getSourceRange();
      Diag(MI.RefExpr->getExprLoc(), diag::note_used_here)
          << MI.RefExpr->getSourceRange();
      continue;
    }

    // OpenMP [2.14.5, Restrictions, C/C++, p.7]
    //  A list item must have a mappable type.
    if (!CheckTypeMappable(VE->getExprLoc(), VE->getSourceRange(), *this,
                           DSAStack, Type)) {
      continue;
    }

    std::pair<Expr *, Expr *> WholeSize = Checker.CalculateWholeSize();
    if (!WholeSize.first || !WholeSize.second) {
      continue;
    }
    std::pair<Expr *, Expr *> CopySize = Checker.CalculateCopySize();
    if (!CopySize.first || !CopySize.second) {
      continue;
    }

    Vars.push_back(*I);
    WholeBegins.push_back(WholeSize.first);
    WholeEnds.push_back(WholeSize.second);
    CopyBegins.push_back(CopySize.first);
    CopyEnds.push_back(CopySize.second);
    MI.RefExpr = *I;
    MI.IsCEAN = Checker.IsCEANExpr();
    DSAStack->addMapInfoForVar(VD, MI);
  }

  if (Vars.empty())
    return 0;

  return OMPMapClause::Create(Context, StartLoc, EndLoc, Vars, WholeBegins,
                              WholeEnds, CopyBegins, CopyEnds, Kind, KindLoc);
}

OMPClause *Sema::ActOnOpenMPToClause(ArrayRef<Expr *> VarList,
                                     SourceLocation StartLoc,
                                     SourceLocation EndLoc) {
  SmallVector<Expr *, 4> Vars;
  SmallVector<Expr *, 4> WholeBegins;
  SmallVector<Expr *, 4> WholeEnds;
  SmallVector<Expr *, 4> CopyBegins;
  SmallVector<Expr *, 4> CopyEnds;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "Null expr in omp to");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      WholeBegins.push_back(0);
      WholeEnds.push_back(0);
      CopyBegins.push_back(0);
      CopyEnds.push_back(0);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();

    // OpenMP [2.9.3, Restrictions]
    //  A variable that is part of another variable (such as field of a
    //  structure) but is not an array element or an array section cannot appear
    //  as a list item in a clause of a target update construct.
    Expr *VE = (*I)->IgnoreParenLValueCasts();

    if (VE->isValueDependent() || VE->isTypeDependent() ||
        VE->isInstantiationDependent() ||
        VE->containsUnexpandedParameterPack()) {
      // It will be analyzed later.
      Vars.push_back(*I);
      WholeBegins.push_back(0);
      WholeEnds.push_back(0);
      CopyBegins.push_back(0);
      CopyEnds.push_back(0);
      continue;
    }

    MapArrayItemChecker Checker(*this);
    VarDecl *VD = 0;
    DeclRefExpr *DE = 0;
    if (Checker.Visit(VE) || !(VD = Checker.getBaseDecl()) ||
        !(DE = Checker.getDeclRefExprForBaseDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name_or_array_item)
          << (*I)->getSourceRange();
      continue;
    }

    // threadprivate variables cannot appear in a map clause.
    DeclRefExpr *DRE = 0;
    if (DSAStack->IsThreadprivate(VD, DRE)) {
      SourceLocation Loc = DRE ? DRE->getLocation() : VD->getLocation();
      Diag(Loc, diag::err_omp_threadprivate_in_target);
      Diag(DE->getLocStart(), diag::note_used_here) << DE->getSourceRange();
      continue;
    }

    // OpenMP [2.9.3, Restrictions, p.6]
    //  A list item in a to or from clause must have a mappable type.
    QualType Type = VD->getType();
    if (!CheckTypeMappable(VE->getExprLoc(), VE->getSourceRange(), *this,
                           DSAStack, Type)) {
      continue;
    }

    // OpenMP [2.9.3, Restrictions, p.6]
    // A list item can only appear in a to or from clause, but not both.
    DSAStackTy::MapInfo MI = DSAStack->IsMappedInCurrentRegion(VD);
    if (MI.RefExpr) {
      Diag(DE->getExprLoc(), diag::err_omp_once_referenced_in_target_update)
          << DE->getSourceRange();
      Diag(MI.RefExpr->getExprLoc(), diag::note_used_here)
          << MI.RefExpr->getSourceRange();
      continue;
    }

    std::pair<Expr *, Expr *> WholeSize = Checker.CalculateWholeSize();
    if (!WholeSize.first || !WholeSize.second) {
      continue;
    }
    std::pair<Expr *, Expr *> CopySize = Checker.CalculateCopySize();
    if (!CopySize.first || !CopySize.second) {
      continue;
    }

    Vars.push_back(*I);
    WholeBegins.push_back(WholeSize.first);
    WholeEnds.push_back(WholeSize.second);
    CopyBegins.push_back(CopySize.first);
    CopyEnds.push_back(CopySize.second);
    MI.RefExpr = *I;
    MI.IsCEAN = Checker.IsCEANExpr();
    DSAStack->addMapInfoForVar(VD, MI);
  }

  if (Vars.empty())
    return 0;

  return OMPToClause::Create(Context, StartLoc, EndLoc, Vars, WholeBegins,
                             WholeEnds, CopyBegins, CopyEnds);
}

OMPClause *Sema::ActOnOpenMPFromClause(ArrayRef<Expr *> VarList,
                                       SourceLocation StartLoc,
                                       SourceLocation EndLoc) {
  SmallVector<Expr *, 4> Vars;
  SmallVector<Expr *, 4> WholeBegins;
  SmallVector<Expr *, 4> WholeEnds;
  SmallVector<Expr *, 4> CopyBegins;
  SmallVector<Expr *, 4> CopyEnds;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "Null expr in omp from");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      WholeBegins.push_back(0);
      WholeEnds.push_back(0);
      CopyBegins.push_back(0);
      CopyEnds.push_back(0);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();

    // OpenMP [2.9.3, Restrictions]
    //  A variable that is part of another variable (such as field of a
    //  structure) but is not an array element or an array section cannot appear
    //  as a list item in a clause of a target update construct.
    Expr *VE = (*I)->IgnoreParenLValueCasts();

    if (VE->isValueDependent() || VE->isTypeDependent() ||
        VE->isInstantiationDependent() ||
        VE->containsUnexpandedParameterPack()) {
      // It will be analyzed later.
      Vars.push_back(*I);
      WholeBegins.push_back(0);
      WholeEnds.push_back(0);
      CopyBegins.push_back(0);
      CopyEnds.push_back(0);
      continue;
    }

    MapArrayItemChecker Checker(*this);
    VarDecl *VD = 0;
    DeclRefExpr *DE = 0;
    if (Checker.Visit(VE) || !(VD = Checker.getBaseDecl()) ||
        !(DE = Checker.getDeclRefExprForBaseDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name_or_array_item)
          << (*I)->getSourceRange();
      continue;
    }

    // threadprivate variables cannot appear in a map clause.
    DeclRefExpr *DRE = 0;
    if (DSAStack->IsThreadprivate(VD, DRE)) {
      SourceLocation Loc = DRE ? DRE->getLocation() : VD->getLocation();
      Diag(Loc, diag::err_omp_threadprivate_in_target);
      Diag(DE->getLocStart(), diag::note_used_here) << DE->getSourceRange();
      continue;
    }

    // OpenMP [2.9.3, Restrictions, p.6]
    //  A list item in a to or from clause must have a mappable type.
    QualType Type = VD->getType();
    if (!CheckTypeMappable(VE->getExprLoc(), VE->getSourceRange(), *this,
                           DSAStack, Type)) {
      continue;
    }

    // OpenMP [2.9.3, Restrictions, p.6]
    // A list item can only appear in a to or from clause, but not both.
    DSAStackTy::MapInfo MI = DSAStack->IsMappedInCurrentRegion(VD);
    if (MI.RefExpr) {
      Diag(DE->getExprLoc(), diag::err_omp_once_referenced_in_target_update)
          << DE->getSourceRange();
      Diag(MI.RefExpr->getExprLoc(), diag::note_used_here)
          << MI.RefExpr->getSourceRange();
      continue;
    }

    std::pair<Expr *, Expr *> WholeSize = Checker.CalculateWholeSize();
    if (!WholeSize.first || !WholeSize.second) {
      continue;
    }
    std::pair<Expr *, Expr *> CopySize = Checker.CalculateCopySize();
    if (!CopySize.first || !CopySize.second) {
      continue;
    }

    Vars.push_back(*I);
    WholeBegins.push_back(WholeSize.first);
    WholeEnds.push_back(WholeSize.second);
    CopyBegins.push_back(CopySize.first);
    CopyEnds.push_back(CopySize.second);
    MI.RefExpr = *I;
    MI.IsCEAN = Checker.IsCEANExpr();
    DSAStack->addMapInfoForVar(VD, MI);
  }

  if (Vars.empty())
    return 0;

  return OMPFromClause::Create(Context, StartLoc, EndLoc, Vars, WholeBegins,
                               WholeEnds, CopyBegins, CopyEnds);
}

OMPClause *Sema::ActOnOpenMPLinearClause(ArrayRef<Expr *> VarList,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc, Expr *Step,
                                         SourceLocation StepLoc) {
  // Checks that apply to both private and linear variables.
  SmallVector<Expr *, 4> Vars;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {

    assert(*I && "Null expr in omp linear");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      continue;
    }

    // OpenMP [2.14.3.7, linear clause]
    // A list item that appears in a linear clause is subject to the private
    // clause semantics described in Section 2.14.3.3 on page 159 except as
    // noted. In addition, the value of the new list item on each iteration
    // of the associated loop(s) corresponds to the value of the original
    // list item before entering the construct plus the logical number of
    // the iteration times linear-step.

    SourceLocation ELoc = (*I)->getExprLoc();
    // OpenMP [2.1, C/C++]
    //  A list item is a variable name.
    // OpenMP  [2.14.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      Diag(ELoc, diag::err_omp_expected_var_name) << (*I)->getSourceRange();
      continue;
    }

    VarDecl *VD = cast<VarDecl>(DE->getDecl());
    // OpenMP [2.14.3.7, linear clause]
    // - A list-item cannot appear in more than one linear clause.
    // - A list-item that appears in a linear clause cannot appear in any
    //   other data-sharing attribute clause.
    DeclRefExpr *PrevRef;
    OpenMPClauseKind Kind = DSAStack->getTopDSA(VD, PrevRef);
    if (PrevRef && (Kind == OMPC_linear || Kind == OMPC_private ||
                    Kind == OMPC_lastprivate || Kind == OMPC_reduction)) {
      Diag(ELoc, diag::err_omp_wrong_dsa) << getOpenMPClauseName(Kind)
                                          << getOpenMPClauseName(OMPC_linear);
      Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
          << getOpenMPClauseName(Kind);
      continue;
    }

    //  A variable that appears in a private clause must not have an incomplete
    //  type or a reference type.
    QualType QTy = VD->getType().getCanonicalType();
    if (RequireCompleteType(ELoc, QTy, diag::err_omp_linear_incomplete_type)) {
      continue;
    }
    if (QTy->isReferenceType()) {
      Diag(ELoc, diag::err_omp_clause_ref_type_arg)
          << getOpenMPClauseName(OMPC_linear);
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    //  A list item that appears in a private clause must not be
    //  const-qualified.
    if (QTy.isConstant(Context)) {
      Diag(ELoc, diag::err_omp_const_variable)
          << getOpenMPClauseName(OMPC_linear);
      bool IsDecl =
          VD->isThisDeclarationADefinition(Context) == VarDecl::DeclarationOnly;
      Diag(VD->getLocation(),
           IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << VD;
      continue;
    }

    // - A list-item that appears in a linear clause must be of integral
    //   or pointer type.
    QTy = QTy.getUnqualifiedType().getCanonicalType();
    const Type *Ty = QTy.getTypePtrOrNull();
    if (!Ty || (!Ty->isDependentType() && !Ty->isIntegralType(Context) &&
                !Ty->isPointerType())) {
      Diag(ELoc, diag::err_omp_expected_int_or_ptr) << (*I)->getSourceRange();
      continue;
    }

    DSAStack->addDSA(VD, DE, OMPC_linear);

    Vars.push_back(DE);
  }

  if (Vars.empty())
    return 0;

  if (Step && Step->isIntegerConstantExpr(Context)) {
    Step = ActOnConstantLinearStep(Step);
    if (!Step)
      return 0;
  }

  return OMPLinearClause::Create(Context, StartLoc, EndLoc, Vars, Step,
                                 StepLoc);
}

OMPClause *Sema::ActOnOpenMPAlignedClause(ArrayRef<Expr *> VarList,
                                          SourceLocation StartLoc,
                                          SourceLocation EndLoc,
                                          Expr *Alignment,
                                          SourceLocation AlignmentLoc) {
  SmallVector<Expr *, 4> Vars;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {

    assert(*I && "Null expr in omp aligned");
    if (*I && isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      continue;
    }

    SourceLocation ELoc = (*I)->getExprLoc();
    DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I);
    if (!DE || !isa<VarDecl>(DE->getDecl())) {
      // OpenMP [2.1, C/C++]
      //  A list item is a variable name.
      Diag(ELoc, diag::err_omp_expected_var_name) << (*I)->getSourceRange();
      continue;
    }
    // OpenMP  [2.8.1, simd construct, Restrictions]
    // The type of list items appearing in the aligned clause must be
    // array, pointer, reference to array, or reference to pointer.
    QualType QTy = DE->getType()
                       .getNonReferenceType()
                       .getUnqualifiedType()
                       .getCanonicalType();
    const Type *Ty = QTy.getTypePtrOrNull();
    if (!Ty || (!Ty->isDependentType() && !Ty->isArrayType() &&
                !Ty->isPointerType())) {
      Diag(ELoc, diag::err_omp_expected_array_or_ptr) << (*I)->getSourceRange();
      continue;
    }
    // OpenMP  [2.8.1, simd construct, Restrictions]
    // A list-item cannot appear in more than one aligned clause.
    DeclRefExpr *PrevRef = DE;
    if (!DSAStack->addUniqueAligned(cast<VarDecl>(DE->getDecl()), PrevRef)) {
      Diag(ELoc, diag::err_omp_wrong_dsa) << getOpenMPClauseName(OMPC_aligned)
                                          << getOpenMPClauseName(OMPC_aligned);
      Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
          << getOpenMPClauseName(OMPC_aligned);
      continue;
    }

    Vars.push_back(DE);
  }

  if (Vars.empty())
    return 0;

  // OpenMP [2.8.1, simd construct, Description]
  // The optional parameter of the aligned clause, alignment, must be
  // a constant positive integer expression.
  if (Alignment) {
    Alignment = ActOnConstantPositiveSubExpressionInClause(Alignment);
    if (!Alignment)
      return 0;
  }

  return OMPAlignedClause::Create(Context, StartLoc, EndLoc, Vars, Alignment,
                                  AlignmentLoc);
}

OMPClause *Sema::ActOnOpenMPReadClause(SourceLocation StartLoc,
                                       SourceLocation EndLoc) {
  return new (Context) OMPReadClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPWriteClause(SourceLocation StartLoc,
                                        SourceLocation EndLoc) {
  return new (Context) OMPWriteClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPUpdateClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return new (Context) OMPUpdateClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPCaptureClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OMPCaptureClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPSeqCstClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return new (Context) OMPSeqCstClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPInBranchClause(SourceLocation StartLoc,
                                           SourceLocation EndLoc) {
  return new (Context) OMPInBranchClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPNotInBranchClause(SourceLocation StartLoc,
                                              SourceLocation EndLoc) {
  return new (Context) OMPNotInBranchClause(StartLoc, EndLoc);
}

OMPClause *Sema::ActOnOpenMPFlushClause(ArrayRef<Expr *> VarList,
                                        SourceLocation StartLoc,
                                        SourceLocation EndLoc) {
  SmallVector<Expr *, 4> Vars;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "Null expr in omp flush");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      continue;
    }

    if (DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I))
      Vars.push_back(DE);
  }

  if (Vars.empty())
    return 0;

  return OMPFlushClause::Create(Context, StartLoc, EndLoc, Vars);
}

OMPClause *Sema::ActOnOpenMPUniformClause(ArrayRef<Expr *> VarList,
                                          SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  SmallVector<Expr *, 4> Vars;
  for (ArrayRef<Expr *>::iterator I = VarList.begin(), E = VarList.end();
       I != E; ++I) {
    assert(*I && "Null expr in omp uniform");
    if (isa<DependentScopeDeclRefExpr>(*I)) {
      // It will be analyzed later.
      Vars.push_back(*I);
      continue;
    }

    if (DeclRefExpr *DE = dyn_cast_or_null<DeclRefExpr>(*I))
      Vars.push_back(DE);
  }

  if (Vars.empty())
    return 0;

  return OMPUniformClause::Create(Context, StartLoc, EndLoc, Vars);
}

namespace {
class ForInitChecker : public StmtVisitor<ForInitChecker, Decl *> {
  class ForInitVarChecker : public StmtVisitor<ForInitVarChecker, Decl *> {
  public:
    VarDecl *VisitDeclRefExpr(DeclRefExpr *E) {
      return dyn_cast_or_null<VarDecl>(E->getDecl());
    }
    Decl *VisitStmt(Stmt *S) { return 0; }
    ForInitVarChecker() {}
  } VarChecker;
  Expr *InitValue;

public:
  Decl *VisitBinaryOperator(BinaryOperator *BO) {
    if (BO->getOpcode() != BO_Assign)
      return 0;

    InitValue = BO->getRHS();
    return VarChecker.Visit(BO->getLHS());
  }
  Decl *VisitDeclStmt(DeclStmt *S) {
    if (S->isSingleDecl()) {
      VarDecl *Var = dyn_cast_or_null<VarDecl>(S->getSingleDecl());
      if (Var && Var->hasInit()) {
        if (CXXConstructExpr *Init =
                dyn_cast<CXXConstructExpr>(Var->getInit())) {
          if (Init->getNumArgs() != 1)
            return 0;
          InitValue = Init->getArg(0);
        } else {
          InitValue = Var->getInit();
        }
        return Var;
      }
    }
    return 0;
  }
  Decl *VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
    switch (E->getOperator()) {
    case OO_Equal:
      InitValue = E->getArg(1);
      return VarChecker.Visit(E->getArg(0));
    default:
      break;
    }
    return 0;
  }
  Decl *VisitStmt(Stmt *S) { return 0; }
  ForInitChecker() : VarChecker(), InitValue(0) {}
  Expr *getInitValue() { return InitValue; }
};

class ForVarChecker : public StmtVisitor<ForVarChecker, bool> {
  Decl *InitVar;

public:
  bool VisitDeclRefExpr(DeclRefExpr *E) { return E->getDecl() == InitVar; }
  bool VisitImplicitCastExpr(ImplicitCastExpr *E) {
    return Visit(E->getSubExpr());
  }
  bool VisitStmt(Stmt *S) { return false; }
  ForVarChecker(Decl *D) : InitVar(D) {}
};

class ForTestChecker : public StmtVisitor<ForTestChecker, bool> {
  ForVarChecker VarChecker;
  Expr *CheckValue;
  bool IsLessOp;
  bool IsStrictOp;

public:
  bool VisitBinaryOperator(BinaryOperator *BO) {
    if (!BO->isRelationalOp())
      return false;
    if (VarChecker.Visit(BO->getLHS())) {
      CheckValue = BO->getRHS();
      IsLessOp = BO->getOpcode() == BO_LT || BO->getOpcode() == BO_LE;
      IsStrictOp = BO->getOpcode() == BO_LT || BO->getOpcode() == BO_GT;
    } else if (VarChecker.Visit(BO->getRHS())) {
      CheckValue = BO->getLHS();
      IsLessOp = BO->getOpcode() == BO_GT || BO->getOpcode() == BO_GE;
      IsStrictOp = BO->getOpcode() == BO_LT || BO->getOpcode() == BO_GT;
    }
    return CheckValue != 0;
  }
  bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
    switch (E->getOperator()) {
    case OO_Greater:
    case OO_GreaterEqual:
    case OO_Less:
    case OO_LessEqual:
      break;
    default:
      return false;
    }
    if (E->getNumArgs() != 2)
      return false;

    if (VarChecker.Visit(E->getArg(0))) {
      CheckValue = E->getArg(1);
      IsLessOp =
          E->getOperator() == OO_Less || E->getOperator() == OO_LessEqual;
      IsStrictOp = E->getOperator() == OO_Less;
    } else if (VarChecker.Visit(E->getArg(1))) {
      CheckValue = E->getArg(0);
      IsLessOp =
          E->getOperator() == OO_Greater || E->getOperator() == OO_GreaterEqual;
      IsStrictOp = E->getOperator() == OO_Greater;
    }

    return CheckValue != 0;
  }
  bool VisitStmt(Stmt *S) { return false; }
  ForTestChecker(Decl *D)
      : VarChecker(D), CheckValue(0), IsLessOp(false), IsStrictOp(false) {}
  Expr *getCheckValue() { return CheckValue; }
  bool isLessOp() const { return IsLessOp; }
  bool isStrictOp() const { return IsStrictOp; }
};

class ForIncrChecker : public StmtVisitor<ForIncrChecker, bool> {
  ForVarChecker VarChecker;
  class ForIncrExprChecker : public StmtVisitor<ForIncrExprChecker, bool> {
    ForVarChecker VarChecker;
    Expr *StepValue;
    bool IsIncrement;

  public:
    bool VisitBinaryOperator(BinaryOperator *BO) {
      if (!BO->isAdditiveOp())
        return false;
      if (BO->getOpcode() == BO_Add) {
        IsIncrement = true;
        if (VarChecker.Visit(BO->getLHS()))
          StepValue = BO->getRHS();
        else if (VarChecker.Visit(BO->getRHS()))
          StepValue = BO->getLHS();
        return StepValue != 0;
      }
      // BO_Sub
      if (VarChecker.Visit(BO->getLHS()))
        StepValue = BO->getRHS();
      return StepValue != 0;
    }
    bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
      switch (E->getOperator()) {
      case OO_Plus:
        IsIncrement = true;
        if (VarChecker.Visit(E->getArg(0)))
          StepValue = E->getArg(1);
        else if (VarChecker.Visit(E->getArg(1)))
          StepValue = E->getArg(0);
        return StepValue != 0;
      case OO_Minus:
        if (VarChecker.Visit(E->getArg(0)))
          StepValue = E->getArg(1);
        return StepValue != 0;
      default:
        return false;
      }
    }
    bool VisitStmt(Stmt *S) { return false; }
    ForIncrExprChecker(ForVarChecker &C)
        : VarChecker(C), StepValue(0), IsIncrement(false) {}
    Expr *getStepValue() { return StepValue; }
    bool isIncrement() const { return IsIncrement; }
  } ExprChecker;
  Expr *StepValue;
  Sema &Actions;
  bool IsLessOp, IsCompatibleWithTest;

public:
  bool VisitUnaryOperator(UnaryOperator *UO) {
    if (!UO->isIncrementDecrementOp())
      return false;
    if (VarChecker.Visit(UO->getSubExpr())) {
      IsCompatibleWithTest = (IsLessOp && UO->isIncrementOp()) ||
                             (!IsLessOp && UO->isDecrementOp());
      if (!IsCompatibleWithTest && IsLessOp)
        StepValue = Actions.ActOnIntegerConstant(SourceLocation(), -1).get();
      else
        StepValue = Actions.ActOnIntegerConstant(SourceLocation(), 1).get();
    }
    return StepValue != 0;
  }
  bool VisitBinaryOperator(BinaryOperator *BO) {
    IsCompatibleWithTest = (IsLessOp && BO->getOpcode() == BO_AddAssign) ||
                           (!IsLessOp && BO->getOpcode() == BO_SubAssign);
    switch (BO->getOpcode()) {
    case BO_AddAssign:
    case BO_SubAssign:
      if (VarChecker.Visit(BO->getLHS())) {
        StepValue = BO->getRHS();
        IsCompatibleWithTest = (IsLessOp && BO->getOpcode() == BO_AddAssign) ||
                               (!IsLessOp && BO->getOpcode() == BO_SubAssign);
      }
      return StepValue != 0;
    case BO_Assign:
      if (VarChecker.Visit(BO->getLHS()) && ExprChecker.Visit(BO->getRHS())) {
        StepValue = ExprChecker.getStepValue();
        IsCompatibleWithTest = IsLessOp == ExprChecker.isIncrement();
      }
      return StepValue != 0;
    default:
      break;
    }
    return false;
  }
  bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
    switch (E->getOperator()) {
    case OO_PlusPlus:
    case OO_MinusMinus:
      if (VarChecker.Visit(E->getArg(0))) {
        IsCompatibleWithTest = (IsLessOp && E->getOperator() == OO_PlusPlus) ||
                               (!IsLessOp && E->getOperator() == OO_MinusMinus);
        if (!IsCompatibleWithTest && IsLessOp)
          StepValue = Actions.ActOnIntegerConstant(SourceLocation(), -1).get();
        else
          StepValue = Actions.ActOnIntegerConstant(SourceLocation(), 1).get();
      }
      return StepValue != 0;
    case OO_PlusEqual:
    case OO_MinusEqual:
      if (VarChecker.Visit(E->getArg(0))) {
        StepValue = E->getArg(1);
        IsCompatibleWithTest = (IsLessOp && E->getOperator() == OO_PlusEqual) ||
                               (!IsLessOp && E->getOperator() == OO_MinusEqual);
      }
      return StepValue != 0;
    case OO_Equal:
      if (VarChecker.Visit(E->getArg(0)) && ExprChecker.Visit(E->getArg(1))) {
        StepValue = ExprChecker.getStepValue();
        IsCompatibleWithTest = IsLessOp == ExprChecker.isIncrement();
      }
      return StepValue != 0;
    default:
      break;
    }
    return false;
  }
  bool VisitStmt(Stmt *S) { return false; }
  ForIncrChecker(Decl *D, Sema &S, bool LessOp)
      : VarChecker(D), ExprChecker(VarChecker), StepValue(0), Actions(S),
        IsLessOp(LessOp), IsCompatibleWithTest(false) {}
  Expr *getStepValue() { return StepValue; }
  bool isCompatibleWithTest() const { return IsCompatibleWithTest; }
};
}

bool Sema::isNotOpenMPCanonicalLoopForm(Stmt *S, OpenMPDirectiveKind Kind,
                                        Expr *&NewEnd, Expr *&NewIncr,
                                        Expr *&InitVal, Expr *&VarCnt,
                                        BinaryOperatorKind &OpKind) {
  // assert(S && "non-null statement must be specified");
  // OpenMP [2.9.5, Canonical Loop Form]
  //  for (init-expr; test-expr; incr-expr) structured-block
  OpKind = BO_Assign;
  ForStmt *For = dyn_cast_or_null<ForStmt>(S);
  if (!For) {
    Diag(S->getLocStart(), diag::err_omp_not_for)
        << getOpenMPDirectiveName(Kind);
    return true;
  }
  Stmt *Body = For->getBody();
  if (!Body) {
    Diag(S->getLocStart(), diag::err_omp_directive_nonblock)
        << getOpenMPDirectiveName(Kind);
    return true;
  }

  // OpenMP [2.9.5, Canonical Loop Form]
  //  init-expr One of the following:
  //  var = lb
  //  integer-type var = lb
  //  random-access-iterator-type var = lb
  //  pointer-type var = lb
  ForInitChecker InitChecker;
  Stmt *Init = For->getInit();
  VarDecl *Var;
  if (!Init || !(Var = dyn_cast_or_null<VarDecl>(InitChecker.Visit(Init)))) {
    Diag(Init ? Init->getLocStart() : For->getForLoc(),
         diag::err_omp_not_canonical_for)
        << 0;
    return true;
  }
  SourceLocation InitLoc = Init->getLocStart();

  // OpenMP [2.11.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++]
  // The loop iteration variable(s) in the associated for-loop(s) of a for or
  // parallel for construct may be listed in a private or lastprivate clause.
  bool HasErrors = false;
  DeclRefExpr *PrevRef;
  OpenMPDirectiveKind CurrentDir = DSAStack->getCurrentDirective();
  OpenMPClauseKind CKind = DSAStack->getTopDSA(Var, PrevRef);
  if (CKind == OMPC_threadprivate) {
    //    Diag(InitLoc, diag::err_omp_for_loop_var_dsa)
    //         << getOpenMPClauseName(CKind);
    //    if (PrevRef)
    //      Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
    //           << getOpenMPClauseName(CKind);
    //    HasErrors = true;
  } else if (CKind != OMPC_unknown && CKind != OMPC_private &&
             CKind != OMPC_lastprivate &&
             (CurrentDir == OMPD_for || CurrentDir == OMPD_parallel_for ||
              CurrentDir == OMPD_distribute ||
              CurrentDir == OMPD_distribute_parallel_for ||
              CurrentDir == OMPD_teams_distribute ||
              CurrentDir == OMPD_target_teams_distribute ||
              CurrentDir == OMPD_teams_distribute_parallel_for ||
              CurrentDir == OMPD_target_teams_distribute_parallel_for)) {
    Diag(InitLoc, diag::err_omp_for_loop_var_dsa) << getOpenMPClauseName(CKind);
    if (PrevRef) {
      Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
          << getOpenMPClauseName(CKind);
    } else {
      Diag(Var->getLocation(), diag::note_omp_predetermined_dsa)
          << getOpenMPClauseName(CKind);
    }
    HasErrors = true;
  } else if (CKind != OMPC_unknown && CKind != OMPC_linear &&
             CKind != OMPC_lastprivate &&
             (CurrentDir == OMPD_simd || CurrentDir == OMPD_for_simd ||
              CurrentDir == OMPD_parallel_for_simd ||
              CurrentDir == OMPD_distribute_parallel_for_simd ||
              CurrentDir == OMPD_teams_distribute_parallel_for_simd ||
              CurrentDir == OMPD_target_teams_distribute_parallel_for_simd ||
              CurrentDir == OMPD_distribute_simd ||
              CurrentDir == OMPD_teams_distribute_simd ||
              CurrentDir == OMPD_target_teams_distribute_simd)) {
    // OpenMP [2.11.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, C/C++]
    // The loop iteration variable in the associated for-loop of a simd
    // construct with just one associated for-loop may be listed in a linear
    // clause with a constant-linear-step that is the increment of the
    // associated for-loop.
    Diag(InitLoc, diag::err_omp_for_loop_var_dsa) << getOpenMPClauseName(CKind);
    if (PrevRef) {
      Diag(PrevRef->getExprLoc(), diag::note_omp_explicit_dsa)
          << getOpenMPClauseName(CKind);
    } else {
      Diag(Var->getLocation(), diag::note_omp_predetermined_dsa)
          << getOpenMPClauseName(CKind);
    }
    HasErrors = true;
  } else {
    // OpenMP [2.11.1.1, Data-sharing Attribute Rules for Variables Referenced
    // in a Construct, C/C++]
    // The loop iteration variable(s) in the associated for-loop(s)of a for or
    // parallel for construct is (are) private.
    DSAStack->addDSA(Var, 0, OMPC_private);
  }

  // OpenMP [2.9.5, Canonical Loop Form]
  // Var One of the following
  // A variable of signed or unsigned integer type
  // For C++, a variable of a random access iterator type.
  // For C, a variable of a pointer type.
  QualType Type = Var->getType()
                      .getNonReferenceType()
                      .getCanonicalType()
                      .getUnqualifiedType();
  if (!Type->isIntegerType() && !Type->isPointerType() &&
      (!getLangOpts().CPlusPlus || !Type->isOverloadableType())) {
    Diag(Init->getLocStart(), diag::err_omp_for_variable)
        << getLangOpts().CPlusPlus;
    HasErrors = true;
  }

  // OpenMP [2.9.5, Canonical Loop Form]
  //  test-expr One of the following:
  //  var relational-op b
  //  b relational-op var
  ForTestChecker TestChecker(Var);
  Stmt *Cond = For->getCond();
  bool TestCheckCorrect = false;
  if (!Cond || !(TestCheckCorrect = TestChecker.Visit(Cond))) {
    Diag(Cond ? Cond->getLocStart() : For->getForLoc(),
         diag::err_omp_not_canonical_for)
        << 1;
    HasErrors = true;
  }

  // OpenMP [2.9.5, Canonical Loop Form]
  //  incr-expr One of the following:
  //  ++var
  //  var++
  //  --var
  //  var--
  //  var += incr
  //  var -= incr
  //  var = var + incr
  //  var = incr + var
  //  var = var - incr
  ForIncrChecker IncrChecker(Var, *this, TestChecker.isLessOp());
  Stmt *Incr = For->getInc();
  bool IncrCheckCorrect = false;
  if (!Incr || !(IncrCheckCorrect = IncrChecker.Visit(Incr))) {
    Diag(Incr ? Incr->getLocStart() : For->getForLoc(),
         diag::err_omp_not_canonical_for)
        << 2;
    HasErrors = true;
  }

  // OpenMP [2.9.5, Canonical Loop Form]
  //  lb and b Loop invariant expressions of a type compatible with the type
  //  of var.
  Expr *InitValue = InitChecker.getInitValue();
  //  QualType InitTy =
  //    InitValue ? InitValue->getType().getNonReferenceType().
  //                                  getCanonicalType().getUnqualifiedType() :
  //                QualType();
  //  if (InitValue &&
  //      Context.mergeTypes(Type, InitTy, false, true).isNull()) {
  //    Diag(InitValue->getExprLoc(), diag::err_omp_for_type_not_compatible)
  //      << InitValue->getType()
  //      << Var << Var->getType();
  //    HasErrors = true;
  //  }
  Expr *CheckValue = TestChecker.getCheckValue();
  //  QualType CheckTy =
  //    CheckValue ? CheckValue->getType().getNonReferenceType().
  //                                  getCanonicalType().getUnqualifiedType() :
  //                 QualType();
  //  if (CheckValue &&
  //      Context.mergeTypes(Type, CheckTy, false, true).isNull()) {
  //    Diag(CheckValue->getExprLoc(), diag::err_omp_for_type_not_compatible)
  //      << CheckValue->getType()
  //      << Var << Var->getType();
  //    HasErrors = true;
  //  }

  // OpenMP [2.9.5, Canonical Loop Form]
  //  incr A loop invariant integer expression.
  Expr *Step = IncrChecker.getStepValue();
  if (Step && !Step->getType()->isIntegralOrEnumerationType()) {
    Diag(Step->getExprLoc(), diag::err_omp_for_incr_not_integer);
    HasErrors = true;
  }
  // llvm::APSInt Result;
  // if (Step && Step->isIntegerConstantExpr(Result, Context) &&
  //    !Result.isStrictlyPositive()) {
  //  Diag(Step->getExprLoc(), diag::err_negative_expression_in_clause);
  //  HasErrors = true;
  //}

  // OpenMP [2.9.5, Canonical Loop Form, Restrictions]
  //  If test-expr is of form var relational-op b and relational-op is < or
  //  <= then incr-expr must cause var to increase on each iteration of the
  //  loop. If test-expr is of form var relational-op b and relational-op is
  //  > or >= then incr-expr must cause var to decrease on each iteration of the
  //  loop.
  //  If test-expr is of form b relational-op var and relational-op is < or
  //  <= then incr-expr must cause var to decrease on each iteration of the
  //  loop. If test-expr is of form b relational-op var and relational-op is
  //  > or >= then incr-expr must cause var to increase on each iteration of the
  //  loop.
  if (Incr && TestCheckCorrect && IncrCheckCorrect &&
      !IncrChecker.isCompatibleWithTest()) {
    // Additional type checking.
    llvm::APSInt Result;
    bool IsConst = Step->isIntegerConstantExpr(Result, getASTContext());
    bool IsConstNeg = IsConst && Result.isSigned() && Result.isNegative();
    bool IsSigned = Step->getType()->hasSignedIntegerRepresentation();
    if ((TestChecker.isLessOp() && IsConst && IsConstNeg) ||
        (!TestChecker.isLessOp() &&
         ((IsConst && !IsConstNeg) || (!IsConst && !IsSigned)))) {
      Diag(Incr->getLocStart(), diag::err_omp_for_incr_not_compatible)
          << Var << TestChecker.isLessOp();
      HasErrors = true;
    } else {
      Step = CreateBuiltinUnaryOp(Step->getExprLoc(), UO_Minus, Step).get();
    }
  }
  if (HasErrors)
    return true;

  // Build expression for number of iterations.
  // if (getLangOpts().CPlusPlus && !StdNamespace && !Type->isIntegerType()) {
  //  Diag(Var->getLocation(), diag::err_omp_type_not_rai);
  //  return true;
  //}

  ExprResult Diff;
  assert(Step && "Null expr in Step in OMP FOR");
  Step = Step->IgnoreParenImpCasts();
  CheckValue = CheckValue->IgnoreParenImpCasts();
  InitValue = InitValue->IgnoreParenImpCasts();
  if (Step->getType()->isDependentType() ||
      CheckValue->getType()->isDependentType() ||
      InitValue->getType()->isDependentType()) {
    NewEnd = CheckValue;
    NewIncr = Step;
    InitVal = InitValue;
    VarCnt = CheckValue;
    return false;
  }
  if (getLangOpts().CPlusPlus && !Type->isIntegerType() &&
      !Type->isPointerType()) {
    // Check that var type is a random access iterator, i.e.
    // we can apply 'std::distance' to the init and test arguments
    // of the for-loop.
    CXXScopeSpec SS;
    SS.Extend(Context, getOrCreateStdNamespace(), SourceLocation(),
              SourceLocation());
    IdentifierInfo *IIT = &Context.Idents.get("iterator_traits");
    DeclarationNameInfo DNIIT(IIT, SourceLocation());
    LookupResult RIT(*this, DNIIT, LookupNestedNameSpecifierName);
    TemplateDecl *D;
    if (!LookupParsedName(RIT, DSAStack->getCurScope(), &SS) ||
        !RIT.isSingleResult() || !(D = RIT.getAsSingle<TemplateDecl>())) {
      Diag(Var->getLocation(), diag::err_omp_type_not_rai);
      return true;
    }

    TemplateArgumentListInfo Args;
    TemplateArgument Arg(Type);
    TemplateArgumentLoc ArgLoc(Arg, Context.CreateTypeSourceInfo(Type));
    Args.addArgument(ArgLoc);
    QualType T = CheckTemplateIdType(TemplateName(D), SourceLocation(), Args);
    CXXRecordDecl *TRDType;
    if (T.isNull() || RequireCompleteType(Var->getLocation(), T, 0) ||
        !(TRDType = T->getAsCXXRecordDecl())) {
      Diag(Var->getLocation(), diag::err_omp_type_not_rai);
      return true;
    }

    IdentifierInfo *IIRAI = &Context.Idents.get("random_access_iterator_tag");
    DeclarationNameInfo DNIRAI(IIRAI, SourceLocation());
    LookupResult RRAI(*this, DNIRAI, LookupOrdinaryName);
    TypeDecl *TDRAI;
    CXXRecordDecl *RDType = Type->getAsCXXRecordDecl();
    if (!LookupParsedName(RRAI, DSAStack->getCurScope(), &SS) ||
        !RRAI.isSingleResult() || !(TDRAI = RRAI.getAsSingle<TypeDecl>()) ||
        !RDType) {
      Diag(Var->getLocation(), diag::err_omp_type_not_rai);
      return true;
    }

    IdentifierInfo *IIC = &Context.Idents.get("iterator_category");
    DeclarationNameInfo DNIIC(IIC, SourceLocation());
    LookupResult RIC(*this, DNIIC, LookupOrdinaryName);
    TypeDecl *TDIC;
    if (!LookupQualifiedName(RIC, TRDType) || !RIC.isSingleResult() ||
        !(TDIC = RIC.getAsSingle<TypeDecl>()) ||
        !Context.hasSameType(Context.getTypeDeclType(TDRAI),
                             Context.getTypeDeclType(TDIC))) {
      Diag(Var->getLocation(), diag::err_omp_type_not_rai);
      return true;
    }

    IdentifierInfo *IID = &Context.Idents.get("distance");
    DeclarationNameInfo DNID(IID, SourceLocation());
    ExprResult ER = BuildQualifiedTemplateIdExpr(SS, InitLoc, DNID, &Args);
    Expr *CallArgs[2] = {TestChecker.isLessOp() ? InitValue : CheckValue,
                         TestChecker.isLessOp() ? CheckValue : InitValue};
    Diff = ActOnCallExpr(DSAStack->getCurScope(), ER.get(), InitLoc, CallArgs,
                         InitLoc);
    if (Diff.isInvalid()) {
      Diag(Var->getLocation(), diag::err_omp_type_not_rai);
      return true;
    }
  } else {
    Diff = BuildBinOp(DSAStack->getCurScope(), InitLoc, BO_Sub,
                      TestChecker.isLessOp() ? CheckValue : InitValue,
                      TestChecker.isLessOp() ? InitValue : CheckValue);
  }
  if (Diff.isUsable() && TestChecker.isStrictOp()) {
    Diff = BuildBinOp(DSAStack->getCurScope(), InitLoc, BO_Sub, Diff.get(),
                      ActOnIntegerConstant(SourceLocation(), 1).get());
  }
  if (Diff.isUsable()) {
    Diff =
        BuildBinOp(DSAStack->getCurScope(), InitLoc, BO_Add, Diff.get(), Step);
  }
  if (Diff.isUsable()) {
    Diff =
        BuildBinOp(DSAStack->getCurScope(), InitLoc, BO_Div, Diff.get(), Step);
  }
  bool Signed = Type->hasSignedIntegerRepresentation();
  uint64_t TypeSize = Context.getTypeSize(Type);
  if (TypeSize < 32)
    TypeSize = 32;
  else if (TypeSize > 64)
    TypeSize = 64;
  QualType DiffType = Context.getIntTypeForBitwidth(TypeSize, Signed);
  TypeSourceInfo *TSI = Context.getTrivialTypeSourceInfo(DiffType);
  NewEnd = BuildCStyleCastExpr(SourceLocation(), TSI, SourceLocation(),
                               Diff.get()).get();
  NewIncr =
      BuildCStyleCastExpr(SourceLocation(), TSI, SourceLocation(), Step).get();
  InitVal =
      PerformImplicitConversion(InitValue, Type, AA_Initializing, true).get();
  // NamedDecl *ND = Var;
  VarCnt =
      DeclRefExpr::Create(Context, NestedNameSpecifierLoc(), SourceLocation(),
                          Var, false, SourceLocation(), Type, VK_LValue);
  // if (!isDeclInScope(ND, CurContext, DSAStack->getCurScope())) {
  //  DeclContext *SavedCurContext = CurContext;
  //  CurContext = Var->getDeclContext();
  //  VarCnt = BuildDeclRefExpr(Var, Type, VK_LValue, InitLoc).get();
  //  CurContext = SavedCurContext;
  //}
  OpKind = TestChecker.isLessOp() ? BO_Add : BO_Sub;
  return false;
}

namespace {
class CEANExprChecker : public StmtVisitor<CEANExprChecker, bool> {
public:
  bool VisitCEANIndexExpr(CEANIndexExpr *E) { return true; }
  bool VisitOpaqueValueExpr(OpaqueValueExpr *E) {
    return E->getSourceExpr() && Visit(E->getSourceExpr());
  }
  bool VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
    return E->getExpr() && Visit(E->getExpr());
  }
  bool VisitCXXDefaultInitExpr(CXXDefaultInitExpr *E) {
    return E->getExpr() && Visit(E->getExpr());
  }
  bool VisitExpressionTraitExpr(ExpressionTraitExpr *E) {
    return E->getQueriedExpression() && Visit(E->getQueriedExpression());
  }
  unsigned VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E) {
    if (!E->isArgumentType())
      return (E->getKind() == UETT_SizeOf) ? false
                                           : Visit(E->getArgumentExpr());
    return VisitStmt(E);
  }
  unsigned VisitLambdaExpr(LambdaExpr *E) { return false; }
  bool VisitStmt(Stmt *S) {
    for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I != E;
         ++I) {
      if (*I && Visit(*I))
        return true;
    }
    return false;
  }
  CEANExprChecker() {}
};
}

ExprResult Sema::ActOnCEANIndexExpr(Scope *S, Expr *Base, Expr *LowerBound,
                                    SourceLocation ColonLoc, Expr *Length) {
  bool ArgsDep =
      (Base && (Base->isTypeDependent() || Base->isValueDependent() ||
                Base->isInstantiationDependent() ||
                Base->containsUnexpandedParameterPack())) ||
      (LowerBound &&
       (LowerBound->isTypeDependent() || LowerBound->isValueDependent() ||
        LowerBound->isInstantiationDependent() ||
        LowerBound->containsUnexpandedParameterPack())) ||
      (Length && (Length->isTypeDependent() || Length->isValueDependent() ||
                  Length->isInstantiationDependent() ||
                  Length->containsUnexpandedParameterPack()));

  if (ArgsDep)
    return new (Context)
        CEANIndexExpr(Base, LowerBound, ColonLoc, Length, Context.IntTy);

  SourceLocation SLoc;
  if (LowerBound)
    SLoc = LowerBound->getExprLoc();
  else
    SLoc = ColonLoc;
  SourceLocation ELoc;
  if (Length)
    ELoc = Length->getLocEnd();
  else
    ELoc = ColonLoc;

  QualType BaseType =
      Base ? Base->getType().getNonReferenceType().getCanonicalType()
           : QualType();
  if (Base && ((Base->isGLValue() && Base->getObjectKind() != OK_Ordinary) ||
               !BaseType->isCompoundType())) {
    Diag(SLoc, diag::err_cean_not_in_statement) << SourceRange(SLoc, ELoc);
    return ExprError();
  }

  if (!LowerBound)
    LowerBound = ActOnIntegerConstant(ColonLoc, 0).get();
  else {
    CEANExprChecker Checker;
    if (Checker.Visit(LowerBound)) {
      Diag(LowerBound->getExprLoc(), diag::err_cean_not_in_statement)
          << LowerBound->getSourceRange();
      return ExprError();
    }
  }
  if (!Length) {
    if (!Base)
      return ExprError();
    QualType Type = Base->getType().getCanonicalType();
    if (DeclRefExpr *DRE =
            dyn_cast_or_null<DeclRefExpr>(Base->IgnoreParenLValueCasts())) {
      if (ParmVarDecl *PVD = dyn_cast_or_null<ParmVarDecl>(DRE->getDecl())) {
        Type = PVD->getOriginalType().getNonReferenceType().getCanonicalType();
      }
    }
    if (!Type->isConstantArrayType() && !Type->isVariableArrayType()) {
      Diag(ColonLoc, diag::err_cean_no_length_for_non_array) << Base->getType();
      return ExprError();
    }
    const ArrayType *ArrType = Type->castAsArrayTypeUnsafe();
    if (const ConstantArrayType *ConstArrType =
            dyn_cast<ConstantArrayType>(ArrType))
      Length = ActOnIntegerConstant(
                   ColonLoc, ConstArrType->getSize().getZExtValue()).get();
    else if (const VariableArrayType *VarArrType =
                 dyn_cast<VariableArrayType>(ArrType))
      Length = VarArrType->getSizeExpr();
    Length = CreateBuiltinBinOp(ColonLoc, BO_Sub, Length, LowerBound).get();
    if (!Length)
      return ExprError();
  } else {
    CEANExprChecker Checker;
    if (Checker.Visit(Length)) {
      Diag(Length->getExprLoc(), diag::err_cean_not_in_statement)
          << Length->getSourceRange();
      return ExprError();
    }
  }

  if (!LowerBound->getType()->isIntegerType()) {
    Diag(LowerBound->getExprLoc(), diag::err_cean_lower_bound_not_integer)
        << LowerBound->getType();
    return ExprError();
  }
  if (!Length->getType()->isIntegerType()) {
    Diag(Length->getExprLoc(), diag::err_cean_length_not_integer)
        << Length->getType();
    return ExprError();
  }

  ExprResult LowerBoundRes(LowerBound);
  ExprResult LengthRes(Length);
  QualType ResType = UsualArithmeticConversions(LowerBoundRes, LengthRes);
  LowerBoundRes = PerformImplicitConversion(LowerBound, ResType, AA_Converting);
  LengthRes = PerformImplicitConversion(Length, ResType, AA_Converting);
  return new (Context) CEANIndexExpr(Base, LowerBoundRes.get(), ColonLoc,
                                     LengthRes.get(), ResType);
}
