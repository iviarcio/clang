//===--- Stmt.cpp - Statement AST Node Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt class and statement subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/Type.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

static struct StmtClassNameTable {
  const char *Name;
  unsigned Counter;
  unsigned Size;
} StmtClassInfo[Stmt::lastStmtConstant + 1];

static StmtClassNameTable &getStmtInfoTableEntry(Stmt::StmtClass E) {
  static bool Initialized = false;
  if (Initialized)
    return StmtClassInfo[E];

  // Intialize the table on the first use.
  Initialized = true;
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT)                                                    \
  StmtClassInfo[(unsigned)Stmt::CLASS##Class].Name = #CLASS;                   \
  StmtClassInfo[(unsigned)Stmt::CLASS##Class].Size = sizeof(CLASS);
#include "clang/AST/StmtNodes.inc"

  return StmtClassInfo[E];
}

void *Stmt::operator new(size_t bytes, const ASTContext &C,
                         unsigned alignment) {
  return ::operator new(bytes, C, alignment);
}

const char *Stmt::getStmtClassName() const {
  return getStmtInfoTableEntry((StmtClass)StmtBits.sClass).Name;
}

void Stmt::PrintStats() {
  // Ensure the table is primed.
  getStmtInfoTableEntry(Stmt::NullStmtClass);

  unsigned sum = 0;
  llvm::errs() << "\n*** Stmt/Expr Stats:\n";
  for (int i = 0; i != Stmt::lastStmtConstant+1; i++) {
    if (StmtClassInfo[i].Name == nullptr) continue;
    sum += StmtClassInfo[i].Counter;
  }
  llvm::errs() << "  " << sum << " stmts/exprs total.\n";
  sum = 0;
  for (int i = 0; i != Stmt::lastStmtConstant+1; i++) {
    if (StmtClassInfo[i].Name == nullptr) continue;
    if (StmtClassInfo[i].Counter == 0) continue;
    llvm::errs() << "    " << StmtClassInfo[i].Counter << " "
                 << StmtClassInfo[i].Name << ", " << StmtClassInfo[i].Size
                 << " each ("
                 << StmtClassInfo[i].Counter * StmtClassInfo[i].Size
                 << " bytes)\n";
    sum += StmtClassInfo[i].Counter * StmtClassInfo[i].Size;
  }

  llvm::errs() << "Total bytes = " << sum << "\n";
}

void Stmt::addStmtClass(StmtClass s) { ++getStmtInfoTableEntry(s).Counter; }

bool Stmt::StatisticsEnabled = false;
void Stmt::EnableStatistics() { StatisticsEnabled = true; }

Stmt *Stmt::IgnoreImplicit() {
  Stmt *s = this;

  if (ExprWithCleanups *ewc = dyn_cast<ExprWithCleanups>(s))
    s = ewc->getSubExpr();

  while (ImplicitCastExpr *ice = dyn_cast<ImplicitCastExpr>(s))
    s = ice->getSubExpr();

  return s;
}

/// \brief Skip no-op (attributed, compound) container stmts and skip captured
/// stmt at the top, if \a IgnoreCaptured is true.
Stmt *Stmt::IgnoreContainers(bool IgnoreCaptured) {
    Stmt *S = this;
    if (IgnoreCaptured)
        if (auto CapS = dyn_cast_or_null<CapturedStmt>(S))
            S = CapS->getCapturedStmt();
    while (true) {
        if (auto AS = dyn_cast_or_null<AttributedStmt>(S))
            S = AS->getSubStmt();
        else if (auto CS = dyn_cast_or_null<CompoundStmt>(S)) {
            if (CS->size() != 1)
                break;
            S = CS->body_back();
        } else
            break;
    }
    return S;
}

/// \brief Strip off all label-like statements.
///
/// This will strip off label statements, case statements, attributed
/// statements and default statements recursively.
const Stmt *Stmt::stripLabelLikeStatements() const {
  const Stmt *S = this;
  while (true) {
    if (const LabelStmt *LS = dyn_cast<LabelStmt>(S))
      S = LS->getSubStmt();
    else if (const SwitchCase *SC = dyn_cast<SwitchCase>(S))
      S = SC->getSubStmt();
    else if (const AttributedStmt *AS = dyn_cast<AttributedStmt>(S))
      S = AS->getSubStmt();
    else
      return S;
  }
}

namespace {
struct good {};
struct bad {};

// These silly little functions have to be static inline to suppress
// unused warnings, and they have to be defined to suppress other
// warnings.
static inline good is_good(good) { return good(); }

typedef Stmt::child_range children_t();
template <class T> good implements_children(children_t T::*) { return good(); }
LLVM_ATTRIBUTE_UNUSED
static inline bad implements_children(children_t Stmt::*) { return bad(); }

typedef SourceLocation getLocStart_t() const;
template <class T> good implements_getLocStart(getLocStart_t T::*) {
  return good();
}
LLVM_ATTRIBUTE_UNUSED
static inline bad implements_getLocStart(getLocStart_t Stmt::*) {
  return bad();
}

typedef SourceLocation getLocEnd_t() const;
template <class T> good implements_getLocEnd(getLocEnd_t T::*) {
  return good();
}
LLVM_ATTRIBUTE_UNUSED
static inline bad implements_getLocEnd(getLocEnd_t Stmt::*) { return bad(); }

#define ASSERT_IMPLEMENTS_children(type)                                       \
  (void) is_good(implements_children(&type::children))
#define ASSERT_IMPLEMENTS_getLocStart(type)                                    \
  (void) is_good(implements_getLocStart(&type::getLocStart))
#define ASSERT_IMPLEMENTS_getLocEnd(type)                                      \
  (void) is_good(implements_getLocEnd(&type::getLocEnd))
}

/// Check whether the various Stmt classes implement their member
/// functions.
LLVM_ATTRIBUTE_UNUSED
static inline void check_implementations() {
#define ABSTRACT_STMT(type)
#define STMT(type, base)                                                       \
  ASSERT_IMPLEMENTS_children(type);                                            \
  ASSERT_IMPLEMENTS_getLocStart(type);                                         \
  ASSERT_IMPLEMENTS_getLocEnd(type);
#include "clang/AST/StmtNodes.inc"
}

Stmt::child_range Stmt::children() {
  switch (getStmtClass()) {
  case Stmt::NoStmtClass:
    llvm_unreachable("statement without class");
#define ABSTRACT_STMT(type)
#define STMT(type, base)                                                       \
  case Stmt::type##Class:                                                      \
    return static_cast<type *>(this)->children();
#include "clang/AST/StmtNodes.inc"
  }
  llvm_unreachable("unknown statement kind!");
}

// Amusing macro metaprogramming hack: check whether a class provides
// a more specific implementation of getSourceRange.
//
// See also Expr.cpp:getExprLoc().
namespace {
/// This implementation is used when a class provides a custom
/// implementation of getSourceRange.
template <class S, class T>
SourceRange getSourceRangeImpl(const Stmt *stmt, SourceRange (T::*v)() const) {
  return static_cast<const S *>(stmt)->getSourceRange();
}

/// This implementation is used when a class doesn't provide a custom
/// implementation of getSourceRange.  Overload resolution should pick it over
/// the implementation above because it's more specialized according to
/// function template partial ordering.
template <class S>
SourceRange getSourceRangeImpl(const Stmt *stmt,
                               SourceRange (Stmt::*v)() const) {
  return SourceRange(static_cast<const S *>(stmt)->getLocStart(),
                     static_cast<const S *>(stmt)->getLocEnd());
}
}

SourceRange Stmt::getSourceRange() const {
  switch (getStmtClass()) {
  case Stmt::NoStmtClass:
    llvm_unreachable("statement without class");
#define ABSTRACT_STMT(type)
#define STMT(type, base)                                                       \
  case Stmt::type##Class:                                                      \
    return getSourceRangeImpl<type>(this, &type::getSourceRange);
#include "clang/AST/StmtNodes.inc"
  }
  llvm_unreachable("unknown statement kind!");
}

SourceLocation Stmt::getLocStart() const {
  //  llvm::errs() << "getLocStart() for " << getStmtClassName() << "\n";
  switch (getStmtClass()) {
  case Stmt::NoStmtClass:
    llvm_unreachable("statement without class");
#define ABSTRACT_STMT(type)
#define STMT(type, base)                                                       \
  case Stmt::type##Class:                                                      \
    return static_cast<const type *>(this)->getLocStart();
#include "clang/AST/StmtNodes.inc"
  }
  llvm_unreachable("unknown statement kind");
}

SourceLocation Stmt::getLocEnd() const {
  switch (getStmtClass()) {
  case Stmt::NoStmtClass:
    llvm_unreachable("statement without class");
#define ABSTRACT_STMT(type)
#define STMT(type, base)                                                       \
  case Stmt::type##Class:                                                      \
    return static_cast<const type *>(this)->getLocEnd();
#include "clang/AST/StmtNodes.inc"
  }
  llvm_unreachable("unknown statement kind");
}

CompoundStmt::CompoundStmt(const ASTContext &C, ArrayRef<Stmt *> Stmts,
                           SourceLocation LB, SourceLocation RB)
    : Stmt(CompoundStmtClass), LBracLoc(LB), RBracLoc(RB) {
  CompoundStmtBits.NumStmts = Stmts.size();
  assert(CompoundStmtBits.NumStmts == Stmts.size() &&
         "NumStmts doesn't fit in bits of CompoundStmtBits.NumStmts!");

  if (Stmts.size() == 0) {
    Body = nullptr;
    return;
  }

  Body = new (C) Stmt *[Stmts.size()];
  std::copy(Stmts.begin(), Stmts.end(), Body);
}

void CompoundStmt::setStmts(const ASTContext &C, Stmt **Stmts,
                            unsigned NumStmts) {
  if (this->Body)
    C.Deallocate(Body);
  this->CompoundStmtBits.NumStmts = NumStmts;

  Body = new (C) Stmt *[NumStmts];
  memcpy(Body, Stmts, sizeof(Stmt *) * NumStmts);
}

const char *LabelStmt::getName() const {
  return getDecl()->getIdentifier()->getNameStart();
}

AttributedStmt *AttributedStmt::Create(const ASTContext &C, SourceLocation Loc,
                                       ArrayRef<const Attr *> Attrs,
                                       Stmt *SubStmt) {
  assert(!Attrs.empty() && "Attrs should not be empty");
  void *Mem = C.Allocate(sizeof(AttributedStmt) + sizeof(Attr *) * Attrs.size(),
                         llvm::alignOf<AttributedStmt>());
  return new (Mem) AttributedStmt(Loc, Attrs, SubStmt);
}

AttributedStmt *AttributedStmt::CreateEmpty(const ASTContext &C,
                                            unsigned NumAttrs) {
  assert(NumAttrs > 0 && "NumAttrs should be greater than zero");
  void *Mem = C.Allocate(sizeof(AttributedStmt) + sizeof(Attr *) * NumAttrs,
                         llvm::alignOf<AttributedStmt>());
  return new (Mem) AttributedStmt(EmptyShell(), NumAttrs);
}

std::string AsmStmt::generateAsmString(const ASTContext &C) const {
  if (const GCCAsmStmt *gccAsmStmt = dyn_cast<GCCAsmStmt>(this))
    return gccAsmStmt->generateAsmString(C);
  if (const MSAsmStmt *msAsmStmt = dyn_cast<MSAsmStmt>(this))
    return msAsmStmt->generateAsmString(C);
  llvm_unreachable("unknown asm statement kind!");
}

StringRef AsmStmt::getOutputConstraint(unsigned i) const {
  if (const GCCAsmStmt *gccAsmStmt = dyn_cast<GCCAsmStmt>(this))
    return gccAsmStmt->getOutputConstraint(i);
  if (const MSAsmStmt *msAsmStmt = dyn_cast<MSAsmStmt>(this))
    return msAsmStmt->getOutputConstraint(i);
  llvm_unreachable("unknown asm statement kind!");
}

const Expr *AsmStmt::getOutputExpr(unsigned i) const {
  if (const GCCAsmStmt *gccAsmStmt = dyn_cast<GCCAsmStmt>(this))
    return gccAsmStmt->getOutputExpr(i);
  if (const MSAsmStmt *msAsmStmt = dyn_cast<MSAsmStmt>(this))
    return msAsmStmt->getOutputExpr(i);
  llvm_unreachable("unknown asm statement kind!");
}

StringRef AsmStmt::getInputConstraint(unsigned i) const {
  if (const GCCAsmStmt *gccAsmStmt = dyn_cast<GCCAsmStmt>(this))
    return gccAsmStmt->getInputConstraint(i);
  if (const MSAsmStmt *msAsmStmt = dyn_cast<MSAsmStmt>(this))
    return msAsmStmt->getInputConstraint(i);
  llvm_unreachable("unknown asm statement kind!");
}

const Expr *AsmStmt::getInputExpr(unsigned i) const {
  if (const GCCAsmStmt *gccAsmStmt = dyn_cast<GCCAsmStmt>(this))
    return gccAsmStmt->getInputExpr(i);
  if (const MSAsmStmt *msAsmStmt = dyn_cast<MSAsmStmt>(this))
    return msAsmStmt->getInputExpr(i);
  llvm_unreachable("unknown asm statement kind!");
}

StringRef AsmStmt::getClobber(unsigned i) const {
  if (const GCCAsmStmt *gccAsmStmt = dyn_cast<GCCAsmStmt>(this))
    return gccAsmStmt->getClobber(i);
  if (const MSAsmStmt *msAsmStmt = dyn_cast<MSAsmStmt>(this))
    return msAsmStmt->getClobber(i);
  llvm_unreachable("unknown asm statement kind!");
}

/// getNumPlusOperands - Return the number of output operands that have a "+"
/// constraint.
unsigned AsmStmt::getNumPlusOperands() const {
  unsigned Res = 0;
  for (unsigned i = 0, e = getNumOutputs(); i != e; ++i)
    if (isOutputPlusConstraint(i))
      ++Res;
  return Res;
}

StringRef GCCAsmStmt::getClobber(unsigned i) const {
  return getClobberStringLiteral(i)->getString();
}

Expr *GCCAsmStmt::getOutputExpr(unsigned i) { return cast<Expr>(Exprs[i]); }

/// getOutputConstraint - Return the constraint string for the specified
/// output operand.  All output constraints are known to be non-empty (either
/// '=' or '+').
StringRef GCCAsmStmt::getOutputConstraint(unsigned i) const {
  return getOutputConstraintLiteral(i)->getString();
}

Expr *GCCAsmStmt::getInputExpr(unsigned i) {
  return cast<Expr>(Exprs[i + NumOutputs]);
}
void GCCAsmStmt::setInputExpr(unsigned i, Expr *E) {
  Exprs[i + NumOutputs] = E;
}

/// getInputConstraint - Return the specified input constraint.  Unlike output
/// constraints, these can be empty.
StringRef GCCAsmStmt::getInputConstraint(unsigned i) const {
  return getInputConstraintLiteral(i)->getString();
}

void GCCAsmStmt::setOutputsAndInputsAndClobbers(
    const ASTContext &C, IdentifierInfo **Names, StringLiteral **Constraints,
    Stmt **Exprs, unsigned NumOutputs, unsigned NumInputs,
    StringLiteral **Clobbers, unsigned NumClobbers) {
  this->NumOutputs = NumOutputs;
  this->NumInputs = NumInputs;
  this->NumClobbers = NumClobbers;

  unsigned NumExprs = NumOutputs + NumInputs;

  C.Deallocate(this->Names);
  this->Names = new (C) IdentifierInfo *[NumExprs];
  std::copy(Names, Names + NumExprs, this->Names);

  C.Deallocate(this->Exprs);
  this->Exprs = new (C) Stmt *[NumExprs];
  std::copy(Exprs, Exprs + NumExprs, this->Exprs);

  C.Deallocate(this->Constraints);
  this->Constraints = new (C) StringLiteral *[NumExprs];
  std::copy(Constraints, Constraints + NumExprs, this->Constraints);

  C.Deallocate(this->Clobbers);
  this->Clobbers = new (C) StringLiteral *[NumClobbers];
  std::copy(Clobbers, Clobbers + NumClobbers, this->Clobbers);
}

/// getNamedOperand - Given a symbolic operand reference like %[foo],
/// translate this into a numeric value needed to reference the same operand.
/// This returns -1 if the operand name is invalid.
int GCCAsmStmt::getNamedOperand(StringRef SymbolicName) const {
  unsigned NumPlusOperands = 0;

  // Check if this is an output operand.
  for (unsigned i = 0, e = getNumOutputs(); i != e; ++i) {
    if (getOutputName(i) == SymbolicName)
      return i;
  }

  for (unsigned i = 0, e = getNumInputs(); i != e; ++i)
    if (getInputName(i) == SymbolicName)
      return getNumOutputs() + NumPlusOperands + i;

  // Not found.
  return -1;
}

/// AnalyzeAsmString - Analyze the asm string of the current asm, decomposing
/// it into pieces.  If the asm string is erroneous, emit errors and return
/// true, otherwise return false.
unsigned GCCAsmStmt::AnalyzeAsmString(SmallVectorImpl<AsmStringPiece> &Pieces,
                                      const ASTContext &C,
                                      unsigned &DiagOffs) const {
  StringRef Str = getAsmString()->getString();
  const char *StrStart = Str.begin();
  const char *StrEnd = Str.end();
  const char *CurPtr = StrStart;

  // "Simple" inline asms have no constraints or operands, just convert the asm
  // string to escape $'s.
  if (isSimple()) {
    std::string Result;
    for (; CurPtr != StrEnd; ++CurPtr) {
      switch (*CurPtr) {
      case '$':
        Result += "$$";
        break;
      default:
        Result += *CurPtr;
        break;
      }
    }
    Pieces.push_back(AsmStringPiece(Result));
    return 0;
  }

  // CurStringPiece - The current string that we are building up as we scan the
  // asm string.
  std::string CurStringPiece;

  bool HasVariants = !C.getTargetInfo().hasNoAsmVariants();

  while (1) {
    // Done with the string?
    if (CurPtr == StrEnd) {
      if (!CurStringPiece.empty())
        Pieces.push_back(AsmStringPiece(CurStringPiece));
      return 0;
    }

    char CurChar = *CurPtr++;
    switch (CurChar) {
    case '$':
      CurStringPiece += "$$";
      continue;
    case '{':
      CurStringPiece += (HasVariants ? "$(" : "{");
      continue;
    case '|':
      CurStringPiece += (HasVariants ? "$|" : "|");
      continue;
    case '}':
      CurStringPiece += (HasVariants ? "$)" : "}");
      continue;
    case '%':
      break;
    default:
      CurStringPiece += CurChar;
      continue;
    }

    // Escaped "%" character in asm string.
    if (CurPtr == StrEnd) {
      // % at end of string is invalid (no escape).
      DiagOffs = CurPtr - StrStart - 1;
      return diag::err_asm_invalid_escape;
    }

    char EscapedChar = *CurPtr++;
    if (EscapedChar == '%') { // %% -> %
      // Escaped percentage sign.
      CurStringPiece += '%';
      continue;
    }

    if (EscapedChar == '=') { // %= -> Generate an unique ID.
      CurStringPiece += "${:uid}";
      continue;
    }

    // Otherwise, we have an operand.  If we have accumulated a string so far,
    // add it to the Pieces list.
    if (!CurStringPiece.empty()) {
      Pieces.push_back(AsmStringPiece(CurStringPiece));
      CurStringPiece.clear();
    }

    // Handle %x4 and %x[foo] by capturing x as the modifier character.
    char Modifier = '\0';
    if (isLetter(EscapedChar)) {
      if (CurPtr == StrEnd) { // Premature end.
        DiagOffs = CurPtr - StrStart - 1;
        return diag::err_asm_invalid_escape;
      }
      Modifier = EscapedChar;
      EscapedChar = *CurPtr++;
    }

    if (isDigit(EscapedChar)) {
      // %n - Assembler operand n
      unsigned N = 0;

      --CurPtr;
      while (CurPtr != StrEnd && isDigit(*CurPtr))
        N = N * 10 + ((*CurPtr++) - '0');

      unsigned NumOperands =
          getNumOutputs() + getNumPlusOperands() + getNumInputs();
      if (N >= NumOperands) {
        DiagOffs = CurPtr - StrStart - 1;
        return diag::err_asm_invalid_operand_number;
      }

      Pieces.push_back(AsmStringPiece(N, Modifier));
      continue;
    }

    // Handle %[foo], a symbolic operand reference.
    if (EscapedChar == '[') {
      DiagOffs = CurPtr - StrStart - 1;

      // Find the ']'.
      const char *NameEnd = (const char*)memchr(CurPtr, ']', StrEnd-CurPtr);
      if (NameEnd == nullptr)
        return diag::err_asm_unterminated_symbolic_operand_name;
      if (NameEnd == CurPtr)
        return diag::err_asm_empty_symbolic_operand_name;

      StringRef SymbolicName(CurPtr, NameEnd - CurPtr);

      int N = getNamedOperand(SymbolicName);
      if (N == -1) {
        // Verify that an operand with that name exists.
        DiagOffs = CurPtr - StrStart;
        return diag::err_asm_unknown_symbolic_operand_name;
      }
      Pieces.push_back(AsmStringPiece(N, Modifier));

      CurPtr = NameEnd + 1;
      continue;
    }

    DiagOffs = CurPtr - StrStart - 1;
    return diag::err_asm_invalid_escape;
  }
}

/// Assemble final IR asm string (GCC-style).
std::string GCCAsmStmt::generateAsmString(const ASTContext &C) const {
  // Analyze the asm string to decompose it into its pieces.  We know that Sema
  // has already done this, so it is guaranteed to be successful.
  SmallVector<GCCAsmStmt::AsmStringPiece, 4> Pieces;
  unsigned DiagOffs;
  AnalyzeAsmString(Pieces, C, DiagOffs);

  std::string AsmString;
  for (unsigned i = 0, e = Pieces.size(); i != e; ++i) {
    if (Pieces[i].isString())
      AsmString += Pieces[i].getString();
    else if (Pieces[i].getModifier() == '\0')
      AsmString += '$' + llvm::utostr(Pieces[i].getOperandNo());
    else
      AsmString += "${" + llvm::utostr(Pieces[i].getOperandNo()) + ':' +
                   Pieces[i].getModifier() + '}';
  }
  return AsmString;
}

/// Assemble final IR asm string (MS-style).
std::string MSAsmStmt::generateAsmString(const ASTContext &C) const {
  // FIXME: This needs to be translated into the IR string representation.
  return AsmStr;
}

Expr *MSAsmStmt::getOutputExpr(unsigned i) { return cast<Expr>(Exprs[i]); }

Expr *MSAsmStmt::getInputExpr(unsigned i) {
  return cast<Expr>(Exprs[i + NumOutputs]);
}
void MSAsmStmt::setInputExpr(unsigned i, Expr *E) { Exprs[i + NumOutputs] = E; }

QualType CXXCatchStmt::getCaughtType() const {
  if (ExceptionDecl)
    return ExceptionDecl->getType();
  return QualType();
}

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

GCCAsmStmt::GCCAsmStmt(const ASTContext &C, SourceLocation asmloc,
                       bool issimple, bool isvolatile, unsigned numoutputs,
                       unsigned numinputs, IdentifierInfo **names,
                       StringLiteral **constraints, Expr **exprs,
                       StringLiteral *asmstr, unsigned numclobbers,
                       StringLiteral **clobbers, SourceLocation rparenloc)
    : AsmStmt(GCCAsmStmtClass, asmloc, issimple, isvolatile, numoutputs,
              numinputs, numclobbers),
      RParenLoc(rparenloc), AsmStr(asmstr) {

  unsigned NumExprs = NumOutputs + NumInputs;

  Names = new (C) IdentifierInfo *[NumExprs];
  std::copy(names, names + NumExprs, Names);

  Exprs = new (C) Stmt *[NumExprs];
  std::copy(exprs, exprs + NumExprs, Exprs);

  Constraints = new (C) StringLiteral *[NumExprs];
  std::copy(constraints, constraints + NumExprs, Constraints);

  Clobbers = new (C) StringLiteral *[NumClobbers];
  std::copy(clobbers, clobbers + NumClobbers, Clobbers);
}

MSAsmStmt::MSAsmStmt(const ASTContext &C, SourceLocation asmloc,
                     SourceLocation lbraceloc, bool issimple, bool isvolatile,
                     ArrayRef<Token> asmtoks, unsigned numoutputs,
                     unsigned numinputs, ArrayRef<StringRef> constraints,
                     ArrayRef<Expr *> exprs, StringRef asmstr,
                     ArrayRef<StringRef> clobbers, SourceLocation endloc)
    : AsmStmt(MSAsmStmtClass, asmloc, issimple, isvolatile, numoutputs,
              numinputs, clobbers.size()),
      LBraceLoc(lbraceloc), EndLoc(endloc), NumAsmToks(asmtoks.size()) {

  initialize(C, asmstr, asmtoks, constraints, exprs, clobbers);
}

static StringRef copyIntoContext(const ASTContext &C, StringRef str) {
  size_t size = str.size();
  char *buffer = new (C) char[size];
  memcpy(buffer, str.data(), size);
  return StringRef(buffer, size);
}

void MSAsmStmt::initialize(const ASTContext &C, StringRef asmstr,
                           ArrayRef<Token> asmtoks,
                           ArrayRef<StringRef> constraints,
                           ArrayRef<Expr *> exprs,
                           ArrayRef<StringRef> clobbers) {
  assert(NumAsmToks == asmtoks.size());
  assert(NumClobbers == clobbers.size());

  unsigned NumExprs = exprs.size();
  assert(NumExprs == NumOutputs + NumInputs);
  assert(NumExprs == constraints.size());

  AsmStr = copyIntoContext(C, asmstr);

  Exprs = new (C) Stmt *[NumExprs];
  for (unsigned i = 0, e = NumExprs; i != e; ++i)
    Exprs[i] = exprs[i];

  AsmToks = new (C) Token[NumAsmToks];
  for (unsigned i = 0, e = NumAsmToks; i != e; ++i)
    AsmToks[i] = asmtoks[i];

  Constraints = new (C) StringRef[NumExprs];
  for (unsigned i = 0, e = NumExprs; i != e; ++i) {
    Constraints[i] = copyIntoContext(C, constraints[i]);
  }

  Clobbers = new (C) StringRef[NumClobbers];
  for (unsigned i = 0, e = NumClobbers; i != e; ++i) {
    // FIXME: Avoid the allocation/copy if at all possible.
    Clobbers[i] = copyIntoContext(C, clobbers[i]);
  }
}

ObjCForCollectionStmt::ObjCForCollectionStmt(Stmt *Elem, Expr *Collect,
                                             Stmt *Body, SourceLocation FCL,
                                             SourceLocation RPL)
    : Stmt(ObjCForCollectionStmtClass) {
  SubExprs[ELEM] = Elem;
  SubExprs[COLLECTION] = Collect;
  SubExprs[BODY] = Body;
  ForLoc = FCL;
  RParenLoc = RPL;
}

ObjCAtTryStmt::ObjCAtTryStmt(SourceLocation atTryLoc, Stmt *atTryStmt,
                             Stmt **CatchStmts, unsigned NumCatchStmts,
                             Stmt *atFinallyStmt)
  : Stmt(ObjCAtTryStmtClass), AtTryLoc(atTryLoc),
    NumCatchStmts(NumCatchStmts), HasFinally(atFinallyStmt != nullptr) {
  Stmt **Stmts = getStmts();
  Stmts[0] = atTryStmt;
  for (unsigned I = 0; I != NumCatchStmts; ++I)
    Stmts[I + 1] = CatchStmts[I];

  if (HasFinally)
    Stmts[NumCatchStmts + 1] = atFinallyStmt;
}

ObjCAtTryStmt *ObjCAtTryStmt::Create(const ASTContext &Context,
                                     SourceLocation atTryLoc, Stmt *atTryStmt,
                                     Stmt **CatchStmts, unsigned NumCatchStmts,
                                     Stmt *atFinallyStmt) {
  unsigned Size = sizeof(ObjCAtTryStmt) +
    (1 + NumCatchStmts + (atFinallyStmt != nullptr)) * sizeof(Stmt *);
  void *Mem = Context.Allocate(Size, llvm::alignOf<ObjCAtTryStmt>());
  return new (Mem) ObjCAtTryStmt(atTryLoc, atTryStmt, CatchStmts, NumCatchStmts,
                                 atFinallyStmt);
}

ObjCAtTryStmt *ObjCAtTryStmt::CreateEmpty(const ASTContext &Context,
                                          unsigned NumCatchStmts,
                                          bool HasFinally) {
  unsigned Size =
      sizeof(ObjCAtTryStmt) + (1 + NumCatchStmts + HasFinally) * sizeof(Stmt *);
  void *Mem = Context.Allocate(Size, llvm::alignOf<ObjCAtTryStmt>());
  return new (Mem) ObjCAtTryStmt(EmptyShell(), NumCatchStmts, HasFinally);
}

SourceLocation ObjCAtTryStmt::getLocEnd() const {
  if (HasFinally)
    return getFinallyStmt()->getLocEnd();
  if (NumCatchStmts)
    return getCatchStmt(NumCatchStmts - 1)->getLocEnd();
  return getTryBody()->getLocEnd();
}

CXXTryStmt *CXXTryStmt::Create(const ASTContext &C, SourceLocation tryLoc,
                               Stmt *tryBlock, ArrayRef<Stmt *> handlers) {
  std::size_t Size = sizeof(CXXTryStmt);
  Size += ((handlers.size() + 1) * sizeof(Stmt));

  void *Mem = C.Allocate(Size, llvm::alignOf<CXXTryStmt>());
  return new (Mem) CXXTryStmt(tryLoc, tryBlock, handlers);
}

CXXTryStmt *CXXTryStmt::Create(const ASTContext &C, EmptyShell Empty,
                               unsigned numHandlers) {
  std::size_t Size = sizeof(CXXTryStmt);
  Size += ((numHandlers + 1) * sizeof(Stmt));

  void *Mem = C.Allocate(Size, llvm::alignOf<CXXTryStmt>());
  return new (Mem) CXXTryStmt(Empty, numHandlers);
}

CXXTryStmt::CXXTryStmt(SourceLocation tryLoc, Stmt *tryBlock,
                       ArrayRef<Stmt *> handlers)
    : Stmt(CXXTryStmtClass), TryLoc(tryLoc), NumHandlers(handlers.size()) {
  Stmt **Stmts = reinterpret_cast<Stmt **>(this + 1);
  Stmts[0] = tryBlock;
  std::copy(handlers.begin(), handlers.end(), Stmts + 1);
}

CXXForRangeStmt::CXXForRangeStmt(DeclStmt *Range, DeclStmt *BeginEndStmt,
                                 Expr *Cond, Expr *Inc, DeclStmt *LoopVar,
                                 Stmt *Body, SourceLocation FL,
                                 SourceLocation CL, SourceLocation RPL)
    : Stmt(CXXForRangeStmtClass), ForLoc(FL), ColonLoc(CL), RParenLoc(RPL) {
  SubExprs[RANGE] = Range;
  SubExprs[BEGINEND] = BeginEndStmt;
  SubExprs[COND] = Cond;
  SubExprs[INC] = Inc;
  SubExprs[LOOPVAR] = LoopVar;
  SubExprs[BODY] = Body;
}

Expr *CXXForRangeStmt::getRangeInit() {
  DeclStmt *RangeStmt = getRangeStmt();
  VarDecl *RangeDecl = dyn_cast_or_null<VarDecl>(RangeStmt->getSingleDecl());
  assert(RangeDecl && "for-range should have a single var decl");
  return RangeDecl->getInit();
}

const Expr *CXXForRangeStmt::getRangeInit() const {
  return const_cast<CXXForRangeStmt *>(this)->getRangeInit();
}

VarDecl *CXXForRangeStmt::getLoopVariable() {
  Decl *LV = cast<DeclStmt>(getLoopVarStmt())->getSingleDecl();
  assert(LV && "No loop variable in CXXForRangeStmt");
  return cast<VarDecl>(LV);
}

const VarDecl *CXXForRangeStmt::getLoopVariable() const {
  return const_cast<CXXForRangeStmt *>(this)->getLoopVariable();
}

IfStmt::IfStmt(const ASTContext &C, SourceLocation IL, VarDecl *var, Expr *cond,
               Stmt *then, SourceLocation EL, Stmt *elsev)
    : Stmt(IfStmtClass), IfLoc(IL), ElseLoc(EL) {
  setConditionVariable(C, var);
  SubExprs[COND] = cond;
  SubExprs[THEN] = then;
  SubExprs[ELSE] = elsev;
}

VarDecl *IfStmt::getConditionVariable() const {
  if (!SubExprs[VAR])
    return nullptr;

  DeclStmt *DS = cast<DeclStmt>(SubExprs[VAR]);
  return cast<VarDecl>(DS->getSingleDecl());
}

void IfStmt::setConditionVariable(const ASTContext &C, VarDecl *V) {
  if (!V) {
    SubExprs[VAR] = nullptr;
    return;
  }

  SourceRange VarRange = V->getSourceRange();
  SubExprs[VAR] =
      new (C) DeclStmt(DeclGroupRef(V), VarRange.getBegin(), VarRange.getEnd());
}

ForStmt::ForStmt(const ASTContext &C, Stmt *Init, Expr *Cond, VarDecl *condVar,
                 Expr *Inc, Stmt *Body, SourceLocation FL, SourceLocation LP,
                 SourceLocation RP)
    : Stmt(ForStmtClass), ForLoc(FL), LParenLoc(LP), RParenLoc(RP) {
  SubExprs[INIT] = Init;
  setConditionVariable(C, condVar);
  SubExprs[COND] = Cond;
  SubExprs[INC] = Inc;
  SubExprs[BODY] = Body;
}

VarDecl *ForStmt::getConditionVariable() const {
  if (!SubExprs[CONDVAR])
    return nullptr;

  DeclStmt *DS = cast<DeclStmt>(SubExprs[CONDVAR]);
  return cast<VarDecl>(DS->getSingleDecl());
}

void ForStmt::setConditionVariable(const ASTContext &C, VarDecl *V) {
  if (!V) {
    SubExprs[CONDVAR] = nullptr;
    return;
  }

  SourceRange VarRange = V->getSourceRange();
  SubExprs[CONDVAR] =
      new (C) DeclStmt(DeclGroupRef(V), VarRange.getBegin(), VarRange.getEnd());
}

SwitchStmt::SwitchStmt(const ASTContext &C, VarDecl *Var, Expr *cond)
  : Stmt(SwitchStmtClass), FirstCase(nullptr), AllEnumCasesCovered(0)
{
  setConditionVariable(C, Var);
  SubExprs[COND] = cond;
  SubExprs[BODY] = nullptr;
}

VarDecl *SwitchStmt::getConditionVariable() const {
  if (!SubExprs[VAR])
    return nullptr;

  DeclStmt *DS = cast<DeclStmt>(SubExprs[VAR]);
  return cast<VarDecl>(DS->getSingleDecl());
}

void SwitchStmt::setConditionVariable(const ASTContext &C, VarDecl *V) {
  if (!V) {
    SubExprs[VAR] = nullptr;
    return;
  }

  SourceRange VarRange = V->getSourceRange();
  SubExprs[VAR] =
      new (C) DeclStmt(DeclGroupRef(V), VarRange.getBegin(), VarRange.getEnd());
}

Stmt *SwitchCase::getSubStmt() {
  if (isa<CaseStmt>(this))
    return cast<CaseStmt>(this)->getSubStmt();
  return cast<DefaultStmt>(this)->getSubStmt();
}

WhileStmt::WhileStmt(const ASTContext &C, VarDecl *Var, Expr *cond, Stmt *body,
                     SourceLocation WL)
    : Stmt(WhileStmtClass) {
  setConditionVariable(C, Var);
  SubExprs[COND] = cond;
  SubExprs[BODY] = body;
  WhileLoc = WL;
}

VarDecl *WhileStmt::getConditionVariable() const {
  if (!SubExprs[VAR])
    return nullptr;

  DeclStmt *DS = cast<DeclStmt>(SubExprs[VAR]);
  return cast<VarDecl>(DS->getSingleDecl());
}

void WhileStmt::setConditionVariable(const ASTContext &C, VarDecl *V) {
  if (!V) {
    SubExprs[VAR] = nullptr;
    return;
  }

  SourceRange VarRange = V->getSourceRange();
  SubExprs[VAR] = new (C) DeclStmt(DeclGroupRef(V), VarRange.getBegin(),
                                   VarRange.getEnd());
}

// IndirectGotoStmt
LabelDecl *IndirectGotoStmt::getConstantTarget() {
  if (AddrLabelExpr *E =
        dyn_cast<AddrLabelExpr>(getTarget()->IgnoreParenImpCasts()))
    return E->getLabel();
  return nullptr;
}

// ReturnStmt
const Expr* ReturnStmt::getRetValue() const {
  return cast_or_null<Expr>(RetExpr);
}
Expr* ReturnStmt::getRetValue() {
  return cast_or_null<Expr>(RetExpr);
}

SEHTryStmt::SEHTryStmt(bool IsCXXTry, SourceLocation TryLoc, Stmt *TryBlock,
                       Stmt *Handler, int HandlerIndex, int HandlerParentIndex)
    : Stmt(SEHTryStmtClass), IsCXXTry(IsCXXTry), TryLoc(TryLoc),
      HandlerIndex(HandlerIndex), HandlerParentIndex(HandlerParentIndex) {
  Children[TRY] = TryBlock;
  Children[HANDLER] = Handler;
}

SEHTryStmt *SEHTryStmt::Create(const ASTContext &C, bool IsCXXTry,
                               SourceLocation TryLoc, Stmt *TryBlock,
                               Stmt *Handler, int HandlerIndex,
                               int HandlerParentIndex) {
  return new (C) SEHTryStmt(IsCXXTry, TryLoc, TryBlock, Handler, HandlerIndex,
                            HandlerParentIndex);
}

SEHExceptStmt* SEHTryStmt::getExceptHandler() const {
  return dyn_cast<SEHExceptStmt>(getHandler());
}

SEHFinallyStmt* SEHTryStmt::getFinallyHandler() const {
  return dyn_cast<SEHFinallyStmt>(getHandler());
}

SEHExceptStmt::SEHExceptStmt(SourceLocation Loc,
                             Expr *FilterExpr,
                             Stmt *Block)
  : Stmt(SEHExceptStmtClass),
    Loc(Loc)
{
  Children[FILTER_EXPR] = FilterExpr;
  Children[BLOCK]       = Block;
}

SEHExceptStmt* SEHExceptStmt::Create(const ASTContext &C, SourceLocation Loc,
                                     Expr *FilterExpr, Stmt *Block) {
  return new(C) SEHExceptStmt(Loc,FilterExpr,Block);
}

SEHFinallyStmt::SEHFinallyStmt(SourceLocation Loc,
                               Stmt *Block)
  : Stmt(SEHFinallyStmtClass),
    Loc(Loc),
    Block(Block)
{}

SEHFinallyStmt* SEHFinallyStmt::Create(const ASTContext &C, SourceLocation Loc,
                                       Stmt *Block) {
  return new(C)SEHFinallyStmt(Loc,Block);
}

StmtRange OMPClause::children() {
  switch(getClauseKind()) {
  default : break;
#define OPENMP_CLAUSE(Name, Class)                                       \
  case OMPC_ ## Name : return static_cast<Class *>(this)->children();
#include "clang/Basic/OpenMPKinds.def"
  }
  llvm_unreachable("unknown OMPClause");
}

void OMPPrivateClause::setDefaultInits(ArrayRef<Expr *> DefaultInits) {
  assert(DefaultInits.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(DefaultInits.begin(), DefaultInits.end(), varlist_end());
}

OMPPrivateClause *OMPPrivateClause::Create(const ASTContext &C,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc,
                                           ArrayRef<Expr *> VL,
                                           ArrayRef<Expr *> DefaultInits) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPPrivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * 2 * VL.size());
  OMPPrivateClause *Clause =
      new (Mem) OMPPrivateClause(StartLoc, EndLoc, VL.size());
  Clause->setVars(VL);
  Clause->setDefaultInits(DefaultInits);
  return Clause;
}

OMPPrivateClause *OMPPrivateClause::CreateEmpty(const ASTContext &C,
                                                unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPPrivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * 2 * N);
  return new (Mem) OMPPrivateClause(N);
}

void OMPFirstPrivateClause::setPseudoVars(ArrayRef<DeclRefExpr *> PseudoVars) {
  assert(PseudoVars.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(PseudoVars.begin(), PseudoVars.end(), varlist_end());
}

void OMPFirstPrivateClause::setInits(ArrayRef<Expr *> Inits) {
  assert(Inits.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(Inits.begin(), Inits.end(), getPseudoVars().end());
}

OMPFirstPrivateClause *
OMPFirstPrivateClause::Create(const ASTContext &C, SourceLocation StartLoc,
                              SourceLocation EndLoc, ArrayRef<Expr *> VL,
                              ArrayRef<DeclRefExpr *> PseudoVars,
                              ArrayRef<Expr *> Inits) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFirstPrivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size() * 3);
  OMPFirstPrivateClause *Clause =
      new (Mem) OMPFirstPrivateClause(StartLoc, EndLoc, VL.size());
  Clause->setVars(VL);
  Clause->setPseudoVars(PseudoVars);
  Clause->setInits(Inits);
  return Clause;
}

OMPFirstPrivateClause *OMPFirstPrivateClause::CreateEmpty(const ASTContext &C,
                                                          unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFirstPrivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N * 3);
  return new (Mem) OMPFirstPrivateClause(N);
}

void OMPLastPrivateClause::setPseudoVars1(ArrayRef<DeclRefExpr *> PseudoVars) {
  assert(PseudoVars.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(PseudoVars.begin(), PseudoVars.end(), varlist_end());
}

void OMPLastPrivateClause::setPseudoVars2(ArrayRef<DeclRefExpr *> PseudoVars) {
  assert(PseudoVars.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(PseudoVars.begin(), PseudoVars.end(), getPseudoVars1().end());
}

void OMPLastPrivateClause::setDefaultInits(ArrayRef<Expr *> DefaultInits) {
  assert(DefaultInits.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(DefaultInits.begin(), DefaultInits.end(), getPseudoVars2().end());
}

void OMPLastPrivateClause::setAssignments(ArrayRef<Expr *> Assignments) {
  assert(Assignments.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(Assignments.begin(), Assignments.end(), getDefaultInits().end());
}

OMPLastPrivateClause *OMPLastPrivateClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<Expr *> VL, ArrayRef<DeclRefExpr *> PseudoVars1,
    ArrayRef<DeclRefExpr *> PseudoVars2, ArrayRef<Expr *> Assignments) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPLastPrivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size() * 5);
  OMPLastPrivateClause *Clause =
      new (Mem) OMPLastPrivateClause(StartLoc, EndLoc, VL.size());
  Clause->setVars(VL);
  Clause->setPseudoVars1(PseudoVars1);
  Clause->setPseudoVars2(PseudoVars2);
  Clause->setAssignments(Assignments);
  llvm::SmallVector<Expr *, 8> DefaultInits(VL.size(), 0);
  Clause->setDefaultInits(DefaultInits);
  return Clause;
}

OMPLastPrivateClause *OMPLastPrivateClause::CreateEmpty(const ASTContext &C,
                                                        unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPLastPrivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N * 5);
  return new (Mem) OMPLastPrivateClause(N);
}

OMPSharedClause *OMPSharedClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSharedClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size());
  OMPSharedClause *Clause =
      new (Mem) OMPSharedClause(StartLoc, EndLoc, VL.size());
  Clause->setVars(VL);
  return Clause;
}

OMPSharedClause *OMPSharedClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSharedClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N);
  return new (Mem) OMPSharedClause(N);
}

void OMPCopyinClause::setPseudoVars1(ArrayRef<DeclRefExpr *> PseudoVars) {
  assert(PseudoVars.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(PseudoVars.begin(), PseudoVars.end(), varlist_end());
}

void OMPCopyinClause::setPseudoVars2(ArrayRef<DeclRefExpr *> PseudoVars) {
  assert(PseudoVars.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(PseudoVars.begin(), PseudoVars.end(), getPseudoVars1().end());
}

void OMPCopyinClause::setAssignments(ArrayRef<Expr *> Assignments) {
  assert(Assignments.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(Assignments.begin(), Assignments.end(), getPseudoVars2().end());
}

OMPCopyinClause *OMPCopyinClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<Expr *> VL, ArrayRef<DeclRefExpr *> PseudoVars1,
    ArrayRef<DeclRefExpr *> PseudoVars2, ArrayRef<Expr *> Assignments) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCopyinClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size() * 4);
  OMPCopyinClause *Clause =
      new (Mem) OMPCopyinClause(StartLoc, EndLoc, VL.size());
  Clause->setVars(VL);
  Clause->setPseudoVars1(PseudoVars1);
  Clause->setPseudoVars2(PseudoVars2);
  Clause->setAssignments(Assignments);
  return Clause;
}

OMPCopyinClause *OMPCopyinClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCopyinClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N * 4);
  return new (Mem) OMPCopyinClause(N);
}

void OMPCopyPrivateClause::setPseudoVars1(ArrayRef<DeclRefExpr *> PseudoVars) {
  assert(PseudoVars.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(PseudoVars.begin(), PseudoVars.end(),
            varlist_end());
}

void OMPCopyPrivateClause::setPseudoVars2(ArrayRef<DeclRefExpr *> PseudoVars) {
  assert(PseudoVars.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(PseudoVars.begin(), PseudoVars.end(),
            getPseudoVars1().end());
}

void OMPCopyPrivateClause::setAssignments(ArrayRef<Expr *> Assignments) {
  assert(Assignments.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(Assignments.begin(), Assignments.end(),
            getPseudoVars2().end());
}

OMPCopyPrivateClause *OMPCopyPrivateClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<Expr *> VL, ArrayRef<DeclRefExpr *> PseudoVars1,
    ArrayRef<DeclRefExpr *> PseudoVars2, ArrayRef<Expr *> Assignments) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCopyPrivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size() * 4);
  OMPCopyPrivateClause *Clause =
      new (Mem) OMPCopyPrivateClause(StartLoc, EndLoc, VL.size());
  Clause->setVars(VL);
  Clause->setPseudoVars1(PseudoVars1);
  Clause->setPseudoVars2(PseudoVars2);
  Clause->setAssignments(Assignments);
  return Clause;
}

OMPCopyPrivateClause *OMPCopyPrivateClause::CreateEmpty(const ASTContext &C,
                                                        unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCopyPrivateClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N * 4);
  return new (Mem) OMPCopyPrivateClause(N);
}

OMPReductionClause *OMPReductionClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<Expr *> VL, ArrayRef<Expr *> OpExprs,
    ArrayRef<Expr *> HelperParams1, ArrayRef<Expr *> HelperParams2,
    ArrayRef<Expr *> DefaultInits, OpenMPReductionClauseOperator Op,
    NestedNameSpecifierLoc S, DeclarationNameInfo OpName) {
  assert(VL.size() == OpExprs.size() &&
         "Number of expressions is not the same as number of variables!");
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPReductionClause),
                                                  llvm::alignOf<Expr *>()) +
                         5 * sizeof(Expr *) * VL.size());
  OMPReductionClause *Clause =
      new (Mem) OMPReductionClause(StartLoc, EndLoc, VL.size(), Op, S, OpName);
  Clause->setVars(VL);
  Clause->setOpExprs(OpExprs);
  Clause->setHelperParameters1st(HelperParams1);
  Clause->setHelperParameters2nd(HelperParams2);
  Clause->setDefaultInits(DefaultInits);
  return Clause;
}

OMPReductionClause *OMPReductionClause::CreateEmpty(const ASTContext &C,
                                                    unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPReductionClause),
                                                  llvm::alignOf<Expr *>()) +
                         5 * sizeof(Expr *) * N);
  return new (Mem) OMPReductionClause(N);
}

void OMPReductionClause::setOpExprs(ArrayRef<Expr *> OpExprs) {
  assert(OpExprs.size() == numberOfVariables() &&
         "Number of expressions is not the same as the number of variables.");
  std::copy(OpExprs.begin(), OpExprs.end(), varlist_end());
}

void OMPReductionClause::setHelperParameters1st(ArrayRef<Expr *> HelperParams) {
  assert(HelperParams.size() == numberOfVariables() &&
         "Number of expressions is not the same as the number of variables.");
  std::copy(HelperParams.begin(), HelperParams.end(), getOpExprs().end());
}

void OMPReductionClause::setHelperParameters2nd(ArrayRef<Expr *> HelperParams) {
  assert(HelperParams.size() == numberOfVariables() &&
         "Number of expressions is not the same as the number of variables.");
  std::copy(HelperParams.begin(), HelperParams.end(),
            getHelperParameters1st().end());
}

void OMPReductionClause::setDefaultInits(ArrayRef<Expr *> DefaultInits) {
  assert(DefaultInits.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(DefaultInits.begin(), DefaultInits.end(),
            getHelperParameters2nd().end());
}

OMPScanClause *OMPScanClause::Create(
        const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
        ArrayRef<Expr *> VL, ArrayRef<Expr *> OpExprs,
        ArrayRef<Expr *> HelperParams1, ArrayRef<Expr *> HelperParams2,
        ArrayRef<Expr *> DefaultInits, OpenMPScanClauseOperator Op,
        NestedNameSpecifierLoc S, DeclarationNameInfo OpName) {
    assert(VL.size() == OpExprs.size() &&
           "Number of expressions is not the same as number of variables!");
    void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPScanClause),
                                                    llvm::alignOf<Expr *>()) +
                           5 * sizeof(Expr *) * VL.size());
    OMPScanClause *Clause =
            new(Mem) OMPScanClause(StartLoc, EndLoc, VL.size(), Op, S, OpName);
    Clause->setVars(VL);
    Clause->setOpExprs(OpExprs);
    Clause->setHelperParameters1st(HelperParams1);
    Clause->setHelperParameters2nd(HelperParams2);
    Clause->setDefaultInits(DefaultInits);
    return Clause;
}

OMPScanClause *OMPScanClause::CreateEmpty(const ASTContext &C,
                                          unsigned N) {
    void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPScanClause),
                                                    llvm::alignOf<Expr *>()) +
                           5 * sizeof(Expr *) * N);
    return new(Mem) OMPScanClause(N);
}

void OMPScanClause::setOpExprs(ArrayRef<Expr *> OpExprs) {
    assert(OpExprs.size() == numberOfVariables() &&
           "Number of expressions is not the same as the number of variables.");
    std::copy(OpExprs.begin(), OpExprs.end(), varlist_end());
}

void OMPScanClause::setHelperParameters1st(ArrayRef<Expr *> HelperParams) {
    assert(HelperParams.size() == numberOfVariables() &&
           "Number of expressions is not the same as the number of variables.");
    std::copy(HelperParams.begin(), HelperParams.end(), getOpExprs().end());
}

void OMPScanClause::setHelperParameters2nd(ArrayRef<Expr *> HelperParams) {
    assert(HelperParams.size() == numberOfVariables() &&
           "Number of expressions is not the same as the number of variables.");
    std::copy(HelperParams.begin(), HelperParams.end(),
              getHelperParameters1st().end());
}

void OMPScanClause::setDefaultInits(ArrayRef<Expr *> DefaultInits) {
    assert(DefaultInits.size() == varlist_size() &&
           "Number of inits is not the same as the preallocated buffer");
    std::copy(DefaultInits.begin(), DefaultInits.end(),
              getHelperParameters2nd().end());
}

void
OMPMapClause::setWholeStartAddresses(ArrayRef<Expr *> WholeStartAddresses) {
  assert(WholeStartAddresses.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(WholeStartAddresses.begin(), WholeStartAddresses.end(),
            varlist_end());
}

void OMPMapClause::setWholeSizesEndAddresses(
    ArrayRef<Expr *> WholeSizesEndAddresses) {
  assert(WholeSizesEndAddresses.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(WholeSizesEndAddresses.begin(), WholeSizesEndAddresses.end(),
            getWholeStartAddresses().end());
}

void
OMPMapClause::setCopyingStartAddresses(ArrayRef<Expr *> CopyingStartAddresses) {
  assert(CopyingStartAddresses.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(CopyingStartAddresses.begin(), CopyingStartAddresses.end(),
            getWholeSizesEndAddresses().end());
}

void OMPMapClause::setCopyingSizesEndAddresses(
    ArrayRef<Expr *> CopyingSizesEndAddresses) {
  assert(CopyingSizesEndAddresses.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(CopyingSizesEndAddresses.begin(), CopyingSizesEndAddresses.end(),
            getCopyingStartAddresses().end());
}

OMPMapClause *OMPMapClause::Create(const ASTContext &C, SourceLocation StartLoc,
                                   SourceLocation EndLoc, ArrayRef<Expr *> VL,
                                   ArrayRef<Expr *> WholeStartAddresses,
                                   ArrayRef<Expr *> WholeSizesEndAddresses,
                                   ArrayRef<Expr *> CopyingStartAddresses,
                                   ArrayRef<Expr *> CopyingSizesEndAddresses,
                                   OpenMPMapClauseKind Kind,
                                   SourceLocation KindLoc) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPMapClause), llvm::alignOf<Expr *>()) +
      sizeof(Expr *) * VL.size() * 5);
  OMPMapClause *Clause =
      new (Mem) OMPMapClause(StartLoc, EndLoc, VL.size(), Kind, KindLoc);
  Clause->setVars(VL);
  Clause->setWholeStartAddresses(WholeStartAddresses);
  Clause->setWholeSizesEndAddresses(WholeSizesEndAddresses);
  Clause->setCopyingStartAddresses(CopyingStartAddresses);
  Clause->setCopyingSizesEndAddresses(CopyingSizesEndAddresses);
  Clause->setKind(Kind);
  Clause->setKindLoc(KindLoc);
  return Clause;
}

OMPMapClause *OMPMapClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPMapClause), llvm::alignOf<Expr *>()) +
      sizeof(Expr *) * N * 5);
  return new (Mem) OMPMapClause(N);
}

void OMPToClause::setWholeStartAddresses(ArrayRef<Expr *> WholeStartAddresses) {
  assert(WholeStartAddresses.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(WholeStartAddresses.begin(), WholeStartAddresses.end(),
            varlist_end());
}

void OMPToClause::setWholeSizesEndAddresses(
    ArrayRef<Expr *> WholeSizesEndAddresses) {
  assert(WholeSizesEndAddresses.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(WholeSizesEndAddresses.begin(), WholeSizesEndAddresses.end(),
            getWholeStartAddresses().end());
}

void
OMPToClause::setCopyingStartAddresses(ArrayRef<Expr *> CopyingStartAddresses) {
  assert(CopyingStartAddresses.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(CopyingStartAddresses.begin(), CopyingStartAddresses.end(),
            getWholeSizesEndAddresses().end());
}

void OMPToClause::setCopyingSizesEndAddresses(
    ArrayRef<Expr *> CopyingSizesEndAddresses) {
  assert(CopyingSizesEndAddresses.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(CopyingSizesEndAddresses.begin(), CopyingSizesEndAddresses.end(),
            getCopyingStartAddresses().end());
}

OMPToClause *OMPToClause::Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation EndLoc, ArrayRef<Expr *> VL,
                                 ArrayRef<Expr *> WholeStartAddresses,
                                 ArrayRef<Expr *> WholeSizesEndAddresses,
                                 ArrayRef<Expr *> CopyingStartAddresses,
                                 ArrayRef<Expr *> CopyingSizesEndAddresses) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPToClause), llvm::alignOf<Expr *>()) +
      sizeof(Expr *) * VL.size() * 5);
  OMPToClause *Clause = new (Mem) OMPToClause(StartLoc, EndLoc, VL.size());
  Clause->setVars(VL);
  Clause->setWholeStartAddresses(WholeStartAddresses);
  Clause->setWholeSizesEndAddresses(WholeSizesEndAddresses);
  Clause->setCopyingStartAddresses(CopyingStartAddresses);
  Clause->setCopyingSizesEndAddresses(CopyingSizesEndAddresses);
  return Clause;
}

OMPToClause *OMPToClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPToClause), llvm::alignOf<Expr *>()) +
      sizeof(Expr *) * N * 5);
  return new (Mem) OMPToClause(N);
}

void
OMPFromClause::setWholeStartAddresses(ArrayRef<Expr *> WholeStartAddresses) {
  assert(WholeStartAddresses.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(WholeStartAddresses.begin(), WholeStartAddresses.end(),
            varlist_end());
}

void OMPFromClause::setWholeSizesEndAddresses(
    ArrayRef<Expr *> WholeSizesEndAddresses) {
  assert(WholeSizesEndAddresses.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(WholeSizesEndAddresses.begin(), WholeSizesEndAddresses.end(),
            getWholeStartAddresses().end());
}

void OMPFromClause::setCopyingStartAddresses(
    ArrayRef<Expr *> CopyingStartAddresses) {
  assert(CopyingStartAddresses.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(CopyingStartAddresses.begin(), CopyingStartAddresses.end(),
            getWholeSizesEndAddresses().end());
}

void OMPFromClause::setCopyingSizesEndAddresses(
    ArrayRef<Expr *> CopyingSizesEndAddresses) {
  assert(CopyingSizesEndAddresses.size() == varlist_size() &&
         "Number of vars is not the same as the preallocated buffer");
  std::copy(CopyingSizesEndAddresses.begin(), CopyingSizesEndAddresses.end(),
            getCopyingStartAddresses().end());
}

OMPFromClause *
OMPFromClause::Create(const ASTContext &C, SourceLocation StartLoc,
                      SourceLocation EndLoc, ArrayRef<Expr *> VL,
                      ArrayRef<Expr *> WholeStartAddresses,
                      ArrayRef<Expr *> WholeSizesEndAddresses,
                      ArrayRef<Expr *> CopyingStartAddresses,
                      ArrayRef<Expr *> CopyingSizesEndAddresses) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPFromClause), llvm::alignOf<Expr *>()) +
      sizeof(Expr *) * VL.size() * 5);
  OMPFromClause *Clause = new (Mem) OMPFromClause(StartLoc, EndLoc, VL.size());
  Clause->setVars(VL);
  Clause->setWholeStartAddresses(WholeStartAddresses);
  Clause->setWholeSizesEndAddresses(WholeSizesEndAddresses);
  Clause->setCopyingStartAddresses(CopyingStartAddresses);
  Clause->setCopyingSizesEndAddresses(CopyingSizesEndAddresses);
  return Clause;
}

OMPFromClause *OMPFromClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPFromClause), llvm::alignOf<Expr *>()) +
      sizeof(Expr *) * N * 5);
  return new (Mem) OMPFromClause(N);
}

OMPFlushClause *OMPFlushClause::Create(const ASTContext &C,
                                       SourceLocation StartLoc,
                                       SourceLocation EndLoc,
                                       ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFlushClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size());
  OMPFlushClause *Clause =
      new (Mem) OMPFlushClause(StartLoc, EndLoc, VL.size());
  Clause->setVars(VL);
  return Clause;
}

OMPFlushClause *OMPFlushClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFlushClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N);
  return new (Mem) OMPFlushClause(N);
}

OMPDependClause *
OMPDependClause::Create(const ASTContext &C, SourceLocation StartLoc,
                        SourceLocation EndLoc, ArrayRef<Expr *> VL,
                        ArrayRef<Expr *> Begins, ArrayRef<Expr *> SizeInBytes,
                        OpenMPDependClauseType Ty, SourceLocation TyLoc) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPDependClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size() * 3);
  OMPDependClause *Clause =
      new (Mem) OMPDependClause(StartLoc, EndLoc, VL.size(), Ty, TyLoc);
  Clause->setVars(VL);
  Clause->setBegins(Begins);
  Clause->setSizeInBytes(SizeInBytes);
  return Clause;
}

OMPDependClause *OMPDependClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPDependClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N * 3);
  OMPDependClause *Clause = new (Mem) OMPDependClause(N);
  return Clause;
}

void OMPDependClause::setBegins(ArrayRef<Expr *> Begins) {
  assert(Begins.size() == varlist_size() &&
         "Number of exprs is not the same as the preallocated buffer");
  std::copy(Begins.begin(), Begins.end(), varlist_end());
}

void OMPDependClause::setSizeInBytes(ArrayRef<Expr *> SizeInBytes) {
  assert(SizeInBytes.size() == varlist_size() &&
         "Number of exprs is not the same as the preallocated buffer");
  std::copy(SizeInBytes.begin(), SizeInBytes.end(),
            varlist_end() + varlist_size());
}

Expr *OMPDependClause::getBegins(unsigned Index) {
  assert(Index < varlist_size() &&
         "Index greter or equal maximum number of expressions.");
  return varlist_end()[Index];
}

Expr *OMPDependClause::getSizeInBytes(unsigned Index) {
  assert(Index < varlist_size() &&
         "Index greter or equal maximum number of expressions.");
  return varlist_end()[varlist_size() + Index];
}

OMPUniformClause *OMPUniformClause::Create(const ASTContext &C,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc,
                                           ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPUniformClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size());
  OMPUniformClause *Clause = new (Mem) OMPUniformClause(StartLoc,
                                                        EndLoc,
                                                        VL.size());
  Clause->setVars(VL);
  return Clause;
}

OMPUniformClause *
OMPUniformClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPUniformClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * N);
  return new (Mem) OMPUniformClause(N);
}

OMPLinearClause *OMPLinearClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL, Expr *St,
                                         SourceLocation StLoc) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPLinearClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size() + sizeof(Expr *));
  OMPLinearClause *Clause =
      new (Mem) OMPLinearClause(StartLoc, EndLoc, VL.size(), StLoc);
  Clause->setVars(VL);
  Clause->setStep(St);
  return Clause;
}

OMPLinearClause *OMPLinearClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPLinearClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * (N + 1));
  return new (Mem) OMPLinearClause(N);
}

OMPAlignedClause *OMPAlignedClause::Create(const ASTContext &C,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc,
                                           ArrayRef<Expr *> VL,
                                           Expr *A,
                                           SourceLocation ALoc) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPAlignedClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * VL.size() +
                         sizeof(Expr *));
  OMPAlignedClause *Clause = new (Mem) OMPAlignedClause(StartLoc,
                                                        EndLoc,
                                                        VL.size(),
                                                        ALoc);
  Clause->setVars(VL);
  Clause->setAlignment(A);
  return Clause;
}

OMPAlignedClause *
OMPAlignedClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPAlignedClause),
                                                  llvm::alignOf<Expr *>()) +
                         sizeof(Expr *) * (N + 1));
  return new (Mem) OMPAlignedClause(N);
}

void OMPExecutableDirective::setClauses(ArrayRef<OMPClause *> CL) {
  assert(CL.size() == NumClauses &&
         "Number of clauses is not the same as the preallocated buffer");
  std::copy(CL.begin(), CL.end(), Clauses);
}

OMPParallelDirective *OMPParallelDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPParallelDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPParallelDirective *Dir =
      new (Mem) OMPParallelDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPParallelDirective *
OMPParallelDirective::CreateEmpty(const ASTContext &C, unsigned N, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPParallelDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *));
  return new (Mem) OMPParallelDirective(N);
}

OMPForDirective *OMPForDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, ArrayRef<Expr *> VarCnts) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPForDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * VarCnts.size());
  OMPForDirective *Dir = new (Mem)
      OMPForDirective(StartLoc, EndLoc, VarCnts.size(), Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPForDirective *OMPForDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                              unsigned CollapsedNum,
                                              EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPForDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPForDirective(CollapsedNum, N);
}

OMPParallelForDirective *OMPParallelForDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, ArrayRef<Expr *> VarCnts) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPParallelForDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * VarCnts.size());
  OMPParallelForDirective *Dir = new (Mem)
      OMPParallelForDirective(StartLoc, EndLoc, VarCnts.size(), Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPParallelForDirective *
OMPParallelForDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                     unsigned CollapsedNum, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPParallelForDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPParallelForDirective(CollapsedNum, N);
}

OMPSimdDirective *OMPSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, ArrayRef<Expr *> VarCnts) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSimdDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * VarCnts.size());
  OMPSimdDirective *Dir = new (Mem)
      OMPSimdDirective(StartLoc, EndLoc, VarCnts.size(), Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPSimdDirective *OMPSimdDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                                unsigned CollapsedNum,
                                                EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSimdDirective),
                                          llvm::alignOf<OMPClause *>()) +
                     sizeof(OMPClause *) * N + sizeof(Stmt *) * 5 +
                     sizeof(Stmt *) * CollapsedNum,
                 llvm::alignOf<OMPSimdDirective>());
  return new (Mem) OMPSimdDirective(CollapsedNum, N);
}

OMPForSimdDirective *OMPForSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, ArrayRef<Expr *> VarCnts) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPForSimdDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * VarCnts.size());
  OMPForSimdDirective *Dir = new (Mem)
      OMPForSimdDirective(StartLoc, EndLoc, VarCnts.size(), Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPForSimdDirective *OMPForSimdDirective::CreateEmpty(const ASTContext &C,
                                                      unsigned N,
                                                      unsigned CollapsedNum,
                                                      EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPForSimdDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPForSimdDirective(CollapsedNum, N);
}

OMPParallelForSimdDirective *OMPParallelForSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, ArrayRef<Expr *> VarCnts) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPParallelForSimdDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * VarCnts.size());
  OMPParallelForSimdDirective *Dir = new (Mem) OMPParallelForSimdDirective(
      StartLoc, EndLoc, VarCnts.size(), Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPParallelForSimdDirective *
OMPParallelForSimdDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                         unsigned CollapsedNum, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPParallelForSimdDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPParallelForSimdDirective(CollapsedNum, N);
}

OMPDistributeSimdDirective *OMPDistributeSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, ArrayRef<Expr *> VarCnts) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPDistributeSimdDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * VarCnts.size());
  OMPDistributeSimdDirective *Dir = new (Mem) OMPDistributeSimdDirective(
      StartLoc, EndLoc, VarCnts.size(), Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPDistributeSimdDirective *
OMPDistributeSimdDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                        unsigned CollapsedNum, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPDistributeSimdDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPDistributeSimdDirective(CollapsedNum, N);
}

OMPDistributeParallelForDirective *OMPDistributeParallelForDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, Expr *LowerBound,
    Expr *UpperBound, ArrayRef<Expr *> VarCnts) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPDistributeParallelForDirective),
                               llvm::alignOf<OMPClause *>()) +
      sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 7 +
      sizeof(Stmt *) * VarCnts.size() + 2);
  OMPDistributeParallelForDirective *Dir = new (Mem)
      OMPDistributeParallelForDirective(StartLoc, EndLoc, VarCnts.size(),
                                        Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setLowerBound(LowerBound);
  Dir->setUpperBound(UpperBound);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPDistributeParallelForDirective *
OMPDistributeParallelForDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                               unsigned CollapsedNum,
                                               EmptyShell) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPDistributeParallelForDirective),
                               llvm::alignOf<OMPClause *>()) +
      sizeof(OMPClause *) * N + sizeof(Stmt *) * 7 +
      sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPDistributeParallelForDirective(CollapsedNum, N);
}

OMPDistributeParallelForSimdDirective *
OMPDistributeParallelForSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, Expr *LowerBound,
    Expr *UpperBound, ArrayRef<Expr *> VarCnts) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPDistributeParallelForSimdDirective),
                               llvm::alignOf<OMPClause *>()) +
      sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 7 +
      sizeof(Stmt *) * VarCnts.size() + 2);
  OMPDistributeParallelForSimdDirective *Dir = new (Mem)
      OMPDistributeParallelForSimdDirective(StartLoc, EndLoc, VarCnts.size(),
                                            Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setLowerBound(LowerBound);
  Dir->setUpperBound(UpperBound);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPDistributeParallelForSimdDirective *
OMPDistributeParallelForSimdDirective::CreateEmpty(const ASTContext &C,
                                                   unsigned N,
                                                   unsigned CollapsedNum,
                                                   EmptyShell) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPDistributeParallelForSimdDirective),
                               llvm::alignOf<OMPClause *>()) +
      sizeof(OMPClause *) * N + sizeof(Stmt *) * 7 +
      sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPDistributeParallelForSimdDirective(CollapsedNum, N);
}

OMPTeamsDistributeParallelForDirective *
OMPTeamsDistributeParallelForDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, Expr *LowerBound,
    Expr *UpperBound, ArrayRef<Expr *> VarCnts) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPTeamsDistributeParallelForDirective),
                               llvm::alignOf<OMPClause *>()) +
      sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 7 +
      sizeof(Stmt *) * VarCnts.size() + 2);
  OMPTeamsDistributeParallelForDirective *Dir = new (Mem)
      OMPTeamsDistributeParallelForDirective(StartLoc, EndLoc, VarCnts.size(),
                                             Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setLowerBound(LowerBound);
  Dir->setUpperBound(UpperBound);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPTeamsDistributeParallelForDirective *
OMPTeamsDistributeParallelForDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned N,
                                                    unsigned CollapsedNum,
                                                    EmptyShell) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPTeamsDistributeParallelForDirective),
                               llvm::alignOf<OMPClause *>()) +
      sizeof(OMPClause *) * N + sizeof(Stmt *) * 7 +
      sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPTeamsDistributeParallelForDirective(CollapsedNum, N);
}

OMPTeamsDistributeParallelForSimdDirective *
OMPTeamsDistributeParallelForSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, Expr *LowerBound,
    Expr *UpperBound, ArrayRef<Expr *> VarCnts) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(
                     sizeof(OMPTeamsDistributeParallelForSimdDirective),
                     llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 7 +
                 sizeof(Stmt *) * VarCnts.size() + 2);
  OMPTeamsDistributeParallelForSimdDirective *Dir =
      new (Mem) OMPTeamsDistributeParallelForSimdDirective(
          StartLoc, EndLoc, VarCnts.size(), Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setLowerBound(LowerBound);
  Dir->setUpperBound(UpperBound);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPTeamsDistributeParallelForSimdDirective *
OMPTeamsDistributeParallelForSimdDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned N,
                                                        unsigned CollapsedNum,
                                                        EmptyShell) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(
                             sizeof(OMPTeamsDistributeParallelForSimdDirective),
                             llvm::alignOf<OMPClause *>()) +
                         sizeof(OMPClause *) * N + sizeof(Stmt *) * 7 +
                         sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPTeamsDistributeParallelForSimdDirective(CollapsedNum, N);
}

OMPTargetTeamsDistributeParallelForDirective *
OMPTargetTeamsDistributeParallelForDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, Expr *LowerBound,
    Expr *UpperBound, ArrayRef<Expr *> VarCnts) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(
                     sizeof(OMPTargetTeamsDistributeParallelForDirective),
                     llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 7 +
                 sizeof(Stmt *) * VarCnts.size() + 2);
  OMPTargetTeamsDistributeParallelForDirective *Dir =
      new (Mem) OMPTargetTeamsDistributeParallelForDirective(
          StartLoc, EndLoc, VarCnts.size(), Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setLowerBound(LowerBound);
  Dir->setUpperBound(UpperBound);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPTargetTeamsDistributeParallelForDirective *
OMPTargetTeamsDistributeParallelForDirective::CreateEmpty(const ASTContext &C,
                                                          unsigned N,
                                                          unsigned CollapsedNum,
                                                          EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(
                     sizeof(OMPTargetTeamsDistributeParallelForDirective),
                     llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *) * 7 +
                 sizeof(Stmt *) * CollapsedNum);
  return new (Mem)
      OMPTargetTeamsDistributeParallelForDirective(CollapsedNum, N);
}

OMPTargetTeamsDistributeParallelForSimdDirective *
OMPTargetTeamsDistributeParallelForSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, Expr *LowerBound,
    Expr *UpperBound, ArrayRef<Expr *> VarCnts) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(
                     sizeof(OMPTargetTeamsDistributeParallelForSimdDirective),
                     llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 7 +
                 sizeof(Stmt *) * VarCnts.size() + 2);
  OMPTargetTeamsDistributeParallelForSimdDirective *Dir =
      new (Mem) OMPTargetTeamsDistributeParallelForSimdDirective(
          StartLoc, EndLoc, VarCnts.size(), Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setLowerBound(LowerBound);
  Dir->setUpperBound(UpperBound);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPTargetTeamsDistributeParallelForSimdDirective *
OMPTargetTeamsDistributeParallelForSimdDirective::CreateEmpty(
    const ASTContext &C, unsigned N, unsigned CollapsedNum, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(
                     sizeof(OMPTargetTeamsDistributeParallelForSimdDirective),
                     llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *) * 7 +
                 sizeof(Stmt *) * CollapsedNum);
  return new (Mem)
      OMPTargetTeamsDistributeParallelForSimdDirective(CollapsedNum, N);
}

OMPSectionsDirective *OMPSectionsDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSectionsDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPSectionsDirective *Dir =
      new (Mem) OMPSectionsDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPSectionsDirective *
OMPSectionsDirective::CreateEmpty(const ASTContext &C, unsigned N, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSectionsDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *));
  return new (Mem) OMPSectionsDirective(N);
}

OMPParallelSectionsDirective *OMPParallelSectionsDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPParallelSectionsDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPParallelSectionsDirective *Dir =
      new (Mem) OMPParallelSectionsDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPParallelSectionsDirective *
OMPParallelSectionsDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                          EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPParallelSectionsDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *));
  return new (Mem) OMPParallelSectionsDirective(N);
}

OMPSectionDirective *OMPSectionDirective::Create(const ASTContext &C,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc,
                                                 Stmt *AssociatedStmt) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSectionDirective),
                                                  llvm::alignOf<Stmt *>()) +
                         sizeof(Stmt *));
  OMPSectionDirective *Dir = new (Mem) OMPSectionDirective(StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPSectionDirective *OMPSectionDirective::CreateEmpty(const ASTContext &C,
                                                      EmptyShell) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSectionDirective),
                                                  llvm::alignOf<Stmt *>()) +
                         sizeof(Stmt *));
  return new (Mem) OMPSectionDirective();
}

OMPSingleDirective *OMPSingleDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               ArrayRef<OMPClause *> Clauses,
                                               Stmt *AssociatedStmt) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSingleDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPSingleDirective *Dir =
      new (Mem) OMPSingleDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPSingleDirective *OMPSingleDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned N, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPSingleDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *));
  return new (Mem) OMPSingleDirective(N);
}

OMPTaskDirective *OMPTaskDirective::Create(const ASTContext &C,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc,
                                           ArrayRef<OMPClause *> Clauses,
                                           Stmt *AssociatedStmt) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTaskDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTaskDirective *Dir =
      new (Mem) OMPTaskDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTaskDirective *OMPTaskDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                                EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTaskDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *));
  return new (Mem) OMPTaskDirective(N);
}

OMPTaskyieldDirective *OMPTaskyieldDirective::Create(const ASTContext &C,
                                                     SourceLocation StartLoc,
                                                     SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OMPTaskyieldDirective),
                         llvm::alignOf<OMPTaskyieldDirective>());
  return new (Mem) OMPTaskyieldDirective(StartLoc, EndLoc);
}

OMPTaskyieldDirective *OMPTaskyieldDirective::CreateEmpty(const ASTContext &C,
                                                          EmptyShell) {
  void *Mem = C.Allocate(sizeof(OMPTaskyieldDirective),
                         llvm::alignOf<OMPTaskyieldDirective>());
  return new (Mem) OMPTaskyieldDirective();
}

OMPMasterDirective *OMPMasterDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               Stmt *AssociatedStmt) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPMasterDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(Stmt *));
  OMPMasterDirective *Dir = new (Mem) OMPMasterDirective(StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPMasterDirective *OMPMasterDirective::CreateEmpty(const ASTContext &C,
                                                    EmptyShell) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPMasterDirective),
                                                  llvm::alignOf<Stmt *>()) +
                         sizeof(Stmt *));
  return new (Mem) OMPMasterDirective();
}

OMPCriticalDirective *OMPCriticalDirective::Create(const ASTContext &C,
                                                   DeclarationNameInfo Name,
                                                   SourceLocation StartLoc,
                                                   SourceLocation EndLoc,
                                                   Stmt *AssociatedStmt) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCriticalDirective),
                                                  llvm::alignOf<Stmt *>()) +
                         sizeof(Stmt *));
  OMPCriticalDirective *Dir =
      new (Mem) OMPCriticalDirective(Name, StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setDirectiveName(Name);
  return Dir;
}

OMPCriticalDirective *OMPCriticalDirective::CreateEmpty(const ASTContext &C,
                                                        EmptyShell) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCriticalDirective),
                                                  llvm::alignOf<Stmt *>()) +
                         sizeof(Stmt *));
  return new (Mem) OMPCriticalDirective();
}

OMPBarrierDirective *OMPBarrierDirective::Create(const ASTContext &C,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OMPBarrierDirective),
                         llvm::alignOf<OMPBarrierDirective>());
  return new (Mem) OMPBarrierDirective(StartLoc, EndLoc);
}

OMPBarrierDirective *OMPBarrierDirective::CreateEmpty(const ASTContext &C,
                                                      EmptyShell) {
  void *Mem = C.Allocate(sizeof(OMPBarrierDirective),
                         llvm::alignOf<OMPBarrierDirective>());
  return new (Mem) OMPBarrierDirective();
}

OMPTaskwaitDirective *OMPTaskwaitDirective::Create(const ASTContext &C,
                                                   SourceLocation StartLoc,
                                                   SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OMPTaskwaitDirective),
                         llvm::alignOf<OMPTaskwaitDirective>());
  return new (Mem) OMPTaskwaitDirective(StartLoc, EndLoc);
}

OMPTaskwaitDirective *OMPTaskwaitDirective::CreateEmpty(const ASTContext &C,
                                                        EmptyShell) {
  void *Mem = C.Allocate(sizeof(OMPTaskwaitDirective),
                         llvm::alignOf<OMPTaskwaitDirective>());
  return new (Mem) OMPTaskwaitDirective();
}

OMPTaskgroupDirective *OMPTaskgroupDirective::Create(const ASTContext &C,
                                                     SourceLocation StartLoc,
                                                     SourceLocation EndLoc,
                                                     Stmt *AssociatedStmt) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTaskgroupDirective),
                                                  llvm::alignOf<Stmt *>()) +
                         sizeof(Stmt *));
  OMPTaskgroupDirective *Dir =
      new (Mem) OMPTaskgroupDirective(StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTaskgroupDirective *OMPTaskgroupDirective::CreateEmpty(const ASTContext &C,
                                                          EmptyShell) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTaskgroupDirective),
                                                  llvm::alignOf<Stmt *>()) +
                         sizeof(Stmt *));
  return new (Mem) OMPTaskgroupDirective();
}

OMPAtomicDirective *OMPAtomicDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *V, Expr *X,
    Expr *OpExpr, BinaryOperatorKind Op, bool CaptureAfter, bool Reversed) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPAtomicDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 4);
  OMPAtomicDirective *Dir =
      new (Mem) OMPAtomicDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setOperator(Op);
  Dir->setV(V);
  Dir->setX(X);
  Dir->setExpr(OpExpr);
  Dir->setCaptureAfter(CaptureAfter);
  Dir->setReversed(Reversed);
  return Dir;
}

OMPAtomicDirective *OMPAtomicDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned N, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPAtomicDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *) * 4);
  return new (Mem) OMPAtomicDirective(N);
}

OMPFlushDirective *OMPFlushDirective::Create(const ASTContext &C,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc,
                                             ArrayRef<OMPClause *> Clauses) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFlushDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size());
  OMPFlushDirective *Dir =
      new (Mem) OMPFlushDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  return Dir;
}

OMPFlushDirective *OMPFlushDirective::CreateEmpty(const ASTContext &C,
                                                  unsigned N, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPFlushDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N);
  return new (Mem) OMPFlushDirective(N);
}

OMPOrderedDirective *OMPOrderedDirective::Create(const ASTContext &C,
                                                 SourceLocation StartLoc,
                                                 SourceLocation EndLoc,
                                                 Stmt *AssociatedStmt) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPOrderedDirective),
                                                  llvm::alignOf<Stmt *>()) +
                         sizeof(Stmt *));
  OMPOrderedDirective *Dir = new (Mem) OMPOrderedDirective(StartLoc, EndLoc);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPOrderedDirective *OMPOrderedDirective::CreateEmpty(const ASTContext &C,
                                                      EmptyShell) {
  void *Mem = C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPOrderedDirective),
                                                  llvm::alignOf<Stmt *>()) +
                         sizeof(Stmt *));
  return new (Mem) OMPOrderedDirective();
}

OMPTeamsDirective *OMPTeamsDirective::Create(const ASTContext &C,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc,
                                             ArrayRef<OMPClause *> Clauses,
                                             Stmt *AssociatedStmt) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTeamsDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTeamsDirective *Dir =
      new (Mem) OMPTeamsDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTeamsDirective *OMPTeamsDirective::CreateEmpty(const ASTContext &C,
                                                  unsigned N, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTeamsDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *));
  return new (Mem) OMPTeamsDirective(N);
}

OMPDistributeDirective *OMPDistributeDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, ArrayRef<Expr *> VarCnts) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPDistributeDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * VarCnts.size());
  OMPDistributeDirective *Dir = new (Mem)
      OMPDistributeDirective(StartLoc, EndLoc, VarCnts.size(), Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPDistributeDirective *
OMPDistributeDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                    unsigned CollapsedNum, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPDistributeDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPDistributeDirective(CollapsedNum, N);
}

OMPCancelDirective *
OMPCancelDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                           SourceLocation EndLoc, ArrayRef<OMPClause *> Clauses,
                           OpenMPDirectiveKind ConstructType) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCancelDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size());
  OMPCancelDirective *Dir = new (Mem)
      OMPCancelDirective(StartLoc, EndLoc, Clauses.size(), ConstructType);
  Dir->setClauses(Clauses);
  return Dir;
}

OMPCancelDirective *
OMPCancelDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                OpenMPDirectiveKind ConstructType, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPCancelDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N);
  return new (Mem) OMPCancelDirective(N, ConstructType);
}

OMPCancellationPointDirective *OMPCancellationPointDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    OpenMPDirectiveKind ConstructType) {
  void *Mem = C.Allocate(sizeof(OMPCancellationPointDirective),
                         llvm::alignOf<OMPCancellationPointDirective>());
  OMPCancellationPointDirective *Dir =
      new (Mem) OMPCancellationPointDirective(StartLoc, EndLoc, ConstructType);
  return Dir;
}

OMPCancellationPointDirective *OMPCancellationPointDirective::CreateEmpty(
    const ASTContext &C, OpenMPDirectiveKind ConstructType, EmptyShell) {
  void *Mem = C.Allocate(sizeof(OMPCancellationPointDirective),
                         llvm::alignOf<OMPCancellationPointDirective>());
  return new (Mem) OMPCancellationPointDirective(ConstructType);
}

OMPTargetDirective *OMPTargetDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               ArrayRef<OMPClause *> Clauses,
                                               Stmt *AssociatedStmt) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTargetDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTargetDirective *Dir =
      new (Mem) OMPTargetDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTargetDirective *OMPTargetDirective::CreateEmpty(const ASTContext &C,
                                                    unsigned N, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTargetDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *));
  return new (Mem) OMPTargetDirective(N);
}

OMPTargetDataDirective *OMPTargetDataDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTargetDataDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTargetDataDirective *Dir =
      new (Mem) OMPTargetDataDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTargetDataDirective *OMPTargetDataDirective::CreateEmpty(const ASTContext &C,
                                                            unsigned N,
                                                            EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTargetDataDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *));
  return new (Mem) OMPTargetDataDirective(N);
}

OMPTeamsDistributeDirective *OMPTeamsDistributeDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, ArrayRef<Expr *> VarCnts) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTeamsDistributeDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * VarCnts.size());
  OMPTeamsDistributeDirective *Dir = new (Mem) OMPTeamsDistributeDirective(
      StartLoc, EndLoc, VarCnts.size(), Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPTeamsDistributeDirective *
OMPTeamsDistributeDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                         unsigned CollapsedNum, EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTeamsDistributeDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *) * 5 +
                 sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPTeamsDistributeDirective(CollapsedNum, N);
}

OMPTeamsDistributeSimdDirective *OMPTeamsDistributeSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, ArrayRef<Expr *> VarCnts) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPTeamsDistributeSimdDirective),
                               llvm::alignOf<OMPClause *>()) +
      sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 5 +
      sizeof(Stmt *) * VarCnts.size());
  OMPTeamsDistributeSimdDirective *Dir =
      new (Mem) OMPTeamsDistributeSimdDirective(StartLoc, EndLoc,
                                                VarCnts.size(), Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPTeamsDistributeSimdDirective *OMPTeamsDistributeSimdDirective::CreateEmpty(
    const ASTContext &C, unsigned N, unsigned CollapsedNum, EmptyShell) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPTeamsDistributeSimdDirective),
                               llvm::alignOf<OMPClause *>()) +
      sizeof(OMPClause *) * N + sizeof(Stmt *) * 5 +
      sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPTeamsDistributeSimdDirective(CollapsedNum, N);
}

OMPTargetTeamsDistributeDirective *OMPTargetTeamsDistributeDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, ArrayRef<Expr *> VarCnts) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPTargetTeamsDistributeDirective),
                               llvm::alignOf<OMPClause *>()) +
      sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 5 +
      sizeof(Stmt *) * VarCnts.size());
  OMPTargetTeamsDistributeDirective *Dir = new (Mem)
      OMPTargetTeamsDistributeDirective(StartLoc, EndLoc, VarCnts.size(),
                                        Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPTargetTeamsDistributeDirective *
OMPTargetTeamsDistributeDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                               unsigned CollapsedNum,
                                               EmptyShell) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPTargetTeamsDistributeDirective),
                               llvm::alignOf<OMPClause *>()) +
      sizeof(OMPClause *) * N + sizeof(Stmt *) * 5 +
      sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPTargetTeamsDistributeDirective(CollapsedNum, N);
}

OMPTargetTeamsDistributeSimdDirective *
OMPTargetTeamsDistributeSimdDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt, Expr *NewIterVar,
    Expr *NewIterEnd, Expr *Init, Expr *Final, ArrayRef<Expr *> VarCnts) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPTargetTeamsDistributeSimdDirective),
                               llvm::alignOf<OMPClause *>()) +
      sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *) * 5 +
      sizeof(Stmt *) * VarCnts.size());
  OMPTargetTeamsDistributeSimdDirective *Dir = new (Mem)
      OMPTargetTeamsDistributeSimdDirective(StartLoc, EndLoc, VarCnts.size(),
                                            Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  Dir->setNewIterVar(NewIterVar);
  Dir->setNewIterEnd(NewIterEnd);
  Dir->setInit(Init);
  Dir->setFinal(Final);
  Dir->setCounters(VarCnts);
  return Dir;
}

OMPTargetTeamsDistributeSimdDirective *
OMPTargetTeamsDistributeSimdDirective::CreateEmpty(const ASTContext &C,
                                                   unsigned N,
                                                   unsigned CollapsedNum,
                                                   EmptyShell) {
  void *Mem = C.Allocate(
      llvm::RoundUpToAlignment(sizeof(OMPTargetTeamsDistributeSimdDirective),
                               llvm::alignOf<OMPClause *>()) +
      sizeof(OMPClause *) * N + sizeof(Stmt *) * 5 +
      sizeof(Stmt *) * CollapsedNum);
  return new (Mem) OMPTargetTeamsDistributeSimdDirective(CollapsedNum, N);
}

CapturedStmt::Capture *CapturedStmt::getStoredCaptures() const {
  unsigned Size = sizeof(CapturedStmt) + sizeof(Stmt *) * (NumCaptures + 1);

  // Offset of the first Capture object.
  unsigned FirstCaptureOffset =
    llvm::RoundUpToAlignment(Size, llvm::alignOf<Capture>());

  return reinterpret_cast<Capture *>(
      reinterpret_cast<char *>(const_cast<CapturedStmt *>(this))
      + FirstCaptureOffset);
}

OMPTargetUpdateDirective *
OMPTargetUpdateDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation EndLoc,
                                 ArrayRef<OMPClause *> Clauses) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTargetUpdateDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size());
  OMPTargetUpdateDirective *Dir =
      new (Mem) OMPTargetUpdateDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  return Dir;
}

OMPTargetUpdateDirective *
OMPTargetUpdateDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                      EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTargetUpdateDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N);
  return new (Mem) OMPTargetUpdateDirective(N);
}

OMPTargetTeamsDirective *OMPTargetTeamsDirective::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OMPClause *> Clauses, Stmt *AssociatedStmt) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTargetTeamsDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * Clauses.size() + sizeof(Stmt *));
  OMPTargetTeamsDirective *Dir =
      new (Mem) OMPTargetTeamsDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AssociatedStmt);
  return Dir;
}

OMPTargetTeamsDirective *
OMPTargetTeamsDirective::CreateEmpty(const ASTContext &C, unsigned N,
                                     EmptyShell) {
  void *Mem =
      C.Allocate(llvm::RoundUpToAlignment(sizeof(OMPTargetTeamsDirective),
                                          llvm::alignOf<OMPClause *>()) +
                 sizeof(OMPClause *) * N + sizeof(Stmt *));
  return new (Mem) OMPTargetTeamsDirective(N);
}

CapturedStmt::CapturedStmt(Stmt *S, CapturedRegionKind Kind,
                           ArrayRef<Capture> Captures,
                           ArrayRef<Expr *> CaptureInits,
                           CapturedDecl *CD,
                           RecordDecl *RD)
  : Stmt(CapturedStmtClass), NumCaptures(Captures.size()),
    TheCapturedDecl(CD), RegionKind(Kind), TheRecordDecl(RD) {
  assert( S && "null captured statement");
  assert(CD && "null captured declaration for captured statement");
  assert(RD && "null record declaration for captured statement");

  // Copy initialization expressions.
  Stmt **Stored = getStoredStmts();
  for (unsigned I = 0, N = NumCaptures; I != N; ++I)
    *Stored++ = CaptureInits[I];

  // Copy the statement being captured.
  *Stored = S;

  // Copy all Capture objects.
  Capture *Buffer = getStoredCaptures();
  std::copy(Captures.begin(), Captures.end(), Buffer);
}

CapturedStmt::CapturedStmt(EmptyShell Empty, unsigned NumCaptures)
  : Stmt(CapturedStmtClass, Empty), NumCaptures(NumCaptures),
    TheCapturedDecl(0), RegionKind(CR_Default), TheRecordDecl(nullptr) {
  getStoredStmts()[NumCaptures] = nullptr;
}

CapturedStmt *CapturedStmt::Create(const ASTContext &Context, Stmt *S,
                                   CapturedRegionKind Kind,
                                   ArrayRef<Capture> Captures,
                                   ArrayRef<Expr *> CaptureInits,
                                   CapturedDecl *CD, RecordDecl *RD) {
  // The layout is
  //
  // -----------------------------------------------------------
  // | CapturedStmt, Init, ..., Init, S, Capture, ..., Capture |
  // ----------------^-------------------^----------------------
  //                 getStoredStmts()    getStoredCaptures()
  //
  // where S is the statement being captured.
  //
  assert(CaptureInits.size() == Captures.size() && "wrong number of arguments");

  unsigned Size = sizeof(CapturedStmt) + sizeof(Stmt *) * (Captures.size() + 1);
  if (!Captures.empty()) {
    // Realign for the following Capture array.
    Size = llvm::RoundUpToAlignment(Size, llvm::alignOf<Capture>());
    Size += sizeof(Capture) * Captures.size();
  }

  void *Mem = Context.Allocate(Size);
  return new (Mem) CapturedStmt(S, Kind, Captures, CaptureInits, CD, RD);
}

CapturedStmt *CapturedStmt::CreateDeserialized(const ASTContext &Context,
                                               unsigned NumCaptures) {
  unsigned Size = sizeof(CapturedStmt) + sizeof(Stmt *) * (NumCaptures + 1);
  if (NumCaptures > 0) {
    // Realign for the following Capture array.
    Size = llvm::RoundUpToAlignment(Size, llvm::alignOf<Capture>());
    Size += sizeof(Capture) * NumCaptures;
  }

  void *Mem = Context.Allocate(Size);
  return new (Mem) CapturedStmt(EmptyShell(), NumCaptures);
}

Stmt::child_range CapturedStmt::children() {
  // Children are captured field initilizers.
  return child_range(getStoredStmts(), getStoredStmts() + NumCaptures);
}

bool CapturedStmt::capturesVariable(const VarDecl *Var) const {
  for (const auto &I : captures()) {
    if (!I.capturesVariable())
      continue;

    // This does not handle variable redeclarations. This should be
    // extended to capture variables with redeclarations, for example
    // a thread-private variable in OpenMP.
    if (I.getCapturedVar() == Var)
      return true;
  }

  return false;
}

