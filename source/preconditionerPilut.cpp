#include "preconditionerPilut.h"

/* ----------------- PreconditionPilut ------------------------ */

PreconditionPilut::AdditionalData::AdditionalData(
  const unsigned int maxiter,
  const unsigned int factorrowsize,
  const double tolerance)
  : maxiter(maxiter), factorrowsize(factorrowsize), tolerance(tolerance)
{
}

PreconditionPilut::PreconditionPilut(const PETScWrappers::MatrixBase &matrix,
                                     const AdditionalData &additional_data)
{
  initialize(matrix, additional_data);
}

void PreconditionPilut::initialize(const PETScWrappers::MatrixBase &matrix_,
                                   const AdditionalData &additional_data_)
{
  clear();

  matrix = static_cast<Mat>(matrix_);
  additional_data = additional_data_;

  MPI_Comm comm = matrix_.get_mpi_communicator();

  PetscErrorCode ierr = PCCreate(comm, &pc);
  AssertThrow(ierr == 0, ExcPETScError(ierr));

  ierr = PCSetOperators(pc, matrix, matrix);
  AssertThrow(ierr == 0, ExcPETScError(ierr));

  ierr = PCSetType(pc, const_cast<char *>(PCHYPRE));
  AssertThrow(ierr == 0, ExcPETScError(ierr));

  ierr = PCHYPRESetType(pc, "pilut");

  std::stringstream ssStream;

  PETScWrappers::set_option_value(
    "-pc_hypre_pilut_maxiter", Utilities::to_string(additional_data.maxiter));

  ssStream << additional_data.factorrowsize;
  PETScWrappers::set_option_value("-pc_hypre_pilut_factorrowsize",
                                  ssStream.str());

  ssStream.str(""); // empty the stringstream
  ssStream << additional_data.tolerance;
  PETScWrappers::set_option_value("-pc_hypre_pilut_tol", ssStream.str());

  ierr = PCSetFromOptions(pc);
  AssertThrow(ierr == 0, ExcPETScError(ierr));

  ierr = PCSetUp(pc);
  AssertThrow(ierr == 0, ExcPETScError(ierr));
}