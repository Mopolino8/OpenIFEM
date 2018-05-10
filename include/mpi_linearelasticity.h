#ifndef MPI_LINEAR_ELASTICITY
#define MPI_LINEAR_ELASTICITY

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <fstream>
#include <iostream>

#include "linearElasticMaterial.h"
#include "parameters.h"
#include "utilities.h"

namespace Solid
{
  using namespace dealii;

  extern template class LinearElasticMaterial<2>;
  extern template class LinearElasticMaterial<3>;

  /*! \brief A fully distributed parallel time-dependent solver for linear
   * elasticity.
   *
   * Both the triangulation and the dofs are fully distributed, the algebraic
   * operations are done using PETSc wrappers offered by deal.II.
   * The output is also parallelized: every processor writes its own output,
   * ParaView is able to group them together.
   * The mesh refinement is parallelized too.
   *
   * The solution vectors, for example displacement, are declared as non-ghosted
   * vectors.
   * There are two reasons for this: 1. we do not need the dofs not owned by the
   * current
   * processor in assembly since the system matrix is not dependent on the dofs
   * at all;
   * 2. we need to do matrix/vector operations on these vectors, for example add
   * and vmult,
   * this requires the vectors to be non-ghosted.
   *
   * Algorithm-wise, this class is not different from the serial version,
   * Newmark-beta method is used for time-discretization and displacement-based
   * finite element is used for space-discretization.
   */
  template <int dim>
  class ParallelLinearElasticity
  {
  public:
    /*! \brief Constructor.
     *
     * The triangulation can either be generated using dealii functions or
     * from Abaqus input file. It is fully distributed.
     * Also we use a parameter handler to specify all the input parameters.
     */
    ParallelLinearElasticity(parallel::distributed::Triangulation<dim> &,
                             const Parameters::AllParameters &);
    /*! \brief Destructor. */
    ~ParallelLinearElasticity() { dof_handler.clear(); }
    void run();

  private:
    /**
     * Set up the DofHandler, reorder the grid, sparsity pattern.
     */
    void setup_dofs();

    /**
     * Initialize the matrix, solution, and rhs.
     */
    void initialize_system();

    /**
     * Assembles lhs and rhs. At time step 0, the lhs is the mass matrix;
     * at all the following steps, it is \f$ M + \beta{\Delta{t}}^2K \f$.
     */
    void assemble_system(bool is_initial);

    /**
     * Solve the linear system. Returns the number of
     * CG iterations and the final residual.
     */
    std::pair<unsigned int, double>
    solve(const PETScWrappers::MPI::SparseMatrix &,
          PETScWrappers::MPI::Vector &,
          const PETScWrappers::MPI::Vector &);

    /**
     * Output the time-dependent solution in vtu format.
     */
    void output_results(const unsigned int) const;

    /**
     * Refine mesh and transfer solution.
     * Max and min levels of refinement are required.
     */
    void refine_mesh(const unsigned int, const unsigned int);

    LinearElasticMaterial<dim> material;

    const double gamma; //!< Newton-beta parameter
    const double beta;  //!< Newton-beta parameter

    const unsigned int
      degree; //!< Polynomial degree, also determines quadrature order.

    const double tolerance; //!< Absolute tolerance

    parallel::distributed::Triangulation<dim> &triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;

    const QGauss<dim>
      volume_quad_formula; //!< Quadrature formula for volume integration.

    const QGauss<dim - 1>
      face_quad_formula; //!< Quadrature formula for face integration.

    /**
     * Constraints to handle both hanging nodes and Dirichlet boundary
     * conditions.
     */
    ConstraintMatrix constraints;

    PETScWrappers::MPI::SparseMatrix
      system_matrix; //!< \f$ M + \beta{\Delta{t}}^2K \f$.
    PETScWrappers::MPI::SparseMatrix stiffness_matrix; //!< System stiffness
    PETScWrappers::MPI::Vector system_rhs; //!< The rhs due to external load

    /**
     * In the Newmark-beta method, acceleration is the variable to solve at
     * every
     * timestep. But displacement and velocity also contribute to the rhs of the
     * equation.
     * For the sake of clarity, we explicitly store two sets of accleration,
     * velocity
     * and displacement.
     * As explained before, these are going to be non-ghosted vectors.
     */
    PETScWrappers::MPI::Vector current_acceleration;
    PETScWrappers::MPI::Vector current_velocity;
    PETScWrappers::MPI::Vector current_displacement;
    PETScWrappers::MPI::Vector previous_acceleration;
    PETScWrappers::MPI::Vector previous_velocity;
    PETScWrappers::MPI::Vector previous_displacement;

    Utils::Time time;

    Parameters::AllParameters parameters;

    MPI_Comm mpi_communicator;

    /** Standard output that works only on one specific processor. */
    ConditionalOStream pcout;

    /**
     * Set of dofs that are owned by the current processor,
     * which means this processor is responsible of writing to
     * the corresponding lhs, rhs, solution etc.
     */
    IndexSet locally_owned_dofs;
    /**
     * Besides the owned dofs, the current processor needs information
     * from dofs owned by other processors, i.e., ghost dofs.
     * They are called relevant dofs.
     */
    IndexSet locally_relevant_dofs;

    mutable TimerOutput timer;
  };
} // namespace Solid

#endif
