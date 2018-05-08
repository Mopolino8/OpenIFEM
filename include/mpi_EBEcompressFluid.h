#ifndef PARALLEL_EBE_COMP_FLUID
#define PARALLEL_EBE_COMP_FLUID

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/lac/petsc_parallel_block_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "parameters.h"
#include "preconditionerPilut.h"
#include "utilities.h"

namespace Fluid
{
  using namespace dealii;

  /** \brief The compressible Navier Stokes equation solver.
   *
   * This program is built upon dealii tutorials step-57, step-22, step-20.
   * We use fully implicit scheme for time stepping.
   * At each time step, Newton's method is used to solve for the update,
   * so we define two variables: the present solution and the update.
   * Additionally, the evaluation point is for temporarily holding
   * the updated solution.
   * We use one ConstraintMatrix for Dirichlet boundary conditions
   * at the initial step and a zero ConstraintMatrix for the rest steps.
   * Although the density does not matter in the incompressible flow, we still
   * include it in the formulation in order to be consistent with the
   * slightly compressible flow. Correspondingly the viscosity represents
   * the dynamic visocity \f$\mu\f$ instead of the kinetic visocity \f$\nu\f$,
   * and the pressure block in the solution is the real pressure.
   */
  template <int dim>
  class EBECompressibleFluid
  {
  public:
    /** \brief Constructor.
     *
     * We do not want to modify the solver every time we change the
     * triangulation.
     * So we pass in a Triangulation<dim> that is either generated using dealii
     * functions or by reading Abaqus input file. Also, a parameter handler is
     * required to specify all the input parameters.
     */
    EBECompressibleFluid(parallel::distributed::Triangulation<dim> &,
                         const Parameters::AllParameters &);
    /*! \brief Destructor. */
    ~EBECompressibleFluid() { dof_handler.clear(); };
    /**
     * This function implements the Newton iteration with given tolerance
     * and maximum number of iterations.
     */
    void run();

  private:
    /**
     * Currently the Dirichlet BCs in the input file can only be constant
     * values.
     * Space/time-dependent Dirichlet BCs are hard-coded in this class.
     */
    class BoundaryValues;
    
    class BlockDiagonalPreconditioner;
    /**
     * This function initializes the DoFHandler and constraints.
     */
    void setup_dofs();
    /**
     * Specify the sparsity pattern and reinit matrices and vectors.
     * It is separated from setup_dofs because when we do mesh refinement
     * we need to transfer the solution from old grid to the new one.
     */
    void initialize_system();
    /**
     * This function builds the system matrix and right hand side that we
     * currently work on. The initial_step argument is used to determine
     * which set of constraints we apply (nonzero for the initial step and zero
     * for the others). The assemble_matrix flag determines whether to
     * assemble the whole system or only the right hand side vector,
     * respectively.
     */
    void assemble(const bool, const bool);
    void assemble_system(const bool);
    void assemble_rhs(const bool);
    /**
     * In this function, we use GMRES solver with the block preconditioner,
     * which is defined at the beginning of the program, to solve the linear
     * system. What we obtain at this step is the solution update.
     * For the initial step, nonzero constraints are applied in order to
     * make sure boundary conditions are satisfied.
     * In the following steps, we will solve for the Newton update so zero
     * constraints are used.
     */
    std::pair<unsigned int, double> solve(const bool);
    /**
     * After finding a good initial guess on the coarse mesh, we hope to
     * decrease the error through refining the mesh. Here we do adaptive
     * refinement based on the Kelly estimator on the velocity only.
     * We also need to transfer the current solution to the
     * next mesh using the SolutionTransfer class.
     */
    void refine_mesh(const unsigned int, const unsigned int);
    /**
     * Write a vtu file for the current solution, as well as a pvtu file to
     * organize them.
     */
    void output_results(const unsigned int) const;

    double viscosity; //!< Dynamic viscosity
    double rho;
    const unsigned int degree;
    std::vector<types::global_dof_index> dofs_per_block;

    parallel::distributed::Triangulation<dim> &triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    QGauss<dim> volume_quad_formula;
    QGauss<dim - 1> face_quad_formula;

    ConstraintMatrix zero_constraints;
    ConstraintMatrix nonzero_constraints;
    ConstraintMatrix increment_constraints;

    BlockSparsityPattern sparsity_pattern;
    PETScWrappers::MPI::BlockSparseMatrix system_matrix;
    PETScWrappers::MPI::BlockSparseMatrix diag_preconditioner;

    /// The latest known solution.
    PETScWrappers::MPI::BlockVector present_solution;
    /// The increment at a certain Newton iteration.
    PETScWrappers::MPI::BlockVector newton_update;
    /**
     * The latest know solution plus the cumulation of all newton_updates
     * in the current time step, which approaches to the new present_solution.
     */
    PETScWrappers::MPI::BlockVector evaluation_point;
    PETScWrappers::MPI::BlockVector system_rhs;

    const double tolerance;
    const unsigned int max_iteration;

    Parameters::AllParameters parameters;

    MPI_Comm mpi_communicator;

    ConditionalOStream pcout;

    /// The IndexSets of owned velocity and pressure respectively.
    std::vector<IndexSet> owned_partitioning;

    /// The IndexSets of relevant velocity and pressure respectively.
    std::vector<IndexSet> relevant_partitioning;

    /// The IndexSet of all relevant dofs. This seems to be redundant but handy.
    IndexSet locally_relevant_dofs;

    /// The BlockIncompSchurPreconditioner for the whole system:
    std::shared_ptr<BlockDiagonalPreconditioner> preconditioner;

    Utils::Time time;
    mutable TimerOutput timer;

    /** \brief Helper function to specify Dirchlet boundary conditions.
     *
     *  It specifies a parabolic velocity profile at the left side boundary,
     *  and all the remaining boundaries are considered as walls
     *  except for the right side one.
     */
    class BoundaryValues : public Function<dim>
    {
    public:
      BoundaryValues() : Function<dim>(dim + 1) {}
      virtual double value(const Point<dim> &p,
                           const unsigned int component) const;

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const;
    };

    /** \brief Helper function to specity time dependent Dirichlet
    * boundary conditions.
    *
    * It specifies a gaussian waveat the left side boundary,
    * and all the remaining boundaries are considered as walls
    * except for the right side one.
    */
    class TimeDependentBoundaryValues : public Function<dim>
    {
    public:
      TimeDependentBoundaryValues()
        : Function<dim>(dim + 1), time(0), dt(0), increment(false)
      {
      }
      TimeDependentBoundaryValues(double t, double dt_, bool inc)
        : Function<dim>(dim + 1), time(t), dt(dt_), increment(inc)
      {
      }
      virtual double value(const Point<dim> &p,
                           const unsigned int component) const;

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const;

    private:
      double time_value(const Point<dim> &p,
                        const unsigned int component,
                        double t) const;
      double time;
      double dt;
      bool increment;
    };

    class BlockDiagonalPreconditioner : public Subscriptor
    {
    public:
      /// constructor.
      BlockDiagonalPreconditioner(
        TimerOutput &timer,
        const PETScWrappers::MPI::BlockSparseMatrix &system);
      /// The matrix-vector multiplication must be defined.
      void vmult(PETScWrappers::MPI::BlockVector &dst,
                 const PETScWrappers::MPI::BlockVector &src) const;
    private:
      class innerBlock;
      TimerOutput &timer;
      std::shared_ptr<innerBlock> M;
      const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix>
            system_matrix;
      class innerBlock : public Subscriptor
      {
      public:
        /// constructor
        innerBlock(
          TimerOutput &timer,
          const PETScWrappers::MPI::BlockSparseMatrix &system);
        /// The matrix-vector multiplication must be defined.
        void vmult(PETScWrappers::MPI::BlockVector &dst,
                 const PETScWrappers::MPI::BlockVector &src) const;
      private:
        TimerOutput &timer;
        const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix>
            system_matrix;
      };
    };
  };
}

#endif
