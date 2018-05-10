#ifndef PARALLEL_COMP_FLUID
#define PARALLEL_COMP_FLUID
/*this macro is used to specify which
  preconditioner to use. If ILU is
  defined, Euclid ILU(k) will be used.
  Otherwise, no ILU will be performed
*/

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
  class ParallelCompressibleFluid
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
    ParallelCompressibleFluid(parallel::distributed::Triangulation<dim> &,
                              const Parameters::AllParameters &);
    /*! \brief Destructor. */
    ~ParallelCompressibleFluid() { dof_handler.clear(); };
    /**
     * This function implements the Newton iteration with given tolerance
     * and maximum number of iterations.
     */
    void run();

    //! Return the solution for testing.
    PETScWrappers::MPI::BlockVector get_current_solution() const;

  private:
    /**
     * Currently the Dirichlet BCs in the input file can only be constant
     * values.
     * Space/time-dependent Dirichlet BCs are hard-coded in this class.
     */
    class BoundaryValues;
    /**
     * The blcok preconditioner for the whole linear system.
     * It is a private member of NavierStokes<dim>.
     */
    class BlockIncompSchurPreconditioner;

    //! Set up the dofs based on the finite element and renumber them.
    void setup_dofs();

    //! Set up the nonzero and zero constraints.
    void make_constraints();

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
    void assemble(const bool use_nonzero_constraints);
    /**
     * In this function, we use GMRES solver with the block preconditioner,
     * which is defined at the beginning of the program, to solve the linear
     * system. What we obtain at this step is the solution update.
     * For the initial step, nonzero constraints are applied in order to
     * make sure boundary conditions are satisfied.
     * In the following steps, we will solve for the Newton update so zero
     * constraints are used.
     */
    std::pair<unsigned int, double> solve(const bool use_nonzero_constraints);
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

    /*! \brief Run the simulation for one time step.
     *
     *  If the Dirichlet BC is time-dependent, nonzero constraints must be
     * applied
     *  at every first Newton iteration in every time step. If it is not, only
     *  apply nonzero constraints at the first iteration in the first time step.
     *  A boolean argument controls whether nonzero constraints should be
     *  applied in a certain time step.
     */
    void run_one_step(bool apply_nonzero_constraints);

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
    PETScWrappers::MPI::SparseMatrix Pvv;
    PETScWrappers::MPI::SparseMatrix Abs_A_matrix;
    PETScWrappers::MPI::SparseMatrix schur_matrix;
    PETScWrappers::MPI::SparseMatrix B2pp_matrix;

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
    std::shared_ptr<BlockIncompSchurPreconditioner> preconditioner;

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
                        const double t) const;
      double time;
      double dt;
      bool increment;
    };

    /** \brief Incomplete Schur Complement Block Preconditioner
     * The format of this preconditioner is as follow:
     *
     * |Pvv^-1  -Pvv^-1*Avp*Tpp^-1|*|I            0|
     * |                          | |              |
     * |0            Tpp^-1       | |-Apv*Pvv^-1  I|
     * With Pvv the ILU(0) of Avv,
     * and Tpp the incomplete Schur complement.
     * The evaluation for Tpp is in SchurComplementTpp class,
     * and its inverse is solved by performing some GMRES iterations
     * By using B2pp = ILU(0) of (App - Apv*(rowsum|Avv|)^-1*Avp
     * as preconditioner.
     * This preconditioner is proposed in:
     * T. Washio et al., A robust preconditioner for fluid–structure
     * interaction problems, Comput. Methods Appl. Mech. Engrg.
     * 194 (2005) 4027–4047
     */
    class BlockIncompSchurPreconditioner : public Subscriptor
    {
    public:
      /// Constructor.
      BlockIncompSchurPreconditioner(
        TimerOutput &timer,
        const std::vector<IndexSet> &owned_partitioning,
        const PETScWrappers::MPI::BlockSparseMatrix &system,
        PETScWrappers::MPI::SparseMatrix &absA,
        PETScWrappers::MPI::SparseMatrix &schur,
        PETScWrappers::MPI::SparseMatrix &B2pp);

      /// The matrix-vector multiplication must be defined.
      void vmult(PETScWrappers::MPI::BlockVector &dst,
                 const PETScWrappers::MPI::BlockVector &src) const;
      /// Accessors for the blocks of the system matrix for clearer
      /// representation
      const PETScWrappers::MPI::SparseMatrix &Avv() const
      {
        return system_matrix->block(0, 0);
      }
      const PETScWrappers::MPI::SparseMatrix &Avp() const
      {
        return system_matrix->block(0, 1);
      }
      const PETScWrappers::MPI::SparseMatrix &Apv() const
      {
        return system_matrix->block(1, 0);
      }
      const PETScWrappers::MPI::SparseMatrix &App() const
      {
        return system_matrix->block(1, 1);
      }
      int get_Tpp_itr_count() const { return Tpp_itr; }
      void Erase_Tpp_count() { Tpp_itr = 0; }

    private:
      class SchurComplementTpp;

      /// We would like to time the BlockSchuPreconditioner in detail.
      TimerOutput &timer;

      /// dealii smart pointer checks if an object is still being referenced
      /// when it is destructed therefore is safer than plain reference.
      const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix>
        system_matrix;
      const SmartPointer<PETScWrappers::MPI::SparseMatrix> Abs_A_matrix;
      const SmartPointer<PETScWrappers::MPI::SparseMatrix> schur_matrix;
      const SmartPointer<PETScWrappers::MPI::SparseMatrix> B2pp_matrix;

      PreconditionEuclid Pvv_inverse;
      PreconditionEuclid B2pp_inverse;

      std::shared_ptr<SchurComplementTpp> Tpp;
      // iteration counter for solving Tpp
      mutable int Tpp_itr;
      class SchurComplementTpp : public Subscriptor
      {
      public:
        SchurComplementTpp(TimerOutput &timer,
                           const std::vector<IndexSet> &owned_partitioning,
                           const PETScWrappers::MPI::BlockSparseMatrix &system,
                           const PETScWrappers::PreconditionerBase &Pvvinv);
        void vmult(PETScWrappers::MPI::Vector &dst,
                   const PETScWrappers::MPI::Vector &src) const;

      private:
        TimerOutput &timer;
        const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix>
          system_matrix;
        const PETScWrappers::PreconditionerBase *Pvv_inverse;
        PETScWrappers::MPI::BlockVector dumb_vector;
      };
    };
  };
} // namespace Fluid

#endif
