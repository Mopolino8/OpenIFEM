#ifndef COMP_FLUID
#define COMP_FLUID

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
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

// To transfer solutions between meshes, this file is included:
#include <deal.II/numerics/solution_transfer.h>

// This file includes UMFPACK: the direct solver:
#include <deal.II/lac/sparse_direct.h>

// And the one for ILU preconditioner:
#include <deal.II/lac/sparse_ilu.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "parameters.h"
#include "utilities.h"

template <int>
class FSI;

namespace Fluid
{
  using namespace dealii;

  /*! \brief The incompressible Navier Stokes equation solver.
   *
   * This program is built upon dealii tutorials step-57, step-22, step-20.
   * Although the density does not matter in the incompressible flow, we still
   * include it in the formulation in order to be consistent with the
   * slightly compressible flow. Correspondingly the viscosity represents
   * the dynamic visocity \f$\mu\f$ instead of the kinetic visocity \f$\nu\f$,
   * and the pressure block in the solution is the non-normalized pressure.
   *
   * Fully implicit scheme is used for time stepping. Newton's method is applied
   * to solve the nonlinear system, thus the actual dofs being solved is the
   * velocity and pressure increment.
   *
   * The final linear system to be solved is nonsymmetric. GMRES solver with
   * Grad-Div right preconditioner is applied, which does modify the linear
   * system
   * a little bit, and requires the velocity shape functions to be one order
   * higher
   * than that of the pressure.
   */
  template <int dim>
  class CompressibleFluid
  {
  public:
    //! Constructor
    CompressibleFluid(Triangulation<dim> &, const Parameters::AllParameters &);

    //! Run the simulation
    void run();

    //! Destructor
    ~CompressibleFluid() { timer.print_summary(); }

    //! Return the solution for testing.
    BlockVector<double> get_current_solution() const;

  private:
    class BoundaryValues;
    class TimeDependentBoundaryValues;
    class SigmaPMLField;
    class BlockIncompSchurPreconditioner;
    struct CellProperty;

    //! Set up the dofs based on the finite element and renumber them.
    void setup_dofs();

    //! Set up the nonzero and zero constraints.
    void make_constraints();

    //! Initialize the cell properties, which only matters in FSI applications.
    void setup_cell_property();

    /// Specify the sparsity pattern and reinit matrices and vectors based on
    /// the dofs and constraints.
    void initialize_system();

    /*! \brief Assemble the system matrix, mass mass matrix, and the RHS.
     *
     *  Since backward Euler method is used, the linear system must be
     * reassembled
     *  at every Newton iteration. The Dirichlet BCs are applied at the same
     * time
     *  as the cell matrix and rhs are distributed to the global matrix and rhs,
     *  which is optimal according to the deal.II documentation.
     *  The boolean argument is used to determine whether nonzero constraints
     *  or zero constraints should be used.
     */
    void assemble(const bool use_nonzero_constraints);

    /*! \brief Solve the linear system using FGMRES solver plus block
     * preconditioner.
     *
     *  After solving the linear system, the same ConstraintMatrix as used
     *  in assembly must be used again, to set the solution to the right value
     *  at the constrained dofs.
     */
    std::pair<unsigned int, double> solve(const bool use_nonzero_constraints);

    /// Mesh adaption.
    void refine_mesh(const unsigned int, const unsigned int);

    /// Output in vtu format.
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
    double SigmaPMLMax;
    double PMLLength;
    const unsigned int degree;
    std::vector<types::global_dof_index> dofs_per_block;

    Triangulation<dim> &triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    QGauss<dim> volume_quad_formula;
    QGauss<dim - 1> face_quad_formula;

    ConstraintMatrix zero_constraints;
    ConstraintMatrix nonzero_constraints;

    BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    SparsityPattern schur_pattern;
    SparseMatrix<double> schur_matrix;
    SparsityPattern Tpp_pattern;
    SparseMatrix<double> B2pp_matrix;

    /// The latest known solution.
    BlockVector<double> present_solution;
    /// The increment at a certain Newton iteration.
    BlockVector<double> newton_update;
    /**
     * The latest know solution plus the cumulation of all newton_updates
     * in the current time step, which approaches to the new present_solution.
     */
    BlockVector<double> evaluation_point;
    BlockVector<double> system_rhs;

    const double tolerance;
    const unsigned int max_iteration;

    Utils::Time time;
    mutable TimerOutput timer;

    Parameters::AllParameters parameters;

    /// The BlockSchurPreconditioner for the entire system.
    std::shared_ptr<BlockIncompSchurPreconditioner> preconditioner;

    CellDataStorage<typename Triangulation<dim>::cell_iterator, CellProperty>
      cell_property;

    /*! \brief Helper class to specify space/time-dependent Dirichlet BCs,
     *         as the input file can only handle constant BC values.
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
      TimeDependentBoundaryValues() : Function<dim>(dim + 1) { time = 0; }
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
      double increment;
    };

    /** \brief sigmaPML field
     * This sigma PML indicates the strength of the PML field.
     * Usually it is a quadratic increasing function with space.
     * the larger the sigma PML is, the stronger absorption is
     * expected.
     */
    class SigmaPMLField : public Function<dim>
    {
    public:
      SigmaPMLField(double sig, double l)
        : Function<dim>(), SigmaPMLMax(sig), PMLLength(l)
      {
      }
      virtual double value(const Point<dim> &p,
                           const unsigned int component = 0) const;
      virtual void value_list(const std::vector<Point<dim>> &points,
                              std::vector<double> &values,
                              const unsigned int component = 0) const;

    private:
      double SigmaPMLMax;
      double PMLLength;
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
      BlockIncompSchurPreconditioner(TimerOutput &timer,
                                     const BlockSparseMatrix<double> &system,
                                     SparseMatrix<double> &schur,
                                     SparseMatrix<double> &B2pp);
      void vmult(BlockVector<double> &dst,
                 const BlockVector<double> &src) const;
      const SparseMatrix<double> &Avv() const
      {
        return system_matrix->block(0, 0);
      }
      const SparseMatrix<double> &Avp() const
      {
        return system_matrix->block(0, 1);
      }
      const SparseMatrix<double> &Apv() const
      {
        return system_matrix->block(1, 0);
      }
      const SparseMatrix<double> &App() const
      {
        return system_matrix->block(1, 1);
      }
      int get_Tpp_itr_count() const { return Tpp_itr; }
      void Erase_Tpp_count() { Tpp_itr = 0; }

    private:
      class SchurComplementTpp;

      /// We would like to time the BlockSchuPreconditioner in detail.
      TimerOutput &timer;

      const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
      const SmartPointer<SparseMatrix<double>> schur_matrix;
      const SmartPointer<SparseMatrix<double>> B2pp_matrix;
      SparseILU<double> Pvv_inverse;
      SparseILU<double> B2pp_inverse;
      std::shared_ptr<SchurComplementTpp> Tpp;
      mutable int Tpp_itr; // iteration counter for solving Tpp
      class SchurComplementTpp : public Subscriptor
      {
      public:
        SchurComplementTpp(TimerOutput &timer,
                           const BlockSparseMatrix<double> &system,
                           const SparseILU<double> &Pvvinv);
        void vmult(Vector<double> &dst, const Vector<double> &src) const;
        const SparseMatrix<double> &Avv() const
        {
          return system_matrix->block(0, 0);
        }
        const SparseMatrix<double> &Avp() const
        {
          return system_matrix->block(0, 1);
        }
        const SparseMatrix<double> &Apv() const
        {
          return system_matrix->block(1, 0);
        }
        const SparseMatrix<double> &App() const
        {
          return system_matrix->block(1, 1);
        }

      private:
        TimerOutput &timer;
        const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
        const SmartPointer<const SparseILU<double>> Pvv_inverse;
      };
    };

    /// A data structure that caches the real/artificial fluid indicator,
    /// FSI stress, and FSI acceleration terms at quadrature points, that
    /// will only be used in FSI simulations.
    struct CellProperty
    {
      int indicator; //!< Domain indicator: 1 for artificial fluid 0 for real
                     //! fluid.
      Tensor<1, dim> fsi_acceleration; //!< The acceleration term in FSI force.
      SymmetricTensor<2, dim> fsi_stress; //!< The stress term in FSI force.
    };
  };
} // namespace Fluid

#endif