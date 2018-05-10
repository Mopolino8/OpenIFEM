#include "mpi_compressibleFluid.h"

namespace Fluid
{
  template <int dim>
  double ParallelCompressibleFluid<dim>::TimeDependentBoundaryValues::value(
    const Point<dim> &p, const unsigned int component) const
  {
    if (increment)
      {
        return time_value(p, component, time) -
               time_value(p, component, time - dt);
      }
    else
      {
        return time_value(p, component, time);
      }
  }

  template <int dim>
  void
  ParallelCompressibleFluid<dim>::TimeDependentBoundaryValues::vector_value(
    const Point<dim> &p, Vector<double> &values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = TimeDependentBoundaryValues::value(p, c);
  }

  template <int dim>
  double
  ParallelCompressibleFluid<dim>::TimeDependentBoundaryValues::time_value(
    const Point<dim> &p, const unsigned int component, const double t) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));
    if (component == 0 && std::abs(p[0]) < 1e-10)
      {
        return 6.0 * exp(-0.5 * pow((t - 0.5e-4) / 0.15e-4, 2));
      }
    return 0;
  }

  template <int dim>
  double ParallelCompressibleFluid<dim>::BoundaryValues::value(
    const Point<dim> &p, const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));
    double left_boundary = (dim == 2 ? 0.3 : 0.0);
    if (component == 0 && std::abs(p[0] - left_boundary) < 1e-10)
      {
        double U = 1.5;
        double y = p[1];
        double value = 4 * U * y * (0.41 - y) / (0.41 * 0.41);
        if (dim == 3)
          {
            value *= 4 * p[2] * (0.41 - p[2]);
          }
        return value;
      }
    return 0;
  }

  template <int dim>
  void ParallelCompressibleFluid<dim>::BoundaryValues::vector_value(
    const Point<dim> &p, Vector<double> &values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = BoundaryValues::value(p, c);
  }

  template <int dim>
  ParallelCompressibleFluid<dim>::BlockIncompSchurPreconditioner::
    SchurComplementTpp::SchurComplementTpp(
      TimerOutput &timer,
      const std::vector<IndexSet> &owned_partitioning,
      const PETScWrappers::MPI::BlockSparseMatrix &system,
      const PETScWrappers::PreconditionerBase &Pvvinv)
    : timer(timer), system_matrix(&system), Pvv_inverse(&Pvvinv)
  {
    dumb_vector.reinit(owned_partitioning,
                       system_matrix->get_mpi_communicator());
  }

  template <int dim>
  void ParallelCompressibleFluid<dim>::BlockIncompSchurPreconditioner::
    SchurComplementTpp::vmult(PETScWrappers::MPI::Vector &dst,
                              const PETScWrappers::MPI::Vector &src) const
  {
    TimerOutput::Scope timer_section(timer, "Tpp vmult");
    // this is the exact representation of Tpp = App - Apv * Pvv * Avp.
    PETScWrappers::MPI::Vector tmp1(dumb_vector.block(0)),
      tmp2(dumb_vector.block(0)), tmp3(src);
    system_matrix->block(0, 1).vmult(tmp1, src);
    Pvv_inverse->vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(tmp3, tmp2);
    system_matrix->block(1, 1).vmult(dst, src);
    dst -= tmp3;
  }

  template <int dim>
  ParallelCompressibleFluid<dim>::BlockIncompSchurPreconditioner::
    BlockIncompSchurPreconditioner(
      TimerOutput &timer,
      const std::vector<IndexSet> &owned_partitioning,
      const PETScWrappers::MPI::BlockSparseMatrix &system,
      PETScWrappers::MPI::SparseMatrix &absA,
      PETScWrappers::MPI::SparseMatrix &schur,
      PETScWrappers::MPI::SparseMatrix &B2pp)
    : timer(timer),
      system_matrix(&system),
      Abs_A_matrix(&absA),
      schur_matrix(&schur),
      B2pp_matrix(&B2pp),
      Tpp_itr(0)
  {
    // Initialize the Pvv inverse (the ILU(0) factorization of Avv)
    Pvv_inverse.initialize(system_matrix->block(0, 0));
    // Initialize Tpp
    Tpp.reset(new SchurComplementTpp(
      timer, owned_partitioning, *system_matrix, Pvv_inverse));

    // Compute B2pp matrix App - Apv*rowsum(|Avv|)^(-1)*Avp
    // as the preconditioner to solve Tpp^-1
    PETScWrappers::MPI::BlockVector IdentityVector, RowSumAvv, ReverseRowSum;
    IdentityVector.reinit(owned_partitioning,
                          system_matrix->get_mpi_communicator());
    RowSumAvv.reinit(owned_partitioning, system_matrix->get_mpi_communicator());
    ReverseRowSum.reinit(owned_partitioning,
                         system_matrix->get_mpi_communicator());
    // Want to set ReverseRowSum to 1 to calculate the Rowsum first
    IdentityVector.block(0) = 1;
    // iterate the Avv matrix to set everything to positive.
    Abs_A_matrix->add(1, system_matrix->block(0, 0));
    Abs_A_matrix->compress(VectorOperation::add);

    // local information of the matrix is in unit of row, so we want to know
    // the range of global row indices that the local rank has.
    unsigned int row_start = Abs_A_matrix->local_range().first;
    unsigned int row_end = Abs_A_matrix->local_range().second;
    unsigned int row_range = row_end - row_start;
    // A temporal vector to cache the columns and values to be set.
    std::vector<std::vector<unsigned int>> cache_columns;
    std::vector<std::vector<double>> cache_values;
    cache_columns.resize(row_range);
    cache_values.resize(row_range);
    for (auto r = Abs_A_matrix->local_range().first;
         r < Abs_A_matrix->local_range().second;
         ++r)
      {
        // Allocation of memory for the input values
        cache_columns[r - row_start].resize(Abs_A_matrix->row_length(r));
        cache_values[r - row_start].resize(Abs_A_matrix->row_length(r));
        unsigned int col_count = 0;
        auto itr = Abs_A_matrix->begin(r);
        while (col_count < Abs_A_matrix->row_length(r))
          {
            cache_columns[r - row_start].push_back(itr->column());
            cache_values[r - row_start].push_back(std::abs(itr->value()));
            ++col_count;
            if (col_count == Abs_A_matrix->row_length(r))
              break;
            ++itr;
          }
      }
    for (auto r = Abs_A_matrix->local_range().first;
         r < Abs_A_matrix->local_range().second;
         ++r)
      {
        Abs_A_matrix->set(
          r, cache_columns[r - row_start], cache_values[r - row_start], true);
      }
    Abs_A_matrix->compress(VectorOperation::insert);

    // Compute the diag vector rowsum(|Avv|)^(-1)
    Abs_A_matrix->vmult(RowSumAvv.block(0), IdentityVector.block(0));
    // Reverse the vector and store in ReverseRowSum
    ReverseRowSum.block(0).ratio(IdentityVector.block(0), RowSumAvv.block(0));
    // Compute Schur matrix Apv*rowsum(|Avv|)^(-1)*Avp
    system_matrix->block(1, 0).mmult(
      *schur_matrix, system_matrix->block(0, 1), ReverseRowSum.block(0));
    // Add in numbers to B2pp
    B2pp_matrix->add(-1, *schur_matrix);
    B2pp_matrix->add(1, system_matrix->block(1, 1));
    B2pp_matrix->compress(VectorOperation::add);
    B2pp_inverse.initialize(*B2pp_matrix);
  }

  /**
   * The vmult operation strictly follows the definition of
   * BlockSchurPreconditioner. Conceptually it computes \f$u = P^{-1}v\f$.
   */
  template <int dim>
  void ParallelCompressibleFluid<dim>::BlockIncompSchurPreconditioner::vmult(
    PETScWrappers::MPI::BlockVector &dst,
    const PETScWrappers::MPI::BlockVector &src) const
  {
    // Compute the intermediate vector:
    //      |I           0|*|src(0)| = |src(0)|
    //      |-ApvPvv^-1  I| |src(1)|   |ptmp  |
    /////////////////////////////////////////
    PETScWrappers::MPI::Vector ptmp1(src.block(0)), ptmp(src.block(1));
    Pvv_inverse.vmult(ptmp1, src.block(0));
    this->Apv().vmult(ptmp, ptmp1);
    ptmp *= -1.0;
    ptmp += src.block(1);

    // Compute the final vector:
    //      |Pvv^-1     -Pvv^-1*Avp*Tpp^-1|*|src(0)|
    //      |0           Tpp^-1           | |ptmp  |
    //                        =   |Pvv^-1*src(0) - Pvv^-1*Avp*Tpp^-1*ptmp|
    //                            |Tpp^-1 * ptmp                         |
    //////////////////////////////////////////
    // Compute Tpp^-1 * ptmp first, which is equal to the problem Tpp*x = ptmp
    // Set up initial guess first
    {
      PETScWrappers::MPI::Vector c(ptmp), Sc(ptmp);
      Tpp->vmult(Sc, c);
      double alpha = (ptmp * c) / (Sc * c);
      c *= alpha;
      dst.block(1) = c;
    }
    // Compute the multiplication
    timer.enter_subsection("Solving Tpp");
    SolverControl solver_control(
      ptmp.size(), 1e-3 * ptmp.l2_norm(), true, true);
    GrowingVectorMemory<PETScWrappers::MPI::Vector> vector_memory;
    SolverGMRES<PETScWrappers::MPI::Vector> gmres(
      solver_control,
      vector_memory,
      SolverGMRES<PETScWrappers::MPI::Vector>::AdditionalData(200));
    gmres.solve(*Tpp, dst.block(1), ptmp, B2pp_inverse);
    // B2pp_inverse.vmult(dst.block(1), ptmp);
    // Count iterations for this solver solving Tpp inverse
    Tpp_itr += solver_control.last_step();

    timer.leave_subsection("Solving Tpp");

    // Compute Pvv^-1*src(0) - Pvv^-1*Avp*dst(1)
    PETScWrappers::MPI::Vector utmp1(src.block(0)), utmp2(src.block(0));
    this->Avp().vmult(utmp1, dst.block(1));
    Pvv_inverse.vmult(utmp2, utmp1);
    Pvv_inverse.vmult(dst.block(0), src.block(0));
    dst.block(0) -= utmp2;
  }

  template <int dim>
  PETScWrappers::MPI::BlockVector
  ParallelCompressibleFluid<dim>::get_current_solution() const
  {
    return present_solution;
  }

  template <int dim>
  ParallelCompressibleFluid<dim>::ParallelCompressibleFluid(
    parallel::distributed::Triangulation<dim> &tria,
    const Parameters::AllParameters &parameters)
    : viscosity(parameters.viscosity),
      rho(parameters.fluid_rho),
      degree(parameters.fluid_degree),
      triangulation(tria),
      fe(FE_Q<dim>(degree), dim, FE_Q<dim>(degree), 1),
      dof_handler(triangulation),
      volume_quad_formula(degree + 2),
      face_quad_formula(degree + 2),
      tolerance(parameters.fluid_tolerance),
      max_iteration(parameters.fluid_max_iterations),
      parameters(parameters),
      mpi_communicator(MPI_COMM_WORLD),
      pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
      time(parameters.end_time,
           parameters.time_step,
           parameters.output_interval,
           parameters.refinement_interval),
      timer(
        mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
  {
  }

  template <int dim>
  void ParallelCompressibleFluid<dim>::setup_dofs()
  {
    // The first step is to associate DoFs with a given mesh.
    dof_handler.distribute_dofs(fe);

    // We renumber the components to have all velocity DoFs come before
    // the pressure DoFs to be able to split the solution vector in two blocks
    // which are separately accessed in the block preconditioner.
    DoFRenumbering::Cuthill_McKee(dof_handler);
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    dofs_per_block.resize(2);
    DoFTools::count_dofs_per_block(
      dof_handler, dofs_per_block, block_component);
    unsigned int dof_u = dofs_per_block[0];
    unsigned int dof_p = dofs_per_block[1];

    // This part is new compared to serial code: we need to split up the
    // IndexSet
    // based on how we want to create the block matrices and vectors
    owned_partitioning.resize(2);
    owned_partitioning[0] = dof_handler.locally_owned_dofs().get_view(0, dof_u);
    owned_partitioning[1] =
      dof_handler.locally_owned_dofs().get_view(dof_u, dof_u + dof_p);

    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    relevant_partitioning.resize(2);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, dof_u);
    relevant_partitioning[1] =
      locally_relevant_dofs.get_view(dof_u, dof_u + dof_p);

    pcout << "   Number of active fluid cells: "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
          << dof_u << '+' << dof_p << ')' << std::endl;
  }

  template <int dim>
  void ParallelCompressibleFluid<dim>::make_constraints()
  {

    // In Newton's scheme, we first apply the boundary condition on the solution
    // obtained from the initial step. To make sure the boundary conditions
    // remain
    // satisfied during Newton's iteration, zero boundary conditions are used
    // for
    // the update \f$\delta u^k\f$. Therefore we set up two different constraint
    // objects.
    // Dirichlet boundary conditions are applied to both boundaries 0 and 1.

    // For inhomogeneous BC, only constant input values can be read from
    // the input file. If time or space dependent Dirichlet BCs are
    // desired, this block of code has to be modified.
    {
      nonzero_constraints.clear();
      increment_constraints.clear();
      zero_constraints.clear();
      nonzero_constraints.reinit(locally_relevant_dofs);
      increment_constraints.reinit(locally_relevant_dofs);
      zero_constraints.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
      DoFTools::make_hanging_node_constraints(dof_handler,
                                              increment_constraints);
      DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
      for (auto itr = parameters.fluid_dirichlet_bcs.begin();
           itr != parameters.fluid_dirichlet_bcs.end();
           ++itr)
        {
          // First get the id, flag and value from the input file
          unsigned int id = itr->first;
          unsigned int flag = itr->second.first;
          std::vector<double> value = itr->second.second;

          // To make VectorTools::interpolate_boundary_values happy,
          // a vector of bool and a vector of double which are of size
          // dim + 1 are required.
          std::vector<bool> mask(dim + 1, false);
          std::vector<double> augmented_value(dim + 1, 0.0);
          // 1-x, 2-y, 3-xy, 4-z, 5-xz, 6-yz, 7-xyz
          switch (flag)
            {
            case 1:
              mask[0] = true;
              augmented_value[0] = value[0];
              break;
            case 2:
              mask[1] = true;
              augmented_value[1] = value[0];
              break;
            case 3:
              mask[0] = true;
              mask[1] = true;
              augmented_value[0] = value[0];
              augmented_value[1] = value[1];
              break;
            case 4:
              mask[2] = true;
              augmented_value[2] = value[0];
              break;
            case 5:
              mask[0] = true;
              mask[2] = true;
              augmented_value[0] = value[0];
              augmented_value[2] = value[1];
              break;
            case 6:
              mask[1] = true;
              mask[2] = true;
              augmented_value[1] = value[0];
              augmented_value[2] = value[1];
              break;
            case 7:
              mask[0] = true;
              mask[1] = true;
              mask[2] = true;
              augmented_value[0] = value[0];
              augmented_value[1] = value[1];
              augmented_value[2] = value[2];
              break;
            default:
              AssertThrow(false, ExcMessage("Unrecogonized component flag!"));
              break;
            }
          VectorTools::interpolate_boundary_values(
            dof_handler,
            id,
            // Functions::ConstantFunction<dim>(augmented_value),
            // BoundaryValues(),
            TimeDependentBoundaryValues(
              time.current(), time.get_delta_t(), false),
            nonzero_constraints,
            ComponentMask(mask));
          VectorTools::interpolate_boundary_values(
            dof_handler,
            id,
            TimeDependentBoundaryValues(
              time.current(), time.get_delta_t(), true),
            increment_constraints,
            ComponentMask(mask));
          VectorTools::interpolate_boundary_values(
            dof_handler,
            id,
            Functions::ZeroFunction<dim>(dim + 1),
            zero_constraints,
            ComponentMask(mask));
        }
    }
    nonzero_constraints.close();
    increment_constraints.close();
    zero_constraints.close();
  }

  template <int dim>
  void ParallelCompressibleFluid<dim>::initialize_system()
  {
    preconditioner.reset();
    system_matrix.clear();
    Abs_A_matrix.clear();
    schur_matrix.clear();
    B2pp_matrix.clear();

    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
    sparsity_pattern.copy_from(dsp);
    SparsityTools::distribute_sparsity_pattern(
      dsp,
      dof_handler.locally_owned_dofs_per_processor(),
      mpi_communicator,
      locally_relevant_dofs);

    system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
    Abs_A_matrix.reinit(owned_partitioning[0],
                        owned_partitioning[0],
                        dsp.block(0, 0),
                        mpi_communicator);

    // Compute the sparsity pattern for mass schur in advance.
    // The only nonzero block is (1, 1), which is the same as \f$BB^T\f$.
    DynamicSparsityPattern schur_dsp(dofs_per_block[1], dofs_per_block[1]);
    schur_dsp.compute_mmult_pattern(sparsity_pattern.block(1, 0),
                                    sparsity_pattern.block(0, 1));

    // Compute the pattern for B2pp perconditioner
    for (auto itr = sparsity_pattern.block(1, 1).begin();
         itr != sparsity_pattern.block(1, 1).end();
         ++itr)
      {
        schur_dsp.add(itr->row(), itr->column());
      }

    B2pp_matrix.reinit(owned_partitioning[1],
                       owned_partitioning[1],
                       schur_dsp,
                       mpi_communicator);
    schur_matrix.reinit(owned_partitioning[1],
                        owned_partitioning[1],
                        schur_dsp,
                        mpi_communicator);

    // present_solution is ghosted because it is used in the
    // output and mesh refinement functions.
    present_solution.reinit(
      owned_partitioning, relevant_partitioning, mpi_communicator);
    // newton_update is non-ghosted because the linear solver needs
    // a completely distributed vector.
    newton_update.reinit(owned_partitioning, mpi_communicator);
    // evaluation_point is ghosted because it is used in the assembly.
    evaluation_point.reinit(
      owned_partitioning, relevant_partitioning, mpi_communicator);
    // system_rhs is non-ghosted because it is only used in the linear
    // solver and residual evaluation.
    system_rhs.reinit(owned_partitioning, mpi_communicator);
  }

  template <int dim>
  void
  ParallelCompressibleFluid<dim>::assemble(const bool use_nonzero_constraints)
  {
    TimerOutput::Scope timer_section(timer, "Assemble system");

    system_matrix = 0;
    Abs_A_matrix = 0;
    schur_matrix = 0;
    B2pp_matrix = 0;

    system_rhs = 0;

    FEValues<dim> fe_values(fe,
                            volume_quad_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quad_formula,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int u_dofs = fe.base_element(0).dofs_per_cell;
    const unsigned int p_dofs = fe.base_element(1).dofs_per_cell;
    const unsigned int n_q_points = volume_quad_formula.size();
    const unsigned int n_face_q_points = face_quad_formula.size();

    Assert(u_dofs * dim + p_dofs == dofs_per_cell,
           ExcMessage("Wrong partitioning of dofs!"));

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // For the linearized system, we create temporary storage for current
    // velocity
    // and gradient, current pressure, and present velocity. In practice, they
    // are
    // all obtained through their shape functions at quadrature points.

    std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);
    std::vector<Tensor<2, dim>> current_velocity_gradients(n_q_points);
    std::vector<double> current_pressure_values(n_q_points);
    std::vector<Tensor<1, dim>> current_pressure_gradients(n_q_points);
    std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
    std::vector<double> present_pressure_values(n_q_points);

    std::vector<double> div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double> phi_p(dofs_per_cell);
    std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

    // The parameters that is used in isentropic continuity equation:
    // heat capacity ratio and atmospheric pressure.
    double cp_to_cv = 1.4;
    double atm = 1013250;

    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            local_matrix = 0;
            local_rhs = 0;

            fe_values[velocities].get_function_values(evaluation_point,
                                                      current_velocity_values);

            fe_values[velocities].get_function_gradients(
              evaluation_point, current_velocity_gradients);

            fe_values[pressure].get_function_values(evaluation_point,
                                                    current_pressure_values);

            fe_values[pressure].get_function_gradients(
              evaluation_point, current_pressure_gradients);

            fe_values[velocities].get_function_values(present_solution,
                                                      present_velocity_values);

            fe_values[pressure].get_function_values(present_solution,
                                                    present_pressure_values);

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    div_phi_u[k] = fe_values[velocities].divergence(k, q);
                    grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                    phi_u[k] = fe_values[velocities].value(k, q);
                    phi_p[k] = fe_values[pressure].value(k, q);
                    grad_phi_p[k] = fe_values[pressure].gradient(k, q);
                  }

                // Define the UGN based SUPG parameters (Tezduyar):
                // tau_SUPG and tau_PSPG. They are
                // evaluated based on the results from the last Newton
                // iteration.
                double tau_SUPG, tau_PSPG;
                // the length scale h is the length of the element in the
                // direction
                // of convection
                double h = 0;
                for (unsigned int a = 0; a < dofs_per_cell / fe.dofs_per_vertex;
                     ++a)
                  {
                    h += abs(present_velocity_values[q] *
                             fe_values.shape_grad(a, q));
                  }
                if (h)
                  h = 2 * present_velocity_values[q].norm() / h;
                else
                  h = 0;
                double nu = viscosity / rho;
                double v_norm = present_velocity_values[q].norm();
                if (h)
                  tau_SUPG = 1 / sqrt((pow(2 / time.get_delta_t(), 2) +
                                       pow(2 * v_norm / h, 2) +
                                       pow(4 * nu / pow(h, 2), 2)));
                else
                  tau_SUPG = time.get_delta_t() / 2;
                tau_PSPG = tau_SUPG / rho;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    double current_velocity_divergence =
                      trace(current_velocity_gradients[q]);
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        // Let the linearized diffusion, continuity
                        // terms be written as
                        // the bilinear operator: \f$A = a((\delta{u},
                        // \delta{p}), (\delta{v}, \delta{q}))\f$,
                        // the linearized convection term be: \f$C =
                        // c(u;\delta{u}, \delta{v})\f$,
                        // and the linearized inertial term be:
                        // \f$M = m(\delta{u}, \delta{v})$, then LHS is: $(A
                        // +
                        // C) + M/{\Delta{t}}\f$
                        local_matrix(i, j) +=
                          ((viscosity *
                              scalar_product(grad_phi_u[j], grad_phi_u[i]) +
                            rho * current_velocity_gradients[q] * phi_u[j] *
                              phi_u[i] +
                            rho * grad_phi_u[j] * current_velocity_values[q] *
                              phi_u[i] -
                            div_phi_u[i] * phi_p[j]) +
                           rho * phi_u[i] * phi_u[j] / time.get_delta_t()) *
                          fe_values.JxW(q);
                        // Add SUPG and PSPG stabilization
                        local_matrix(i, j) +=
                          // SUPG Convection
                          (tau_SUPG * rho *
                             (current_velocity_values[q] * grad_phi_u[i]) *
                             (phi_u[j] * current_velocity_gradients[q]) +
                           tau_SUPG * rho *
                             (current_velocity_values[q] * grad_phi_u[i]) *
                             (current_velocity_values[q] * grad_phi_u[j]) +
                           tau_SUPG * rho * (phi_u[j] * grad_phi_u[i]) *
                             (current_velocity_values[q] *
                              current_velocity_gradients[q]) +
                           // SUPG Acceleration
                           tau_SUPG * rho * current_velocity_values[q] *
                             grad_phi_u[i] * phi_u[j] / time.get_delta_t() +
                           tau_SUPG * rho * phi_u[j] * grad_phi_u[i] *
                             (current_velocity_values[q] -
                              present_velocity_values[q]) /
                             time.get_delta_t() +
                           // SUPG Pressure
                           tau_SUPG * current_velocity_values[q] *
                             grad_phi_u[i] * grad_phi_p[j] +
                           tau_SUPG * phi_u[j] * grad_phi_u[i] *
                             current_pressure_gradients[q] +
                           // PSPG Convection
                           tau_PSPG * rho * grad_phi_p[i] *
                             (phi_u[j] * current_velocity_gradients[q]) +
                           tau_PSPG * rho * grad_phi_p[i] *
                             (current_velocity_values[q] * grad_phi_u[j]) +
                           // PSPG Acceleration
                           tau_PSPG * rho * grad_phi_p[i] * phi_u[j] /
                             time.get_delta_t() +
                           // PSPG Pressure
                           tau_PSPG * grad_phi_p[i] * grad_phi_p[j]) *
                          fe_values.JxW(q);
                        // For more clear demonstration, write continuity
                        // equation
                        // separately.
                        // The original strong form is:
                        // \f$p_{,t} + \frac{C_p}{C_v} * (p_0 + p) * (\nabla
                        // \times u) + u (\nabla p) = 0\f$
                        local_matrix(i, j) +=
                          (cp_to_cv * (atm + current_pressure_values[q]) *
                             div_phi_u[j] * phi_p[i] +
                           cp_to_cv * phi_p[j] * current_velocity_divergence *
                             phi_p[i] +
                           current_velocity_values[q] * grad_phi_p[j] *
                             phi_p[i] +
                           phi_u[j] * current_pressure_gradients[q] * phi_p[i] +
                           phi_p[i] * phi_p[j] / time.get_delta_t()) /
                          (cp_to_cv * atm) * fe_values.JxW(q);
                      }

                    // RHS is \f$-(A_{current} + C_{current}) -
                    // M_{present-current}/\Delta{t}\f$.
                    local_rhs(i) +=
                      ((-viscosity *
                          scalar_product(current_velocity_gradients[q],
                                         grad_phi_u[i]) -
                        rho * current_velocity_gradients[q] *
                          current_velocity_values[q] * phi_u[i] +
                        current_pressure_values[q] * div_phi_u[i]) -
                       rho *
                         (current_velocity_values[q] -
                          present_velocity_values[q]) *
                         phi_u[i] / time.get_delta_t()) *
                      fe_values.JxW(q);
                    local_rhs(i) +=
                      -(cp_to_cv * (atm + current_pressure_values[q]) *
                          current_velocity_divergence * phi_p[i] +
                        current_velocity_values[q] *
                          current_pressure_gradients[q] * phi_p[i] +
                        (current_pressure_values[q] -
                         present_pressure_values[q]) *
                          phi_p[i] / time.get_delta_t()) /
                      (cp_to_cv * atm) * fe_values.JxW(q);
                    // Add SUPG and PSPS rhs terms.
                    local_rhs(i) +=
                      -((tau_SUPG * current_velocity_values[q] *
                         grad_phi_u[i]) *
                          (rho * ((current_velocity_values[q] -
                                   present_velocity_values[q]) /
                                    time.get_delta_t() +
                                  current_velocity_values[q] *
                                    current_velocity_gradients[q]) +
                           current_pressure_gradients[q]) +
                        (tau_PSPG * grad_phi_p[i]) *
                          (rho * ((current_velocity_values[q] -
                                   present_velocity_values[q]) /
                                    time.get_delta_t() +
                                  current_velocity_values[q] *
                                    current_velocity_gradients[q]) +
                           current_pressure_gradients[q])) *
                      fe_values.JxW(q);
                  }
              }

            // Impose pressure boundary here if specified, loop over faces on
            // the
            // cell
            // and apply pressure boundary conditions:
            // \f$\int_{\Gamma_n} -p\bold{n}d\Gamma\f$
            if (parameters.n_fluid_neumann_bcs != 0)
              {
                for (unsigned int face_n = 0;
                     face_n < GeometryInfo<dim>::faces_per_cell;
                     ++face_n)
                  {
                    if (cell->at_boundary(face_n) &&
                        parameters.fluid_neumann_bcs.find(
                          cell->face(face_n)->boundary_id()) !=
                          parameters.fluid_neumann_bcs.end())
                      {
                        fe_face_values.reinit(cell, face_n);
                        unsigned int p_bc_id =
                          cell->face(face_n)->boundary_id();
                        double boundary_values_p =
                          parameters.fluid_neumann_bcs[p_bc_id];
                        for (unsigned int q = 0; q < n_face_q_points; ++q)
                          {
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                              {
                                local_rhs(i) +=
                                  -(fe_face_values[velocities].value(i, q) *
                                    fe_face_values.normal_vector(q) *
                                    boundary_values_p * fe_face_values.JxW(q));
                              }
                          }
                      }
                  }
              }

            cell->get_dof_indices(local_dof_indices);

            const ConstraintMatrix &constraints_used = use_nonzero_constraints
                                                         ? increment_constraints
                                                         : zero_constraints;

            constraints_used.distribute_local_to_global(local_matrix,
                                                        local_rhs,
                                                        local_dof_indices,
                                                        system_matrix,
                                                        system_rhs);
          }
      }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  std::pair<unsigned int, double>
  ParallelCompressibleFluid<dim>::solve(const bool use_nonzero_constraints)
  {
    // This section includes the work done in the preconditioner
    // and GMRES solver.
    TimerOutput::Scope timer_section(timer, "Solve linear system");
    preconditioner.reset(new BlockIncompSchurPreconditioner(timer,
                                                            owned_partitioning,
                                                            system_matrix,
                                                            Abs_A_matrix,
                                                            schur_matrix,
                                                            B2pp_matrix));

    SolverControl solver_control(
      system_matrix.m(), 1e-6 * system_rhs.l2_norm(), true);

    // Because PETScWrappers::SolverGMRES requires preconditioner derived
    // from PETScWrappers::PreconditionBase, we use dealii SolverFGMRES.
    GrowingVectorMemory<PETScWrappers::MPI::BlockVector> vector_memory;
    SolverFGMRES<PETScWrappers::MPI::BlockVector> gmres(solver_control,
                                                        vector_memory);

    // The solution vector must be non-ghosted
    gmres.solve(system_matrix, newton_update, system_rhs, *preconditioner);

    const ConstraintMatrix &constraints_used =
      use_nonzero_constraints ? increment_constraints : zero_constraints;
    constraints_used.distribute(newton_update);

    return {solver_control.last_step(), solver_control.last_value()};
  }

  template <int dim>
  void
  ParallelCompressibleFluid<dim>::refine_mesh(const unsigned int min_grid_level,
                                              const unsigned int max_grid_level)
  {
    TimerOutput::Scope timer_section(timer, "Refine mesh");
    pcout << "Refining mesh..." << std::endl;

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    FEValuesExtractors::Vector velocity(0);

    // Evaluate the error
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       face_quad_formula,
                                       typename FunctionMap<dim>::type(),
                                       present_solution,
                                       estimated_error_per_cell,
                                       fe.component_mask(velocity));

    // Set the refine and coarsen flag
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
      triangulation, estimated_error_per_cell, 0.6, 0.4);
    if (triangulation.n_levels() > max_grid_level)
      {
        for (auto cell = triangulation.begin_active(max_grid_level);
             cell != triangulation.end();
             ++cell)
          {
            cell->clear_refine_flag();
          }
      }
    for (auto cell = triangulation.begin_active(min_grid_level);
         cell != triangulation.end_active(min_grid_level);
         ++cell)
      {
        cell->clear_coarsen_flag();
      }

    // Prepare to transfer
    parallel::distributed::SolutionTransfer<dim,
                                            PETScWrappers::MPI::BlockVector>
      trans(dof_handler);

    triangulation.prepare_coarsening_and_refinement();

    trans.prepare_for_coarsening_and_refinement(present_solution);

    // Refine the mesh
    triangulation.execute_coarsening_and_refinement();

    // Reinitialize the system
    setup_dofs();
    make_constraints();
    initialize_system();

    // Transfer solution
    // Need a non-ghosted vector for interpolation
    PETScWrappers::MPI::BlockVector tmp(newton_update);
    tmp = 0;
    trans.interpolate(tmp);
    nonzero_constraints.distribute(tmp);
    present_solution = tmp;
  }

  template <int dim>
  void
  ParallelCompressibleFluid<dim>::run_one_step(bool apply_nonzero_constraints)
  {
    std::cout.precision(6);
    std::cout.width(12);

    if (time.get_timestep() == 0)
      {
        output_results(0);
      }

    time.increment();
    pcout << std::string(96, '*') << std::endl
          << "Time step = " << time.get_timestep()
          << ", at t = " << std::scientific << time.current() << std::endl;

    // Resetting
    double current_residual = 1.0;
    double initial_residual = 1.0;
    double relative_residual = 1.0;
    unsigned int outer_iteration = 0;
    evaluation_point = present_solution;
    while (relative_residual > tolerance && current_residual > 1e-14)
      {
        AssertThrow(outer_iteration < max_iteration,
                    ExcMessage("Too many Newton iterations!"));

        newton_update = 0;

        // Since evaluation_point changes at every iteration,
        // we have to reassemble both the lhs and rhs of the system
        // before solving it.
        // If the Dirichlet BCs are time-dependent, nonzero_constraints
        // should be applied at the first iteration of every time step;
        // if they are time-independent, nonzero_constraints should be
        // applied only at the first iteration of the first time step.
        assemble(apply_nonzero_constraints && outer_iteration == 0);
        auto state = solve(apply_nonzero_constraints && outer_iteration == 0);
        current_residual = system_rhs.l2_norm();

        // Update evaluation_point. Since newton_update has been set to
        // the correct bc values, there is no need to distribute the
        // evaluation_point again. Note we have to use a non-ghosted
        // vector as a buffer in order to do addition.
        PETScWrappers::MPI::BlockVector tmp;
        tmp.reinit(owned_partitioning, mpi_communicator);
        tmp = evaluation_point;
        tmp += newton_update;
        nonzero_constraints.distribute(tmp);
        evaluation_point = tmp;

        if (outer_iteration == 0)
          {
            initial_residual = current_residual;
          }
        relative_residual = current_residual / initial_residual;

        pcout << std::scientific << std::left << " ITR = " << std::setw(2)
              << outer_iteration << " ABS_RES = " << current_residual
              << " REL_RES = " << relative_residual
              << " GMRES_ITR = " << std::setw(3) << state.first
              << " GMRES_RES = " << state.second
              << " INNER_GMRES_ITR = " << std::setw(3)
              << preconditioner->get_Tpp_itr_count() << std::endl;
        outer_iteration++;
      }
    // Newton iteration converges, update time and solution
    present_solution = evaluation_point;
    // Output
    if (time.time_to_output())
      {
        output_results(time.get_timestep());
      }
    if (time.time_to_refine())
      {
        refine_mesh(1, 3);
      }
  }

  template <int dim>
  void ParallelCompressibleFluid<dim>::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    triangulation.refine_global(parameters.global_refinement);
    setup_dofs();
    make_constraints();
    initialize_system();

    // Time loop.
    // use_nonzero_constraints is set to true only at the first time step,
    // which means nonzero_constraints will be applied at the first iteration
    // in the first time step only, and never be used again.
    // This corresponds to time-independent Dirichlet BCs.
    run_one_step(true);
    while (time.end() - time.current() > 1e-12)
      {
        make_constraints();
        // for time dependent problems, run with applying nonzero constraints.
        run_one_step(true);
      }
  }

  template <int dim>
  void ParallelCompressibleFluid<dim>::output_results(
    const unsigned int output_index) const
  {
    TimerOutput::Scope timer_section(timer, "Output results");

    pcout << "Writing results..." << std::endl;
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    // vector to be output must be ghosted
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      {
        subdomain(i) = triangulation.locally_owned_subdomain();
      }
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();

    std::string basename =
      "navierstokes" + Utilities::int_to_string(output_index, 6) + "-";

    std::string filename =
      basename +
      Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
      ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);

    static std::vector<std::pair<double, std::string>> times_and_names;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        for (unsigned int i = 0;
             i < Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          {
            times_and_names.push_back(
              {time.current(),
               basename + Utilities::int_to_string(i, 4) + ".vtu"});
          }
        std::ofstream pvd_output("navierstokes.pvd");
        DataOutBase::write_pvd_record(pvd_output, times_and_names);
      }
  }

  template class ParallelCompressibleFluid<2>;
  template class ParallelCompressibleFluid<3>;
} // namespace Fluid
