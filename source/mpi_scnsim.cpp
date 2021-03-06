#include "mpi_scnsim.h"

namespace Fluid
{
  namespace MPI
  {
    template <int dim>
    SCnsIM<dim>::BlockIncompSchurPreconditioner::SchurComplementTpp::
      SchurComplementTpp(TimerOutput &timer,
                         const std::vector<IndexSet> &owned_partitioning,
                         const PETScWrappers::MPI::BlockSparseMatrix &system,
                         const PETScWrappers::PreconditionerBase &Pvvinv)
      : timer(timer), system_matrix(&system), Pvv_inverse(&Pvvinv)
    {
      dumb_vector.reinit(owned_partitioning,
                         system_matrix->get_mpi_communicator());
    }

    template <int dim>
    void SCnsIM<dim>::BlockIncompSchurPreconditioner::SchurComplementTpp::vmult(
      PETScWrappers::MPI::Vector &dst,
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
    SCnsIM<dim>::BlockIncompSchurPreconditioner::BlockIncompSchurPreconditioner(
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
      RowSumAvv.reinit(owned_partitioning,
                       system_matrix->get_mpi_communicator());
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
      std::vector<double> cache_vector(ReverseRowSum.block(0).local_size());
      std::vector<unsigned int> cache_rows(ReverseRowSum.block(0).local_size());
      for (auto r = ReverseRowSum.block(0).local_range().first;
           r < ReverseRowSum.block(0).local_range().second;
           ++r)
        {
          cache_vector.push_back(1/(RowSumAvv.block(0)(r)));
          cache_rows.push_back(r);
        }
      ReverseRowSum.block(0).set(cache_rows, cache_vector);
      ReverseRowSum.compress(VectorOperation::insert);

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
    void SCnsIM<dim>::BlockIncompSchurPreconditioner::vmult(
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
    SCnsIM<dim>::SCnsIM(parallel::distributed::Triangulation<dim> &tria,
                        const Parameters::AllParameters &parameters,
                        std::shared_ptr<Function<dim>> bc,
                        std::shared_ptr<Function<dim>> pml)
      : FluidSolver<dim>(tria, parameters, bc), sigma_pml_field(pml)
    {
      AssertThrow(parameters.fluid_velocity_degree ==
                    parameters.fluid_pressure_degree,
                  ExcMessage("Velocity degree must the same as pressure!"));
    }

    template <int dim>
    void SCnsIM<dim>::initialize_system()
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

      // Cell property
      setup_cell_property();
    }

    template <int dim>
    void SCnsIM<dim>::assemble(const bool use_nonzero_constraints)
    {
      TimerOutput::Scope timer_section(timer, "Assemble system");

      const double viscosity = parameters.viscosity;
      const double rho = parameters.fluid_rho;

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
      std::vector<double> sigma_pml(n_q_points);

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
              auto p = cell_property.get_data(cell);

              fe_values.reinit(cell);

              local_matrix = 0;
              local_rhs = 0;

              fe_values[velocities].get_function_values(
                evaluation_point, current_velocity_values);

              fe_values[velocities].get_function_gradients(
                evaluation_point, current_velocity_gradients);

              fe_values[pressure].get_function_values(evaluation_point,
                                                      current_pressure_values);

              fe_values[pressure].get_function_gradients(
                evaluation_point, current_pressure_gradients);

              fe_values[velocities].get_function_values(
                present_solution, present_velocity_values);

              fe_values[pressure].get_function_values(present_solution,
                                                      present_pressure_values);

              sigma_pml_field->value_list(
                fe_values.get_quadrature_points(), sigma_pml, 0);

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  const int ind = p[q]->indicator;
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
                  for (unsigned int a = 0;
                       a < dofs_per_cell / fe.dofs_per_vertex;
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
                          // PML attenuation
                          local_matrix(i, j) +=
                            (rho * sigma_pml[q] * phi_u[j] * phi_u[i] +
                             sigma_pml[q] * phi_p[j] * phi_p[i] /
                               (cp_to_cv * atm)) *
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
                             // SUPG PML
                             tau_SUPG * rho * current_velocity_values[q] *
                               grad_phi_u[i] * sigma_pml[q] * phi_u[j] +
                             tau_SUPG * rho * phi_u[j] * grad_phi_u[i] *
                               sigma_pml[q] * current_velocity_values[q] +
                             // PSPG Convection
                             tau_PSPG * rho * grad_phi_p[i] *
                               (phi_u[j] * current_velocity_gradients[q]) +
                             tau_PSPG * rho * grad_phi_p[i] *
                               (current_velocity_values[q] * grad_phi_u[j]) +
                             // PSPG Acceleration
                             tau_PSPG * rho * grad_phi_p[i] * phi_u[j] /
                               time.get_delta_t() +
                             // PSPG Pressure
                             tau_PSPG * grad_phi_p[i] * grad_phi_p[j] +
                             // PSPG PML
                             tau_PSPG * rho * grad_phi_p[i] * sigma_pml[q] *
                               phi_u[j]) *
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
                             phi_u[j] * current_pressure_gradients[q] *
                               phi_p[i] +
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
                        -(rho * sigma_pml[q] * current_velocity_values[q] *
                            phi_u[i] +
                          sigma_pml[q] * current_pressure_values[q] * phi_p[i] /
                            (cp_to_cv * atm)) *
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
                             current_pressure_gradients[q] +
                             rho * sigma_pml[q] * current_velocity_values[q]) +
                          (tau_PSPG * grad_phi_p[i]) *
                            (rho * ((current_velocity_values[q] -
                                     present_velocity_values[q]) /
                                      time.get_delta_t() +
                                    current_velocity_values[q] *
                                      current_velocity_gradients[q]) +
                             current_pressure_gradients[q] +
                             rho * sigma_pml[q] * current_velocity_values[q])) *
                        fe_values.JxW(q);
                      if (ind == 1)
                        {
                          local_rhs(i) +=
                            (scalar_product(grad_phi_u[i], p[q]->fsi_stress) +
                             (p[q]->fsi_acceleration * rho * phi_u[i])) *
                            fe_values.JxW(q);
                        }
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
                                  local_rhs(i) += -(
                                    fe_face_values[velocities].value(i, q) *
                                    fe_face_values.normal_vector(q) *
                                    boundary_values_p * fe_face_values.JxW(q));
                                }
                            }
                        }
                    }
                }

              cell->get_dof_indices(local_dof_indices);

              const ConstraintMatrix &constraints_used = use_nonzero_constraints
                                                           ? nonzero_constraints
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
    SCnsIM<dim>::solve(const bool use_nonzero_constraints)
    {
      // This section includes the work done in the preconditioner
      // and GMRES solver.
      TimerOutput::Scope timer_section(timer, "Solve linear system");
      preconditioner.reset(
        new BlockIncompSchurPreconditioner(timer,
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
        use_nonzero_constraints ? nonzero_constraints : zero_constraints;
      constraints_used.distribute(newton_update);

      return {solver_control.last_step(), solver_control.last_value()};
    }

    template <int dim>
    void SCnsIM<dim>::run_one_step(bool apply_nonzero_constraints,
                                   bool assemble_system)
    {
      (void)assemble_system;
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
      while (relative_residual > parameters.fluid_tolerance &&
             current_residual > 1e-14)
        {
          AssertThrow(outer_iteration < parameters.fluid_max_iterations,
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
    void SCnsIM<dim>::run()
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
          if (parameters.use_hard_coded_values)
            {
              // Only for time dependent BCs!
              // Advance the time by delta_t and make constraints
              boundary_values->advance_time(time.get_delta_t());
              make_constraints();
              run_one_step(true);
            }
          else
            run_one_step(false);
        }
    }
    template class SCnsIM<2>;
    template class SCnsIM<3>;
  } // namespace MPI
} // namespace Fluid