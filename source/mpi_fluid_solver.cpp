#include "mpi_fluid_solver.h"

namespace Fluid
{
  namespace MPI
  {
    template <int dim>
    PETScWrappers::MPI::BlockVector
    FluidSolver<dim>::get_current_solution() const
    {
      return present_solution;
    }

    template <int dim>
    FluidSolver<dim>::FluidSolver(
      parallel::distributed::Triangulation<dim> &tria,
      const Parameters::AllParameters &parameters,
      std::shared_ptr<Function<dim>> bc)
      : triangulation(tria),
        fe(FE_Q<dim>(parameters.fluid_velocity_degree),
           dim,
           FE_Q<dim>(parameters.fluid_pressure_degree),
           1),
        dof_handler(triangulation),
        volume_quad_formula(parameters.fluid_velocity_degree + 1),
        face_quad_formula(parameters.fluid_velocity_degree + 1),
        parameters(parameters),
        mpi_communicator(MPI_COMM_WORLD),
        pcout(std::cout,
              Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
        time(parameters.end_time,
             parameters.time_step,
             parameters.output_interval,
             parameters.refinement_interval),
        timer(
          mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times),
        boundary_values(bc)
    {
    }

    template <int dim>
    void FluidSolver<dim>::setup_dofs()
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
      owned_partitioning[0] =
        dof_handler.locally_owned_dofs().get_view(0, dof_u);
      owned_partitioning[1] =
        dof_handler.locally_owned_dofs().get_view(dof_u, dof_u + dof_p);

      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);

      relevant_partitioning.resize(2);
      relevant_partitioning[0] = locally_relevant_dofs.get_view(0, dof_u);
      relevant_partitioning[1] =
        locally_relevant_dofs.get_view(dof_u, dof_u + dof_p);

      pcout << "   Number of active fluid cells: "
            << triangulation.n_global_active_cells() << std::endl
            << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << " (" << dof_u << '+' << dof_p << ')' << std::endl;
    }

    template <int dim>
    void FluidSolver<dim>::make_constraints()
    {
      // In Newton's scheme, we first apply the boundary condition on the
      // solution obtained from the initial step. To make sure the boundary
      // conditions remain satisfied during Newton's iteration, zero boundary
      // conditions are used for the update \f$\delta u^k\f$. Therefore we set
      // up two different constraint objects. Dirichlet boundary conditions are
      // applied to both boundaries 0 and 1.

      // For inhomogeneous BC, only constant input values can be read from
      // the input file. If time or space dependent Dirichlet BCs are
      // desired, they must be implemented in BoundaryValues.
      {
        nonzero_constraints.clear();
        zero_constraints.clear();
        nonzero_constraints.reinit(locally_relevant_dofs);
        zero_constraints.reinit(locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints(dof_handler,
                                                nonzero_constraints);
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
            if (parameters.use_hard_coded_values == 1)
              {
                VectorTools::interpolate_boundary_values(dof_handler,
                                                         id,
                                                         *boundary_values,
                                                         nonzero_constraints,
                                                         ComponentMask(mask));
              }
            else
              {
                VectorTools::interpolate_boundary_values(
                  dof_handler,
                  id,
                  Functions::ConstantFunction<dim>(augmented_value),
                  nonzero_constraints,
                  ComponentMask(mask));
              }
            VectorTools::interpolate_boundary_values(
              dof_handler,
              id,
              Functions::ZeroFunction<dim>(dim + 1),
              zero_constraints,
              ComponentMask(mask));
          }
      }
      nonzero_constraints.close();
      zero_constraints.close();
    }

    template <int dim>
    void FluidSolver<dim>::setup_cell_property()
    {
      pcout << "   Setting up cell property..." << std::endl;
      const unsigned int n_q_points = volume_quad_formula.size();
      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              cell_property.initialize(cell, n_q_points);
              const std::vector<std::shared_ptr<CellProperty>> p =
                cell_property.get_data(cell);
              Assert(p.size() == n_q_points,
                     ExcMessage("Wrong number of cell property!"));
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  p[q]->indicator = 0;
                  p[q]->fsi_acceleration = 0;
                  p[q]->fsi_stress = 0;
                }
            }
        }
    }

    template <int dim>
    void FluidSolver<dim>::initialize_system()
    {
      system_matrix.clear();
      mass_matrix.clear();
      mass_schur.clear();

      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
      sparsity_pattern.copy_from(dsp);
      SparsityTools::distribute_sparsity_pattern(
        dsp,
        dof_handler.locally_owned_dofs_per_processor(),
        mpi_communicator,
        locally_relevant_dofs);

      system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
      mass_matrix.reinit(owned_partitioning, dsp, mpi_communicator);

      // Compute the sparsity pattern for mass schur in advance.
      // The only nonzero block is (1, 1), which is the same as \f$BB^T\f$.
      BlockDynamicSparsityPattern schur_dsp(dofs_per_block, dofs_per_block);
      schur_dsp.block(1, 1).compute_mmult_pattern(sparsity_pattern.block(1, 0),
                                                  sparsity_pattern.block(0, 1));
      mass_schur.reinit(owned_partitioning, schur_dsp, mpi_communicator);

      // present_solution is ghosted because it is used in the
      // output and mesh refinement functions.
      present_solution.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);
      // system_rhs is non-ghosted because it is only used in the linear
      // solver and residual evaluation.
      system_rhs.reinit(owned_partitioning, mpi_communicator);

      // Cell property
      setup_cell_property();
    }

    template <int dim>
    void FluidSolver<dim>::refine_mesh(const unsigned int min_grid_level,
                                       const unsigned int max_grid_level)
    {
      TimerOutput::Scope timer_section(timer, "Refine mesh");

      Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
      FEValuesExtractors::Vector velocity(0);
      KellyErrorEstimator<dim>::estimate(dof_handler,
                                         face_quad_formula,
                                         typename FunctionMap<dim>::type(),
                                         present_solution,
                                         estimated_error_per_cell,
                                         fe.component_mask(velocity));
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
      PETScWrappers::MPI::BlockVector tmp;
      tmp.reinit(owned_partitioning, mpi_communicator);
      tmp = 0;
      trans.interpolate(tmp);
      nonzero_constraints.distribute(tmp); // Is this line necessary?
      present_solution = tmp;
    }

    template <int dim>
    void FluidSolver<dim>::output_results(const unsigned int output_index) const
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

      // Partition
      Vector<float> subdomain(triangulation.n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        {
          subdomain(i) = triangulation.locally_owned_subdomain();
        }
      data_out.add_data_vector(subdomain, "subdomain");

      // Indicator
      Vector<float> ind(triangulation.n_active_cells());
      int cnt = 0;
      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              auto p = cell_property.get_data(cell);
              bool artificial = false;
              for (auto ptr : p)
                {
                  if (ptr->indicator == 1)
                    {
                      artificial = true;
                      break;
                    }
                }
              ind[cnt++] = artificial;
            }
        }
      data_out.add_data_vector(ind, "Indicator");
      data_out.build_patches(parameters.fluid_velocity_degree);

      std::string basename =
        "fluid" + Utilities::int_to_string(output_index, 6) + "-";

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
          std::ofstream pvd_output("fluid.pvd");
          DataOutBase::write_pvd_record(pvd_output, times_and_names);
        }
    }

    template class FluidSolver<2>;
    template class FluidSolver<3>;
  } // namespace MPI
} // namespace Fluid
