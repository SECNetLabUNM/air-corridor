#!/bin/bash
source /home/kun/anaconda3/bin/activate torch

# Current time for logging or file naming
current_time=$(date +"%Y%m%d%H%M%S")

# specified as the folder to track intermedia results and performance
desired_directory="/mnt/storage/result"

# Create directory if it doesn't exist
if [ ! -d "$desired_directory" ]; then
    mkdir -p "$desired_directory"
fi

cd "$desired_directory"

# Parameters initialization
param1_values=(5) # num of agents
param2_values=(8)
param3_values=(16)
param4_values=('True')  # token query, work for trans only
param5_values=('True')  # reduce state or original state, original state is not well maintained
param6_values=(1 2 ) # num of encoder
param7_values=(1) # time slot
param8_values=(13) #reset level;     level 2: random 1 piece;   level 3: random 2 pieces; level 13: random 3 pieces;
param9_values=('fc')
param10_values=(True)  # whether to share layers
param11_values=(1.0 )  # beta base
param12_values=(1.1)  # beta range
param13_values=(1)  # num of future corridors in state, at least 1
param14_values=(0.3 )  # num of future corridors in state, at least 1
param15_values=(1101)  # num of future corridors in state, at least 1
param16_values=('True' 'False')  # num of future corridors in state, at least 1
max_concurrent=100
concurrent_processes=0
num_executions=1

# Task distribution counters
num_task_gpu0=0
num_task_gpu1=0

# Main loop for parameter combination execution
for i in $(seq $num_executions); do
  for param1 in "${param1_values[@]}"; do
    for param2 in "${param2_values[@]}"; do
      for param3 in "${param3_values[@]}"; do
        for param4 in "${param4_values[@]}"; do
          for param5 in "${param5_values[@]}"; do
            for param6 in "${param6_values[@]}"; do
              for param7 in "${param7_values[@]}"; do
                for param8 in "${param8_values[@]}"; do
                  for param9 in "${param9_values[@]}"; do
                    for param10 in "${param10_values[@]}"; do
                      for param11 in "${param11_values[@]}"; do
                        for param12 in "${param12_values[@]}"; do
                          for param13 in "${param13_values[@]}"; do
                            for param14 in "${param14_values[@]}"; do
                              for param15 in "${param15_values[@]}"; do
                                for param16 in "${param16_values[@]}"; do
                        if awk -v p11="$param11" -v p12="$param12" 'BEGIN{ exit !(p11 == 1e-5 && p12 == 1.1) }'; then
                          continue  # Skip this combination
                        fi

                      gpu_index=0  # Set default GPU index

                      # Construct experiment name
                      exp_name="simulation:dynamic_${param16}index_${param15}acc${param14}_future${param13}_share${param10}_mod${param9}_horizon${param2}_batch${param3}_enc${param6}_dt${param7}_space${param5}_level${param8}_capacity${param1}_beta_base${param11}_beta_adaptor_coefficient${param12}"

                      # Run the Python script with parameters
                      CUDA_VISIBLE_DEVICES=$gpu_index python /home/kun/PycharmProjects/air-corridor/rl_multi_3d_trans/main.py \
                          --seed 8 \
                          --time ${current_time} \
                          --multiply_horrizion ${param2} \
                          --multiply_batch ${param3} \
                          --token_query ${param4} \
                          --reduce_space ${param5} \
                          --num_enc ${param6} \
                          --dt ${param7} \
                          --level ${param8} \
                          --curriculum True  \
                          --net_width 256 \
                          --base_difficulty 0.1\
                          --corridor_index_awareness ${param15}\
                          --num_agents ${param1} \
                          --liability True \
                          --collision_free False \
                          --dynamic_minor_radius ${param16} \
                          --num_corridor_in_state ${param13} \
                          --acceleration_max ${param14} \
                          --share_layer_flag ${param10} \
                          --beta_base ${param11} \
                          --beta_adaptor_coefficient ${param12} \
                          --net_model ${param9} \
                          --exp-name ${exp_name}  &

                      # Manage concurrent processes
                      concurrent_processes=$((concurrent_processes + 1))

                      # Limit the number of concurrent processes
                      if [ "$concurrent_processes" -ge "$max_concurrent" ]; then
                        wait
                        concurrent_processes=0
                      fi
                      done
                      done
                      done
                      done
                      done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

# Wait for all background processes to finish
wait